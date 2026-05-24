from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .coder import CoderAgent
from .kernelbench import build_kernelbench_prompt, load_problem
from .llm import LLMConfig, OpenAIChatClient
from .planner import PlannerAgent
from .types import CandidateAttempt, KernelBenchProblem, VerificationResult
from .verifier import KernelBenchVerifier


@dataclass
class FixedWorkflowConfig:
    level: int
    problem_id: int
    model: str
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 65536
    reasoning_effort: str = ""
    kernelbench_root: str = ""
    output_root: str = "runs"
    run_name: str = ""
    prompt_option: str = "zero_shot_safe_glue"
    gpu_arch: str = "Blackwell"
    target_sm: str = ""
    device: int | None = None
    max_attempts: int = 3
    max_replans: int = 2
    replan_after_failed_attempts: int = 2
    replan_after_stagnant_attempts: int = 2
    target_speedup: float = 1.0
    num_correct_trials: int = 5
    num_perf_trials: int = 10
    measure_performance: bool = True
    verifier_timeout_s: int = 1800


class StitchCUDAWorkflow:
    """Fixed KernelBench planner -> coder -> verifier workflow."""

    def __init__(self, config: FixedWorkflowConfig):
        self.config = config
        llm = OpenAIChatClient(
            LLMConfig(
                model=config.model,
                api_base=config.api_base,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                reasoning_effort=config.reasoning_effort,
            )
        )
        self.planner = PlannerAgent(llm)
        self.coder = CoderAgent(llm, kernelbench_root=config.kernelbench_root)
        self.verifier = KernelBenchVerifier(
            kernelbench_root=config.kernelbench_root,
            gpu_arch=config.gpu_arch,
            device=config.device,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            measure_performance=config.measure_performance,
            timeout_s=config.verifier_timeout_s,
        )

    def run(self) -> dict[str, Any]:
        cfg = self.config
        problem = load_problem(
            level=cfg.level,
            problem_id=cfg.problem_id,
            kernelbench_root=cfg.kernelbench_root,
        )
        run_dir = self._create_run_dir(problem)
        hardware = self._hardware_summary()

        (run_dir / "reference.py").write_text(problem.reference_code, encoding="utf-8")
        _save_json(run_dir / "problem.json", problem.to_dict())
        _save_json(run_dir / "hardware.json", hardware)
        _save_json(run_dir / "config.json", _config_to_json(cfg))

        kb_prompt = build_kernelbench_prompt(
            problem.reference_code,
            prompt_option=cfg.prompt_option,
            kernelbench_root=cfg.kernelbench_root,
            gpu_name=str(hardware.get("gpu_name", "")),
            target_arch_context=_target_arch_context(hardware),
            include_hardware=False,
        )
        (run_dir / "kernelbench_prompt.txt").write_text(kb_prompt, encoding="utf-8")

        plan = self.planner.run(problem, hardware_summary=hardware)
        plan_version = 0
        _save_json(run_dir / f"plan_v{plan_version:02d}.json", plan.to_dict())
        _save_json(run_dir / "plan.json", plan.to_dict())

        attempts: list[CandidateAttempt] = []
        previous_code = ""
        previous_result = VerificationResult(error="no previous verifier result")
        best: CandidateAttempt | None = None
        replan_count = 0
        stop_reason = "max_attempts_reached"

        for attempt_idx in range(max(1, cfg.max_attempts)):
            stage = "draft" if attempt_idx == 0 else _next_stage(previous_result, cfg.target_speedup)
            if attempt_idx == 0:
                code = self.coder.draft(
                    problem,
                    kernelbench_prompt=kb_prompt,
                    plan=plan,
                    hardware_summary=hardware,
                )
            else:
                code = self.coder.revise(
                    problem,
                    kernelbench_prompt=kb_prompt,
                    plan=plan,
                    previous_code=previous_code,
                    verifier_result=previous_result,
                    hardware_summary=hardware,
                    target_speedup=cfg.target_speedup,
                )

            solution_path = run_dir / f"attempt_{attempt_idx:02d}_{stage}.py"
            solution_path.write_text(code, encoding="utf-8")
            result = self.verifier.verify(
                problem,
                code_path=solution_path,
                output_dir=run_dir,
                attempt=attempt_idx,
            )
            attempt = CandidateAttempt(
                attempt=attempt_idx,
                plan_version=plan_version,
                stage=stage,
                solution_path=solution_path,
                result=result,
            )
            attempts.append(attempt)
            _save_json(run_dir / f"attempt_{attempt_idx:02d}_summary.json", attempt.to_dict())

            if result.correct and (best is None or result.speedup > best.result.speedup):
                best = attempt
                shutil.copy2(solution_path, run_dir / "best_solution.py")

            if _meets_target(result, cfg.target_speedup):
                stop_reason = "target_reached"
                break

            replan_reason = _replan_reason(
                attempts,
                target_speedup=cfg.target_speedup,
                failed_threshold=cfg.replan_after_failed_attempts,
                stagnant_threshold=cfg.replan_after_stagnant_attempts,
            )
            if replan_reason and replan_count < max(0, cfg.max_replans) and attempt_idx < max(1, cfg.max_attempts) - 1:
                replan_count += 1
                plan_version += 1
                plan = self.planner.replan(
                    problem,
                    hardware_summary=hardware,
                    previous_plan=plan,
                    attempts=attempts,
                    reason=replan_reason,
                )
                _save_json(run_dir / f"plan_v{plan_version:02d}.json", plan.to_dict())
                _save_json(run_dir / "plan.json", plan.to_dict())

            previous_code = code
            previous_result = result

        summary = {
            "run_dir": str(run_dir),
            "problem": problem.to_dict(),
            "stop_reason": stop_reason,
            "replan_count": replan_count,
            "best_attempt": best.to_dict() if best else None,
            "attempts": [attempt.to_dict() for attempt in attempts],
        }
        _save_json(run_dir / "summary.json", summary)
        return summary

    def _create_run_dir(self, problem: KernelBenchProblem) -> Path:
        root = Path(self.config.output_root).expanduser().resolve()
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config.run_name or f"stitchcuda_L{problem.level}_P{problem.problem_id}_{stamp}"
        run_dir = root / _slug(name)
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _hardware_summary(self) -> dict[str, Any]:
        env_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        summary: dict[str, Any] = {
            "requested_device": self.config.device,
            "cuda_visible_devices": env_device,
            "kernelbench_gpu_arch": self.config.gpu_arch,
        }
        try:
            query = "name,compute_cap,memory.total,driver_version"
            cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
            if self.config.device is not None:
                cmd.extend(["-i", str(self.config.device)])
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            first = out.splitlines()[0] if out else ""
            parts = [part.strip() for part in first.split(",")]
            if len(parts) >= 4:
                summary.update(
                    {
                        "gpu_name": parts[0],
                        "compute_capability": parts[1],
                        "memory_total": parts[2],
                        "driver_version": parts[3],
                    }
                )
        except Exception as exc:
            summary["nvidia_smi_error"] = str(exc)
        target_sm = _normalize_sm(self.config.target_sm) or _sm_from_compute_capability(
            str(summary.get("compute_capability", ""))
        )
        if target_sm:
            summary["target_sm"] = target_sm
            summary["target_compute"] = target_sm.replace("sm_", "compute_")
        return summary


def _next_stage(result: VerificationResult, target_speedup: float) -> str:
    if not result.compiled:
        return "repair_compile"
    if not result.correct:
        return "repair_correctness"
    if target_speedup > 0 and result.speedup < target_speedup:
        return "optimize_performance"
    return "optimize"


def _meets_target(result: VerificationResult, target_speedup: float) -> bool:
    if not result.correct:
        return False
    if target_speedup <= 0:
        return True
    return result.speedup >= target_speedup or math.isclose(result.speedup, target_speedup)


def _replan_reason(
    attempts: list[CandidateAttempt],
    *,
    target_speedup: float,
    failed_threshold: int,
    stagnant_threshold: int,
) -> str:
    if not attempts:
        return ""

    failed_threshold = max(1, failed_threshold)
    stagnant_threshold = max(1, stagnant_threshold)
    recent_failures = attempts[-failed_threshold:]
    if len(recent_failures) == failed_threshold and all(not item.result.compiled for item in recent_failures):
        return (
            f"last {failed_threshold} attempts failed to compile; current implementation plan is not producing "
            "valid KernelBench code"
        )
    if len(recent_failures) == failed_threshold and all(item.result.compiled and not item.result.correct for item in recent_failures):
        return (
            f"last {failed_threshold} attempts compiled but failed correctness; planner should change the "
            "correctness strategy before more coding"
        )

    correct_attempts = [item for item in attempts if item.result.correct]
    if target_speedup > 0 and len(correct_attempts) >= stagnant_threshold:
        recent_correct = correct_attempts[-stagnant_threshold:]
        if all(item.result.speedup < target_speedup for item in recent_correct):
            speeds = [item.result.speedup for item in recent_correct]
            if max(speeds) - min(speeds) < 0.05 * max(1.0, max(speeds)):
                return (
                    f"last {stagnant_threshold} correct attempts are below target and show little speedup "
                    "improvement; planner should choose a different optimization strategy"
                )
    return ""


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _config_to_json(config: FixedWorkflowConfig) -> dict[str, Any]:
    data = asdict(config)
    if data.get("api_key"):
        data["api_key"] = "<redacted>"
    return data


def _target_arch_context(hardware: dict[str, Any]) -> str:
    parts = []
    for key in ("gpu_name", "kernelbench_gpu_arch", "target_sm", "target_compute", "compute_capability"):
        value = hardware.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _normalize_sm(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text.startswith("sm_"):
        numeric = text[3:]
    elif text.startswith("sm"):
        numeric = text[2:]
    else:
        numeric = text
    numeric = numeric.replace(".", "")
    if not numeric.isdigit():
        return ""
    return f"sm_{numeric}"


def _sm_from_compute_capability(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    numeric = text.replace(".", "")
    if numeric.isdigit():
        return f"sm_{numeric}"
    return ""


def _slug(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in "._-":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "stitchcuda_run"
