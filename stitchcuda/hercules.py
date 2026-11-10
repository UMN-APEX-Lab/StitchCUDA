"""CUDA-Hercules backend for StitchCUDA (Class 1 and Class 2 tasks).

The fixed planner -> coder -> verifier loop of `workflow.py` is reused unchanged; this module
swaps its KernelBench-specific parts:

- problems are CUDA-Hercules tasks (`task.yaml`); the planner and coder see only the task
  prompt built by CUDA-Hercules (description, function signature, rules) and never any
  reference code;
- the coder writes the task's solution file (`solution.h` for Class 1, `solution.cu` for
  Class 2) as one ```cuda block;
- the verifier runs the task's own harness through `cuda_hercules.runner.run_task` in a fresh
  subprocess (static anti-cheat rules, correctness on every size, timing, per-task geometric
  mean speedup), and then the CUDA-Hercules hacking audit (`scripts/hacking_audit.py`, a judge
  LLM) on every candidate that passes. A "hacking" verdict turns the attempt into a failure
  whose reasons are fed back to the coder.

Batch usage (the API key is read from an environment variable or --env-file, never from argv):

    python -m stitchcuda.hercules --hercules-root /path/to/CUDAHercules \\
        --task-list tasks.txt --model gpt-6-luna --api-base-env OPENAI_BASE_URL \\
        --api-key-env OPENAI_API_KEY --env-file /path/to/.env --device 1 \\
        --max-attempts 15 --output runs/hercules --run-name my_run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .coder import CoderAgent, target_arch_context
from .llm import LLMConfig, OpenAIChatClient, parse_json_object
from .ncu_profiler import NcuProfiler
from .planner import PlannerAgent
from .schemas import PlanPayload
from .templates import render_template
from .types import Plan, VerificationResult
from .workflow import FixedWorkflowConfig, StitchCUDAWorkflow, _slug

FEEDBACK_CHARS = 6000


# --------------------------------------------------------------------------- CUDA-Hercules access
def _use_hercules(root: str) -> None:
    for p in (os.path.join(root, "src"), os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _extract_cuda(response: str) -> str:
    from cuda_hercules.utils import extract_cuda_code
    return extract_cuda_code(response)


@dataclass
class HerculesProblem:
    level: int              # task class (1 or 2)
    problem_id: str         # CUDA-Hercules task id, e.g. class1/general/00_basic_gemm
    name: str
    task_dir: str
    solution_file: str
    task_prompt: str
    reference_code: str = ""   # never used: references are not shown to the agents

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("reference_code")
        d.pop("task_prompt")
        return d


def load_hercules_problems(root: str, task_ids: list[str]) -> list[HerculesProblem]:
    _use_hercules(root)
    from cuda_hercules.prompt_builder import build_prompt
    from cuda_hercules.runner import discover_tasks

    configs = {t.task_id: t for t in discover_tasks(root)}
    out = []
    for tid in task_ids:
        if tid not in configs:
            raise KeyError(f"unknown CUDA-Hercules task id: {tid}")
        cfg = configs[tid]
        backend = cfg.runner.backend
        if backend not in ("class1_make", "class2_defpy"):
            raise ValueError(f"{tid}: backend {backend} is not supported (Class 1 / Class 2 only)")
        task_dir = os.path.join(root, cfg.runner.workdir)
        out.append(HerculesProblem(
            level=1 if backend == "class1_make" else 2, problem_id=tid, name=tid.rsplit("/", 1)[-1],
            task_dir=task_dir, solution_file=cfg.runner.solution_file,
            task_prompt=build_prompt(task_dir, cfg)))
    return out


# --------------------------------------------------------------------------- agents
class HerculesPlanner(PlannerAgent):
    SYSTEM = "You are the planner in a fixed CUDA kernel optimization workflow. Return strict JSON."

    def run(self, problem, *, hardware_summary: dict) -> Plan:
        prompt = render_template("hercules_planner.md", problem_json=json.dumps(problem.to_dict(), indent=2),
                                 hardware_json=json.dumps(hardware_summary, indent=2),
                                 task_prompt=problem.task_prompt)
        return self._plan(prompt)

    def replan(self, problem, *, hardware_summary: dict, previous_plan: Plan, attempt_memory_context: str,
               reason: str) -> Plan:
        prompt = render_template("hercules_replanner.md", problem_json=json.dumps(problem.to_dict(), indent=2),
                                 hardware_json=json.dumps(hardware_summary, indent=2),
                                 previous_plan_json=json.dumps(previous_plan.to_dict(), indent=2),
                                 attempt_memory_context=attempt_memory_context, replan_reason=reason,
                                 task_prompt=problem.task_prompt)
        return self._plan(prompt)

    def _plan(self, prompt: str, tries: int = 3) -> Plan:
        for i in range(tries):
            raw = self.llm.chat([{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": prompt}])
            try:
                payload = PlanPayload.model_validate(parse_json_object(raw))
                return Plan.from_dict(payload.model_dump())
            except Exception:
                if i == tries - 1:
                    raise


class HerculesCoder(CoderAgent):
    def draft(self, problem, *, kernelbench_prompt: str, plan: Plan, hardware_summary: dict) -> str:
        prompt = render_template("hercules_coder_initial.md", task_prompt=kernelbench_prompt,
                                 plan_json=json.dumps(plan.to_dict(), indent=2),
                                 hardware_json=json.dumps(hardware_summary, indent=2),
                                 target_arch_context=target_arch_context(hardware_summary),
                                 solution_file=problem.solution_file)
        return self._code("You are the coder in a fixed CUDA kernel workflow. Return one complete solution file.",
                          prompt)

    def revise(self, problem, *, kernelbench_prompt: str, plan: Plan, baseline_code: str,
               attempt_memory_context: str, hardware_summary: dict, target_speedup: float) -> str:
        prompt = render_template("hercules_coder_revise.md", task_prompt=kernelbench_prompt,
                                 plan_json=json.dumps(plan.to_dict(), indent=2),
                                 hardware_json=json.dumps(hardware_summary, indent=2),
                                 target_arch_context=target_arch_context(hardware_summary),
                                 baseline_code=baseline_code, attempt_memory_context=attempt_memory_context,
                                 target_speedup=target_speedup, solution_file=problem.solution_file)
        return self._code("You are the coder in a fixed CUDA kernel workflow. Repair or optimize the full "
                          "solution file.", prompt)

    def _code(self, system: str, prompt: str) -> str:
        raw = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return _extract_cuda(raw)


# --------------------------------------------------------------------------- verifier
_CHILD = r"""
import json, os, sys
sys.path.insert(0, os.path.join({root!r}, "src"))
from cuda_hercules.runner import discover_tasks, run_task
tasks = {{t.task_id: t for t in discover_tasks({root!r})}}
tr = run_task(tasks[{task_id!r}], {sol!r}, measure_perf=True, verbose=False)
payload = {{
    "status": tr.status.value if tr.status else "ERROR",
    "compiled": bool(tr.compiled), "correct": bool(tr.correct),
    "speedup": tr.speedup or 0.0,
    "latency_ms": tr.latency_mean_ms, "ref_latency_ms": tr.ref_latency_mean_ms,
    "per_size": tr.per_size or [],
    "error_msg": tr.error_msg or "",
    "correctness_detail": tr.correctness_detail or {{}},
}}
sys.stdout.write("__RESULT_JSON__" + json.dumps(payload, default=str) + "\n")
"""


class HackingAuditor:
    """CUDA-Hercules hacking audit (scripts/hacking_audit.py) with a judge LLM."""

    def __init__(self, root: str, llm: OpenAIChatClient | None):
        _use_hercules(root)
        import hacking_audit  # CUDA-Hercules scripts/hacking_audit.py
        self.ha = hacking_audit
        self.llm = llm

    def audit(self, problem: HerculesProblem, solution: Path) -> dict[str, Any]:
        if self.llm is None:
            return {"verdict": "skipped"}
        try:
            system, user = self.ha.build_prompt(problem.problem_id, Path(problem.task_dir), solution)
            raw = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            verdict = self.ha.parse_verdict(raw)
        except BaseException as exc:  # build_prompt raises SystemExit on missing files
            if isinstance(exc, KeyboardInterrupt):
                raise
            verdict = {"verdict": "audit_error", "error": f"{exc.__class__.__name__}: {exc}"}
        verdict.setdefault("verdict", "parse_error")
        verdict["judge_model"] = self.llm.config.model
        verdict["audited_solution"] = solution.name
        verdict["audited_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return verdict


class HerculesVerifier:
    """Run the task's CUDA-Hercules harness in a fresh process, then the hacking audit."""

    def __init__(self, *, root: str, device: int | None, timeout_s: int, auditor: HackingAuditor | None,
                 profiler=None):
        self.root = root
        self.device = device
        self.timeout_s = timeout_s
        self.auditor = auditor
        self.profiler = profiler

    def verify(self, problem: HerculesProblem, *, code_path: Path, output_dir: Path, attempt) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        code_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
        tag = f"attempt_{attempt:02d}" if isinstance(attempt, int) else str(attempt)
        build_dir = output_dir / "torch_extensions" / f"{tag}_{code_hash[:12]}"
        build_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(build_dir))
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)
        script = _CHILD.format(root=self.root, task_id=problem.problem_id, sol=str(code_path))
        try:
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                  timeout=self.timeout_s, cwd=self.root, env=env)
        except subprocess.TimeoutExpired:
            return VerificationResult(error_kind="timeout", code_hash=code_hash,
                                      error=f"evaluation timeout after {self.timeout_s}s")
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT_JSON__"):
                payload = json.loads(line[len("__RESULT_JSON__"):])
        if payload is None:
            return VerificationResult(error_kind="worker_failed", code_hash=code_hash,
                                      error=f"evaluation process exited {proc.returncode} without a result",
                                      stderr_tail=proc.stderr[-4000:])
        result = _to_result(payload, code_hash)
        result.metadata["stdout_tail"] = proc.stdout[-2000:]
        if result.correct and self.auditor is not None:
            verdict = self.auditor.audit(problem, code_path)
            result.metadata["audit_verdict"] = verdict
            if verdict.get("verdict") == "hacking":
                reasons = "; ".join(str(r) for r in (verdict.get("reasons") or []))
                evidence = "; ".join(str(e) for e in (verdict.get("evidence") or []))
                result.correct = False
                result.metadata["harness_speedup"] = result.speedup
                result.speedup = 0.0
                result.error_kind = "hacking"
                result.error = _clip(f"hacking audit verdict: hacking (the harness passed, but the solution is "
                                     f"scored as a failure). Reasons: {reasons}. Evidence: {evidence}")
            elif verdict.get("verdict") == "suspicious":
                result.warnings.append("hacking audit verdict: suspicious; " +
                                       "; ".join(str(r) for r in (verdict.get("reasons") or [])))
        if result.correct and self.profiler is not None:
            # Nsight Compute analysis -> ranked next-step suggestions (only these go back to the agents).
            prof = self.profiler.profile(problem, code_path, output_dir / f"{tag}_ncu", result)
            result.suggestions = prof.get("suggestions", "")
            result.metadata["ncu"] = {k: v for k, v in prof.items() if k != "suggestions"}
        return result


def _clip(text: str, n: int = FEEDBACK_CHARS) -> str:
    return text if len(text) <= n else text[: n // 2] + "\n...[truncated]...\n" + text[-n // 2:]


def _to_result(p: dict[str, Any], code_hash: str) -> VerificationResult:
    status = p.get("status", "ERROR")
    msg = p.get("error_msg") or ""
    correct = status == "PASS" and bool(p.get("correct"))
    compiled = bool(p.get("compiled"))
    if status == "SKIP_ARCH":
        kind = "skip_arch"
    elif msg.startswith("Static check failed"):
        kind, compiled = "static_forbidden", False
    elif not compiled:
        kind = "compile_error"
    elif "timed out" in msg.lower():
        kind = "timeout"
    elif not p.get("correct"):
        kind = "correctness_error"
    elif not correct:
        kind = "runtime_error"
    else:
        kind = ""
    detail = p.get("correctness_detail") or {}
    error = ""
    if not correct:
        parts = [msg]
        if detail:
            parts.append("correctness detail: " + json.dumps(detail, default=str)[:3000])
        error = _clip("\n".join(x for x in parts if x) or f"status {status}")
    lat, ref = p.get("latency_ms"), p.get("ref_latency_ms")
    sizes = [x for x in (p.get("per_size") or []) if isinstance(x, dict)]
    if not lat and sizes:  # Class 1 reports per-size minima only: use their sum over the sizes
        lat = sum(float(x.get("kernel_min_ms") or 0.0) for x in sizes) or None
        ref = sum(float(x.get("ref_min_ms") or 0.0) for x in sizes) or None
    return VerificationResult(
        compiled=compiled, correct=correct, speedup=float(p.get("speedup") or 0.0) if correct else 0.0,
        runtime_us=lat * 1000.0 if lat else -1.0, ref_runtime_us=ref * 1000.0 if ref else -1.0,
        error_kind=kind, error=error, code_hash=code_hash,
        metadata={"status": status, "per_size": p.get("per_size") or [], "correctness_detail": detail})


# --------------------------------------------------------------------------- workflow
class HerculesWorkflow(StitchCUDAWorkflow):
    task_prompt_filename = "task_prompt.md"

    def __init__(self, config: FixedWorkflowConfig, *, problem: HerculesProblem, root: str,
                 auditor: HackingAuditor | None, profile_args: dict | None = None, events=None):
        super().__init__(config, events=events)
        llm = self.planner.llm
        self.problem = problem
        self.solution_suffix = os.path.splitext(problem.solution_file)[1] or ".cu"
        self.planner = HerculesPlanner(llm)
        self.coder = HerculesCoder(llm, kernelbench_root="")
        profiler = NcuProfiler(root=root, llm=llm, device=config.device, **profile_args) if profile_args else None
        self.verifier = HerculesVerifier(root=root, device=config.device, timeout_s=config.verifier_timeout_s,
                                         auditor=auditor, profiler=profiler)

    def _load_problem(self):
        return self.problem

    def _build_task_prompt(self, problem, hardware: dict[str, Any]) -> str:
        return problem.task_prompt

    def _default_run_name(self, problem, stamp: str) -> str:
        return problem.problem_id.replace("/", "_")

    def _hardware_summary(self) -> dict[str, Any]:
        summary = super()._hardware_summary()
        summary.pop("kernelbench_gpu_arch", None)
        return summary


# --------------------------------------------------------------------------- batch driver
def _read_env_file(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if path and os.path.isfile(os.path.expanduser(path)):
        for line in open(os.path.expanduser(path)):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


def _env(name: str, file_env: dict[str, str]) -> str:
    return os.environ.get(name) or file_env.get(name, "")


def _llm_usage(run_dir: Path) -> dict[str, int]:
    tot = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    log = run_dir / "events.jsonl"
    if log.is_file():
        for line in log.open():
            ev = json.loads(line)
            u = (ev.get("llm") or {}).get("usage") or {}
            if u:
                tot["llm_calls"] += 1
                tot["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                tot["completion_tokens"] += int(u.get("completion_tokens") or 0)
    return tot


def run_task(problem: HerculesProblem, args, file_env: dict[str, str], out_root: Path) -> dict[str, Any]:
    task_dir = out_root / problem.problem_id.replace("/", "_")
    if (task_dir / "result.json").is_file():
        return json.loads((task_dir / "result.json").read_text())
    if task_dir.exists():  # interrupted earlier: keep it, start over
        aborted = out_root / "_aborted"
        aborted.mkdir(exist_ok=True)
        shutil.move(str(task_dir), str(aborted / f"{task_dir.name}_{dt.datetime.now():%Y%m%d_%H%M%S}"))
    api_key = _env(args.api_key_env, file_env)
    api_base = args.api_base or _env(args.api_base_env, file_env)
    cfg = FixedWorkflowConfig(
        level=problem.level, problem_id=problem.problem_id, model=args.model, api_base=api_base, api_key=api_key,
        temperature=args.temperature, max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
        output_root=str(out_root), run_name=task_dir.name, device=args.device,
        max_attempts=args.max_attempts, max_replans=args.max_replans,
        replan_after_failed_attempts=args.replan_after_failed_attempts,
        replan_after_stagnant_attempts=args.replan_after_stagnant_attempts,
        target_speedup=args.target_speedup, verifier_timeout_s=args.verifier_timeout_s)
    auditor = None
    if args.audit_model:
        judge = OpenAIChatClient(LLMConfig(model=args.audit_model, api_base=api_base, api_key=api_key,
                                           temperature=0.0, max_tokens=args.audit_max_tokens))
        auditor = HackingAuditor(args.hercules_root, judge)
    profile_args = None if args.no_ncu else {"skill_dir": args.ncu_skill_dir, "timeout_s": args.ncu_timeout_s,
                                              "ncu": args.ncu}
    wf = HerculesWorkflow(cfg, problem=problem, root=args.hercules_root, auditor=auditor, profile_args=profile_args)
    t0 = dt.datetime.now()
    summary = wf.run()
    wall_s = (dt.datetime.now() - t0).total_seconds()

    best = summary.get("best_attempt")
    result: dict[str, Any] = {
        "task_id": problem.problem_id, "agent": "stitchcuda", "model": args.model,
        "audit_model": args.audit_model or None, "attempts_used": len(summary.get("attempts") or []),
        "max_attempts": args.max_attempts, "replans": summary.get("replan_count"),
        "in_loop_best_attempt": best["attempt"] if best else None,
        "in_loop_best_speedup": best["result"]["speedup"] if best else 0.0,
        "wall_s": round(wall_s, 1), **_llm_usage(task_dir),
    }
    final_status, final_speedup = "FAIL", 0.0
    if best and not args.no_final_reeval:
        # Clean re-evaluation (fresh process) of the in-loop best: this is the task's score.
        sol = task_dir / f"best_solution{wf.solution_suffix}"
        fin = HerculesVerifier(root=args.hercules_root, device=args.device, timeout_s=args.verifier_timeout_s,
                               auditor=None).verify(problem, code_path=sol, output_dir=task_dir, attempt="final")
        result["final_reeval"] = fin.to_dict()
        result["audit_verdict"] = best["result"].get("metadata", {}).get("audit_verdict")
        if fin.correct:
            final_status, final_speedup = "PASS", fin.speedup
    elif best:
        final_status, final_speedup = "PASS", best["result"]["speedup"]
    result.update(final_status=final_status, best_speedup=final_speedup,
                  best_solution_file=f"best_solution{wf.solution_suffix}" if best else None)
    (task_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hercules-root", required=True, help="CUDA-Hercules checkout")
    ap.add_argument("--task", action="append", default=[], help="Task id (repeatable)")
    ap.add_argument("--task-list", default="", help="File with one task id per line")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-base-env", default="OPENAI_BASE_URL")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--env-file", default="", help="Optional file holding the API key/base variables")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=15, help="Verifier evaluations per task")
    ap.add_argument("--max-replans", type=int, default=2)
    ap.add_argument("--replan-after-failed-attempts", type=int, default=2)
    ap.add_argument("--replan-after-stagnant-attempts", type=int, default=2)
    ap.add_argument("--target-speedup", type=float, default=1.0,
                    help="Shown to the coder and used for stage selection; never stops the loop")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=65536, help="Per LLM call output limit")
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--verifier-timeout-s", type=int, default=1800)
    ap.add_argument("--audit-model", default="", help="Judge model for the hacking audit; empty = no audit")
    ap.add_argument("--audit-max-tokens", type=int, default=8192)
    ap.add_argument("--no-final-reeval", action="store_true")
    ap.add_argument("--no-ncu", action="store_true", help="Disable the verifier's Nsight Compute analysis")
    ap.add_argument("--ncu-skill-dir", default="",
                    help="KDA ncu-report-skill checkout (default: <hercules-root>/kda/skills/ncu-report-skill)")
    ap.add_argument("--ncu-timeout-s", type=int, default=1200)
    ap.add_argument("--ncu", default="ncu", help="ncu executable")
    ap.add_argument("--output", default="runs/hercules")
    ap.add_argument("--run-name", required=True)
    args = ap.parse_args(argv)
    args.hercules_root = os.path.abspath(args.hercules_root)
    if not args.ncu_skill_dir:
        args.ncu_skill_dir = os.path.join(args.hercules_root, "kda", "skills", "ncu-report-skill")

    ids = list(args.task)
    if args.task_list:
        ids += [l.strip() for l in open(args.task_list) if l.strip() and not l.startswith("#")]
    problems = load_hercules_problems(args.hercules_root, ids)
    file_env = _read_env_file(args.env_file)
    if not _env(args.api_key_env, file_env):
        raise SystemExit(f"missing API key variable {args.api_key_env} (environment or --env-file)")
    out_root = Path(args.output).expanduser().resolve() / _slug(args.run_name)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items() if k not in ("api_base",)}, indent=2))
    print(f"StitchCUDA on CUDA-Hercules: {len(problems)} tasks, model={args.model}, audit={args.audit_model or 'off'}, "
          f"device={args.device}, attempts={args.max_attempts}, out={out_root}", flush=True)

    rows = []
    for i, prob in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] START {prob.problem_id}", flush=True)
        try:
            r = run_task(prob, args, file_env, out_root)
        except Exception as exc:  # keep going; the task is recorded as an infrastructure failure
            r = {"task_id": prob.problem_id, "final_status": "ERROR", "best_speedup": 0.0,
                 "error": f"{exc.__class__.__name__}: {exc}"}
            print(f"    ERROR {r['error'][:300]}", flush=True)
        rows.append(r)
        print(f"[{i + 1}/{len(problems)}] {prob.problem_id}: {r.get('final_status')} "
              f"{r.get('best_speedup', 0.0):.3f}x (in-loop {r.get('in_loop_best_speedup', 0.0):.3f}x, "
              f"attempts {r.get('attempts_used')}, audit {((r.get('audit_verdict') or {}).get('verdict'))})",
              flush=True)
    n = len(rows)
    sp = [float(r.get("best_speedup") or 0.0) if r.get("final_status") == "PASS" else 0.0 for r in rows]
    summary = {"num_tasks": n, "num_pass": sum(s > 0 for s in sp), "mean_speedup_all": sum(sp) / max(1, n),
               "num_ge_1x": sum(s >= 1.0 for s in sp),
               "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
               "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
               "tasks": [{k: r.get(k) for k in ("task_id", "final_status", "best_speedup", "in_loop_best_speedup",
                                                "attempts_used")} for r in rows]}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"PASS {summary['num_pass']}/{n}, mean speedup (fail=0) {summary['mean_speedup_all']:.3f}x, "
          f">=1x {summary['num_ge_1x']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
