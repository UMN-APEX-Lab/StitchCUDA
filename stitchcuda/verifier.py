from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .kernelbench import kernelbench_pythonpath, set_kernelbench_gpu_arch
from .types import KernelBenchProblem, VerificationResult


class KernelBenchVerifier:
    """Run KernelBench eval in a fresh process for each candidate."""

    def __init__(
        self,
        *,
        kernelbench_root: str | None,
        gpu_arch: str = "Blackwell",
        device: int | None = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 10,
        measure_performance: bool = True,
        timeout_s: int = 1800,
    ):
        self.kernelbench_root = kernelbench_root
        self.gpu_arch = gpu_arch
        self.device = device
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
        self.measure_performance = measure_performance
        self.timeout_s = timeout_s

    def verify(
        self,
        problem: KernelBenchProblem,
        *,
        code_path: Path,
        output_dir: Path,
        attempt: int,
    ) -> VerificationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        ref_path = output_dir / f"L{problem.level}_P{problem.problem_id}_reference.py"
        if not ref_path.exists():
            ref_path.write_text(problem.reference_code, encoding="utf-8")

        result_json = output_dir / f"attempt_{attempt:02d}_verifier.json"
        cmd = [
            sys.executable,
            "-m",
            "stitchcuda.verifier",
            "--worker-eval",
            "--kernelbench-root",
            self.kernelbench_root,
            "--gpu-arch",
            self.gpu_arch,
            "--ref-path",
            str(ref_path),
            "--code-path",
            str(code_path),
            "--output-json",
            str(result_json),
            "--num-correct-trials",
            str(self.num_correct_trials),
            "--num-perf-trials",
            str(self.num_perf_trials),
        ]
        if self.measure_performance:
            cmd.append("--measure-performance")

        env = os.environ.copy()
        pythonpath = [str(Path(__file__).resolve().parents[1]), kernelbench_pythonpath(self.kernelbench_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)

        try:
            completed = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                compiled=False,
                correct=False,
                error=f"KernelBench verifier timeout after {self.timeout_s}s",
            )

        if result_json.exists():
            data = json.loads(result_json.read_text(encoding="utf-8"))
            result = VerificationResult.from_dict(data)
        else:
            result = VerificationResult(
                compiled=False,
                correct=False,
                error=f"verifier worker exited {completed.returncode} without result json",
            )
        result.stdout_tail = completed.stdout[-4000:]
        result.stderr_tail = completed.stderr[-4000:]
        return result


def run_worker(args: argparse.Namespace) -> int:
    set_kernelbench_gpu_arch(args.gpu_arch, args.kernelbench_root)
    from kernelbench.eval import eval_kernel_against_ref

    ref_src = Path(args.ref_path).read_text(encoding="utf-8")
    code = Path(args.code_path).read_text(encoding="utf-8")

    try:
        result = eval_kernel_against_ref(
            ref_src,
            code,
            measure_performance=args.measure_performance,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            backend="cuda",
        )
        payload = _kernel_exec_result_to_payload(result)
    except Exception as exc:
        payload = {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "metadata": {},
            "error": f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
        }

    Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StitchCUDA KernelBench verifier worker")
    parser.add_argument("--worker-eval", action="store_true")
    parser.add_argument("--kernelbench-root", default="")
    parser.add_argument("--gpu-arch", default="Blackwell")
    parser.add_argument("--ref-path", default="")
    parser.add_argument("--code-path", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    parser.add_argument("--measure-performance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.worker_eval:
        parser.error("verifier.py is intended to be called with --worker-eval")
    return run_worker(args)


def _kernel_exec_result_to_payload(result: Any) -> dict[str, Any]:
    if result is None:
        return {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "metadata": {"worker_returned_none": True},
            "error": "KernelBench worker returned None",
        }
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif hasattr(result, "dict"):
        data = result.dict()
    else:
        data = dict(result)
    data = json.loads(json.dumps(data, default=str))
    runtime = float(data.get("runtime", -1.0) or -1.0)
    ref_runtime = float(data.get("ref_runtime", -1.0) or -1.0)
    speedup = ref_runtime / runtime if runtime > 0 and ref_runtime > 0 else 0.0
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    error = ""
    if not data.get("compiled", False):
        error = metadata.get("compilation_error") or metadata.get("runtime_error") or "candidate did not compile"
    elif not data.get("correctness", False):
        error = metadata.get("correctness_issue") or metadata.get("runtime_error") or "candidate failed correctness"
    return {
        "compiled": bool(data.get("compiled", False)),
        "correct": bool(data.get("correctness", False)),
        "speedup": speedup,
        "runtime_us": runtime,
        "ref_runtime_us": ref_runtime,
        "metadata": metadata,
        "error": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
