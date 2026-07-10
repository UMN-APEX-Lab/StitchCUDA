from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
        timeout_s: int = 1800,
    ):
        self.kernelbench_root = kernelbench_root
        self.gpu_arch = gpu_arch
        self.device = device
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
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
        code_hash = _sha256_file(code_path)
        build_dir = _attempt_extension_root(output_dir, attempt, code_hash)
        extension_prefix = _extension_prefix(attempt, code_hash)
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
            "--build-dir",
            str(build_dir),
            "--extension-prefix",
            extension_prefix,
            "--num-correct-trials",
            str(self.num_correct_trials),
            "--num-perf-trials",
            str(self.num_perf_trials),
        ]

        env = os.environ.copy()
        pythonpath = [str(Path(__file__).resolve().parents[1]), kernelbench_pythonpath(self.kernelbench_root)]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["TORCH_EXTENSIONS_DIR"] = str(build_dir)
        env["STITCHCUDA_BUILD_DIR"] = str(build_dir)
        env["STITCHCUDA_EXTENSION_PREFIX"] = extension_prefix
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
                error_kind="timeout",
                error=f"KernelBench verifier timeout after {self.timeout_s}s",
                code_hash=code_hash,
                metadata={
                    "extension_build_dir": str(build_dir),
                    "extension_prefix": extension_prefix,
                },
            )

        if result_json.exists():
            try:
                data = json.loads(result_json.read_text(encoding="utf-8"))
                result = VerificationResult.from_dict(data)
            except Exception as exc:
                result = VerificationResult(
                    compiled=False,
                    correct=False,
                    error_kind="verifier_result_parse_error",
                    error=f"failed to parse verifier result json: {exc}",
                )
        else:
            result = VerificationResult(
                compiled=False,
                correct=False,
                error_kind="worker_failed",
                error=f"verifier worker exited {completed.returncode} without result json",
            )
        if not result.code_hash:
            result.code_hash = code_hash
        result.metadata.setdefault("extension_build_dir", str(build_dir))
        result.metadata.setdefault("extension_prefix", extension_prefix)
        result.stdout_tail = completed.stdout[-4000:]
        result.stderr_tail = completed.stderr[-4000:]
        return result


def run_worker(args: argparse.Namespace) -> int:
    build_dir = Path(args.build_dir).expanduser().resolve() if args.build_dir else None
    if build_dir is not None:
        build_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(build_dir)
        os.environ["STITCHCUDA_BUILD_DIR"] = str(build_dir)
    if args.extension_prefix:
        os.environ["STITCHCUDA_EXTENSION_PREFIX"] = args.extension_prefix

    from .runtime import install_cuda_extension_scaffold

    install_cuda_extension_scaffold()
    set_kernelbench_gpu_arch(args.gpu_arch, args.kernelbench_root)
    from kernelbench.eval import eval_kernel_against_ref

    ref_src = Path(args.ref_path).read_text(encoding="utf-8")
    code = Path(args.code_path).read_text(encoding="utf-8")
    static_warnings: list[str] = []

    try:
        from kernelbench.kernel_static_checker import validate_kernel_static

        static_valid, static_errors, static_warnings = validate_kernel_static(
            code,
            backend="cuda",
            precision="fp32",
        )
    except Exception as exc:
        static_valid = True
        static_errors = []
        static_warnings = [f"static checker unavailable: {exc.__class__.__name__}: {exc}"]

    if not static_valid:
        payload = _static_validation_payload(static_errors, static_warnings)
        payload.setdefault("metadata", {}).update(_runtime_metadata())
        _add_extension_metadata(payload, build_dir, args.extension_prefix)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return 0

    try:
        result = eval_kernel_against_ref(
            ref_src,
            code,
            measure_performance=True,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            backend="cuda",
            build_dir=str(build_dir) if build_dir is not None else None,
            check_for_excessive_speedup=True,
            excessive_speedup_threshold=10,
        )
        payload = _kernel_exec_result_to_payload(result)
        payload.setdefault("metadata", {}).update(_runtime_metadata())
        _add_extension_metadata(payload, build_dir, args.extension_prefix)
        if static_warnings:
            payload.setdefault("metadata", {})["static_warnings"] = static_warnings
            payload["warnings"] = [*payload.get("warnings", []), *static_warnings]
    except Exception as exc:
        payload = {
            "compiled": False,
            "correct": False,
            "speedup": 0.0,
            "runtime_us": -1.0,
            "ref_runtime_us": -1.0,
            "error_kind": "verifier_exception",
            "metadata": _runtime_metadata(),
            "error": f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
            "warnings": static_warnings,
        }
        _add_extension_metadata(payload, build_dir, args.extension_prefix)

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
    parser.add_argument("--build-dir", default="")
    parser.add_argument("--extension-prefix", default="")
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
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
            "error_kind": "retryable_worker_result",
            "metadata": {"worker_returned_none": True},
            "error": "KernelBench worker returned None",
            "warnings": [],
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
    compiled = bool(data.get("compiled", False))
    correct = bool(data.get("correctness", False))
    error = ""
    error_kind = ""
    warnings: list[str] = []
    if metadata.get("excessive_speedup") is True:
        correct = False
        error_kind = "excessive_speedup"
        error = "candidate was flagged for excessive speedup; treat as potential reward hacking"
        warnings.append(error)
    elif not compiled:
        error_kind = "compile_error"
        error = metadata.get("compilation_error") or metadata.get("runtime_error") or "candidate did not compile"
    elif metadata.get("runtime_error"):
        error_kind = "runtime_error"
        error = metadata.get("runtime_error") or "candidate raised a runtime error"
    elif not correct:
        error_kind = "correctness_error"
        error = metadata.get("correctness_issue") or metadata.get("runtime_error") or "candidate failed correctness"
    elif metadata.get("error_during_performance"):
        error_kind = "performance_error"
        error = metadata.get("error_during_performance") or "candidate failed during performance measurement"
    return {
        "compiled": compiled,
        "correct": correct,
        "speedup": speedup,
        "runtime_us": runtime,
        "ref_runtime_us": ref_runtime,
        "error_kind": error_kind,
        "metadata": metadata,
        "error": error,
        "warnings": warnings,
    }


def _static_validation_payload(errors: list[str], warnings: list[str]) -> dict[str, Any]:
    return {
        "compiled": False,
        "correct": False,
        "speedup": 0.0,
        "runtime_us": -1.0,
        "ref_runtime_us": -1.0,
        "error_kind": "static_forbidden",
        "metadata": {
            "static_errors": errors,
            "static_warnings": warnings,
        },
        "error": "KernelBench static validation failed: " + "; ".join(errors),
        "warnings": warnings,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_extension_root(output_dir: Path, attempt: int, code_hash: str) -> Path:
    short_hash = (code_hash or "nohash")[:12]
    return output_dir / "torch_extensions" / f"attempt_{attempt:02d}_{short_hash}"


def _extension_prefix(attempt: int, code_hash: str) -> str:
    short_hash = (code_hash or "nohash")[:12]
    return f"stitchcuda_a{attempt:02d}_{short_hash}"


def _add_extension_metadata(payload: dict[str, Any], build_dir: Path | None, extension_prefix: str) -> None:
    metadata = payload.setdefault("metadata", {})
    if build_dir is not None:
        metadata["extension_build_dir"] = str(build_dir)
    if extension_prefix:
        metadata["extension_prefix"] = extension_prefix


def _runtime_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
            }
        )
    except Exception as exc:
        metadata["torch_metadata_error"] = f"{exc.__class__.__name__}: {exc}"
    return metadata


if __name__ == "__main__":
    raise SystemExit(main())
