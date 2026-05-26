"""Environment self-check for StitchCUDA.

``stitchcuda doctor`` runs a fixed battery of probes (Python, CUDA toolchain,
KernelBench submodule, PyTorch, LLM credentials) and prints a status table.
Each probe is isolated so a single failure cannot mask the rest, and each
returns a :class:`Check` describing what was inspected, the verdict, and a
remediation hint when applicable.

The exit code is ``1`` if any probe reports ``fail`` (hard blocker), else
``0`` — ``warn`` is informational and does not change the exit code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from ..kernelbench import default_kernelbench_root
from .display import display_endpoint, display_executable, display_path

Status = str  # one of: "ok", "warn", "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def run_checks() -> list[Check]:
    """Run every probe in a fixed order; never raise."""
    probes: list[Callable[[], Check]] = [
        _check_python,
        _check_nvcc,
        _check_nvidia_smi,
        _check_kernelbench,
        _check_torch_cuda,
        _check_llm_credentials,
    ]
    results: list[Check] = []
    for probe in probes:
        try:
            results.append(probe())
        except Exception as exc:  # defensive: a probe should never break doctor
            results.append(Check(probe.__name__.removeprefix("_check_"), "fail", f"probe crashed: {exc}"))
    return results


def render(checks: list[Check], console: Console) -> int:
    """Print the table and return the conventional shell exit code."""
    table = Table(title="StitchCUDA environment", title_style="bold", header_style="bold")
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Details", overflow="fold")
    for chk in checks:
        table.add_row(chk.name, _status_cell(chk.status), chk.detail)
    console.print(table)
    return 1 if any(chk.status == "fail" for chk in checks) else 0


def _status_cell(status: Status) -> str:
    return {
        "ok": "[green]OK[/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }.get(status, status)


def _check_python() -> Check:
    if sys.version_info < (3, 10):
        return Check("Python", "fail", f"Need Python >=3.10, found {sys.version.split()[0]}")
    return Check("Python", "ok", f"{sys.version.split()[0]} ({display_executable(sys.executable)})")


def _check_nvcc() -> Check:
    path = shutil.which("nvcc")
    if not path:
        return Check("nvcc", "fail", "Not on PATH; install the CUDA toolkit")
    try:
        out = subprocess.check_output([path, "-V"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return Check("nvcc", "warn", f"{display_executable(path)} present but failed to run: {exc}")
    release = next((line for line in out.splitlines() if "release" in line.lower()), "")
    return Check("nvcc", "ok", f"{display_executable(path)} ({release.strip()})" if release else display_executable(path))


def _check_nvidia_smi() -> Check:
    path = shutil.which("nvidia-smi")
    if not path:
        return Check("nvidia-smi", "fail", "Not on PATH; install the NVIDIA driver")
    try:
        out = subprocess.check_output(
            [path, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return Check("nvidia-smi", "fail", f"exited {exc.returncode}")
    except Exception as exc:
        return Check("nvidia-smi", "fail", f"failed to run: {exc}")
    if not out:
        return Check("nvidia-smi", "warn", "No GPUs reported")
    lines = out.splitlines()
    return Check("nvidia-smi", "ok", f"{len(lines)} GPU(s); first: {lines[0]}")


def _check_kernelbench() -> Check:
    root = Path(os.environ.get("KERNELBENCH_ROOT") or default_kernelbench_root())
    src = root / "src"
    if not src.is_dir():
        return Check(
            "KernelBench",
            "fail",
            f"{display_path(src)} missing — run `git submodule update --init --recursive`",
        )
    # Best-effort import; keep sys.path mutation behind try/finally so a broken
    # KernelBench install does not poison the parent process search path.
    inserted = False
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        inserted = True
    try:
        import kernelbench.dataset  # noqa: F401
    except Exception as exc:
        if inserted:
            sys.path.remove(src_str)
        return Check("KernelBench", "warn", f"found at {display_path(root)}, but `import kernelbench.dataset` failed: {exc}")
    return Check("KernelBench", "ok", display_path(root))


def _check_torch_cuda() -> Check:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return Check("PyTorch", "fail", "torch not installed; install a CUDA-enabled build")
    if not torch.cuda.is_available():
        return Check("PyTorch", "fail", f"torch {torch.__version__} installed but CUDA unavailable")
    return Check(
        "PyTorch",
        "ok",
        f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)",
    )


def _check_llm_credentials() -> Check:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("STITCHCUDA_API_BASE", "")
    if api_key and api_base:
        return Check("LLM credentials", "ok", f"OPENAI_API_KEY set; STITCHCUDA_API_BASE={display_endpoint(api_base)}")
    if api_key:
        return Check("LLM credentials", "ok", "OPENAI_API_KEY set (hosted OpenAI endpoint)")
    if api_base:
        return Check(
            "LLM credentials",
            "warn",
            f"STITCHCUDA_API_BASE={display_endpoint(api_base)}; OPENAI_API_KEY unset (local server will accept a dummy key)",
        )
    return Check(
        "LLM credentials",
        "warn",
        "Neither OPENAI_API_KEY nor STITCHCUDA_API_BASE set; pass --api-key / --api-base at run time",
    )
