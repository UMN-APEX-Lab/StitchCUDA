from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from .types import KernelBenchProblem


def safe_glue_prompt_suffix(*, target_arch_context: str = "") -> str:
    target_line = (
        f"7. Target architecture context for this run: {target_arch_context}."
        if target_arch_context
        else "7. Use the evaluator-provided target architecture for this run."
    )
    return f"""
Implementation constraints for this KernelBench task:

1. Return one complete Python file defining ModelNew.
2. If you use a custom CUDA extension, build it with torch.utils.cpp_extension.load_inline.
3. Do not use torch.utils.cpp_extension.load.
4. Do not write PYBIND11_MODULE; let load_inline generate Python bindings.
5. Keep the custom extension self-contained in this single Python file.
6. Use load_inline(..., functions=[...]) to bind exported wrapper functions.
{target_line}
8. Do not hardcode a CUDA architecture different from the target architecture
   context. Prefer relying on the evaluator's configured TORCH_CUDA_ARCH_LIST
   unless a target-specific flag is explicitly required.
""".strip()


def ensure_kernelbench_on_path(kernelbench_root: str | Path | None = None) -> Path:
    root = _resolve_kernelbench_root(kernelbench_root)
    src = root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"KernelBench src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src


def load_dataset_problems(
    *,
    level: int,
    kernelbench_root: str | Path | None = None,
    source: str = "local",
    problem_ids: Iterable[int] | None = None,
) -> list[KernelBenchProblem]:
    ensure_kernelbench_on_path(kernelbench_root)
    from kernelbench.dataset import construct_kernelbench_dataset

    ids = list(problem_ids) if problem_ids is not None else None
    dataset = construct_kernelbench_dataset(level=level, source=source, problem_ids=ids)
    return [_problem_from_kernelbench_item(level, item) for item in dataset]


def load_problem(
    *,
    level: int,
    problem_id: int,
    kernelbench_root: str | Path | None = None,
    source: str = "local",
) -> KernelBenchProblem:
    problems = load_dataset_problems(
        level=level,
        kernelbench_root=kernelbench_root,
        source=source,
        problem_ids=[problem_id],
    )
    for problem in problems:
        if problem.problem_id == problem_id:
            return problem
    raise KeyError(f"KernelBench L{level} problem {problem_id} not found")


def build_kernelbench_prompt(
    reference_code: str,
    *,
    prompt_option: str = "zero_shot_safe_glue",
    kernelbench_root: str | Path | None = None,
    gpu_name: str = "",
    target_arch_context: str = "",
    include_hardware: bool = False,
) -> str:
    ensure_kernelbench_on_path(kernelbench_root)
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend

    safe_glue = prompt_option == "zero_shot_safe_glue"
    base_option = "zero_shot" if safe_glue else prompt_option
    prompt = get_prompt_for_backend(
        reference_code,
        "cuda",
        option=base_option,
        precision="fp32",
        include_hardware=include_hardware,
        gpu_name=gpu_name or None,
    )
    if safe_glue:
        prompt = f"{prompt.rstrip()}\n\n{safe_glue_prompt_suffix(target_arch_context=target_arch_context)}\n"
    return prompt


def extract_solution_code(response: str, kernelbench_root: str | Path | None = None) -> str:
    try:
        ensure_kernelbench_on_path(kernelbench_root)
        from kernelbench.utils import extract_first_code

        code = extract_first_code(response, ["python", "py", "cpp", "cuda", "c++"])
        if code:
            return code.strip()
    except Exception:
        pass

    for lang in ["python", "py", "cpp", "cuda", "c++", "c"]:
        match = re.search(rf"```{re.escape(lang)}\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def set_kernelbench_gpu_arch(arch: str, kernelbench_root: str | Path | None = None) -> None:
    ensure_kernelbench_on_path(kernelbench_root)
    from kernelbench.utils import set_gpu_arch

    set_gpu_arch([arch])


def kernelbench_pythonpath(kernelbench_root: str | Path | None = None) -> str:
    src = ensure_kernelbench_on_path(kernelbench_root)
    return str(src)


def _problem_from_kernelbench_item(level: int, item) -> KernelBenchProblem:
    return KernelBenchProblem(
        level=level,
        problem_id=int(getattr(item, "problem_id")),
        name=str(getattr(item, "name", f"problem_{getattr(item, 'problem_id', '')}")),
        reference_code=str(getattr(item, "code")),
        reference_path=getattr(item, "path", None),
    )


def default_kernelbench_root() -> str:
    return str((Path(__file__).resolve().parents[1] / "third_party" / "KernelBench").resolve())


def _resolve_kernelbench_root(kernelbench_root: str | Path | None = None) -> Path:
    configured = kernelbench_root or os.environ.get("KERNELBENCH_ROOT") or default_kernelbench_root()
    return Path(configured).expanduser().resolve()
