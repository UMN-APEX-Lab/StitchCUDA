from __future__ import annotations

import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from torch.utils import cpp_extension


_ORIGINAL_LOAD_INLINE = cpp_extension.load_inline
_INSTALLED = False
_NAME_COUNTERS: dict[str, int] = defaultdict(int)

_LOAD_INLINE_PARAMS = [
    "name",
    "cpp_sources",
    "cuda_sources",
    "sycl_sources",
    "functions",
    "extra_cflags",
    "extra_cuda_cflags",
    "extra_sycl_cflags",
    "extra_ldflags",
    "extra_include_paths",
    "build_directory",
    "verbose",
    "with_cuda",
    "with_sycl",
    "is_python_module",
    "with_pytorch_error_handling",
    "keep_intermediates",
    "use_pch",
    "no_implicit_headers",
]
_IGNORED_KWARGS = {"force_build"}


def install_cuda_extension_scaffold() -> None:
    """Route inline CUDA extension builds through StitchCUDA's managed scaffold."""

    global _INSTALLED
    if _INSTALLED:
        return
    cpp_extension.load_inline = managed_load_inline
    _INSTALLED = True


def managed_load_inline(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for torch.utils.cpp_extension.load_inline."""

    call_kwargs = _kwargs_from_load_inline_call(args, kwargs)
    prepared = _prepare_load_inline_kwargs(call_kwargs)
    return _ORIGINAL_LOAD_INLINE(**prepared)


def build_load_inline_extension(
    *,
    name: str,
    cuda_sources: str | Iterable[str],
    functions: str | Iterable[str],
    cpp_sources: str | Iterable[str] | None = None,
    extra_cflags: Iterable[str] | None = None,
    extra_cuda_cflags: Iterable[str] | None = None,
    extra_ldflags: Iterable[str] | None = None,
    extra_include_paths: Iterable[str] | None = None,
    verbose: bool = False,
    with_pytorch_error_handling: bool = True,
    keep_intermediates: bool = True,
) -> Any:
    """Build a CUDA extension using the framework-owned load_inline scaffold."""

    return _ORIGINAL_LOAD_INLINE(
        **_prepare_load_inline_kwargs(
            {
                "name": name,
                "cpp_sources": cpp_sources or "",
                "cuda_sources": cuda_sources,
                "functions": functions,
                "extra_cflags": list(extra_cflags or []),
                "extra_cuda_cflags": list(extra_cuda_cflags or []),
                "extra_ldflags": list(extra_ldflags or []),
                "extra_include_paths": list(extra_include_paths or []),
                "verbose": verbose,
                "with_cuda": True,
                "with_pytorch_error_handling": with_pytorch_error_handling,
                "keep_intermediates": keep_intermediates,
            }
        )
    )


build_cuda_extension = build_load_inline_extension


def _kwargs_from_load_inline_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) > len(_LOAD_INLINE_PARAMS):
        raise TypeError(f"load_inline expected at most {len(_LOAD_INLINE_PARAMS)} positional arguments")

    normalized = dict(kwargs)
    for index, value in enumerate(args):
        key = _LOAD_INLINE_PARAMS[index]
        if key in normalized:
            raise TypeError(f"load_inline got multiple values for argument '{key}'")
        normalized[key] = value
    return normalized


def _prepare_load_inline_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(kwargs)
    for key in list(prepared):
        if key in _IGNORED_KWARGS:
            prepared.pop(key)

    unknown = sorted(set(prepared) - set(_LOAD_INLINE_PARAMS))
    if unknown:
        raise TypeError(f"unsupported load_inline arguments under StitchCUDA scaffold: {', '.join(unknown)}")

    name = str(prepared.get("name") or "stitchcuda_extension")
    managed_name = _managed_extension_name(name)
    prepared["name"] = managed_name
    prepared["build_directory"] = str(_managed_build_directory(managed_name))

    cuda_sources = _source_text(prepared.get("cuda_sources"))
    cpp_sources = _source_text(prepared.get("cpp_sources"))
    if _contains_cuda_launch(cpp_sources):
        raise ValueError(
            "cpp_sources must contain declarations or CPU C++ only; CUDA <<<...>>> launches belong in cuda_sources"
        )

    functions = _normalize_functions(prepared.get("functions"))
    if functions:
        prepared["functions"] = functions
        prepared["cpp_sources"] = _merge_cpp_declarations(cpp_sources, cuda_sources, functions)
    elif "functions" in prepared:
        prepared["functions"] = []

    if "cpp_sources" not in prepared:
        prepared["cpp_sources"] = ""
    if prepared.get("with_cuda") is None and cuda_sources:
        prepared["with_cuda"] = True
    return prepared


def _managed_extension_name(name: str) -> str:
    base = _sanitize_identifier(name) or "extension"
    prefix = _sanitize_identifier(os.environ.get("STITCHCUDA_EXTENSION_PREFIX", "")) or "stitchcuda"
    if base.startswith(prefix + "_"):
        stem = base
    else:
        stem = f"{prefix}_{base}"

    count = _NAME_COUNTERS[stem]
    _NAME_COUNTERS[stem] += 1
    return stem if count == 0 else f"{stem}_{count}"


def _managed_build_directory(module_name: str) -> Path:
    root = (
        os.environ.get("STITCHCUDA_BUILD_DIR")
        or os.environ.get("TORCH_EXTENSIONS_DIR")
        or str(Path(tempfile.gettempdir()) / "stitchcuda_extensions")
    )
    path = Path(root).expanduser().resolve() / module_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_functions(value: Any) -> list[str]:
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    names: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", text):
            names.append(text)
            continue
        match = re.search(r"([A-Za-z_]\w*)\s*\(", text)
        if match:
            names.append(match.group(1))
            continue
        raise ValueError(f"invalid exported function name for load_inline: {text!r}")
    return names


def _merge_cpp_declarations(cpp_sources: str, cuda_sources: str, functions: list[str]) -> str:
    declarations = [cpp_sources.strip()] if cpp_sources.strip() else []
    for function in functions:
        if re.search(rf"\b{re.escape(function)}\s*\(", cpp_sources):
            continue
        declaration = _infer_function_declaration(cuda_sources, function)
        if declaration:
            declarations.append(declaration)
    return "\n\n".join(part for part in declarations if part).strip()


def _infer_function_declaration(cuda_sources: str, function: str) -> str:
    pattern = re.compile(
        rf"(?ms)^[ \t]*(?!__global__\b)(?!__device__\b)(?!__host__\b)"
        rf"(?P<ret>[A-Za-z_][\w:\s<>,*&]*?)\s+{re.escape(function)}\s*"
        rf"\((?P<args>[^;{{}}]*)\)\s*\{{"
    )
    match = pattern.search(cuda_sources)
    if not match:
        return ""
    ret = _collapse_ws(match.group("ret"))
    args = _collapse_ws(match.group("args"))
    return f"{ret} {function}({args});"


def _source_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return "\n".join(str(item) for item in value)


def _contains_cuda_launch(text: str) -> bool:
    return "<<<" in text and ">>>" in text


def _sanitize_identifier(value: str) -> str:
    text = re.sub(r"\W+", "_", str(value).strip())
    text = text.strip("_")
    if not text:
        return ""
    if text[0].isdigit():
        text = "_" + text
    return text


def _collapse_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _reset_scaffold_state_for_tests() -> None:
    _NAME_COUNTERS.clear()
