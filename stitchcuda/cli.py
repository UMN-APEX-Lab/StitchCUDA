"""Top-level command-line entry point for StitchCUDA.

This module is intentionally thin: it owns argument rewriting for backward
compatibility and delegates everything else to the Typer application in
:mod:`stitchcuda.tui.app`. Splitting things this way keeps the entry point
import cost low (heavier UI imports happen only after argv is settled) and
isolates the legacy argparse-style invocation contract in one place.

Backward compatibility
----------------------
Earlier releases shipped a single argparse parser, so users learnt to type
``python -m stitchcuda --level 1 --problem-id 1 ...``. In the Typer model
that same call would error because ``--level`` is not a recognised
subcommand. To keep every documented script working unchanged, we detect
that the first argument is a flag (begins with ``-`` and is not the help
sentinel) and prepend the implicit ``run`` subcommand.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


# Subcommands registered on the Typer application. Kept here (rather than
# imported from `tui.app`) so legacy-argv detection does not pull the whole
# UI stack on every invocation.
_KNOWN_SUBCOMMANDS: frozenset[str] = frozenset({"run", "doctor", "runs", "prompts", "profile"})
_HELP_TOKENS: frozenset[str] = frozenset({"-h", "--help", "--version"})


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by ``python -m stitchcuda`` and the ``stitchcuda`` script."""
    raw = list(sys.argv[1:] if argv is None else argv)
    rewritten = _inject_default_subcommand(raw)

    try:
        from .tui.app import app
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(
            f"stitchcuda CLI dependencies are missing ({missing}). "
            "Reinstall with `pip install -e .` to pull in typer/rich/questionary.",
            file=sys.stderr,
        )
        return 2

    try:
        app(args=rewritten, prog_name="stitchcuda")
    except SystemExit as exc:
        return _exit_code(exc.code)
    return 0


def _inject_default_subcommand(argv: list[str]) -> list[str]:
    """Prepend the implicit ``run`` subcommand for legacy flag-only invocations."""
    if not argv:
        return argv
    head = argv[0]
    if head in _KNOWN_SUBCOMMANDS or head in _HELP_TOKENS:
        return argv
    if head.startswith("-"):
        return ["run", *argv]
    return argv


def _exit_code(code: object) -> int:
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return code
    return 0 if code is None else 1
