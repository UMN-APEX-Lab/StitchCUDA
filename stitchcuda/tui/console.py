"""Shared Rich console factories.

Centralising console construction here keeps colour/TTY policy in one place,
so commands do not each rediscover ``NO_COLOR`` or ``--no-color`` semantics.
Rich already honours the ``NO_COLOR`` environment variable and auto-degrades
when stdout is not a TTY; this module exists so future global toggles
(themes, force-no-color flag, log file mirror) have a single seam.
"""

from __future__ import annotations

from rich.console import Console

_stdout_console: Console | None = None
_stderr_console: Console | None = None


def stdout_console() -> Console:
    global _stdout_console
    if _stdout_console is None:
        _stdout_console = Console()
    return _stdout_console


def stderr_console() -> Console:
    global _stderr_console
    if _stderr_console is None:
        _stderr_console = Console(stderr=True)
    return _stderr_console
