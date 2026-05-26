"""Display-only redaction helpers for terminal output."""

from __future__ import annotations

import os
from pathlib import Path


def display_path(value: str | Path | None) -> str:
    """Return a user-safe path for UI output without changing stored values."""
    if value is None:
        return "-"
    text = str(value)
    if not text:
        return "-"
    path = Path(text).expanduser()
    if not path.is_absolute():
        return text

    resolved = path.resolve()
    for root in (_project_root(), Path.cwd().resolve()):
        try:
            return str(resolved.relative_to(root)) or "."
        except ValueError:
            pass

    home = Path.home().resolve()
    try:
        rel_home = resolved.relative_to(home)
        return str(Path("~") / rel_home)
    except ValueError:
        return path.name or "<absolute-path>"


def display_executable(value: str | Path | None) -> str:
    if value is None:
        return "-"
    return Path(str(value)).name


def display_endpoint(value: str | None) -> str:
    if not value:
        return "hosted OpenAI"
    text = str(value)
    for host in (os.environ.get("USER", ""), os.environ.get("HOME", "")):
        if host:
            text = text.replace(host, "<redacted>")
    return text


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
