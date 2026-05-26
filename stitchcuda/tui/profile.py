"""Read and write user run profiles (``~/.config/stitchcuda/profiles.toml``).

A profile is a named, on-disk snapshot of the parameters of a
:class:`RunInvocation`, so a user does not have to retype a long flag list
every session. Profiles are intentionally restricted:

* Secrets (currently ``api_key``) are never persisted; the runtime always
  resolves them from ``--api-key`` or ``$OPENAI_API_KEY``.
* Only fields whose values can round-trip through TOML are stored; ``None``
  entries are dropped on save.

The on-disk schema is::

    [profiles.<name>]
    level = 1
    problem_ids = [1, 2]
    model = "gpt-4o"
    ...

Override the file location with ``$STITCHCUDA_PROFILES`` (mainly useful for
tests).
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:  # pragma: no cover — exercised on Python 3.10 only
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

from .invocation import PROFILE_EXCLUDED_FIELDS, RunInvocation, invocation_field_names

_DEFAULT_PATH = Path.home() / ".config" / "stitchcuda" / "profiles.toml"


def profiles_path() -> Path:
    override = os.environ.get("STITCHCUDA_PROFILES")
    return Path(override).expanduser() if override else _DEFAULT_PATH


def load_all() -> dict[str, dict[str, Any]]:
    path = profiles_path()
    if not path.exists():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError(f"{path}: top-level [profiles] must be a table")
    return {name: dict(values) for name, values in profiles.items()}


def load(name: str) -> dict[str, Any]:
    profiles = load_all()
    if name not in profiles:
        raise KeyError(f"profile {name!r} not found in {profiles_path()}")
    return _filter_known_fields(profiles[name])


def save(name: str, invocation: RunInvocation) -> Path:
    """Persist ``invocation`` under ``name``, overwriting any prior entry."""
    target = profiles_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    profiles = load_all()
    profiles[name] = _serialise(invocation)
    target.write_text(tomli_w.dumps({"profiles": profiles}), encoding="utf-8")
    return target


def delete(name: str) -> Path:
    target = profiles_path()
    profiles = load_all()
    if name not in profiles:
        raise KeyError(f"profile {name!r} not found")
    del profiles[name]
    if profiles:
        target.write_text(tomli_w.dumps({"profiles": profiles}), encoding="utf-8")
    else:
        # Leave an empty table rather than deleting the file so future writes
        # do not need to recreate parent directories.
        target.write_text(tomli_w.dumps({"profiles": {}}), encoding="utf-8")
    return target


def _serialise(invocation: RunInvocation) -> dict[str, Any]:
    data = asdict(invocation)
    for excluded in PROFILE_EXCLUDED_FIELDS:
        data.pop(excluded, None)
    return {key: value for key, value in data.items() if value is not None}


def _filter_known_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop unknown keys so future schema changes do not crash old profiles."""
    known = invocation_field_names()
    return {key: value for key, value in raw.items() if key in known}
