"""``stitchcuda profile`` subcommand group.

Profiles are small TOML snapshots of run parameters. This command module is
only the CLI surface; persistence rules live in :mod:`stitchcuda.tui.profile`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from . import profile as profile_io
from .console import stderr_console, stdout_console
from .display import display_path
from .invocation import RunInvocation, invocation_field_names

app = typer.Typer(help="Manage saved run profiles.", no_args_is_help=True)


@app.command("list")
def list_cmd() -> None:
    """Show saved profile names and key launch parameters."""
    profiles = _load_all_or_exit()
    if not profiles:
        stdout_console().print(f"No profiles found in {display_path(profile_io.profiles_path())}.")
        return

    table = Table(title=f"Profiles in {display_path(profile_io.profiles_path())}", header_style="bold")
    table.add_column("Name")
    table.add_column("Level", justify="right")
    table.add_column("Problems")
    table.add_column("Model")
    table.add_column("GPU arch")
    table.add_column("Target SM")

    for name, values in sorted(profiles.items()):
        table.add_row(
            name,
            str(values.get("level", "?")),
            _format_problem_ids(values.get("problem_ids")),
            str(values.get("model", "?")),
            str(values.get("gpu_arch", "?")),
            str(values.get("target_sm") or "-"),
        )
    stdout_console().print(table)


@app.command("show")
def show_cmd(name: str = typer.Argument(..., help="Profile name.")) -> None:
    """Print one profile as a key/value table."""
    try:
        values = profile_io.load(name)
    except (KeyError, ValueError, OSError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value", overflow="fold")
    for key in sorted(values):
        table.add_row(key, _format_value(values[key]))
    stdout_console().print(Panel(table, title=f"Profile: {name}", border_style="cyan"))


@app.command("save")
def save_cmd(
    name: str = typer.Argument(..., help="Profile name to create or overwrite."),
    from_run: Path | None = typer.Option(
        None,
        "--from-run",
        help="Run directory whose config.json should be saved as this profile.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Save a profile from an existing run directory."""
    if from_run is None:
        _print_error("profile save currently requires --from-run RUN_DIR")
        raise typer.Exit(code=2)

    try:
        invocation = _invocation_from_run(from_run)
        path = profile_io.save(name, invocation)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc
    stdout_console().print(f"Saved profile [bold]{name}[/bold] to {display_path(path)}")


@app.command("rm")
def rm_cmd(name: str = typer.Argument(..., help="Profile name to remove.")) -> None:
    """Remove a saved profile."""
    try:
        path = profile_io.delete(name)
    except (KeyError, ValueError, OSError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc
    stdout_console().print(f"Removed profile [bold]{name}[/bold] from {display_path(path)}")


def _load_all_or_exit() -> dict[str, dict[str, Any]]:
    try:
        return profile_io.load_all()
    except (ValueError, OSError) as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc


def _invocation_from_run(run_dir: Path) -> RunInvocation:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json in {run_dir}")
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: expected a JSON object")

    summary = _read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    problem = summary.get("problem") if isinstance(summary, dict) else {}
    if not isinstance(problem, dict):
        problem = {}

    level = config.get("level", problem.get("level"))
    if level is None:
        raise ValueError(f"{config_path}: missing level")

    raw_problem_ids = config.get("problem_ids")
    if raw_problem_ids is None:
        problem_id = config.get("problem_id", problem.get("problem_id"))
        raw_problem_ids = [problem_id] if problem_id is not None else []
    problem_ids = [int(value) for value in raw_problem_ids if value is not None]
    if not problem_ids:
        raise ValueError(f"{config_path}: missing problem_id/problem_ids")

    known = invocation_field_names()
    kwargs = {key: value for key, value in config.items() if key in known}
    kwargs["level"] = int(level)
    kwargs["problem_ids"] = problem_ids
    return RunInvocation(**kwargs)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_problem_ids(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value == "":
        return "-"
    if isinstance(value, str) and (value.startswith("/") or value.startswith("~") or "/" in value):
        return display_path(value)
    return str(value)


def _print_error(message: str) -> None:
    stderr_console().print(f"[red]{message}[/red]")
