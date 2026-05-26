"""``stitchcuda runs`` subcommand group — browse historical run directories.

A "run" is one of the timestamped subdirectories the workflow writes under
``--output-root`` (default ``runs/``). Each contains a ``summary.json`` that
this module treats as the source of truth, with ``config.json``,
``hardware.json``, and ``best_solution.py`` consulted opportunistically for
extra detail. Missing or malformed files are degraded gracefully so a single
broken run does not poison ``list`` output for the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .console import stderr_console, stdout_console
from .display import display_endpoint, display_path

app = typer.Typer(help="Browse historical runs under the output root.", no_args_is_help=True)


@app.command("list")
def list_cmd(
    output_root: Path = typer.Option(
        Path("runs"),
        "--output-root",
        help="Root directory of stored runs.",
        show_default=True,
    ),
) -> None:
    """Print one row per run under ``--output-root``."""
    if not output_root.is_dir():
        stderr_console().print(f"[red]No such directory: {output_root}[/red]")
        raise typer.Exit(code=1)

    rows = [
        (run_dir, summary)
        for run_dir, summary in _iter_runs(output_root)
    ]
    if not rows:
        stdout_console().print(f"No runs with summary.json found under {output_root}.")
        return

    table = Table(title=f"Runs in {output_root}", header_style="bold")
    table.add_column("Run dir", overflow="fold")
    table.add_column("Level", justify="right")
    table.add_column("Problem", overflow="fold")
    table.add_column("Attempts", justify="right")
    table.add_column("Best speedup", justify="right")
    table.add_column("Stop reason")

    for run_dir, summary in rows:
        problem = summary.get("problem") or {}
        best_result = (summary.get("best_attempt") or {}).get("result") or {}
        speedup = _float_or_zero(best_result.get("speedup"))
        table.add_row(
            run_dir.name,
            str(problem.get("level", "?")),
            f"{problem.get('problem_id', '?')} {problem.get('name', '')}".strip(),
            str(len(summary.get("attempts") or [])),
            _format_speedup(speedup),
            str(summary.get("stop_reason", "?")),
        )
    stdout_console().print(table)


@app.command("show")
def show_cmd(
    run_dir: Path = typer.Argument(..., help="Run directory to inspect."),
    show_solution: bool = typer.Option(
        False, "--solution/--no-solution", help="Also print best_solution.py with syntax highlighting."
    ),
) -> None:
    """Show the configuration, attempt table, and best-attempt details for a run."""
    summary = _read_json(run_dir / "summary.json")
    if summary is None:
        stderr_console().print(f"[red]Missing or unreadable summary.json in {run_dir}[/red]")
        raise typer.Exit(code=1)
    config = _read_json(run_dir / "config.json") or {}
    hardware = _read_json(run_dir / "hardware.json") or {}

    console = stdout_console()
    console.print(Panel(_identity_text(run_dir, summary, config, hardware), title="Run", border_style="cyan"))
    console.print(_attempts_table(summary))

    best = summary.get("best_attempt")
    if best:
        result = best.get("result") or {}
        console.print(Panel(_best_text(best, result), title="Best attempt", border_style="green"))
    else:
        console.print("[yellow]No correct attempt was produced for this run.[/yellow]")

    if show_solution:
        best_path = run_dir / "best_solution.py"
        if not best_path.exists():
            console.print(f"[yellow]No best_solution.py in {run_dir}.[/yellow]")
        else:
            console.print(
                Panel(
                    Syntax(
                        best_path.read_text(encoding="utf-8"),
                        "python",
                        theme="ansi_dark",
                        background_color="default",
                        line_numbers=True,
                    ),
                    title=display_path(best_path),
                    border_style="dim",
                )
            )


@app.command("diff")
def diff_cmd(
    run_a: Path = typer.Argument(..., help="First run directory."),
    run_b: Path = typer.Argument(..., help="Second run directory."),
) -> None:
    """Side-by-side comparison of two runs' key metrics."""
    summary_a = _read_json(run_a / "summary.json")
    summary_b = _read_json(run_b / "summary.json")
    if summary_a is None or summary_b is None:
        stderr_console().print("[red]One or both run directories lack summary.json[/red]")
        raise typer.Exit(code=1)

    panel_a = Panel(_diff_text(run_a, summary_a), title=run_a.name, border_style="cyan")
    panel_b = Panel(_diff_text(run_b, summary_b), title=run_b.name, border_style="magenta")
    stdout_console().print(Columns([panel_a, panel_b], expand=True))


# ---- helpers ------------------------------------------------------------


def _iter_runs(output_root: Path):
    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary = _read_json(run_dir / "summary.json")
        if summary is None:
            continue
        yield run_dir, summary


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _identity_text(run_dir: Path, summary: dict, config: dict, hardware: dict) -> str:
    problem = summary.get("problem") or {}
    return "\n".join(
        [
            f"[bold]Run dir:[/bold] {display_path(run_dir)}",
            f"[bold]Level:[/bold] {problem.get('level', '?')}   "
            f"[bold]Problem:[/bold] {problem.get('problem_id', '?')} {problem.get('name', '')}",
            f"[bold]Model:[/bold] {config.get('model', '?')}",
            f"[bold]Endpoint:[/bold] {display_endpoint(config.get('api_base'))}",
            f"[bold]GPU:[/bold] {hardware.get('gpu_name', '?')} "
            f"(target={hardware.get('target_sm', '?')})",
            f"[bold]Stop reason:[/bold] {summary.get('stop_reason', '?')}   "
            f"[bold]Replans:[/bold] {summary.get('replan_count', 0)}",
        ]
    )


def _attempts_table(summary: dict) -> Table:
    table = Table(title="Attempts", header_style="bold", title_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Stage", no_wrap=True)
    table.add_column("Compiled", no_wrap=True)
    table.add_column("Correct", no_wrap=True)
    table.add_column("Speedup", justify="right", no_wrap=True)
    table.add_column("Runtime µs", justify="right", no_wrap=True)
    table.add_column("Error", overflow="fold")
    for attempt in summary.get("attempts") or []:
        result = attempt.get("result") or {}
        speedup = _float_or_zero(result.get("speedup"))
        runtime = _float_or_neg(result.get("runtime_us"))
        table.add_row(
            str(attempt.get("attempt", "?")),
            str(attempt.get("stage", "?")),
            _bool_cell(result.get("compiled", False)),
            _bool_cell(result.get("correct", False)),
            _format_speedup(speedup),
            f"{runtime:.1f}" if runtime > 0 else "—",
            (result.get("error") or "")[:120],
        )
    return table


def _best_text(best: dict, result: dict) -> str:
    speedup = _float_or_zero(result.get("speedup"))
    runtime = _float_or_neg(result.get("runtime_us"))
    return "\n".join(
        [
            f"Attempt: [bold]{best.get('attempt', '?')}[/bold]   Stage: {best.get('stage', '?')}",
            f"Speedup: [bold green]{speedup:.3f}[/bold green]   "
            f"Runtime: {runtime:.1f} µs" if runtime > 0 else f"Speedup: [bold green]{speedup:.3f}[/bold green]",
            f"Solution: {display_path(best.get('solution_path'))}",
        ]
    )


def _diff_text(run_dir: Path, summary: dict) -> str:
    problem = summary.get("problem") or {}
    best = summary.get("best_attempt") or {}
    result = best.get("result") or {}
    speedup = _float_or_zero(result.get("speedup"))
    runtime = _float_or_neg(result.get("runtime_us"))
    speedup_line = (
        f"Best speedup: [bold green]{speedup:.3f}[/bold green]"
        if speedup
        else "Best speedup: [dim]—[/dim]"
    )
    return "\n".join(
        [
            f"[bold]Run:[/bold] {run_dir.name}",
            f"L{problem.get('level', '?')} P{problem.get('problem_id', '?')} {problem.get('name', '')}",
            f"Stop reason: {summary.get('stop_reason', '?')}",
            f"Attempts: {len(summary.get('attempts') or [])}   "
            f"Replans: {summary.get('replan_count', 0)}",
            speedup_line,
            f"Best stage: {best.get('stage', '—')}",
            f"Runtime µs: {runtime:.1f}" if runtime > 0 else "Runtime µs: —",
            f"Error: {(result.get('error') or '')[:200] or '—'}",
        ]
    )


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_neg(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _format_speedup(value: float) -> str:
    return f"{value:.3f}" if value else "—"


def _bool_cell(value: bool) -> str:
    return "[green]yes[/green]" if value else "[red]no[/red]"
