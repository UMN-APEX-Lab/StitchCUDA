"""``stitchcuda prompts`` subcommand group — inspect and edit prompt templates.

Templates live in ``<repo>/prompts/`` and are loaded by
:func:`stitchcuda.templates.render_template`. This module never duplicates
that path; it imports :data:`stitchcuda.templates.PROMPT_ROOT` so changes
to the storage layout stay in one place.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..templates import PROMPT_ROOT
from .console import stderr_console, stdout_console

app = typer.Typer(help="Inspect or edit prompt templates.", no_args_is_help=True)


@app.command("list")
def list_cmd() -> None:
    """Show one row per ``*.md`` template under the prompts directory."""
    if not PROMPT_ROOT.is_dir():
        stderr_console().print(f"[red]Prompts directory not found: {PROMPT_ROOT}[/red]")
        raise typer.Exit(code=1)
    files = sorted(PROMPT_ROOT.glob("*.md"))
    if not files:
        stdout_console().print(f"No prompts found in {PROMPT_ROOT}.")
        return

    table = Table(title=f"Prompts in {PROMPT_ROOT}", header_style="bold")
    table.add_column("Name")
    table.add_column("Lines", justify="right")
    table.add_column("Bytes", justify="right")
    table.add_column("Last modified")
    for path in files:
        stat = path.stat()
        mtime = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
        table.add_row(path.name, str(line_count), str(stat.st_size), mtime)
    stdout_console().print(table)


@app.command("show")
def show_cmd(name: str = typer.Argument(..., help="Prompt file name, e.g. planner.md")) -> None:
    """Render a prompt template as Markdown."""
    path = _resolve(name)
    text = path.read_text(encoding="utf-8")
    stdout_console().print(Panel(Markdown(text), title=str(path), border_style="dim"))


@app.command("edit")
def edit_cmd(name: str = typer.Argument(..., help="Prompt file name, e.g. planner.md")) -> None:
    """Open a prompt template in $EDITOR (falls back to a system default)."""
    path = _resolve(name)
    typer.edit(filename=str(path))


@app.command("path")
def path_cmd() -> None:
    """Print the absolute path of the prompts directory."""
    stdout_console().print(str(PROMPT_ROOT))


def _resolve(name: str) -> Path:
    path = PROMPT_ROOT / name
    if path.exists():
        return path
    if not name.endswith(".md"):
        alt = PROMPT_ROOT / f"{name}.md"
        if alt.exists():
            return alt
    stderr_console().print(f"[red]Prompt not found: {path}[/red]")
    raise typer.Exit(code=1)
