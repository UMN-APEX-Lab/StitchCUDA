"""Typer application root.

This module wires individual command modules into the top-level ``stitchcuda``
command. Each command lives in its own module so the help surface stays
discoverable and tests can import a command without booting the rest of the
CLI.

Subcommands implemented per development phase:

* Phase 1: ``run`` (back-compat for the legacy argparse CLI), ``doctor``
* Phase 2: ``run`` gains an interactive wizard, plus ``profile`` group
* Phase 4: ``runs`` group (list/show/diff)
* Phase 5: ``prompts`` group (list/edit)
"""

from __future__ import annotations

import typer

from . import doctor as doctor_module
from . import profile_cmd
from . import prompts_cmd
from . import run_cmd
from . import runs_cmd
from .console import stdout_console

app = typer.Typer(
    name="stitchcuda",
    help="StitchCUDA — multi-agent CUDA kernel generation and optimization workflow.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

run_cmd.register(app)
app.add_typer(profile_cmd.app, name="profile")
app.add_typer(runs_cmd.app, name="runs")
app.add_typer(prompts_cmd.app, name="prompts")


@app.command("doctor", help="Check the local environment (CUDA toolchain, KernelBench, credentials).")
def doctor() -> None:
    checks = doctor_module.run_checks()
    exit_code = doctor_module.render(checks, stdout_console())
    raise typer.Exit(code=exit_code)
