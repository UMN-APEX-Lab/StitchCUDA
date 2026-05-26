"""The ``stitchcuda run`` Typer command.

Three entry modes are supported by this one command:

* **Flag-driven** — the legacy CLI surface (``--level``, ``--problem-id``, …).
* **Interactive editor** — automatic when no ``--level`` and no ``--profile``
  are given. Reuses :mod:`stitchcuda.tui.wizard` so the terminal UI can be
  exercised on its own.
* **Profile-driven** — ``--profile NAME`` loads defaults from
  ``profiles.toml``; any explicit flag overrides the corresponding profile
  value, so users can pin a named configuration and selectively tweak one
  field per invocation.

The merge order is, for every field, **explicit CLI > profile > built-in
default**. "Explicit CLI" is detected through Click's
``Context.get_parameter_source`` API so we never have to add sentinel values
to every Typer ``Option`` signature.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, List, Optional

import click
import typer

from ..events import EventSink
from ..kernelbench import default_kernelbench_root
from . import profile as profile_io
from . import wizard
from .console import stderr_console, stdout_console
from .display import display_path
from .invocation import RunInvocation, SinkFactory, execute, resolve_problem_ids

# Typer arg name -> profile key. The plural form is what the profile stores,
# since it captures "the list to actually run", not "what the user typed".
_PROFILE_KEY_OVERRIDES = {"problem_id": "problem_ids"}


def register(app: typer.Typer) -> None:
    """Attach the ``run`` command to a Typer application."""

    @app.command(
        "run",
        help=(
            "Run the planner → coder → verifier workflow. "
            "With no --level and no --profile, an interactive config editor opens."
        ),
    )
    def run(  # noqa: PLR0913 — flag surface mirrors the legacy CLI for back-compat.
        ctx: typer.Context,
        profile: str = typer.Option(
            "", "--profile", help="Load defaults from a saved profile (see `stitchcuda profile list`)."
        ),
        level: Optional[int] = typer.Option(
            None, "--level", help="KernelBench level. Required unless --profile supplies it."
        ),
        problem_id: Optional[List[int]] = typer.Option(
            None, "--problem-id", help="KernelBench problem id; pass multiple times for batch runs."
        ),
        max_problems: int = typer.Option(
            1, "--max-problems", help="Number of problems to run when --problem-id is omitted."
        ),
        model: str = typer.Option(
            os.environ.get("STITCHCUDA_MODEL", "gpt-4o"), "--model", help="LLM model name."
        ),
        api_base: str = typer.Option(
            os.environ.get("STITCHCUDA_API_BASE", ""),
            "--api-base",
            help="OpenAI-compatible base URL. Omit for the hosted OpenAI endpoint.",
            show_default=False,
        ),
        api_key: str = typer.Option(
            os.environ.get("OPENAI_API_KEY", ""),
            "--api-key",
            help="API key. Defaults to $OPENAI_API_KEY. Never stored in profiles.",
            show_default=False,
        ),
        temperature: float = typer.Option(0.7, "--temperature"),
        max_tokens: int = typer.Option(65536, "--max-tokens"),
        reasoning_effort: str = typer.Option("", "--reasoning-effort"),
        kernelbench_root: str = typer.Option(
            os.environ.get("KERNELBENCH_ROOT") or default_kernelbench_root(),
            "--kernelbench-root",
            help="KernelBench checkout root. Defaults to the bundled submodule.",
            show_default=False,
        ),
        output_root: str = typer.Option("runs", "--output-root"),
        run_name: str = typer.Option(
            "", "--run-name", help="Custom run directory name (single-problem runs only)."
        ),
        prompt_option: str = typer.Option("zero_shot_safe_glue", "--prompt-option"),
        gpu_arch: str = typer.Option(
            "Blackwell",
            "--gpu-arch",
            help="KernelBench architecture name (e.g. Blackwell, Hopper, Ada, Ampere).",
        ),
        target_sm: str = typer.Option(
            "", "--target-sm", help="Target SM for prompts, e.g. sm_120. Auto-detected if omitted."
        ),
        device: Optional[int] = typer.Option(None, "--device", help="CUDA device index."),
        max_attempts: int = typer.Option(3, "--max-attempts"),
        max_replans: int = typer.Option(2, "--max-replans"),
        replan_after_failed_attempts: int = typer.Option(2, "--replan-after-failed-attempts"),
        replan_after_stagnant_attempts: int = typer.Option(2, "--replan-after-stagnant-attempts"),
        target_speedup: float = typer.Option(1.0, "--target-speedup"),
        num_correct_trials: int = typer.Option(5, "--num-correct-trials"),
        num_perf_trials: int = typer.Option(10, "--num-perf-trials"),
        no_performance: bool = typer.Option(False, "--no-performance", help="Skip performance measurement."),
        verifier_timeout_s: int = typer.Option(1800, "--verifier-timeout-s"),
        no_live: bool = typer.Option(
            False,
            "--no-live",
            help="Disable the live Rich dashboard. Auto-disabled when stdout is not a TTY.",
        ),
    ) -> None:
        if level is None and not profile:
            invocation = _run_wizard()
        else:
            invocation = _build_from_flags(
                ctx,
                profile_name=profile,
                cli=dict(
                    level=level,
                    problem_id=problem_id,
                    max_problems=max_problems,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    kernelbench_root=kernelbench_root,
                    output_root=output_root,
                    run_name=run_name,
                    prompt_option=prompt_option,
                    gpu_arch=gpu_arch,
                    target_sm=target_sm,
                    device=device,
                    max_attempts=max_attempts,
                    max_replans=max_replans,
                    replan_after_failed_attempts=replan_after_failed_attempts,
                    replan_after_stagnant_attempts=replan_after_stagnant_attempts,
                    target_speedup=target_speedup,
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials,
                    no_performance=no_performance,
                    verifier_timeout_s=verifier_timeout_s,
                ),
            )

        summaries = execute(invocation, make_sink=_choose_sink_factory(no_live=no_live))
        typer.echo(json.dumps({"runs": summaries}, indent=2, default=str))


def _run_wizard() -> RunInvocation:
    invocation = wizard.prompt()
    if invocation is None:
        stderr_console().print(
            "[red]Interactive editor cancelled or no TTY available.[/red] "
            "Pass --level (and other flags) for non-interactive use."
        )
        raise typer.Exit(code=1)
    decision = wizard.prompt_save_decision()
    if decision is None:
        # User cancelled the save prompt; treat as "do not save" but still run.
        decision = ""
    if decision:
        path = profile_io.save(decision, invocation)
        stdout_console().print(f"Saved profile [bold]{decision}[/bold] to {display_path(path)}")
    return invocation


def _build_from_flags(ctx: typer.Context, *, profile_name: str, cli: dict[str, Any]) -> RunInvocation:
    profile_data = profile_io.load(profile_name) if profile_name else {}

    resolved_level = _merge(ctx, profile_data, cli, "level")
    if resolved_level is None:
        raise typer.BadParameter(
            f"profile {profile_name!r} does not set 'level' and --level was not provided"
            if profile_name
            else "--level is required (or use --profile NAME)"
        )

    resolved_kbroot = _merge(ctx, profile_data, cli, "kernelbench_root")
    problem_ids = _resolve_problem_ids(
        ctx,
        profile_data=profile_data,
        cli_problem_ids=cli["problem_id"] or [],
        cli_max_problems=cli["max_problems"],
        level=resolved_level,
        kernelbench_root=resolved_kbroot,
    )

    return RunInvocation(
        level=resolved_level,
        problem_ids=problem_ids,
        model=_merge(ctx, profile_data, cli, "model"),
        api_base=_merge(ctx, profile_data, cli, "api_base"),
        api_key=cli["api_key"],  # secrets never round-trip through profiles
        temperature=_merge(ctx, profile_data, cli, "temperature"),
        max_tokens=_merge(ctx, profile_data, cli, "max_tokens"),
        reasoning_effort=_merge(ctx, profile_data, cli, "reasoning_effort"),
        kernelbench_root=resolved_kbroot,
        output_root=_merge(ctx, profile_data, cli, "output_root"),
        run_name=_merge(ctx, profile_data, cli, "run_name"),
        prompt_option=_merge(ctx, profile_data, cli, "prompt_option"),
        gpu_arch=_merge(ctx, profile_data, cli, "gpu_arch"),
        target_sm=_merge(ctx, profile_data, cli, "target_sm"),
        device=_merge(ctx, profile_data, cli, "device"),
        max_attempts=_merge(ctx, profile_data, cli, "max_attempts"),
        max_replans=_merge(ctx, profile_data, cli, "max_replans"),
        replan_after_failed_attempts=_merge(ctx, profile_data, cli, "replan_after_failed_attempts"),
        replan_after_stagnant_attempts=_merge(ctx, profile_data, cli, "replan_after_stagnant_attempts"),
        target_speedup=_merge(ctx, profile_data, cli, "target_speedup"),
        num_correct_trials=_merge(ctx, profile_data, cli, "num_correct_trials"),
        num_perf_trials=_merge(ctx, profile_data, cli, "num_perf_trials"),
        measure_performance=_resolve_measure_performance(ctx, profile_data, cli["no_performance"]),
        verifier_timeout_s=_merge(ctx, profile_data, cli, "verifier_timeout_s"),
    )


def _merge(ctx: typer.Context, profile_data: dict, cli: dict[str, Any], typer_name: str) -> Any:
    """Pick the value for ``typer_name`` per the explicit > profile > default order."""
    if _was_explicit(ctx, typer_name):
        return cli[typer_name]
    profile_key = _PROFILE_KEY_OVERRIDES.get(typer_name, typer_name)
    if profile_key in profile_data:
        return profile_data[profile_key]
    return cli[typer_name]


def _resolve_measure_performance(ctx: typer.Context, profile_data: dict, no_performance_cli: bool) -> bool:
    # The CLI flag is inverted (--no-performance) but the profile / invocation
    # field is positive (measure_performance). Map carefully across both surfaces.
    if _was_explicit(ctx, "no_performance"):
        return not no_performance_cli
    if "measure_performance" in profile_data:
        return bool(profile_data["measure_performance"])
    return not no_performance_cli  # CLI default = False, so measure = True


def _resolve_problem_ids(
    ctx: typer.Context,
    *,
    profile_data: dict,
    cli_problem_ids: list[int],
    cli_max_problems: int,
    level: int,
    kernelbench_root: str,
) -> list[int]:
    """Apply the explicit > profile > default order to problem id selection."""
    if _was_explicit(ctx, "problem_id") and cli_problem_ids:
        return list(cli_problem_ids)
    profile_ids = profile_data.get("problem_ids") or []
    if profile_ids:
        return [int(i) for i in profile_ids]
    max_problems = (
        cli_max_problems
        if _was_explicit(ctx, "max_problems")
        else int(profile_data.get("max_problems", cli_max_problems))
    )
    return resolve_problem_ids(
        level=level,
        kernelbench_root=kernelbench_root,
        problem_ids=None,
        max_problems=max_problems,
    )


def _was_explicit(ctx: typer.Context, name: str) -> bool:
    try:
        source = ctx.get_parameter_source(name)
    except Exception:
        return False
    return source == click.core.ParameterSource.COMMANDLINE


def _choose_sink_factory(*, no_live: bool) -> SinkFactory | None:
    """Pick the right sink based on user flag and whether stdout is a TTY."""
    if no_live or not sys.stdout.isatty():
        return None
    # Defer the import so non-Live invocations don't pay for it.
    from .sinks import RichLiveSink

    return RichLiveSink
