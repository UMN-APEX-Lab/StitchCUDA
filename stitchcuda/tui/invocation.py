"""Wire format and execution helpers shared by every ``run`` entry point.

The :class:`RunInvocation` dataclass is the single source of truth for what
"one launch of the workflow" looks like. Three callers translate into it:

* the legacy-flag-compatible Typer command in :mod:`stitchcuda.tui.run_cmd`
* the interactive wizard in :mod:`stitchcuda.tui.wizard`
* saved profiles loaded via :mod:`stitchcuda.tui.profile`

Keeping this layer free of Typer/questionary/I-O lets each surface evolve
independently and makes the run logic trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, List, Optional

import typer

from ..events import EventSink, NullSink
from ..kernelbench import load_dataset_problems
from ..workflow import FixedWorkflowConfig, StitchCUDAWorkflow
from .display import display_path

SinkFactory = Callable[[], EventSink]


@dataclass
class RunInvocation:
    """All inputs needed to launch one or more workflow runs."""

    level: int
    problem_ids: List[int] = field(default_factory=list)
    model: str = "gpt-4o"
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 65536
    reasoning_effort: str = ""
    kernelbench_root: str = ""
    output_root: str = "runs"
    run_name: str = ""
    prompt_option: str = "zero_shot_safe_glue"
    gpu_arch: str = "Blackwell"
    target_sm: str = ""
    device: Optional[int] = None
    max_attempts: int = 3
    max_replans: int = 2
    replan_after_failed_attempts: int = 2
    replan_after_stagnant_attempts: int = 2
    target_speedup: float = 1.0
    num_correct_trials: int = 5
    num_perf_trials: int = 10
    measure_performance: bool = True
    verifier_timeout_s: int = 1800


# Fields excluded from saved profiles. Secrets and per-machine paths should
# come from env vars / CLI flags at runtime, never from a versioned profile.
PROFILE_EXCLUDED_FIELDS: frozenset[str] = frozenset({"api_key"})


def invocation_field_names() -> set[str]:
    return {f.name for f in fields(RunInvocation)}


def resolve_problem_ids(
    *,
    level: int,
    kernelbench_root: str,
    problem_ids: list[int] | None,
    max_problems: int,
) -> list[int]:
    """Return the problem ids to run.

    If the caller supplied explicit ids, those are returned verbatim. Otherwise
    the first ``max_problems`` (>= 1) ids from the KernelBench dataset at the
    given level are returned, preserving dataset order.
    """
    if problem_ids:
        return list(problem_ids)
    problems = load_dataset_problems(level=level, kernelbench_root=kernelbench_root, problem_ids=None)
    limit = max(1, int(max_problems))
    return [problem.problem_id for problem in problems[:limit]]


def execute(invocation: RunInvocation, *, make_sink: SinkFactory | None = None) -> list[dict]:
    """Run the workflow for every problem id and return the summaries.

    ``make_sink``, if given, is called once per problem to construct a fresh
    :class:`EventSink` for that run. The sink is used as a context manager so
    it can own short-lived resources (e.g. a Rich Live controller). When
    ``make_sink`` is ``None`` a :class:`NullSink` is used and the workflow
    behaves exactly as the legacy CLI.
    """
    if not invocation.problem_ids:
        raise ValueError("RunInvocation.problem_ids must contain at least one id")

    summaries: list[dict] = []
    multi = len(invocation.problem_ids) > 1
    for problem_id in invocation.problem_ids:
        cfg = _to_workflow_config(invocation, problem_id, multi=multi)
        sink = make_sink() if make_sink is not None else NullSink()
        with sink:
            summary = StitchCUDAWorkflow(cfg, events=sink).run()
        summaries.append(summary)
        _print_short_summary(invocation.level, problem_id, summary)
    return summaries


def _to_workflow_config(invocation: RunInvocation, problem_id: int, *, multi: bool) -> FixedWorkflowConfig:
    # `run_name` labels a single run directory; in batch runs we drop it so
    # every problem gets its own timestamped folder.
    run_name = "" if multi else invocation.run_name
    return FixedWorkflowConfig(
        level=invocation.level,
        problem_id=problem_id,
        model=invocation.model,
        api_base=invocation.api_base,
        api_key=invocation.api_key,
        temperature=invocation.temperature,
        max_tokens=invocation.max_tokens,
        reasoning_effort=invocation.reasoning_effort,
        kernelbench_root=invocation.kernelbench_root,
        output_root=invocation.output_root,
        run_name=run_name,
        prompt_option=invocation.prompt_option,
        gpu_arch=invocation.gpu_arch,
        target_sm=invocation.target_sm,
        device=invocation.device,
        max_attempts=invocation.max_attempts,
        max_replans=invocation.max_replans,
        replan_after_failed_attempts=invocation.replan_after_failed_attempts,
        replan_after_stagnant_attempts=invocation.replan_after_stagnant_attempts,
        target_speedup=invocation.target_speedup,
        num_correct_trials=invocation.num_correct_trials,
        num_perf_trials=invocation.num_perf_trials,
        measure_performance=invocation.measure_performance,
        verifier_timeout_s=invocation.verifier_timeout_s,
    )


def _print_short_summary(level: int, problem_id: int, summary: dict) -> None:
    best = summary.get("best_attempt") or {}
    result = best.get("result") or {}
    speedup = float(result.get("speedup", 0.0) or 0.0)
    typer.echo(
        f"L{level} P{problem_id}: {summary.get('stop_reason', 'unknown')} "
        f"best_speedup={speedup:.3f} run_dir={display_path(summary.get('run_dir'))}"
    )
