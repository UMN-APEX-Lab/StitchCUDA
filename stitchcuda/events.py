"""Workflow event sink interface.

:class:`StitchCUDAWorkflow` does not know how its progress is displayed.
Instead, it calls hooks on an :class:`EventSink` at fixed boundaries — run
start/end, planning, attempt start, code generation, verification, replan.
Anything else (a TUI dashboard, a log file, a remote pubsub stream, a test
spy) plugs in by subclassing :class:`EventSink` and overriding the methods
it cares about.

The default :class:`NullSink` does nothing, so legacy code that constructs
``StitchCUDAWorkflow(cfg)`` without passing a sink keeps its original
behaviour.

Design notes
------------
* Sinks are context managers (``__enter__`` / ``__exit__``) so they can own
  short-lived UI resources (e.g. a ``rich.live.Live`` controller) without
  the workflow having to know about them.
* Event payloads are passed as keyword arguments and use primitive types
  (``int``, ``float``, ``str``, ``dict``, ``Path``). Sinks should never need
  to import workflow-internal dataclasses; that keeps the contract narrow
  and trivially loggable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EventSink:
    """No-op event sink. Override only the hooks you need."""

    def __enter__(self) -> "EventSink":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    # --- lifecycle --------------------------------------------------------

    def on_run_start(
        self,
        *,
        level: int,
        problem_id: int,
        problem_name: str,
        run_dir: Path,
        hardware: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """The workflow has loaded the problem and created the run directory."""

    def on_run_end(
        self,
        *,
        stop_reason: str,
        best_attempt: int | None,
        best_speedup: float,
    ) -> None:
        """The workflow has finished (either cleanly or by giving up)."""

    # --- planning ---------------------------------------------------------

    def on_plan_start(self, *, version: int, reason: str = "") -> None:
        """The planner is about to be invoked (``reason`` empty on initial plan)."""

    def on_plan_done(self, *, version: int, summary: str) -> None:
        """The planner has returned plan version ``version``."""

    # --- per attempt ------------------------------------------------------

    def on_attempt_start(self, *, attempt: int, stage: str, plan_version: int) -> None:
        """A new candidate iteration is starting with the given stage label."""

    def on_code_start(self, *, attempt: int) -> None:
        """The coder agent is about to be invoked for this attempt."""

    def on_code_done(self, *, attempt: int, code_chars: int) -> None:
        """The coder agent has returned a candidate solution."""

    def on_verify_start(self, *, attempt: int) -> None:
        """The verifier subprocess is about to be launched for this attempt."""

    def on_verify_done(
        self,
        *,
        attempt: int,
        compiled: bool,
        correct: bool,
        speedup: float,
        runtime_us: float,
        ref_runtime_us: float,
        error: str,
        is_best: bool,
    ) -> None:
        """The verifier has returned a result for this attempt."""


NullSink = EventSink
