"""Rich-based :class:`EventSink` implementations.

:class:`RichLiveSink` renders a live-updating dashboard while the workflow
runs: a header with the run identity, a status line with the current stage
and elapsed time, an attempts table, and a footer with the best result so
far. It is meant for interactive (TTY) use; callers should fall back to
:class:`stitchcuda.events.NullSink` for non-TTY contexts so output stays
log-friendly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..events import EventSink
from .console import stdout_console
from .display import display_endpoint, display_path


@dataclass
class _AttemptRow:
    attempt: int
    stage: str
    status: str  # one of: running, code, verify, compile-fail, incorrect, correct, best
    speedup: float | None = None
    runtime_us: float | None = None


@dataclass
class _DashboardState:
    header: str = "[bold]StitchCUDA[/bold]"
    stage: str = "starting…"
    attempts: list[_AttemptRow] = field(default_factory=list)
    plan_version: int = 0
    best_attempt: int | None = None
    best_speedup: float = 0.0
    stop_reason: str = "running"
    started_monotonic: float | None = None
    last_plan_summary: str = ""


class RichLiveSink(EventSink):
    """Render workflow progress as a live-updating Rich dashboard.

    The sink owns a :class:`rich.live.Live` controller that is started on
    ``__enter__`` and stopped on ``__exit__``; outside that context the
    sink is inert (events are still accepted but no rendering happens).
    """

    def __init__(self, *, console: Console | None = None, refresh_per_second: int = 6) -> None:
        self._console = console or stdout_console()
        self._refresh_per_second = refresh_per_second
        self._state = _DashboardState()
        self._live: Live | None = None

    # --- context manager -------------------------------------------------

    def __enter__(self) -> "RichLiveSink":
        self._state.started_monotonic = time.monotonic()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=self._refresh_per_second,
            vertical_overflow="visible",
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            # Push the final state once more so the post-Live snapshot reflects
            # everything that happened up to the close.
            self._live.update(self._render(), refresh=True)
            self._live.stop()
            self._live = None
        return None

    # --- workflow events --------------------------------------------------

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
        gpu = hardware.get("gpu_name", "?")
        sm = hardware.get("target_sm") or hardware.get("compute_capability", "?")
        model = config.get("model", "?")
        endpoint = display_endpoint(config.get("api_base"))
        self._state.header = (
            f"[bold cyan]StitchCUDA[/bold cyan] · L{level} P{problem_id} [bold]{problem_name}[/bold]\n"
            f"[dim]model={model} → {endpoint} · gpu={gpu} ({sm}) · run_dir={display_path(run_dir)}[/dim]"
        )
        self._refresh()

    def on_plan_start(self, *, version: int, reason: str = "") -> None:
        self._state.plan_version = version
        label = f"planning v{version}"
        if reason:
            label += f" — {reason}"
        self._state.stage = label
        self._refresh()

    def on_plan_done(self, *, version: int, summary: str) -> None:
        self._state.plan_version = version
        self._state.last_plan_summary = (summary or "").strip()
        self._state.stage = f"plan v{version} ready"
        self._refresh()

    def on_attempt_start(self, *, attempt: int, stage: str, plan_version: int) -> None:
        self._state.attempts.append(_AttemptRow(attempt=attempt, stage=stage, status="running"))
        self._state.stage = f"attempt {attempt} · {stage}"
        self._refresh()

    def on_code_start(self, *, attempt: int) -> None:
        self._update_row(attempt, status="code")
        self._state.stage = f"attempt {attempt} · drafting code"
        self._refresh()

    def on_code_done(self, *, attempt: int, code_chars: int) -> None:
        self._update_row(attempt, status="code-done")
        self._state.stage = f"attempt {attempt} · code ready ({code_chars} chars)"
        self._refresh()

    def on_verify_start(self, *, attempt: int) -> None:
        self._update_row(attempt, status="verify")
        self._state.stage = f"attempt {attempt} · verifying"
        self._refresh()

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
        if not compiled:
            status = "compile-fail"
        elif not correct:
            status = "incorrect"
        elif is_best:
            status = "best"
        else:
            status = "correct"
        if is_best:
            # Demote any previously best row so the table never shows two bests.
            for row in self._state.attempts:
                if row.status == "best" and row.attempt != attempt:
                    row.status = "correct"
        self._update_row(
            attempt,
            status=status,
            speedup=speedup if correct else None,
            runtime_us=runtime_us if runtime_us > 0 else None,
        )
        if is_best:
            self._state.best_attempt = attempt
            self._state.best_speedup = speedup
        self._refresh()

    def on_run_end(self, *, stop_reason: str, best_attempt: int | None, best_speedup: float) -> None:
        self._state.stop_reason = stop_reason
        self._state.best_attempt = best_attempt
        self._state.best_speedup = best_speedup or 0.0
        self._state.stage = f"done · {stop_reason}"
        self._refresh()

    # --- rendering -------------------------------------------------------

    def _update_row(self, attempt: int, **changes: Any) -> None:
        for row in self._state.attempts:
            if row.attempt == attempt:
                for key, value in changes.items():
                    setattr(row, key, value)
                return

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> RenderableType:
        return Group(
            Panel(self._state.header, padding=(0, 1), border_style="cyan"),
            Panel(self._status_text(), padding=(0, 1), title="Status", border_style="dim"),
            self._attempts_table(),
            Panel(self._footer_text(), padding=(0, 1), border_style="dim"),
        )

    def _status_text(self) -> str:
        elapsed = self._elapsed_text()
        return f"[bold]{self._state.stage}[/bold]    [dim]elapsed {elapsed}[/dim]"

    def _elapsed_text(self) -> str:
        if self._state.started_monotonic is None:
            return "—"
        secs = int(time.monotonic() - self._state.started_monotonic)
        return f"{secs // 60:02d}:{secs % 60:02d}"

    def _attempts_table(self) -> Table:
        table = Table(title="Attempts", header_style="bold", title_style="bold")
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Stage")
        table.add_column("Status", no_wrap=True)
        table.add_column("Speedup", justify="right", no_wrap=True)
        table.add_column("Runtime µs", justify="right", no_wrap=True)
        for row in self._state.attempts:
            sp = "—" if row.speedup is None else f"{row.speedup:.3f}"
            rt = "—" if row.runtime_us is None else f"{row.runtime_us:.1f}"
            table.add_row(str(row.attempt), row.stage, _format_status(row.status), sp, rt)
        return table

    def _footer_text(self) -> str:
        if self._state.best_attempt is None:
            return f"Best: [dim]—[/dim]    Stop reason: {self._state.stop_reason}"
        return (
            f"Best: attempt [bold]{self._state.best_attempt}[/bold] · "
            f"speedup [bold green]{self._state.best_speedup:.3f}[/bold green]    "
            f"Stop reason: {self._state.stop_reason}"
        )


_STATUS_STYLES: dict[str, str] = {
    "running": "yellow",
    "code": "yellow",
    "code-done": "yellow",
    "verify": "yellow",
    "compile-fail": "red",
    "incorrect": "red",
    "correct": "green",
    "best": "bold green",
}


def _format_status(status: str) -> str:
    style = _STATUS_STYLES.get(status, "")
    return f"[{style}]{status}[/{style}]" if style else status
