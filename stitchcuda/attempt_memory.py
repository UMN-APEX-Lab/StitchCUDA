from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .types import CandidateAttempt, Plan, VerificationResult


DEFAULT_MEMORY_TOKEN_BUDGET = 65_536


@dataclass(frozen=True)
class AttemptContext:
    """A prompt-ready, budgeted view of attempt history."""

    audience: str
    text: str
    token_budget: int
    estimated_tokens: int
    baseline_code: str = ""
    baseline_attempt: int | None = None
    baseline_kind: str = ""
    over_budget: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "audience": self.audience,
            "token_budget": self.token_budget,
            "estimated_tokens": self.estimated_tokens,
            "baseline_attempt": self.baseline_attempt,
            "baseline_kind": self.baseline_kind,
            "over_budget": self.over_budget,
        }


@dataclass
class _MemoryAttempt:
    attempt: int
    plan_version: int
    stage: str
    code: str
    code_hash: str
    result: VerificationResult
    plan_summary: str

    def compact_dict(self, target_speedup: float) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "plan_version": self.plan_version,
            "stage": self.stage,
            "code_hash": self.code_hash,
            "code_chars": len(self.code),
            "status": _status_text(self.result),
            "failure_kind": _failure_kind(self.result, target_speedup),
        }


@dataclass
class _ErrorCluster:
    fingerprint: str
    kind: str
    summary: str
    count: int = 0
    attempts: list[int] = field(default_factory=list)

    def record(self, attempt: int) -> None:
        self.count += 1
        self.attempts.append(attempt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "kind": self.kind,
            "summary": self.summary,
            "count": self.count,
            "attempts": self.attempts,
        }


@dataclass
class _StrategyFailure:
    key: str
    plan_version: int
    strategy: str
    failure_kind: str
    error_fingerprint: str
    outcome: str
    count: int = 0
    attempts: list[int] = field(default_factory=list)

    def record(self, attempt: int) -> None:
        self.count += 1
        self.attempts.append(attempt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "plan_version": self.plan_version,
            "strategy": self.strategy,
            "failure_kind": self.failure_kind,
            "error_fingerprint": self.error_fingerprint,
            "outcome": self.outcome,
            "count": self.count,
            "attempts": self.attempts,
        }


class AttemptMemory:
    """Structured attempt history with audience-specific prompt compression."""

    def __init__(
        self,
        *,
        target_speedup: float,
        coder_token_budget: int = DEFAULT_MEMORY_TOKEN_BUDGET,
        replanner_token_budget: int = DEFAULT_MEMORY_TOKEN_BUDGET,
    ):
        self.target_speedup = float(target_speedup)
        self.coder_token_budget = max(32, int(coder_token_budget))
        self.replanner_token_budget = max(32, int(replanner_token_budget))
        self._attempts: list[_MemoryAttempt] = []
        self._best_correct: _MemoryAttempt | None = None
        self._error_clusters: dict[str, _ErrorCluster] = {}
        self._strategy_failures: dict[str, _StrategyFailure] = {}

    @property
    def latest_code(self) -> str:
        return self._attempts[-1].code if self._attempts else ""

    @property
    def best_correct_code(self) -> str:
        return self._best_correct.code if self._best_correct is not None else ""

    @property
    def latest_attempt(self) -> int | None:
        return self._attempts[-1].attempt if self._attempts else None

    @property
    def best_correct_attempt(self) -> int | None:
        return self._best_correct.attempt if self._best_correct is not None else None

    def record(self, candidate: CandidateAttempt, *, code: str, plan: Plan) -> None:
        if self._attempts and candidate.attempt <= self._attempts[-1].attempt:
            raise ValueError("AttemptMemory records must be added in strictly increasing attempt order")

        item = _MemoryAttempt(
            attempt=candidate.attempt,
            plan_version=candidate.plan_version,
            stage=candidate.stage,
            code=code,
            code_hash=candidate.code_hash or candidate.result.code_hash,
            result=candidate.result,
            plan_summary=_strategy_summary(plan),
        )
        self._attempts.append(item)

        if item.result.correct and (
            self._best_correct is None or item.result.speedup > self._best_correct.result.speedup
        ):
            self._best_correct = item

        kind = _failure_kind(item.result, self.target_speedup)
        if not kind:
            return

        diagnostic = _diagnostic_summary(item.result, kind, self.target_speedup)
        fingerprint = _fingerprint(kind, diagnostic)
        cluster = self._error_clusters.get(fingerprint)
        if cluster is None:
            cluster = _ErrorCluster(fingerprint=fingerprint, kind=kind, summary=diagnostic)
            self._error_clusters[fingerprint] = cluster
        else:
            cluster.summary = diagnostic
        cluster.record(item.attempt)

        strategy_key = _short_hash(f"{item.plan_summary}|{kind}|{fingerprint}")
        strategy = self._strategy_failures.get(strategy_key)
        if strategy is None:
            strategy = _StrategyFailure(
                key=strategy_key,
                plan_version=item.plan_version,
                strategy=item.plan_summary,
                failure_kind=kind,
                error_fingerprint=fingerprint,
                outcome=diagnostic,
            )
            self._strategy_failures[strategy_key] = strategy
        strategy.record(item.attempt)

    def context_for_coder(self) -> AttemptContext:
        if not self._attempts:
            return AttemptContext(
                audience="coder",
                text="No previous attempts are available.",
                token_budget=self.coder_token_budget,
                estimated_tokens=estimate_tokens("No previous attempts are available."),
            )

        latest = self._attempts[-1]
        baseline = self._best_correct or latest
        baseline_kind = "best_correct" if self._best_correct is not None else "latest"
        baseline_tokens = estimate_tokens(baseline.code)
        summary_budget = max(0, self.coder_token_budget - baseline_tokens)

        lines = [
            "Attempt memory for the coder:",
            (
                f"Selected baseline: attempt {baseline.attempt} ({baseline_kind}), "
                f"{_status_text(baseline.result)}. The source block below is this exact baseline."
            ),
            f"Latest outcome: {_attempt_line(latest, self.target_speedup)}",
        ]
        if baseline.attempt != latest.attempt:
            lines.append(
                f"The latest attempt {latest.attempt} regressed from the selected correct baseline; "
                "do not copy its implementation wholesale. Preserve the baseline and use its failure only as evidence."
            )

        clusters = self._ranked_error_clusters()
        if clusters:
            lines.append("Recurring error fingerprints to avoid:")
            lines.extend(
                f"- [{item.fingerprint}] {item.kind}, count={item.count}, attempts={item.attempts}: {item.summary}"
                for item in clusters[:6]
            )

        failures = self._ranked_strategy_failures()
        if failures:
            lines.append("Failed strategy summaries:")
            lines.extend(
                f"- attempts={item.attempts}, count={item.count}: {item.strategy} -> "
                f"{item.failure_kind} [{item.error_fingerprint}]: {item.outcome}"
                for item in failures[:5]
            )

        lines.append("Recent outcome timeline:")
        lines.extend(f"- {_attempt_line(item, self.target_speedup)}" for item in self._attempts[-4:])
        text = _fit_to_token_budget("\n".join(lines), summary_budget)
        estimated_tokens = baseline_tokens + estimate_tokens(text)
        return AttemptContext(
            audience="coder",
            text=text,
            token_budget=self.coder_token_budget,
            estimated_tokens=estimated_tokens,
            baseline_code=baseline.code,
            baseline_attempt=baseline.attempt,
            baseline_kind=baseline_kind,
            over_budget=estimated_tokens > self.coder_token_budget,
        )

    def context_for_replanner(self) -> AttemptContext:
        if not self._attempts:
            text = "No previous attempts are available."
        else:
            latest = self._attempts[-1]
            lines = [
                "Compressed attempt memory for replanning:",
                f"Latest outcome: {_attempt_line(latest, self.target_speedup)}",
            ]
            if self._best_correct is None:
                lines.append("Best correct candidate: none.")
            else:
                lines.append(
                    f"Best correct candidate: attempt {self._best_correct.attempt}, "
                    f"speedup={self._best_correct.result.speedup:.4f}x, "
                    f"code_hash={_short_code_hash(self._best_correct.code_hash)}."
                )

            clusters = self._ranked_error_clusters()
            if clusters:
                lines.append("Error fingerprint clusters:")
                lines.extend(
                    f"- [{item.fingerprint}] {item.kind}, count={item.count}, attempts={item.attempts}: {item.summary}"
                    for item in clusters
                )

            failures = self._ranked_strategy_failures()
            if failures:
                lines.append("Failed strategy summaries (change strategy when a cluster repeats):")
                lines.extend(
                    f"- plan_v{item.plan_version}, attempts={item.attempts}, count={item.count}: "
                    f"{item.strategy} -> {item.failure_kind} [{item.error_fingerprint}]: {item.outcome}"
                    for item in failures
                )

            lines.append("Outcome timeline:")
            lines.extend(f"- {_attempt_line(item, self.target_speedup)}" for item in self._attempts)
            text = "\n".join(lines)

        text = _fit_to_token_budget(text, self.replanner_token_budget)
        estimated_tokens = estimate_tokens(text)
        return AttemptContext(
            audience="replanner",
            text=text,
            token_budget=self.replanner_token_budget,
            estimated_tokens=estimated_tokens,
            over_budget=estimated_tokens > self.replanner_token_budget,
        )

    def to_dict(self) -> dict[str, Any]:
        latest = self._attempts[-1] if self._attempts else None
        return {
            "target_speedup": self.target_speedup,
            "coder_token_budget": self.coder_token_budget,
            "replanner_token_budget": self.replanner_token_budget,
            "latest": _attempt_reference(latest),
            "best_correct": _attempt_reference(self._best_correct),
            "attempts": [item.compact_dict(self.target_speedup) for item in self._attempts],
            "error_fingerprints": [item.to_dict() for item in self._ranked_error_clusters()],
            "failed_strategy_summaries": [item.to_dict() for item in self._ranked_strategy_failures()],
        }

    def _ranked_error_clusters(self) -> list[_ErrorCluster]:
        return sorted(
            self._error_clusters.values(),
            key=lambda item: (item.count, item.attempts[-1]),
            reverse=True,
        )

    def _ranked_strategy_failures(self) -> list[_StrategyFailure]:
        return sorted(
            self._strategy_failures.values(),
            key=lambda item: (item.count, item.attempts[-1]),
            reverse=True,
        )


def estimate_tokens(text: str) -> int:
    """Return a tokenizer-independent estimate suitable for prompt budgeting."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _fit_to_token_budget(text: str, token_budget: int) -> str:
    token_budget = max(0, int(token_budget))
    if token_budget == 0:
        return ""
    if estimate_tokens(text) <= token_budget:
        return text
    suffix = "\n...[attempt memory truncated to token budget]"
    char_budget = token_budget * 4
    if char_budget <= len(suffix):
        return suffix[:char_budget]
    return text[: char_budget - len(suffix)].rstrip() + suffix


def _failure_kind(result: VerificationResult, target_speedup: float) -> str:
    if not result.compiled:
        return result.error_kind or "compile_error"
    if not result.correct:
        return result.error_kind or "correctness_error"
    if target_speedup > 0 and result.speedup < target_speedup:
        return "performance_below_target"
    return ""


def _diagnostic_summary(result: VerificationResult, kind: str, target_speedup: float) -> str:
    if kind == "performance_below_target":
        return (
            f"correct candidate measured {result.speedup:.4f}x versus "
            f"target {target_speedup:.4f}x (runtime_us={result.runtime_us:.4f}, "
            f"ref_runtime_us={result.ref_runtime_us:.4f})"
        )

    raw_parts = [result.error, *result.warnings, result.stderr_tail, result.stdout_tail]
    raw_lines = [line.strip() for part in raw_parts if part for line in str(part).splitlines() if line.strip()]
    if not raw_lines:
        return kind

    primary_pattern = re.compile(
        r"(?:\berror:|fatal error|undefined symbol|not declared|not found|mismatch|exception|traceback|timeout)",
        re.IGNORECASE,
    )
    secondary_pattern = re.compile(r"(?:failed|missing|invalid|unsupported|returned none)", re.IGNORECASE)
    ordered = (
        [line for line in raw_lines if primary_pattern.search(line)]
        + [line for line in raw_lines if secondary_pattern.search(line)]
        + raw_lines
    )
    selected: list[str] = []
    seen: set[str] = set()
    for line in ordered:
        normalized = _normalize_diagnostic(line)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(_clip(normalized, 280))
        if len(selected) == 4:
            break
    return " | ".join(selected) or kind


def _normalize_diagnostic(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"Error building extension ['\"][^'\"]+['\"]", "Error building extension <extension>", text)
    text = re.sub(r"TORCH_EXTENSION_NAME=\w+", "TORCH_EXTENSION_NAME=<extension>", text)
    text = re.sub(r"stitchcuda_a\d+_[0-9a-f]+", "stitchcuda_<attempt>_<hash>", text)
    text = re.sub(r"\b[0-9a-f]{12,64}\b", "<hash>", text, flags=re.IGNORECASE)
    text = re.sub(r":\d+:\d+(?::|\b)", ":<line>:<col>", text)
    text = re.sub(r"(?<![\w.])/(?:[^\s:'\"]+/)*[^\s:'\"]+", "<path>", text)
    return re.sub(r"\s+", " ", text).strip()


def _fingerprint(kind: str, diagnostic: str) -> str:
    if kind == "performance_below_target":
        return _short_hash(kind)
    return _short_hash(f"{kind}|{diagnostic}")


def _strategy_summary(plan: Plan) -> str:
    parts = [plan.summary.strip()]
    steps = [step.strip() for step in plan.implementation_steps[:3] if step.strip()]
    if steps:
        parts.append("steps=" + "; ".join(steps))
    return _clip(" | ".join(part for part in parts if part), 700) or "unspecified strategy"


def _status_text(result: VerificationResult) -> str:
    if result.correct:
        return f"correct, speedup={result.speedup:.4f}x"
    if result.compiled:
        return f"compiled but incorrect, kind={result.error_kind or 'correctness_error'}"
    return f"not compiled, kind={result.error_kind or 'compile_error'}"


def _attempt_line(item: _MemoryAttempt, target_speedup: float) -> str:
    kind = _failure_kind(item.result, target_speedup)
    diagnostic = _diagnostic_summary(item.result, kind, target_speedup) if kind else "target satisfied"
    fingerprint = _fingerprint(kind, diagnostic) if kind else "none"
    return (
        f"attempt={item.attempt}, plan_v={item.plan_version}, stage={item.stage}, "
        f"code_hash={_short_code_hash(item.code_hash)}, {_status_text(item.result)}, "
        f"fingerprint={fingerprint}, summary={diagnostic}"
    )


def _attempt_reference(item: _MemoryAttempt | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "attempt": item.attempt,
        "plan_version": item.plan_version,
        "stage": item.stage,
        "code_hash": item.code_hash,
        "code_chars": len(item.code),
        "status": _status_text(item.result),
        "speedup": item.result.speedup,
    }


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _short_code_hash(value: str) -> str:
    return value[:12] if value else "unknown"


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
