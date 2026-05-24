from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KernelBenchProblem:
    level: int
    problem_id: int
    name: str
    reference_code: str
    reference_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    summary: str
    implementation_steps: list[str] = field(default_factory=list)
    expected_bottlenecks: list[str] = field(default_factory=list)
    correctness_risks: list[str] = field(default_factory=list)
    kernelbench_constraints: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        return cls(
            summary=str(data.get("summary", "")).strip(),
            implementation_steps=_string_list(data.get("implementation_steps")),
            expected_bottlenecks=_string_list(data.get("expected_bottlenecks")),
            correctness_risks=_string_list(data.get("correctness_risks")),
            kernelbench_constraints=_string_list(data.get("kernelbench_constraints")),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    compiled: bool = False
    correct: bool = False
    speedup: float = 0.0
    runtime_us: float = -1.0
    ref_runtime_us: float = -1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationResult":
        return cls(
            compiled=bool(data.get("compiled", False)),
            correct=bool(data.get("correct", False)),
            speedup=float(data.get("speedup", 0.0) or 0.0),
            runtime_us=float(data.get("runtime_us", -1.0) or -1.0),
            ref_runtime_us=float(data.get("ref_runtime_us", -1.0) or -1.0),
            metadata=dict(data.get("metadata") or {}),
            error=str(data.get("error", "") or ""),
            stdout_tail=str(data.get("stdout_tail", "") or ""),
            stderr_tail=str(data.get("stderr_tail", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAttempt:
    attempt: int
    plan_version: int
    stage: str
    solution_path: Path
    result: VerificationResult

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["solution_path"] = str(self.solution_path)
        data["result"] = self.result.to_dict()
        return data


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []
