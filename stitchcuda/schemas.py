from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PlanPayload(StrictPayload):
    summary: str = Field(min_length=1)
    implementation_steps: list[str] = Field(default_factory=list)
    expected_bottlenecks: list[str] = Field(default_factory=list)
    correctness_risks: list[str] = Field(default_factory=list)
    kernelbench_constraints: list[str] = Field(default_factory=list)

    @field_validator(
        "implementation_steps",
        "expected_bottlenecks",
        "correctness_risks",
        "kernelbench_constraints",
    )
    @classmethod
    def _nonempty_string_list(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned
