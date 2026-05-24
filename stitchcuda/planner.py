from __future__ import annotations

import json

from .llm import OpenAIChatClient, parse_json_object
from .templates import render_template
from .types import CandidateAttempt, KernelBenchProblem, Plan


class PlannerAgent:
    """KernelBench-specific planning stage."""

    def __init__(self, llm: OpenAIChatClient):
        self.llm = llm

    def run(self, problem: KernelBenchProblem, *, hardware_summary: dict) -> Plan:
        prompt = render_template(
            "planner.md",
            problem_json=json.dumps(problem.to_dict(), indent=2),
            hardware_json=json.dumps(hardware_summary, indent=2),
            reference_code=problem.reference_code,
        )
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are the planner in a fixed KernelBench CUDA optimization workflow. Return strict JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        data = parse_json_object(raw)
        return Plan.from_dict(data)

    def replan(
        self,
        problem: KernelBenchProblem,
        *,
        hardware_summary: dict,
        previous_plan: Plan,
        attempts: list[CandidateAttempt],
        reason: str,
    ) -> Plan:
        prompt = render_template(
            "replanner.md",
            problem_json=json.dumps(problem.to_dict(), indent=2),
            hardware_json=json.dumps(hardware_summary, indent=2),
            previous_plan_json=json.dumps(previous_plan.to_dict(), indent=2),
            attempts_json=json.dumps([attempt.to_dict() for attempt in attempts], indent=2, default=str),
            replan_reason=reason,
            reference_code=problem.reference_code,
        )
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are the replanner in a fixed KernelBench CUDA optimization workflow. Return strict JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        data = parse_json_object(raw)
        return Plan.from_dict(data)
