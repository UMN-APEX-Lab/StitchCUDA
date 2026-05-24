from __future__ import annotations

import json

from .kernelbench import extract_solution_code
from .llm import OpenAIChatClient
from .templates import render_template
from .types import KernelBenchProblem, Plan, VerificationResult


class CoderAgent:
    """KernelBench-specific code generation and repair stage."""

    def __init__(self, llm: OpenAIChatClient, *, kernelbench_root: str):
        self.llm = llm
        self.kernelbench_root = kernelbench_root

    def draft(
        self,
        problem: KernelBenchProblem,
        *,
        kernelbench_prompt: str,
        plan: Plan,
        hardware_summary: dict,
    ) -> str:
        prompt = render_template(
            "coder_initial.md",
            kernelbench_prompt=kernelbench_prompt,
            plan_json=json.dumps(plan.to_dict(), indent=2),
            hardware_json=json.dumps(hardware_summary, indent=2),
            target_arch_context=target_arch_context(hardware_summary),
        )
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are the coder in a fixed KernelBench CUDA workflow. Return one complete Python solution.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return extract_solution_code(raw, self.kernelbench_root)

    def revise(
        self,
        problem: KernelBenchProblem,
        *,
        kernelbench_prompt: str,
        plan: Plan,
        previous_code: str,
        verifier_result: VerificationResult,
        hardware_summary: dict,
        target_speedup: float,
    ) -> str:
        prompt = render_template(
            "coder_revise.md",
            kernelbench_prompt=kernelbench_prompt,
            plan_json=json.dumps(plan.to_dict(), indent=2),
            hardware_json=json.dumps(hardware_summary, indent=2),
            target_arch_context=target_arch_context(hardware_summary),
            previous_code=previous_code,
            verifier_feedback=json.dumps(verifier_result.to_dict(), indent=2),
            target_speedup=target_speedup,
        )
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are the coder in a fixed KernelBench CUDA workflow. Repair or optimize the full solution.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return extract_solution_code(raw, self.kernelbench_root)


def target_arch_context(hardware_summary: dict) -> str:
    parts = []
    for key in ("gpu_name", "kernelbench_gpu_arch", "target_sm", "target_compute", "compute_capability"):
        value = hardware_summary.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "use evaluator-provided target architecture"
