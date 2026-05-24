You are replanning a KernelBench CUDA solution after verifier feedback.

Replan trigger:
{{replan_reason}}

KernelBench problem:
{{problem_json}}

Target hardware:
{{hardware_json}}

Previous plan:
{{previous_plan_json}}

Verifier history:
{{attempts_json}}

Reference PyTorch model:
```python
{{reference_code}}
```

Return only one JSON object with this schema:
{
  "summary": "one sentence revised implementation strategy",
  "implementation_steps": ["specific revised coding steps for ModelNew"],
  "expected_bottlenecks": ["main bottlenecks the revised plan addresses"],
  "correctness_risks": ["risks exposed by verifier history and how to avoid them"],
  "kernelbench_constraints": ["constraints the coder must obey"]
}

The revised plan must explicitly respond to the verifier history. Do not repeat a
failed strategy unless the verifier feedback shows the failure was only a small
implementation bug.
