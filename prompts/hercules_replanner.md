You are replanning a CUDA-Hercules kernel solution after verifier feedback.

Replan trigger:
{{replan_reason}}

Task:
{{problem_json}}

Target hardware:
{{hardware_json}}

Previous plan:
{{previous_plan_json}}

Compressed attempt memory:
{{attempt_memory_context}}

Task specification (this is all the information about the operation; the reference
implementation it is compared against is not available):
{{task_prompt}}

Return only one JSON object with this schema:
{
  "summary": "one sentence revised implementation strategy",
  "implementation_steps": ["specific revised coding steps for the solution file"],
  "expected_bottlenecks": ["main bottlenecks the revised plan addresses"],
  "correctness_risks": ["risks exposed by verifier history and how to avoid them"],
  "kernelbench_constraints": ["constraints from the task specification the coder must obey"]
}

The revised plan must explicitly respond to the error fingerprints, failed
strategy summaries, and best correct candidate in attempt memory. Do not repeat
a failed strategy unless the compressed evidence shows the failure was only a
small implementation bug.
