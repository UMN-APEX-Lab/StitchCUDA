You are planning one CUDA kernel solution for a CUDA-Hercules task on the target hardware.

Task:
{{problem_json}}

Target hardware:
{{hardware_json}}

Task specification (this is all the information about the operation; the reference
implementation it is compared against is not available):
{{task_prompt}}

Return only one JSON object with this schema:
{
  "summary": "one sentence implementation strategy",
  "implementation_steps": ["specific coding steps for the solution file"],
  "expected_bottlenecks": ["main GPU performance bottlenecks expected for this operation and these problem sizes"],
  "correctness_risks": ["risks that could fail the task's correctness check"],
  "kernelbench_constraints": ["constraints from the task specification the coder must obey"]
}

Keep the plan specific to this task and this hardware. Do not write code.
