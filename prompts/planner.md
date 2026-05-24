You are planning one KernelBench CUDA solution for the target hardware.

KernelBench problem:
{{problem_json}}

Target hardware:
{{hardware_json}}

Reference PyTorch model:
```python
{{reference_code}}
```

Return only one JSON object with this schema:
{
  "summary": "one sentence implementation strategy",
  "implementation_steps": ["specific coding steps for ModelNew"],
  "expected_bottlenecks": ["main GPU performance bottlenecks expected from the reference"],
  "correctness_risks": ["risks that could fail KernelBench correctness"],
  "kernelbench_constraints": ["constraints the coder must obey"]
}

Keep the plan specific to this problem and this hardware. Do not write code.
