You are revising a KernelBench candidate after verifier feedback.

Target speedup: {{target_speedup}}x

Target hardware:
{{hardware_json}}

Planner output:
{{plan_json}}

Verifier feedback:
{{verifier_feedback}}

Previous solution:
```python
{{previous_code}}
```

KernelBench task prompt:
{{kernelbench_prompt}}

If the candidate failed compilation or correctness, repair the bug first. If it
was correct but below target speed, optimize the measured bottleneck implied by
the verifier feedback while preserving correctness.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.

Return exactly one complete fenced ```python code block and no extra prose.
