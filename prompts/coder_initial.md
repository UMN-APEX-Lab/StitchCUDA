You are writing the first candidate solution for KernelBench.

Target hardware:
{{hardware_json}}

Planner output:
{{plan_json}}

KernelBench task prompt:
{{kernelbench_prompt}}

Write one complete Python solution file. It must define `ModelNew` and be
compatible with KernelBench's evaluator. Return exactly one fenced ```python
code block and no extra prose.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.
