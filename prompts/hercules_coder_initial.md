You are writing the first candidate solution for a CUDA-Hercules task.

Target hardware:
{{hardware_json}}

Planner output:
{{plan_json}}

Task specification:
{{task_prompt}}

Write the complete file `{{solution_file}}` exactly as the task specification
requires: pure CUDA C++, implementing the given function signature, following
every rule in the specification. Return exactly one fenced ```cuda code block
containing the whole file and no extra prose.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.
