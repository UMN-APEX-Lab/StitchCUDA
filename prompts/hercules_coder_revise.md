You are revising a CUDA-Hercules candidate solution after verifier feedback.

Target speedup: {{target_speedup}}x

Target hardware:
{{hardware_json}}

Planner output:
{{plan_json}}

Compressed attempt memory:
{{attempt_memory_context}}

Selected baseline solution:
```cuda
{{baseline_code}}
```

Task specification:
{{task_prompt}}

The selected baseline may be the best correct historical candidate instead of
the latest candidate. Preserve its known-correct behavior. Use the latest
outcome and recurring fingerprints in attempt memory as evidence, and do not
repeat a failed strategy cluster. If no correct baseline exists, repair the
selected latest candidate's compilation or correctness failure first.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.

Return the complete revised file `{{solution_file}}` in exactly one fenced
```cuda code block and no extra prose.
