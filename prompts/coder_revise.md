You are revising a KernelBench candidate after verifier feedback.

Target speedup: {{target_speedup}}x

Target hardware:
{{hardware_json}}

Planner output:
{{plan_json}}

Compressed attempt memory:
{{attempt_memory_context}}

Selected baseline solution:
```python
{{baseline_code}}
```

KernelBench task prompt:
{{kernelbench_prompt}}

The selected baseline may be the best correct historical candidate instead of
the latest candidate. Preserve its known-correct behavior. Use the latest
outcome and recurring fingerprints in attempt memory as evidence, and do not
repeat a failed strategy cluster. If no correct baseline exists, repair the
selected latest candidate's compilation or correctness failure first.

CUDA extension scaffold:
- Prefer `from stitchcuda.runtime import build_load_inline_extension` instead of
  copying raw `load_inline` calls from an earlier solution.
- Build extensions with `build_load_inline_extension(name=..., cpp_sources=..., cuda_sources=..., functions=[...])`.
- `functions` must contain only bare wrapper names such as `"fused_forward"`.
- `cpp_sources` should contain declarations only, for example
  `"torch::Tensor fused_forward(torch::Tensor x);"`; put CUDA launches and
  wrapper definitions in `cuda_sources`.
- Do not pass `build_directory`, `force_build`, or custom extension cache paths.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.

Return exactly one complete fenced ```python code block and no extra prose.
