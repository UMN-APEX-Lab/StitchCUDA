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

CUDA extension scaffold:
- Prefer `from stitchcuda.runtime import build_load_inline_extension`.
- Build extensions with `build_load_inline_extension(name=..., cpp_sources=..., cuda_sources=..., functions=[...])`.
- `functions` must contain only bare wrapper names such as `"fused_forward"`.
- `cpp_sources` should contain declarations only, for example
  `"torch::Tensor fused_forward(torch::Tensor x);"`; put CUDA launches and
  wrapper definitions in `cuda_sources`.
- Do not pass `build_directory`, `force_build`, or custom extension cache paths.

Configured target architecture context:
{{target_arch_context}}

Do not hardcode a CUDA architecture different from the configured target.
