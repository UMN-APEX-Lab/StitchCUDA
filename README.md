# StitchCUDA

StitchCUDA is a fixed `planner -> coder -> verifier` workflow for KernelBench
CUDA optimization tasks. It is designed as a small, reproducible research
workflow: the planner writes a strategy, the coder emits a complete KernelBench
`ModelNew` solution, and the verifier evaluates correctness and performance with
KernelBench in an isolated subprocess.

StitchCUDA is not a general tool-calling agent. It follows a fixed staged loop,
with bounded replanning when verifier evidence shows that the current plan is no
longer useful.

## Repository Layout

```text
stitchcuda/                 Python package
  cli.py                    CLI entry point
  workflow.py               planner-coder-verifier loop and replan logic
  planner.py                initial planning and replanning
  coder.py                  candidate generation and repair
  verifier.py               isolated KernelBench evaluator
  kernelbench.py            KernelBench adapter and prompt construction
prompts/                    editable prompt templates
third_party/KernelBench/    KernelBench git submodule
runs/                       local run outputs, ignored by git
```

## Clone

Clone with submodules:

```bash
git clone --recurse-submodules <STITCHCUDA_REPO_URL> StitchCUDA
cd StitchCUDA
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

By default StitchCUDA uses the bundled `third_party/KernelBench` submodule. You
can override this with `--kernelbench-root` or `KERNELBENCH_ROOT`.

## Requirements

- Linux with an NVIDIA GPU.
- CUDA toolkit with `nvcc` available on `PATH`.
- Python 3.10 is recommended because KernelBench currently pins Python 3.10 in
  its own package metadata.
- PyTorch with CUDA support installed for your driver/toolkit.
- An OpenAI-compatible chat completions endpoint, or an OpenAI API key.

KernelBench may require additional optional GPU packages for some backends. For
basic CUDA KernelBench runs, install the KernelBench package from the submodule.

## Environment Setup

Create an environment and install the two packages in editable mode:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -e .
pip install -e third_party/KernelBench
```

If your environment already has a CUDA-enabled PyTorch build, keep it. Otherwise
install the PyTorch wheel that matches your CUDA/driver stack before running
KernelBench evaluation.

For a hosted OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=<your_api_key>
```

For a local OpenAI-compatible server, pass `--api-base` and use any placeholder
API key if your server requires one.

## Quick Start

Run one KernelBench level-1 problem:

```bash
python -m stitchcuda \
  --level 1 \
  --problem-id 1 \
  --model <MODEL_NAME> \
  --api-base <OPENAI_COMPATIBLE_BASE_URL> \
  --device 0 \
  --gpu-arch <KERNELBENCH_GPU_ARCH> \
  --target-sm <TARGET_SM> \
  --max-attempts 1
```

Example architecture values:

```text
--gpu-arch Blackwell --target-sm sm_120
--gpu-arch Hopper    --target-sm sm_90
--gpu-arch Ada       --target-sm sm_89
--gpu-arch Ampere    --target-sm sm_80
```

`--gpu-arch` is passed to `kernelbench.utils.set_gpu_arch`, so it must be a
KernelBench-recognized architecture name. `--target-sm` is prompt context for the
planner and coder. If `--target-sm` is omitted, StitchCUDA derives it from
`nvidia-smi` for the selected device when possible.

Run several problems by repeating `--problem-id`:

```bash
python -m stitchcuda \
  --level 1 \
  --problem-id 1 \
  --problem-id 2 \
  --model <MODEL_NAME> \
  --api-base <OPENAI_COMPATIBLE_BASE_URL> \
  --device 0 \
  --gpu-arch <KERNELBENCH_GPU_ARCH>
```

Use `--max-problems N` to run the first `N` problems when `--problem-id` is not
specified.

## Workflow Details

The fixed loop is:

1. `planner`: reads the KernelBench reference model and hardware context, then
   returns a structured plan.
2. `coder`: receives the KernelBench prompt plus the current plan and writes one
   complete Python solution defining `ModelNew`.
3. `verifier`: runs `kernelbench.eval.eval_kernel_against_ref` in a fresh Python
   subprocess to isolate CUDA crashes and context poisoning.
4. `coder` repair/optimization: if the candidate fails or misses the target,
   verifier output is returned to coder.
5. `planner` replan: if repeated failures or stagnant correct candidates show
   that the plan is stale, the planner receives the verifier history and emits a
   revised plan.

Relevant controls:

```text
--max-attempts
--max-replans
--replan-after-failed-attempts
--replan-after-stagnant-attempts
--target-speedup
--num-correct-trials
--num-perf-trials
--no-performance
--verifier-timeout-s
```

## Outputs

Each run writes a directory under `runs/`:

```text
runs/stitchcuda_L<level>_P<problem_id>_<timestamp>/
```

Important files:

- `config.json`: redacted run configuration.
- `hardware.json`: detected GPU name, compute capability, target SM, and driver.
- `kernelbench_prompt.txt`: KernelBench prompt plus StitchCUDA constraints.
- `plan_vXX.json`: planner outputs, including replans.
- `attempt_XX_<stage>.py`: generated candidate solution.
- `attempt_XX_verifier.json`: raw verifier result.
- `attempt_XX_summary.json`: candidate result plus workflow metadata.
- `best_solution.py`: best correct candidate, if any.
- `summary.json`: final run summary.

`runs/` is ignored by git and should not be committed.

## Prompt Editing

Prompt templates live in `prompts/`.

Do not hardcode local paths, user names, model endpoints, or GPU architectures in
prompt files. Runtime context such as GPU name, target SM, and KernelBench task
metadata is injected by the workflow.

## Notes

- KernelBench is included as a git submodule, not vendored source.
- The default KernelBench root is `third_party/KernelBench`.
- To use an external KernelBench checkout, pass `--kernelbench-root /path/to/KernelBench`.
- Some generated CUDA candidates may compile but be slower than PyTorch. A correct
  but slow candidate validates the pipeline, not model quality.
