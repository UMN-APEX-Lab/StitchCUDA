# StitchCUDA

StitchCUDA is an open-source multi-agent framework for automated end-to-end
CUDA program generation and optimization. It is the agent framework component
associated with an ICML 2026 paper on automated CUDA programming.

The system decomposes GPU programming into three cooperating agents:

- **Planner**: reasons about the full program and chooses an optimization plan.
- **Coder**: implements or repairs CUDA/Python code for the current task.
- **Verifier**: checks correctness, measures performance, and returns structured
  feedback to drive further optimization.

Paper status: **accepted to ICML 2026**.

- Paper: <https://arxiv.org/abs/2603.02637>
- ICML 2026 listing: <https://icml.cc/Downloads/2026>

This repository provides the KernelBench-based agent framework: a fixed
`planner -> coder -> verifier` workflow with verifier-driven repair,
optimization, and replanning.

## What Is Included

```text
stitchcuda/                 Python package
  cli.py                    Command-line entry point
  workflow.py               Fixed planner-coder-verifier loop and replan logic
  planner.py                Initial planning and replanning agents
  coder.py                  Candidate generation and repair agent
  verifier.py               Isolated KernelBench evaluator
  kernelbench.py            KernelBench adapter and prompt construction
prompts/                    Editable prompt templates
third_party/KernelBench/    KernelBench git submodule
runs/                       Local run outputs, ignored by git
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

StitchCUDA uses `third_party/KernelBench` by default. To use another KernelBench
checkout, pass `--kernelbench-root /path/to/KernelBench` or set
`KERNELBENCH_ROOT`.

## Requirements

- Linux with an NVIDIA GPU.
- CUDA toolkit with `nvcc` available on `PATH`.
- Python 3.10 recommended. KernelBench currently pins Python 3.10 in its package
  metadata.
- CUDA-enabled PyTorch compatible with your driver/toolkit.
- An OpenAI-compatible chat-completions endpoint, or an OpenAI API key.

KernelBench may require extra optional packages for non-CUDA backends. The
default StitchCUDA path uses KernelBench's CUDA backend.

## Environment Setup

Create an environment and install StitchCUDA plus the KernelBench submodule:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -e .
pip install -e third_party/KernelBench
```

Install a CUDA-enabled PyTorch build if your environment does not already have
one. Follow the official PyTorch installation matrix for your CUDA toolkit and
driver.

For a hosted OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=<your_api_key>
```

For a local OpenAI-compatible server, pass `--api-base`. If the server does not
enforce authentication, any placeholder API key is sufficient.

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
KernelBench-recognized architecture name. `--target-sm` is prompt context for
hardware-aware planning and coding. If `--target-sm` is omitted, StitchCUDA
derives it from `nvidia-smi` for the selected device when possible.

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

## Workflow

The current open-source workflow is deterministic:

1. **Plan**: read the KernelBench reference program and hardware context, then
   produce a structured optimization plan.
2. **Code**: generate a full KernelBench-compatible Python solution defining
   `ModelNew`.
3. **Verify**: run KernelBench correctness and performance evaluation in a fresh
   Python subprocess to avoid CUDA context poisoning after failures.
4. **Repair or optimize**: if the candidate fails or misses the target speedup,
   return verifier feedback to the coder.
5. **Replan**: if failures repeat or correct candidates stagnate below the
   target, ask the planner for a revised strategy.

Useful controls:

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

Do not hardcode local paths, personal information, model endpoints, or GPU
architectures in prompts. Runtime context such as GPU name, target SM, model
name, and KernelBench task metadata is injected by the workflow.

## Citation

```bibtex
@inproceedings{li2026stitchcuda,
  title     = {StitchCUDA: An Automated Multi-Agents End-to-End GPU Programming Framework with Rubric-based Agentic Reinforcement Learning},
  author    = {Li, Shiyang and Zhang, Zijian and Chen, Winson and Luo, Yuebo and Hong, Mingyi and Ding, Caiwen},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

The arXiv version is available as `arXiv:2603.02637`.

## Notes

- KernelBench is included as a git submodule, not copied into this repository.
- The default KernelBench root is `third_party/KernelBench`.
- Generated candidates may compile and pass correctness but still be slower than
  PyTorch. That validates the workflow path, not necessarily model quality.
