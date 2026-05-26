# StitchCUDA

StitchCUDA is a source-available, noncommercial multi-agent framework for
automated end-to-end CUDA program generation and optimization. It is the agent
framework component associated with the ICML 2026 paper **StitchCUDA: An
Automated Multi-Agents End-to-End GPU Programming Framework with Rubric-based
Agentic Reinforcement Learning**.

The system decomposes GPU programming into three cooperating agents:

- **Planner**: reasons about the full program and chooses an optimization plan.
- **Coder**: implements or repairs CUDA/Python code for the current task.
- **Verifier**: checks correctness, measures performance, and returns structured
  feedback to drive further optimization.


This repository provides the KernelBench-based agent framework: a fixed
`planner -> coder -> verifier` workflow with verifier-driven repair,
optimization, and replanning.

## What Is Included

```text
stitchcuda/                 Python package
  cli.py                    Command-line entry point
  events.py                 Workflow event sink protocol for live/quiet UIs
  workflow.py               Fixed planner-coder-verifier loop and replan logic
  planner.py                Initial planning and replanning agents
  coder.py                  Candidate generation and repair agent
  verifier.py               Isolated KernelBench evaluator
  kernelbench.py            KernelBench adapter and prompt construction
  tui/                      Typer/Rich terminal frontend and run browser
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

## Interactive CLI

StitchCUDA ships with a Typer-based command line that wraps the same workflow
in interactive subcommands. The legacy form above (`python -m stitchcuda
--level ... --problem-id ...`) keeps working unchanged; under the hood the
flags are routed to the `run` subcommand.

```text
stitchcuda                       # show top-level help
stitchcuda doctor                # check CUDA toolchain, KernelBench, credentials
stitchcuda run                   # interactive config editor (problems, model, GPU, attempts, ...)
stitchcuda run --profile demo    # launch with parameters from a saved profile
stitchcuda run --profile demo --level 2 --model gpt-4o-mini
                                 # any flag overrides the profile per-field
stitchcuda run --no-live         # disable the live Rich dashboard (CI mode)

stitchcuda runs list             # table of every run under runs/
stitchcuda runs show RUN_DIR     # configuration, attempts, and best result
stitchcuda runs show RUN_DIR --solution
                                 # also print best_solution.py with highlighting
stitchcuda runs diff RUN_A RUN_B # side-by-side comparison

stitchcuda prompts list          # *.md templates under prompts/
stitchcuda prompts show planner  # render a template as Markdown
stitchcuda prompts edit planner  # open the template in $EDITOR

stitchcuda profile list          # saved run profiles (~/.config/stitchcuda/profiles.toml)
stitchcuda profile show demo
stitchcuda profile save demo --from-run runs/stitchcuda_L1_P1_20260524_174905
stitchcuda profile rm demo
```

The `run` editor shows the full configuration first, lets you jump directly to
the section you want to change, and offers to save the final settings as a
named profile. The second invocation can simply be `stitchcuda run --profile
<name>`. API keys are never written to the profile file — they are always read
from `--api-key`, `$OPENAI_API_KEY`, or (for local OpenAI-compatible servers)
a dummy placeholder.

Press `Esc` inside a section to return to the previous menu. The model section
can configure a local OpenAI-compatible endpoint such as
`http://localhost:8002/v1` and choose from its `/models` response.

During a `run`, an interactive terminal will show a live dashboard
(planner/coder/verifier stage, attempt history, best speedup so far). The
dashboard auto-disables when stdout is not a TTY, and can be forced off with
`--no-live`. Color is suppressed when the `NO_COLOR` environment variable is
set.

Terminal UI output avoids printing local absolute paths where possible:
repository paths are displayed relative to the project root, and home-directory
paths are displayed with `~`.

## Workflow

The current workflow is deterministic:

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

## License

StitchCUDA is distributed under the **PolyForm Noncommercial License 1.0.0**.
Commercial use is not permitted. See [LICENSE](LICENSE) for the full terms.

## Notes

- KernelBench is included as a git submodule, not copied into this repository.
- The default KernelBench root is `third_party/KernelBench`.
- Generated candidates may compile and pass correctness but still be slower than
  PyTorch. That validates the workflow path, not necessarily model quality.
