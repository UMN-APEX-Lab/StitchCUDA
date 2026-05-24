from __future__ import annotations

import argparse
import json
import os

from .kernelbench import default_kernelbench_root, load_dataset_problems
from .workflow import FixedWorkflowConfig, StitchCUDAWorkflow


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StitchCUDA fixed KernelBench workflow")
    parser.add_argument("--level", type=int, required=True, help="KernelBench level")
    parser.add_argument("--problem-id", type=int, action="append", help="KernelBench problem id; repeatable")
    parser.add_argument("--max-problems", type=int, default=1, help="Used when --problem-id is omitted")
    parser.add_argument("--model", default=os.environ.get("STITCHCUDA_MODEL", "gpt-4o"))
    parser.add_argument("--api-base", default=os.environ.get("STITCHCUDA_API_BASE", ""))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument(
        "--kernelbench-root",
        default=os.environ.get("KERNELBENCH_ROOT", default_kernelbench_root()),
        help="KernelBench repo root. Defaults to the bundled third_party/KernelBench submodule.",
    )
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--prompt-option", default="zero_shot_safe_glue")
    parser.add_argument(
        "--gpu-arch",
        default="Blackwell",
        help="KernelBench architecture name passed to kernelbench.utils.set_gpu_arch, e.g. Blackwell/Ada/Hopper.",
    )
    parser.add_argument(
        "--target-sm",
        default="",
        help="Optional target SM for prompt context, e.g. sm_120 or 12.0. If omitted, StitchCUDA derives it from nvidia-smi.",
    )
    parser.add_argument("--device", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--replan-after-failed-attempts", type=int, default=2)
    parser.add_argument("--replan-after-stagnant-attempts", type=int, default=2)
    parser.add_argument("--target-speedup", type=float, default=1.0)
    parser.add_argument("--num-correct-trials", type=int, default=5)
    parser.add_argument("--num-perf-trials", type=int, default=10)
    parser.add_argument("--no-performance", action="store_true")
    parser.add_argument("--verifier-timeout-s", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    problem_ids = args.problem_id or _default_problem_ids(args)
    summaries = []

    for problem_id in problem_ids:
        cfg = FixedWorkflowConfig(
            level=args.level,
            problem_id=problem_id,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            kernelbench_root=args.kernelbench_root,
            output_root=args.output_root,
            run_name=args.run_name if len(problem_ids) == 1 else "",
            prompt_option=args.prompt_option,
            gpu_arch=args.gpu_arch,
            target_sm=args.target_sm,
            device=args.device,
            max_attempts=args.max_attempts,
            max_replans=args.max_replans,
            replan_after_failed_attempts=args.replan_after_failed_attempts,
            replan_after_stagnant_attempts=args.replan_after_stagnant_attempts,
            target_speedup=args.target_speedup,
            num_correct_trials=args.num_correct_trials,
            num_perf_trials=args.num_perf_trials,
            measure_performance=not args.no_performance,
            verifier_timeout_s=args.verifier_timeout_s,
        )
        summary = StitchCUDAWorkflow(cfg).run()
        summaries.append(summary)
        best = summary.get("best_attempt") or {}
        result = best.get("result") or {}
        print(
            f"L{args.level} P{problem_id}: {summary['stop_reason']} "
            f"best_speedup={float(result.get('speedup', 0.0) or 0.0):.3f} "
            f"run_dir={summary['run_dir']}",
            flush=True,
        )

    print(json.dumps({"runs": summaries}, indent=2, default=str))
    return 0


def _default_problem_ids(args: argparse.Namespace) -> list[int]:
    problems = load_dataset_problems(
        level=args.level,
        kernelbench_root=args.kernelbench_root,
        problem_ids=None,
    )
    limit = max(1, int(args.max_problems))
    return [problem.problem_id for problem in problems[:limit]]


if __name__ == "__main__":
    raise SystemExit(main())
