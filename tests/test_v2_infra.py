from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from stitchcuda.attempt_memory import DEFAULT_MEMORY_TOKEN_BUDGET, AttemptMemory, estimate_tokens
from stitchcuda.coder import CoderAgent
from stitchcuda.event_log import JsonlEventLog
from stitchcuda.events import NullSink
from stitchcuda.planner import PlannerAgent
from stitchcuda.runtime import _prepare_load_inline_kwargs, _reset_scaffold_state_for_tests
from stitchcuda.schemas import PlanPayload
from stitchcuda.types import CandidateAttempt, KernelBenchProblem, Plan, VerificationResult
from stitchcuda.tui.invocation import RunInvocation
from stitchcuda.verifier import _attempt_extension_root, _extension_prefix
from stitchcuda.workflow import FixedWorkflowConfig, StitchCUDAWorkflow, _meets_target, _next_stage


class V2InfraTests(unittest.TestCase):
    def test_all_default_token_budgets_are_65536(self) -> None:
        memory = AttemptMemory(target_speedup=1.0)
        workflow_config = FixedWorkflowConfig(level=2, problem_id=1, model="fake")
        invocation = RunInvocation(level=2)

        self.assertEqual(DEFAULT_MEMORY_TOKEN_BUDGET, 65_536)
        self.assertEqual(memory.coder_token_budget, 65_536)
        self.assertEqual(memory.replanner_token_budget, 65_536)
        self.assertEqual(workflow_config.max_tokens, 65_536)
        self.assertEqual(workflow_config.coder_memory_tokens, 65_536)
        self.assertEqual(workflow_config.replanner_memory_tokens, 65_536)
        self.assertEqual(invocation.max_tokens, 65_536)
        self.assertEqual(invocation.coder_memory_tokens, 65_536)
        self.assertEqual(invocation.replanner_memory_tokens, 65_536)

    def test_plan_payload_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PlanPayload.model_validate(
                {
                    "summary": "use a fused CUDA kernel",
                    "implementation_steps": ["write ModelNew"],
                    "expected_bottlenecks": [],
                    "correctness_risks": [],
                    "kernelbench_constraints": [],
                    "unexpected": "should not be accepted",
                }
            )

    def test_static_forbidden_result_routes_to_static_repair(self) -> None:
        result = VerificationResult.from_dict(
            {
                "compiled": False,
                "correct": False,
                "error_kind": "static_forbidden",
                "warnings": ["precision downgrade warning"],
                "code_hash": "abc123",
            }
        )

        self.assertEqual(result.warnings, ["precision downgrade warning"])
        self.assertEqual(result.code_hash, "abc123")
        self.assertEqual(_next_stage(result, target_speedup=1.0), "repair_static")

    def test_event_log_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            log = JsonlEventLog(path)
            log.record("code_finished", attempt=0, code_hash="abc123")

            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "code_finished")
        self.assertEqual(events[0]["attempt"], 0)
        self.assertEqual(events[0]["code_hash"], "abc123")

    def test_runtime_scaffold_normalizes_extension_contract(self) -> None:
        _reset_scaffold_state_for_tests()
        with tempfile.TemporaryDirectory() as tmp:
            old_build_dir = os.environ.get("STITCHCUDA_BUILD_DIR")
            old_prefix = os.environ.get("STITCHCUDA_EXTENSION_PREFIX")
            os.environ["STITCHCUDA_BUILD_DIR"] = tmp
            os.environ["STITCHCUDA_EXTENSION_PREFIX"] = "attempt_00_deadbeef"
            try:
                prepared = _prepare_load_inline_kwargs(
                    {
                        "name": "fused_ops",
                        "cuda_sources": """
                        #include <torch/extension.h>
                        __global__ void kernel(float* x) {}
                        torch::Tensor fused_forward(torch::Tensor x) {
                            return x;
                        }
                        """,
                        "functions": ["torch::Tensor fused_forward(torch::Tensor x)"],
                        "force_build": True,
                    }
                )
            finally:
                if old_build_dir is None:
                    os.environ.pop("STITCHCUDA_BUILD_DIR", None)
                else:
                    os.environ["STITCHCUDA_BUILD_DIR"] = old_build_dir
                if old_prefix is None:
                    os.environ.pop("STITCHCUDA_EXTENSION_PREFIX", None)
                else:
                    os.environ["STITCHCUDA_EXTENSION_PREFIX"] = old_prefix

        self.assertEqual(prepared["functions"], ["fused_forward"])
        self.assertIn("torch::Tensor fused_forward(torch::Tensor x);", prepared["cpp_sources"])
        self.assertTrue(prepared["name"].startswith("attempt_00_deadbeef_fused_ops"))
        self.assertNotIn("force_build", prepared)
        self.assertIn("attempt_00_deadbeef_fused_ops", prepared["build_directory"])

    def test_verifier_attempt_extension_names_are_unique(self) -> None:
        root = Path("/tmp/stitchcuda_test_run")
        code_hash_a = "a" * 64
        code_hash_b = "b" * 64

        self.assertEqual(
            _attempt_extension_root(root, 3, code_hash_a),
            root / "torch_extensions" / "attempt_03_aaaaaaaaaaaa",
        )
        self.assertNotEqual(
            _attempt_extension_root(root, 3, code_hash_a),
            _attempt_extension_root(root, 3, code_hash_b),
        )
        self.assertEqual(_extension_prefix(3, code_hash_a), "stitchcuda_a03_aaaaaaaaaaaa")

    def test_attempt_memory_keeps_best_correct_baseline_after_regression(self) -> None:
        memory = AttemptMemory(
            target_speedup=2.0,
            coder_token_budget=512,
            replanner_token_budget=256,
        )
        plan = _plan("fuse reduction and activation")
        memory.record(
            _candidate(
                0,
                VerificationResult(compiled=True, correct=True, speedup=1.3),
                code_hash="a" * 64,
            ),
            code="class ModelNew:\n    pass\n# known correct",
            plan=plan,
        )
        memory.record(
            _candidate(
                1,
                VerificationResult(
                    compiled=False,
                    correct=False,
                    error_kind="compile_error",
                    error="kernel.cu:31:7: error: missing wrapper declaration",
                ),
                code_hash="b" * 64,
            ),
            code="class ModelNew:\n    broken",
            plan=plan,
        )

        context = memory.context_for_coder()

        self.assertIn("known correct", memory.best_correct_code)
        self.assertIn("broken", memory.latest_code)
        self.assertEqual(context.baseline_attempt, 0)
        self.assertEqual(context.baseline_kind, "best_correct")
        self.assertEqual(context.baseline_code, memory.best_correct_code)
        self.assertIn("regressed", context.text)
        self.assertLessEqual(context.estimated_tokens, context.token_budget)

    def test_attempt_memory_deduplicates_dynamic_compile_paths(self) -> None:
        memory = AttemptMemory(target_speedup=1.0)
        plan = _plan("build one fused extension")
        errors = [
            "Error building extension 'first': /tmp/run_a/main.cpp:12:3: error: missing wrapper declaration",
            "Error building extension 'second': /tmp/run_b/main.cpp:99:8: error: missing wrapper declaration",
        ]
        for attempt, error in enumerate(errors):
            memory.record(
                _candidate(
                    attempt,
                    VerificationResult(
                        compiled=False,
                        correct=False,
                        error_kind="compile_error",
                        error=error,
                    ),
                    code_hash=str(attempt) * 64,
                ),
                code=f"# candidate {attempt}",
                plan=plan,
            )

        snapshot = memory.to_dict()

        self.assertEqual(len(snapshot["error_fingerprints"]), 1)
        self.assertEqual(snapshot["error_fingerprints"][0]["count"], 2)
        self.assertEqual(len(snapshot["failed_strategy_summaries"]), 1)
        self.assertEqual(snapshot["failed_strategy_summaries"][0]["count"], 2)

    def test_attempt_memory_clusters_below_target_measurements(self) -> None:
        memory = AttemptMemory(target_speedup=2.0)
        plan = _plan("optimize the same fused kernel")
        for attempt, speedup in enumerate((1.10, 1.15, 1.18)):
            memory.record(
                _candidate(
                    attempt,
                    VerificationResult(
                        compiled=True,
                        correct=True,
                        speedup=speedup,
                        runtime_us=10.0 / speedup,
                        ref_runtime_us=10.0,
                    ),
                    code_hash=f"{attempt:x}" * 64,
                ),
                code=f"# correct candidate {attempt}",
                plan=plan,
            )

        clusters = memory.to_dict()["error_fingerprints"]

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["kind"], "performance_below_target")
        self.assertEqual(clusters[0]["count"], 3)

    def test_attempt_memory_builds_distinct_budgeted_contexts(self) -> None:
        memory = AttemptMemory(
            target_speedup=2.0,
            coder_token_budget=160,
            replanner_token_budget=96,
        )
        plan = _plan("use a deliberately verbose strategy " * 30)
        for attempt in range(5):
            memory.record(
                _candidate(
                    attempt,
                    VerificationResult(
                        compiled=True,
                        correct=False,
                        error_kind="correctness_error",
                        error="Output mismatch " + ("details " * 100),
                    ),
                    code_hash=f"{attempt:x}" * 64,
                ),
                code="def candidate():\n    return None\n",
                plan=plan,
            )

        coder_context = memory.context_for_coder()
        replanner_context = memory.context_for_replanner()

        self.assertNotEqual(coder_context.text, replanner_context.text)
        self.assertIn("Selected baseline", coder_context.text)
        self.assertIn("Compressed attempt memory", replanner_context.text)
        self.assertLessEqual(coder_context.estimated_tokens, coder_context.token_budget)
        self.assertLessEqual(estimate_tokens(replanner_context.text), replanner_context.token_budget)

    def test_planner_injects_reference_code_once(self) -> None:
        llm = _CapturingLLM(
            json.dumps(
                {
                    "summary": "fuse operations",
                    "implementation_steps": [],
                    "expected_bottlenecks": [],
                    "correctness_risks": [],
                    "kernelbench_constraints": [],
                }
            )
        )
        problem = KernelBenchProblem(
            level=2,
            problem_id=1,
            name="sentinel",
            reference_code="REFERENCE_CODE_SENTINEL",
        )

        PlannerAgent(llm).run(problem, hardware_summary={"gpu_name": "test"})

        prompt = llm.messages[-1]["content"]
        self.assertEqual(prompt.count("REFERENCE_CODE_SENTINEL"), 1)

    def test_coder_revise_receives_compressed_memory_and_selected_baseline(self) -> None:
        llm = _CapturingLLM("```python\nclass ModelNew:\n    pass\n```")
        coder = CoderAgent(llm, kernelbench_root="")

        with patch("stitchcuda.coder.extract_solution_code", return_value="class ModelNew:\n    pass"):
            code = coder.revise(
                KernelBenchProblem(level=2, problem_id=1, name="test", reference_code="ref"),
                kernelbench_prompt="KERNELBENCH_PROMPT",
                plan=_plan("fuse operations"),
                baseline_code="BASELINE_CODE_SENTINEL",
                attempt_memory_context="MEMORY_CONTEXT_SENTINEL",
                hardware_summary={"gpu_name": "test"},
                target_speedup=2.0,
            )

        prompt = llm.messages[-1]["content"]
        self.assertIn("BASELINE_CODE_SENTINEL", prompt)
        self.assertIn("MEMORY_CONTEXT_SENTINEL", prompt)
        self.assertNotIn("{{", prompt)
        self.assertIn("class ModelNew", code)

    def test_workflow_uses_best_correct_memory_baseline(self) -> None:
        problem = KernelBenchProblem(
            level=2,
            problem_id=1,
            name="memory_integration",
            reference_code="class Model:\n    pass",
        )
        coder = _FakeCoder()
        verifier = _FakeVerifier(
            [
                VerificationResult(compiled=True, correct=True, speedup=1.2),
                VerificationResult(
                    compiled=False,
                    correct=False,
                    error_kind="compile_error",
                    error="missing declaration",
                ),
                VerificationResult(compiled=True, correct=True, speedup=2.2),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            workflow = object.__new__(StitchCUDAWorkflow)
            workflow.config = FixedWorkflowConfig(
                level=2,
                problem_id=1,
                model="fake",
                output_root=tmp,
                run_name="memory_integration",
                max_attempts=3,
                max_replans=0,
                target_speedup=2.0,
            )
            workflow.events = NullSink()
            workflow.planner = _FakePlanner()
            workflow.coder = coder
            workflow.verifier = verifier
            workflow._hardware_summary = lambda: {"gpu_name": "test"}

            with (
                patch("stitchcuda.workflow.load_problem", return_value=problem),
                patch("stitchcuda.workflow.build_kernelbench_prompt", return_value="KB_PROMPT"),
            ):
                summary = workflow.run()

            memory_snapshot = json.loads(
                (Path(summary["run_dir"]) / "attempt_memory.json").read_text(encoding="utf-8")
            )

        self.assertEqual(coder.baselines, ["CODE_0_CORRECT", "CODE_0_CORRECT"])
        self.assertEqual(summary["stop_reason"], "target_reached")
        self.assertEqual(memory_snapshot["latest"]["attempt"], 2)
        self.assertEqual(memory_snapshot["best_correct"]["attempt"], 2)


def _candidate(attempt: int, result: VerificationResult, *, code_hash: str) -> CandidateAttempt:
    return CandidateAttempt(
        attempt=attempt,
        plan_version=0,
        stage="draft" if attempt == 0 else "repair",
        solution_path=Path(f"attempt_{attempt:02d}.py"),
        result=result,
        code_hash=code_hash,
    )


def _plan(summary: str) -> Plan:
    return Plan(
        summary=summary,
        implementation_steps=["write a managed CUDA extension", "preserve output shapes"],
    )


class _CapturingLLM:
    def __init__(self, response: str):
        self.response = response
        self.messages: list[dict[str, str]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.response


class _FakeMetadataLLM:
    def last_metadata_dict(self) -> dict:
        return {}


class _FakePlanner:
    def __init__(self):
        self.llm = _FakeMetadataLLM()

    def run(self, problem: KernelBenchProblem, *, hardware_summary: dict) -> Plan:
        return _plan("start from one fused kernel")

    def replan(self, *args, **kwargs) -> Plan:
        return _plan("replanned strategy")


class _FakeCoder:
    def __init__(self):
        self.llm = _FakeMetadataLLM()
        self.baselines: list[str] = []

    def draft(self, *args, **kwargs) -> str:
        return "CODE_0_CORRECT"

    def revise(self, *args, baseline_code: str, **kwargs) -> str:
        self.baselines.append(baseline_code)
        return "CODE_1_BROKEN" if len(self.baselines) == 1 else "CODE_2_CORRECT"


class _FakeVerifier:
    def __init__(self, results: list[VerificationResult]):
        self.results = results

    def verify(self, problem: KernelBenchProblem, *, code_path: Path, output_dir: Path, attempt: int):
        return self.results[attempt]


if __name__ == "__main__":
    unittest.main()
