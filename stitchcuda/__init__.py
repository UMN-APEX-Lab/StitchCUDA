"""Fixed KernelBench planner-coder-verifier workflow."""

__all__ = ["FixedWorkflowConfig", "StitchCUDAWorkflow"]


def __getattr__(name: str):
    if name in __all__:
        from .workflow import FixedWorkflowConfig, StitchCUDAWorkflow

        return {
            "FixedWorkflowConfig": FixedWorkflowConfig,
            "StitchCUDAWorkflow": StitchCUDAWorkflow,
        }[name]
    raise AttributeError(name)
