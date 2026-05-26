"""Interactive terminal interface for StitchCUDA.

This subpackage hosts the Typer-based CLI, Rich-driven renderers, and
questionary wizards. Core workflow modules (planner/coder/verifier/workflow)
remain free of UI concerns; they only emit structured events via
``stitchcuda.events`` that this subpackage knows how to render.
"""
