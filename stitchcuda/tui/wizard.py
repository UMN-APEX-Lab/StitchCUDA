"""Interactive ``run`` configuration editor built on questionary.

The editor starts with sensible defaults, renders the whole
:class:`RunInvocation`, and lets the user jump directly to the section they
want to change. It never raises on user cancellation; it returns ``None`` so
the caller can choose how to handle the exit (typically a clean
``typer.Exit``).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from typing import Any, Callable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from questionary import utils
from questionary.constants import DEFAULT_QUESTION_PREFIX, DEFAULT_SELECTED_POINTER
from questionary.prompts import common
from questionary.prompts.common import Choice, InquirerControl, Separator
from questionary.question import Question
from questionary.styles import merge_styles_default

from ..kernelbench import default_kernelbench_root
from .console import stdout_console
from .display import display_endpoint, display_path
from .invocation import RunInvocation, resolve_problem_ids

_GPU_ARCH_PRESETS: tuple[str, ...] = ("Blackwell", "Hopper", "Ada", "Ampere")
_OTHER_ARCH_LABEL = "Other (enter manually)"
_BACK = "Back"
_QUIT = "Quit without running"
_ESCAPED = object()
_LOCAL_ENDPOINTS: tuple[str, ...] = (
    "http://localhost:8002/v1",
    "http://127.0.0.1:8002/v1",
    "http://localhost:8004/v1",
    "http://127.0.0.1:8004/v1",
)


def default_invocation() -> RunInvocation:
    """RunInvocation seeded with environment-aware defaults."""
    return RunInvocation(
        level=1,
        model=os.environ.get("STITCHCUDA_MODEL", "gpt-4o"),
        api_base=os.environ.get("STITCHCUDA_API_BASE", ""),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        kernelbench_root=os.environ.get("KERNELBENCH_ROOT") or default_kernelbench_root(),
    )


def prompt(defaults: RunInvocation | None = None, *, console: Console | None = None) -> RunInvocation | None:
    """Interactively gather a :class:`RunInvocation`.

    Returns ``None`` if the editor cannot run (no TTY) or the user cancels.
    """
    if not _is_interactive():
        return None

    console = console or stdout_console()
    current = _ensure_problem_ids(defaults or default_invocation())

    while True:
        _render_summary(current, console)
        action = _select(
            "Choose what to do",
            choices=[
                Choice("Run now", "run"),
                Choice("Problems", "problems"),
                Choice("Model / endpoint", "model"),
                Choice("Hardware", "hardware"),
                Choice("Attempts / verifier", "attempts"),
                Choice("Output / prompts", "output"),
                Choice("Reset defaults", "reset"),
                Choice(_QUIT, "quit"),
            ],
            escape_value="quit",
        ).ask()
        if action is None or action == "quit":
            return None
        if action == "run":
            return current
        if action == "reset":
            current = _ensure_problem_ids(default_invocation())
            continue

        edited = _edit_section(current, action, console)
        if edited is None:
            return None
        current = edited


def prompt_save_decision() -> str | None:
    """Ask whether to save the just-built invocation as a named profile.

    Returns the chosen profile name, ``""`` for "do not save", or ``None``
    if the user cancelled.
    """
    if not _is_interactive():
        return ""
    save = questionary.confirm("Save these settings as a profile?", default=False).ask()
    if save is None:
        return None
    if not save:
        return ""
    name = questionary.text(
        "Profile name:",
        default="default",
        validate=_validate_profile_name,
    ).ask()
    return (name or "").strip()


def _edit_section(current: RunInvocation, section: str, console: Console) -> RunInvocation | None:
    if section == "problems":
        return _edit_problems(current, console)
    if section == "model":
        return _edit_model(current, console)
    if section == "hardware":
        return _edit_hardware(current)
    if section == "attempts":
        return _edit_attempts(current)
    if section == "output":
        return _edit_output(current)
    return current


def _render_summary(invocation: RunInvocation, console: Console) -> None:
    console.clear()
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)

    table.add_row(
        _summary_panel(
            "Run",
            [
                ("Level", str(invocation.level)),
                ("Problems", _format_problem_ids(invocation.problem_ids)),
                ("Output", display_path(invocation.output_root)),
                ("Run name", invocation.run_name or "-"),
            ],
        ),
        _summary_panel(
            "Model",
            [
                ("Model", invocation.model),
                ("API base", display_endpoint(invocation.api_base)),
                ("API key", _api_key_status(invocation.api_key)),
                ("Max tokens", str(invocation.max_tokens)),
                ("Temperature", str(invocation.temperature)),
            ],
        ),
    )
    table.add_row(
        _summary_panel(
            "Hardware",
            [
                ("GPU arch", invocation.gpu_arch),
                ("Target SM", invocation.target_sm or "auto"),
                ("Device", "env/default" if invocation.device is None else str(invocation.device)),
            ],
        ),
        _summary_panel(
            "Verifier",
            [
                ("Max attempts", str(invocation.max_attempts)),
                ("Max replans", str(invocation.max_replans)),
                ("Target speedup", str(invocation.target_speedup)),
                ("Correct trials", str(invocation.num_correct_trials)),
                ("Perf trials", str(invocation.num_perf_trials)),
                ("Measure perf", "yes" if invocation.measure_performance else "no"),
            ],
        ),
    )
    table.add_row(
        _summary_panel(
            "Prompts",
            [
                ("Prompt option", invocation.prompt_option),
                ("Reasoning effort", invocation.reasoning_effort or "-"),
                ("KernelBench", display_path(invocation.kernelbench_root or default_kernelbench_root())),
                ("Timeout", f"{invocation.verifier_timeout_s}s"),
            ],
        ),
        "",
    )
    console.print(
        Panel(
            table,
            title="StitchCUDA run configuration",
            subtitle=f"Select '{_QUIT}' or press Ctrl-C to exit",
            border_style="cyan",
        )
    )


def _summary_panel(title: str, rows: list[tuple[str, str]]) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    return Panel(table, title=title, border_style="dim")


def _edit_problems(current: RunInvocation, console: Console) -> RunInvocation | None:
    level_text = _select(
        "KernelBench level",
        choices=["1", "2", "3"],
        default=str(current.level),
        escape_value="back",
    ).ask()
    if level_text is None:
        return None
    if level_text == "back":
        return current
    level = int(level_text)

    mode = _select(
        "Problem selection",
        choices=[
            Choice("Specific ids", "ids"),
            Choice("First N problems", "first_n"),
            Choice("Whole level", "all"),
            Choice(_BACK, "back"),
        ],
        escape_value="back",
    ).ask()
    if mode is None:
        return None
    if mode == "back":
        return current

    kernelbench_root = current.kernelbench_root or default_kernelbench_root()
    if mode == "ids":
        default_ids = ",".join(str(i) for i in current.problem_ids) or "1"
        ids = _ask_int_list("Problem ids (comma-separated)", default=default_ids)
        if ids is _ESCAPED:
            return current
        if ids is None:
            return None
        return replace(current, level=level, problem_ids=ids)

    if mode == "first_n":
        n = _ask_int("How many problems?", default=max(1, len(current.problem_ids) or 1))
        if n is _ESCAPED:
            return current
        if n is None:
            return None
        try:
            ids = resolve_problem_ids(
                level=level,
                kernelbench_root=kernelbench_root,
                problem_ids=None,
                max_problems=n,
            )
        except Exception as exc:
            console.print(f"[red]Could not load KernelBench dataset: {exc}[/red]")
            questionary.press_any_key_to_continue().ask()
            return current
        return replace(current, level=level, problem_ids=ids)

    try:
        ids = resolve_problem_ids(
            level=level,
            kernelbench_root=kernelbench_root,
            problem_ids=None,
            max_problems=10_000,
        )
    except Exception as exc:
        console.print(f"[red]Could not load KernelBench dataset: {exc}[/red]")
        questionary.press_any_key_to_continue().ask()
        return current
    return replace(current, level=level, problem_ids=ids)


def _edit_model(current: RunInvocation, console: Console) -> RunInvocation | None:
    fields = {
        "model": _Field("model", "Model", "str"),
        "api_base": _Field("api_base", "API base URL", "str"),
        "api_key": _Field("api_key", "API key", "secret"),
        "temperature": _Field("temperature", "Temperature", "float"),
        "max_tokens": _Field("max_tokens", "Max tokens", "int"),
        "reasoning_effort": _Field("reasoning_effort", "Reasoning effort", "str"),
    }
    while True:
        choice = _select(
            "Model / endpoint",
            choices=[
                Choice("Use local endpoint and discover model", "local"),
                Choice("Select model from current endpoint", "discover"),
                Choice(f"Model = {current.model}", "model"),
                Choice(f"API base URL = {display_endpoint(current.api_base)}", "api_base"),
                Choice(f"API key = {_api_key_status(current.api_key)}", "api_key"),
                Choice(f"Temperature = {current.temperature}", "temperature"),
                Choice(f"Max tokens = {current.max_tokens}", "max_tokens"),
                Choice(f"Reasoning effort = {current.reasoning_effort or '-'}", "reasoning_effort"),
                Choice(_BACK, "back"),
            ],
            escape_value="back",
        ).ask()
        if choice is None:
            return None
        if choice == "back":
            return current
        if choice == "local":
            updated = _configure_local_endpoint(current, console)
            if updated is None:
                return None
            current = updated
            continue
        if choice == "discover":
            updated = _select_model_from_endpoint(current, console)
            if updated is None:
                return None
            current = updated
            continue
        current = _edit_one_field(current, fields[choice])
        if current is None:
            return None


def _edit_hardware(current: RunInvocation) -> RunInvocation | None:
    while True:
        field = _select(
            "Hardware",
            choices=[
                Choice(f"GPU arch = {current.gpu_arch}", "gpu_arch"),
                Choice(f"Target SM = {current.target_sm or 'auto'}", "target_sm"),
                Choice(
                    f"CUDA device = {'env/default' if current.device is None else current.device}",
                    "device",
                ),
                Choice(_BACK, "back"),
            ],
            escape_value="back",
        ).ask()
        if field is None:
            return None
        if field == "back":
            return current
        if field == "gpu_arch":
            updated = _pick_gpu_arch(current)
            if updated is None:
                return None
            current = updated
        elif field == "target_sm":
            value = _text("Target SM (empty = auto)", default=current.target_sm)
            if value is _ESCAPED:
                continue
            if value is None:
                return None
            current = replace(current, target_sm=value.strip())
        elif field == "device":
            value = _text(
                "CUDA device index (empty = CUDA_VISIBLE_DEVICES/default)",
                default="" if current.device is None else str(current.device),
                validate=_validate_optional_int,
            )
            if value is _ESCAPED:
                continue
            if value is None:
                return None
            current = replace(current, device=int(value) if value.strip() else None)


def _edit_attempts(current: RunInvocation) -> RunInvocation | None:
    return _edit_fields(
        current,
        "Attempts / verifier",
        [
            _Field("max_attempts", "Max attempts", "int"),
            _Field("max_replans", "Max replans", "int"),
            _Field("replan_after_failed_attempts", "Replan after failed attempts", "int"),
            _Field("replan_after_stagnant_attempts", "Replan after stagnant attempts", "int"),
            _Field("target_speedup", "Target speedup", "float"),
            _Field("num_correct_trials", "Correctness trials", "int"),
            _Field("num_perf_trials", "Performance trials", "int"),
            _Field("measure_performance", "Measure performance", "bool"),
            _Field("verifier_timeout_s", "Verifier timeout seconds", "int"),
        ],
    )


def _edit_output(current: RunInvocation) -> RunInvocation | None:
    return _edit_fields(
        current,
        "Output / prompts",
        [
            _Field("kernelbench_root", "KernelBench root", "str"),
            _Field("output_root", "Output root", "str"),
            _Field("run_name", "Run name", "str"),
            _Field("prompt_option", "Prompt option", "str"),
        ],
    )


class _Field:
    def __init__(self, name: str, label: str, kind: str):
        self.name = name
        self.label = label
        self.kind = kind


def _edit_fields(current: RunInvocation, title: str, fields: list[_Field]) -> RunInvocation | None:
    by_name = {field.name: field for field in fields}
    while True:
        choice = _select(
            title,
            choices=[
                Choice(f"{field.label} = {_display_field_value(current, field)}", field.name)
                for field in fields
            ]
            + [Choice(_BACK, "back")],
            escape_value="back",
        ).ask()
        if choice is None:
            return None
        if choice == "back":
            return current
        current = _edit_one_field(current, by_name[choice])
        if current is None:
            return None


def _edit_one_field(current: RunInvocation, field: _Field) -> RunInvocation | None:
    value = getattr(current, field.name)
    if field.kind == "bool":
        answer = _select(
            field.label,
            choices=[Choice("yes", True), Choice("no", False), Choice(_BACK, "back")],
            default=bool(value),
            escape_value="back",
        ).ask()
        if answer is None:
            return None
        if answer == "back":
            return current
        return replace(current, **{field.name: bool(answer)})
    if field.kind == "secret":
        answer = _password(f"{field.label} (blank keeps current/env)")
        if answer is _ESCAPED:
            return current
        if answer is None:
            return None
        if not answer:
            return current
        return replace(current, **{field.name: answer})
    if field.name in {"kernelbench_root", "output_root"}:
        answer = _text(f"{field.label} (blank keeps {display_path(value)})", default="")
        if answer is _ESCAPED:
            return current
        if answer is None:
            return None
        raw_path = answer.strip()
        if not raw_path:
            return current
        return replace(current, **{field.name: raw_path})

    validator = None
    if field.kind == "int":
        validator = _validate_int_text
    elif field.kind == "float":
        validator = _validate_float_text
    answer = _text(field.label, default=str(value), validate=validator)
    if answer is _ESCAPED:
        return current
    if answer is None:
        return None
    raw = answer.strip()
    if field.kind == "int":
        return replace(current, **{field.name: int(raw)})
    if field.kind == "float":
        return replace(current, **{field.name: float(raw)})
    return replace(current, **{field.name: raw})


def _pick_gpu_arch(current: RunInvocation) -> RunInvocation | None:
    arch_default = current.gpu_arch if current.gpu_arch in _GPU_ARCH_PRESETS else _OTHER_ARCH_LABEL
    choice = _select(
        "GPU architecture",
        choices=[*_GPU_ARCH_PRESETS, _OTHER_ARCH_LABEL],
        default=arch_default,
        escape_value="back",
    ).ask()
    if choice is None:
        return None
    if choice == "back":
        return current
    if choice != _OTHER_ARCH_LABEL:
        return replace(current, gpu_arch=choice)
    manual = _text("Architecture name", default=current.gpu_arch)
    if manual is _ESCAPED:
        return current
    if manual is None:
        return None
    return replace(current, gpu_arch=manual.strip() or current.gpu_arch)


def _configure_local_endpoint(current: RunInvocation, console: Console) -> RunInvocation | None:
    endpoint = _select(
        "Local OpenAI-compatible endpoint",
        choices=[*list(_LOCAL_ENDPOINTS), "Custom endpoint", _BACK],
        default=current.api_base if current.api_base in _LOCAL_ENDPOINTS else _LOCAL_ENDPOINTS[0],
        escape_value=_BACK,
    ).ask()
    if endpoint is None:
        return None
    if endpoint == _BACK:
        return current
    if endpoint == "Custom endpoint":
        answer = _text(
            "Endpoint base URL",
            default=current.api_base or _LOCAL_ENDPOINTS[0],
            validate=_validate_endpoint_text,
        )
        if answer is _ESCAPED:
            return current
        if answer is None:
            return None
        endpoint = answer.strip()

    updated = replace(current, api_base=endpoint, api_key=current.api_key or "")
    return _select_model_from_endpoint(updated, console)


def _select_model_from_endpoint(current: RunInvocation, console: Console) -> RunInvocation | None:
    if not current.api_base:
        console.print("[yellow]Set API base URL before discovering models.[/yellow]")
        questionary.press_any_key_to_continue().ask()
        return current

    models, error = _fetch_model_ids(current.api_base)
    if error:
        console.print(f"[yellow]Could not fetch {display_endpoint(current.api_base)}/models: {error}[/yellow]")
        manual = _text("Model name", default=current.model)
        if manual is _ESCAPED:
            return current
        if manual is None:
            return None
        return replace(current, model=manual.strip() or current.model)
    if not models:
        console.print(f"[yellow]No models returned by {display_endpoint(current.api_base)}/models.[/yellow]")
        manual = _text("Model name", default=current.model)
        if manual is _ESCAPED:
            return current
        if manual is None:
            return None
        return replace(current, model=manual.strip() or current.model)

    choice = _select(
        "Model",
        choices=[*models, _BACK],
        default=current.model if current.model in models else models[0],
        escape_value=_BACK,
    ).ask()
    if choice is None:
        return None
    if choice == _BACK:
        return current
    return replace(current, model=str(choice))


def _fetch_model_ids(api_base: str) -> tuple[list[str], str]:
    url = _models_url(api_base)
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return [], str(exc)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return [], "response did not contain a data list"
    models = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
    return sorted(models), ""


def _models_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    return f"{base}/models"


def _validate_endpoint_text(text: str) -> bool | str:
    value = text.strip()
    if not value:
        return "Endpoint cannot be empty."
    if not (value.startswith("http://") or value.startswith("https://")):
        return "Endpoint must start with http:// or https://."
    return True


def _select(
    message: str,
    *,
    choices: Sequence[str | Choice | dict[str, Any]],
    default: str | Choice | dict[str, Any] | None = None,
    escape_value: Any = _BACK,
) -> Question:
    """Questionary-style select prompt where Esc returns ``escape_value``."""
    ic = InquirerControl(
        choices,
        default,
        pointer=DEFAULT_SELECTED_POINTER,
        use_indicator=False,
        use_shortcuts=False,
        show_selected=False,
        show_description=True,
        use_arrow_keys=True,
        initial_choice=default,
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens = [("class:qmark", DEFAULT_QUESTION_PREFIX), ("class:question", f" {message} ")]
        if ic.is_answered:
            title = ic.get_pointed_at().title
            if isinstance(title, list):
                tokens.append(("class:answer", "".join(token[1] for token in title)))
            else:
                tokens.append(("class:answer", title))
        else:
            tokens.append(("class:instruction", "(Use arrow keys, Enter to select, Esc to go back)"))
        return tokens

    layout = common.create_inquirer_layout(ic, get_prompt_tokens)
    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _(event):
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(Keys.Escape)
    def _(event):
        event.app.exit(result=escape_value)

    def move_cursor_down(event):
        ic.select_next()
        while not ic.is_selection_valid():
            ic.select_next()

    def move_cursor_up(event):
        ic.select_previous()
        while not ic.is_selection_valid():
            ic.select_previous()

    bindings.add(Keys.Down, eager=True)(move_cursor_down)
    bindings.add(Keys.Up, eager=True)(move_cursor_up)
    bindings.add("j", eager=True)(move_cursor_down)
    bindings.add("k", eager=True)(move_cursor_up)
    bindings.add(Keys.ControlN, eager=True)(move_cursor_down)
    bindings.add(Keys.ControlP, eager=True)(move_cursor_up)

    @bindings.add(Keys.ControlM, eager=True)
    def _(event):
        ic.is_answered = True
        event.app.exit(result=ic.get_pointed_at().value)

    @bindings.add(Keys.Any)
    def _(event):
        pass

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merge_styles_default([None]),
            **utils.used_kwargs({}, Application.__init__),
        )
    )


def _text(message: str, *, default: str = "", validate: Any = None) -> str | object | None:
    return questionary.text(
        message,
        default=default,
        validate=validate,
    ).ask()


def _password(message: str) -> str | object | None:
    return questionary.password(
        message,
    ).ask()


def _ensure_problem_ids(invocation: RunInvocation) -> RunInvocation:
    if invocation.problem_ids:
        return invocation
    return replace(invocation, problem_ids=[1])


def _format_problem_ids(problem_ids: list[int]) -> str:
    if not problem_ids:
        return "-"
    if len(problem_ids) <= 8:
        return ", ".join(str(i) for i in problem_ids)
    head = ", ".join(str(i) for i in problem_ids[:5])
    tail = ", ".join(str(i) for i in problem_ids[-2:])
    return f"{len(problem_ids)} selected ({head}, ..., {tail})"


def _api_key_status(api_key: str) -> str:
    if api_key:
        return "set"
    return "from env" if os.environ.get("OPENAI_API_KEY") else "not set"


def _display_field_value(current: RunInvocation, field: _Field) -> str:
    value = getattr(current, field.name)
    if field.kind == "secret":
        return _api_key_status(str(value))
    if value == "":
        return "-"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if field.name in {"kernelbench_root", "output_root"}:
        return display_path(value)
    return str(value)


def _ask_int(
    label: str,
    *,
    default: int,
    validator: Callable[[int], bool | str] | None = None,
) -> int | object | None:
    def _validate(text: str) -> bool | str:
        text = text.strip()
        if not text:
            return "Please enter an integer."
        try:
            value = int(text)
        except ValueError:
            return f"{text!r} is not an integer."
        if validator is None:
            return True
        verdict = validator(value)
        if verdict is True:
            return True
        return verdict if isinstance(verdict, str) else "Invalid value."

    answer = _text(label, default=str(default), validate=_validate)
    if answer is _ESCAPED:
        return _ESCAPED
    if answer is None:
        return None
    return int(answer.strip())


def _validate_int_text(text: str) -> bool | str:
    text = text.strip()
    if not text:
        return "Please enter an integer."
    try:
        int(text)
    except ValueError:
        return f"{text!r} is not an integer."
    return True


def _ask_float(label: str, *, default: float) -> float | object | None:
    def _validate(text: str) -> bool | str:
        text = text.strip()
        if not text:
            return "Please enter a number."
        try:
            float(text)
        except ValueError:
            return f"{text!r} is not a number."
        return True

    answer = _text(label, default=str(default), validate=_validate)
    if answer is _ESCAPED:
        return _ESCAPED
    if answer is None:
        return None
    return float(answer.strip())


def _validate_float_text(text: str) -> bool | str:
    text = text.strip()
    if not text:
        return "Please enter a number."
    try:
        float(text)
    except ValueError:
        return f"{text!r} is not a number."
    return True


def _ask_int_list(label: str, *, default: str) -> list[int] | object | None:
    def _validate(text: str) -> bool | str:
        items = [part.strip() for part in text.split(",") if part.strip()]
        if not items:
            return "Please enter at least one id."
        try:
            [int(part) for part in items]
        except ValueError:
            return f"Could not parse {text!r} as a comma-separated list of integers."
        return True

    answer = _text(label, default=default, validate=_validate)
    if answer is _ESCAPED:
        return _ESCAPED
    if answer is None:
        return None
    return [int(part.strip()) for part in answer.split(",") if part.strip()]


def _validate_optional_int(text: str) -> bool | str:
    text = text.strip()
    if not text:
        return True
    try:
        int(text)
    except ValueError:
        return f"{text!r} is not an integer."
    return True


def _validate_profile_name(text: str) -> bool | str:
    name = text.strip()
    if not name:
        return "Profile name cannot be empty."
    if not all(ch.isalnum() or ch in "._-" for ch in name):
        return "Use only letters, digits, dot, dash, or underscore."
    return True


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
