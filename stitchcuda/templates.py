from __future__ import annotations

from pathlib import Path

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def render_template(name: str, **values: object) -> str:
    path = PROMPT_ROOT / name
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
