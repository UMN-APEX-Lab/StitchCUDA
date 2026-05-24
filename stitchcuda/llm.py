from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMConfig:
    model: str
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 65536
    timeout_s: float = 600.0
    reasoning_effort: str = ""


class OpenAIChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        kwargs: dict[str, Any] = {}
        if config.api_base:
            kwargs["base_url"] = config.api_base
        if config.api_key:
            kwargs["api_key"] = config.api_key
        elif config.api_base and not os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = "dummy"
        self.client = OpenAI(**kwargs)

    def chat(self, messages: list[dict[str, str]]) -> str:
        model = _normalize_model_name(self.config.model)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if _uses_max_completion_tokens(model, self.config.reasoning_effort):
            kwargs["max_completion_tokens"] = self.config.max_tokens
            if self.config.reasoning_effort:
                kwargs["reasoning_effort"] = self.config.reasoning_effort
            else:
                kwargs["temperature"] = self.config.temperature
        else:
            kwargs["max_tokens"] = self.config.max_tokens
            kwargs["temperature"] = self.config.temperature

        response = self.client.chat.completions.create(timeout=self.config.timeout_s, **kwargs)
        content = response.choices[0].message.content
        if content is None:
            content = getattr(response.choices[0].message, "reasoning", None) or ""
        return content


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_fence(text, "json")
    candidates = [stripped, text.strip()]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        for idx, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("LLM response did not contain a JSON object")


def _strip_fence(text: str, lang: str) -> str:
    stripped = text.strip()
    prefix = f"```{lang}"
    if stripped.lower().startswith(prefix) and stripped.endswith("```"):
        return stripped[len(prefix):-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def _normalize_model_name(model: str) -> str:
    return model.removeprefix("openai/")


def _uses_max_completion_tokens(model: str, reasoning_effort: str) -> bool:
    lowered = model.lower()
    return bool(reasoning_effort) or lowered.startswith(("gpt-5", "gpt5", "o1", "o3", "o4"))
