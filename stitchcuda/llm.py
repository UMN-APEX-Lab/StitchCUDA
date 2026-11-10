from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import openai
from openai import OpenAI

# Transient API failures (rate limits, timeouts, dropped connections, 5xx) are retried with
# exponential backoff instead of aborting the whole run.
_RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
              openai.InternalServerError)


@dataclass
class LLMConfig:
    model: str
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 65536
    timeout_s: float = 3600.0  # long non-streamed generations (up to 64K tokens) take well over 10 min
    reasoning_effort: str = ""
    max_retries: int = 8


@dataclass
class LLMCallMetadata:
    response_id: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    content_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIChatClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_metadata: LLMCallMetadata | None = None
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

        response = self._create(kwargs)
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            content = getattr(choice.message, "reasoning", None) or ""
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {})
        self.last_metadata = LLMCallMetadata(
            response_id=str(getattr(response, "id", "") or ""),
            model=str(getattr(response, "model", "") or model),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            usage=usage_dict,
            content_chars=len(content),
        )
        return content

    def _create(self, kwargs: dict[str, Any]):
        delay = 5.0
        for attempt in range(self.config.max_retries + 1):
            if getattr(self, "_drop_temperature", False):
                kwargs.pop("temperature", None)
            try:
                return self.client.chat.completions.create(timeout=self.config.timeout_s, **kwargs)
            except openai.BadRequestError as exc:
                # Reasoning model families accept only their default temperature.
                if "temperature" in kwargs and "temperature" in str(exc).lower() and not getattr(self, "_drop_temperature", False):
                    self._drop_temperature = True
                    continue
                raise
            except _RETRYABLE:
                if attempt >= self.config.max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 300.0)
        raise RuntimeError("unreachable")

    def last_metadata_dict(self) -> dict[str, Any]:
        return self.last_metadata.to_dict() if self.last_metadata else {}


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
    # GPT-5 and later families (gpt-5*, gpt-6*, ...) and the o-series take max_completion_tokens.
    return bool(reasoning_effort) or bool(re.match(r"(gpt-?(?:[5-9]|\d{2,})|o\d)", lowered))
