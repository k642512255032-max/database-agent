"""Minimal Ollama client with structured (JSON-schema) output.

Small models such as qwen2.5-coder:1.5b/3b are far more reliable when Ollama
constrains the output with a JSON schema, so every agent call uses `chat_json`.

Thinking models (qwen3, deepseek-r1, gpt-oss, ...) reason before they answer: for them the
request carries "think": true, Ollama returns the reasoning separately in message.thinking
(kept in `last_thinking` so the trace can show it) and the content stays clean JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests

from .config import settings


class LLMError(RuntimeError):
    pass


def is_thinking_model(name: str) -> bool:
    return bool(re.search(settings.thinking_models, name or "", re.I))


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None, think: bool | None = None):
        self.model = model or settings.model
        self.host = (host or settings.ollama_host).rstrip("/")
        self.think = is_thinking_model(self.model) if think is None else think
        self.last_thinking: str = ""          # reasoning of the last call (thinking models only)
        self.last_stats: dict[str, Any] = {}  # Ollama timings / token counts of the last call (nanoseconds)

    def light(self) -> "OllamaLLM":
        """The same model with thinking off: for steps that only fill a JSON form (charts, memory, brief summary)."""
        return OllamaLLM(self.model, self.host, think=False)

    # ------------------------------------------------------------------ utils
    def health(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
            if not any(n == self.model or n.startswith(self.model + ":") for n in names):
                return False, f"Ollama is running but model '{self.model}' is not pulled. Run: ollama pull {self.model}"
            return True, f"Ollama OK - model {self.model}"
        except Exception as exc:
            return False, f"Cannot reach Ollama at {self.host}: {exc}"

    def warm(self) -> tuple[bool, str]:
        """Load the model now (an empty chat request) so the first real call does not pay the load time."""
        try:
            r = requests.post(f"{self.host}/api/chat", json={"model": self.model, "messages": [],
                                                              "keep_alive": settings.llm_keep_alive}, timeout=120)
            r.raise_for_status()
            return True, f"{self.model} loaded"
        except requests.RequestException as exc:
            return False, f"could not load {self.model}: {exc}"

    # ------------------------------------------------------------------- core
    def chat(self, system: str, user: str, schema: Optional[dict] = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": settings.temperature, "num_ctx": settings.num_ctx},
            "keep_alive": settings.llm_keep_alive,   # keep the model (and its prompt cache) loaded between calls
        }
        if schema is not None:
            payload["format"] = schema  # Ollama >= 0.5 structured outputs
        if self.think:
            payload["think"] = True     # Ollama >= 0.9: reasoning comes back in message.thinking
        timeout = settings.llm_think_timeout_s if self.think else settings.llm_timeout_s
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        body = r.json()
        message = body["message"]
        self.last_thinking = (message.get("thinking") or "").strip()
        self.last_stats = {k: body.get(k, 0) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                                      "prompt_eval_duration", "eval_count", "eval_duration")}
        return message["content"]

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        raw = self.chat(system, user, schema=schema)
        return parse_json(raw)


def light_llm(llm: Any) -> Any:
    """LLM for a light step: thinking off unless THINK_LIGHT_STEPS=1; fakes without .light() are returned as is."""
    if settings.think_light_steps or not hasattr(llm, "light"):
        return llm
    return llm.light()


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def parse_json(raw: str) -> dict:
    """Parse JSON robustly (strips <think> blocks, ``` fences and leading prose if a model adds them)."""
    raw = THINK_BLOCK.sub("", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fenced:
        return json.loads(fenced.group(1))
    brace = re.search(r"\{.*\}", raw, re.S)
    if brace:
        return json.loads(brace.group(0))
    raise LLMError(f"Model did not return JSON: {raw[:300]}")
