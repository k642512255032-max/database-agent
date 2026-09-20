"""Minimal Ollama client with structured (JSON-schema) output.

Small models such as qwen2.5-coder:1.5b/3b are far more reliable when Ollama
constrains the output with a JSON schema, so every agent call uses `chat_json`.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests

from .config import settings


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or settings.model
        self.host = (host or settings.ollama_host).rstrip("/")

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
        }
        if schema is not None:
            payload["format"] = schema  # Ollama >= 0.5 structured outputs
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=settings.llm_timeout_s)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return r.json()["message"]["content"]

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        raw = self.chat(system, user, schema=schema)
        return parse_json(raw)


def parse_json(raw: str) -> dict:
    """Parse JSON robustly (strips ``` fences / leading prose if a model adds them)."""
    raw = raw.strip()
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
