"""Hermetic tests for agent/llm.py: thinking-model support and robust JSON parsing (requests is monkeypatched)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import llm as llm_mod  # noqa: E402
from agent.llm import LLMError, OllamaLLM, is_thinking_model, parse_json  # noqa: E402
from agent.trace import Trace  # noqa: E402


class FakeResponse:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._body


@pytest.fixture
def post(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        thinking = "Let me reason..." if json.get("think") else None
        return FakeResponse({"message": {"role": "assistant", "thinking": thinking, "content": '{"answer": "42"}'}})

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def test_thinking_models_are_detected():
    assert is_thinking_model("qwen3:8b") and is_thinking_model("deepseek-r1:14b") and is_thinking_model("gpt-oss:20b")
    assert not is_thinking_model("qwen2.5:7b-instruct") and not is_thinking_model("qwen2.5-coder:7b")
    assert OllamaLLM("qwen3:8b").think is True and OllamaLLM("qwen2.5:7b-instruct").think is False
    assert OllamaLLM("qwen3:8b", think=False).think is False


def test_thinking_model_sends_think_and_keeps_reasoning(post):
    llm = OllamaLLM("qwen3:8b")
    out = llm.chat_json("sys", "user", {"type": "object"})
    assert out == {"answer": "42"}
    assert post[0]["json"]["think"] is True and post[0]["json"]["format"] == {"type": "object"}
    assert post[0]["timeout"] == llm_mod.settings.llm_think_timeout_s
    assert llm.last_thinking == "Let me reason..."


def test_plain_model_sends_no_think(post):
    llm = OllamaLLM("qwen2.5-coder:7b")
    llm.chat_json("sys", "user", {"type": "object"})
    assert "think" not in post[0]["json"] and post[0]["timeout"] == llm_mod.settings.llm_timeout_s
    assert llm.last_thinking == ""


def test_step_records_thinking(post):
    llm = OllamaLLM("qwen3:8b")
    llm.chat_json("sys", "user", {"type": "object"})
    trace = Trace("q")
    with trace.step("x", "why") as s:
        s.add_thinking(llm)
    assert trace.steps[0].details["thinking"] == "Let me reason..."
    with trace.step("y", "why") as s:
        s.add_thinking(OllamaLLM("qwen2.5-coder:7b"))
    assert "thinking" not in trace.steps[1].details


def test_parse_json_strips_think_blocks_and_fences():
    assert parse_json('<think>\nsome {braces} here\n</think>\n{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure: {"a": 3}') == {"a": 3}
    with pytest.raises(LLMError):
        parse_json("<think>only thoughts</think>")


def test_request_failure_is_llm_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no ollama")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(LLMError, match="Ollama request failed"):
        OllamaLLM("qwen3:8b").chat("s", "u")
