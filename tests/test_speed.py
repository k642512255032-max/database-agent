"""Hermetic tests for the speed work (no Ollama, no MySQL): keep_alive + warm-up, per-call token stats in the trace,
thinking off for the light steps, the router shortcut, and the deferred memory update."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answer_agent import AnswerAgent  # noqa: E402
from agent.briefing import Briefing  # noqa: E402
from agent.config import settings  # noqa: E402
from agent.expert_agent import ExpertAgent  # noqa: E402
from agent.llm import OllamaLLM, light_llm  # noqa: E402
from agent.orchestrator import DataAgent  # noqa: E402
from agent.trace import Step, Trace  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402
from test_expert_flow import FlowLLM, StubDB  # noqa: E402


# ------------------------------------------------------------------ Ollama client
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
        if json.get("messages") == []:            # warm-up request: Ollama only loads the model
            return FakeResponse({"model": json["model"], "message": {"role": "assistant", "content": ""},
                                 "done_reason": "load", "done": True})
        return FakeResponse({"message": {"role": "assistant", "thinking": None, "content": '{"answer": "42"}'},
                             "load_duration": 2_000_000_000, "prompt_eval_count": 40, "prompt_eval_duration": 800_000_000,
                             "eval_count": 10, "eval_duration": 500_000_000, "total_duration": 3_300_000_000})

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def test_chat_sends_keep_alive_and_records_stats(post):
    llm = OllamaLLM("qwen2.5-coder:3b")
    assert llm.chat_json("sys", "user", {"type": "object"}) == {"answer": "42"}
    assert post[0]["json"]["keep_alive"] == settings.llm_keep_alive
    assert llm.last_stats["eval_count"] == 10 and llm.last_stats["prompt_eval_count"] == 40
    step = Step(1, "x", "y").add_thinking(llm)
    assert step.details["llm"] == {"prompt_tokens": 40, "prompt_ms": 800, "gen_tokens": 10, "gen_ms": 500,
                                   "load_ms": 2000, "tok_per_s": 20.0}
    assert "thinking" not in step.details


def test_stats_are_optional(monkeypatch):
    """An old server / error body without counts: no 'llm' detail, no exception."""
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None:
                        FakeResponse({"message": {"role": "assistant", "content": '{"answer": "42"}'}}))
    llm = OllamaLLM("qwen2.5-coder:3b")
    llm.chat_json("sys", "user", {"type": "object"})
    assert "llm" not in Step(1, "x", "y").add_thinking(llm).details


def test_warm_posts_empty_messages(post):
    llm = OllamaLLM("qwen3:4b")
    ok, msg = llm.warm()
    assert ok and "qwen3:4b" in msg
    assert post[-1]["url"].endswith("/api/chat") and post[-1]["json"]["messages"] == []
    assert post[-1]["json"]["keep_alive"] == settings.llm_keep_alive


def test_warm_reports_failure(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: FakeResponse({}, status=500))
    ok, msg = OllamaLLM("qwen3:4b").warm()
    assert not ok and "qwen3:4b" in msg


def test_light_copy_has_thinking_off(monkeypatch):
    llm = OllamaLLM("qwen3:4b")
    assert llm.think is True
    light = llm.light()
    assert light.think is False and light.model == llm.model and light.host == llm.host
    assert llm.think is True                          # the original is untouched
    monkeypatch.setattr(settings, "think_light_steps", False)
    assert light_llm(llm).think is False
    fake = FlowLLM()
    assert light_llm(fake) is fake                    # fakes without .light() are returned as is
    monkeypatch.setattr(settings, "think_light_steps", True)
    assert light_llm(llm) is llm


# ------------------------------------------------------------------ pipeline fakes
class ThinkLLM(FlowLLM):
    """FlowLLM that behaves like a thinking model and tags every call with its think flag."""

    def __init__(self, calls=None, think=True):
        super().__init__()
        self.calls = calls if calls is not None else []
        self.think = think
        self.model = "qwen3:stub"

    def light(self):
        return ThinkLLM(self.calls, think=False)

    def chat_json(self, system, user, schema):
        if system.startswith("You are a data-visualisation planner"):
            self.calls.append((system, user, self.think))
            return {"charts": [{"type": "bar", "x": "dept_name", "y": "avg_salary", "color": "", "aggregate": "none",
                                "title": "t", "why": "w"}]}
        if system.startswith("You maintain a compact memory"):
            self.calls.append((system, user, self.think))
            return {"reasoning": "", "summary": "salary by department", "entities": {}, "filters": [], "metrics": [],
                    "preferences": [], "findings": ["Development pays most"]}
        out = super().chat_json(system, user, schema)
        self.calls[-1] = (*self.calls[-1], self.think)
        return out


def _agent(llm, expert=True, charts=True, **switches) -> DataAgent:
    agent = DataAgent(db=StubDB(), llm=llm, registry=ModelRegistry(), answer_agent=AnswerAgent(llm),
                      expert_agent=ExpertAgent(llm), charts=charts, summarise_data=True, expert=expert, **switches)
    agent._briefing = Briefing("employees", "file", None, "BRIEFING: to_date 9999-01-01 means current")
    return agent


def _prompt_prefix(system: str) -> str:
    return system.split("\n", 1)[0][:40]


def test_light_steps_do_not_think(monkeypatch):
    monkeypatch.setattr(settings, "think_light_steps", False)
    monkeypatch.setattr(settings, "defer_memory_update", False)
    llm = ThinkLLM()
    res = _agent(llm).ask("average salary per department")
    assert res.error is None and res.charts and res.context.turns == 1
    think_by_prompt = {_prompt_prefix(s): t for s, _u, t in llm.calls}
    assert think_by_prompt["You are a data-visualisation planner. Gi"] is False        # charts
    assert think_by_prompt["You maintain a compact memory of a conve"] is False        # memory
    assert think_by_prompt["You summarise the result of a database q"] is False        # brief data-query summary
    expert = [t for s, _u, t in llm.calls if s.startswith("You are the Expert AI")]
    assert expert == [True, True]                                                      # plan + assessment still think
    steps = {s.name: s for s in res.trace.steps}
    assert steps["Plan charts (answer agent)"].details["thinking"] == "off"
    assert steps["Expert assessment (expert agent)"].details.get("thinking") is None


def test_think_light_steps_setting_restores_thinking(monkeypatch):
    monkeypatch.setattr(settings, "think_light_steps", True)
    monkeypatch.setattr(settings, "defer_memory_update", False)
    llm = ThinkLLM()
    _agent(llm).ask("average salary per department")
    assert all(t is True for _s, _u, t in llm.calls)


# ------------------------------------------------------------------ router shortcut
def test_router_shortcut_skips_llm(monkeypatch):
    monkeypatch.setattr(settings, "router_shortcut", True)
    llm = FlowLLM()                                         # its standardiser answers task = "aggregate"
    res = _agent(llm, expert=False, charts=False).ask("average salary per department", build_context=False)
    assert res.error is None and res.intent == "data_query"
    assert not any(s.startswith("You classify") for s, _u in llm.calls)
    route = res.trace.steps[1]
    assert route.name == "Understand the request" and route.details["router"] == "skipped"
    assert "aggregate" in route.reasoning


def test_router_runs_when_standardiser_is_off(monkeypatch):
    monkeypatch.setattr(settings, "router_shortcut", True)
    llm = FlowLLM()
    res = _agent(llm, expert=False, charts=False, standardise=False).ask("average salary per department",
                                                                        build_context=False)
    assert res.error is None
    assert any(s.startswith("You classify") for s, _u in llm.calls)
    assert "router" not in res.trace.steps[1].details


def test_router_not_skipped_for_analysis_words(monkeypatch):
    monkeypatch.setattr(settings, "router_shortcut", True)
    llm = FlowLLM()
    res = _agent(llm, expert=False, charts=False).ask("correlation of salary per department", build_context=False)
    assert res.error is None
    assert any(s.startswith("You classify") for s, _u in llm.calls)


def test_router_shortcut_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "router_shortcut", False)
    llm = FlowLLM()
    _agent(llm, expert=False, charts=False).ask("average salary per department", build_context=False)
    assert any(s.startswith("You classify") for s, _u in llm.calls)


# ------------------------------------------------------------------ trace timing / threads
def test_trace_timing_sums_llm_steps():
    t = Trace("q")
    with t.step("a", "w") as s:
        s.details["llm"] = {"prompt_tokens": 100, "prompt_ms": 10, "gen_tokens": 5, "gen_ms": 20, "load_ms": 0, "tok_per_s": 1}
    with t.step("b", "w"):
        pass
    with t.step("c", "w") as s:
        s.details["llm"] = {"prompt_tokens": 50, "prompt_ms": 5, "gen_tokens": 5, "gen_ms": 5, "load_ms": 30, "tok_per_s": 1}
    tm = t.timing()
    assert tm["llm_calls"] == 2 and tm["llm_ms"] == 70 and tm["prompt_tokens"] == 150 and tm["gen_tokens"] == 10
    assert tm["total_ms"] >= 0
    assert Trace("empty").timing() == {"total_ms": 0, "llm_ms": 0, "prompt_tokens": 0, "gen_tokens": 0, "llm_calls": 0}


def test_steps_from_other_threads_are_not_pushed_live():
    live = []
    t = Trace("q", on_step=lambda s: live.append(s.name))
    with t.step("main", "w"):
        pass

    def work():
        with t.step("worker", "w"):
            pass

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert [s.name for s in t.steps] == ["main", "worker"] and live == ["main"]


# ------------------------------------------------------------------ deferred memory update
def test_deferred_memory_update(monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", True)
    monkeypatch.setattr(settings, "think_light_steps", False)
    llm = ThinkLLM()
    live = []
    agent = _agent(llm, expert=False, charts=False)
    res = agent.ask("average salary per department", on_step=lambda s: live.append(s.name), build_context=True)
    assert res.error is None and res.answer.startswith("final answer")
    assert "Update conversation context" not in live                 # worker steps are not pushed to the UI
    ctx = res.wait_context()
    assert ctx is res.context and ctx.turns == 1 and ctx.summary == "salary by department"
    assert not res.context_pending() and res.wait_context() is ctx  # second wait is a no-op
    assert "Update conversation context" in [s.name for s in res.trace.steps]
    # the next turn reads the updated memory
    res2 = agent.ask("and for Sales only?", history=[res], build_context=False)
    std = next(u for s, u, *_ in llm.calls if s.startswith("You standardise") and "Sales" in u)
    assert "So far: salary by department" in std
    assert res2.error is None


def test_deferred_memory_survives_llm_failure(monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", True)
    llm = FlowLLM()                                         # raises on the memory prompt (unexpected prompt)
    res = _agent(llm, expert=False, charts=False).ask("average salary per department", build_context=True)
    assert res.error is None
    ctx = res.wait_context()
    assert ctx.turns == 1 and ctx.findings                  # deterministic fallback inside the worker


def test_memory_update_is_synchronous_when_not_deferred(monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", False)
    llm = ThinkLLM()
    live = []
    res = _agent(llm, expert=False, charts=False).ask("average salary per department",
                                                      on_step=lambda s: live.append(s.name), build_context=True)
    assert "Update conversation context" in live and res.context.turns == 1 and not res.context_pending()


def test_no_memory_update_when_build_context_is_off(monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", True)
    res = _agent(FlowLLM(), expert=False, charts=False).ask("average salary per department", build_context=False)
    assert res._context_future is None and not res.context_pending() and res.context.turns == 0
