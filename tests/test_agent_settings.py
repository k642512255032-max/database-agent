"""Agent settings: prompt overrides (store, validation, fallback), the prompt catalog, overrides reaching the
pipeline, the knowledge re-wrap after the chat page swaps agents, and a headless run of the page.

    pytest -q tests/test_agent_settings.py        (no Ollama, no database)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import prompt_store, prompts  # noqa: E402
from agent.agent_catalog import PROMPT_SPECS, SPECS, default_model, specs_for  # noqa: E402
from agent.answer_agent import AnswerAgent  # noqa: E402
from agent.config import settings  # noqa: E402
from agent.expert_agent import REVIEW_FORMAT, TASK_FOCUS, ExpertAgent  # noqa: E402
from agent.knowledge import AGENTS, KnowledgeBase, KnowledgeLLM  # noqa: E402
from agent.llm import OllamaLLM  # noqa: E402
from agent.prompt_store import PromptStore, prompt  # noqa: E402
from test_knowledge import Recorder  # noqa: E402
from test_speed import ThinkLLM, _agent  # noqa: E402


@pytest.fixture
def ps(tmp_path, monkeypatch):
    s = PromptStore(tmp_path / "prompts.json")
    monkeypatch.setattr(prompt_store, "store", s)          # prompt() reads the module-level store
    return s


# ------------------------------------------------------------------ store
def test_get_returns_default_without_file(ps):
    assert ps.get("sql.system", "DEFAULT") == "DEFAULT" and not ps.is_custom("sql.system")
    assert prompt("sql.system", "Write {dialect}", dialect="MySQL") == "Write MySQL"


def test_set_get_reset_roundtrip(ps):
    ps.set("router.system", "You classify. Be strict.", "You classify.")
    assert ps.get("router.system", "You classify.") == "You classify. Be strict." and ps.is_custom("router.system")
    assert prompt("router.system", "You classify.") == "You classify. Be strict."
    ps.reset("router.system")
    assert ps.get("router.system", "You classify.") == "You classify." and "router.system" not in ps.path.read_text()


def test_saving_default_text_or_blank_removes_override(ps):
    ps.set("k", "custom", "default")
    ps.set("k", "  default \n", "default")
    assert not ps.is_custom("k")
    ps.set("k", "custom", "default")
    ps.set("k", "   ", "default")
    assert not ps.is_custom("k")


def test_validate_formatted_prompt():
    ok = "You write {dialect} SQL. Return JSON: {{\"sql\": \"...\"}}"
    assert PromptStore.validate(ok, ("dialect",)) == ""
    assert "Keep the placeholder" in PromptStore.validate("You write SQL.", ("dialect",))
    assert "Unknown placeholder" in PromptStore.validate("{dialect} {foo}", ("dialect",))
    assert "Unbalanced brace" in PromptStore.validate("{dialect} {\"sql\": 1", ("dialect",))
    assert PromptStore.validate("   ", ("dialect",)) == "The prompt is empty."


def test_validate_unformatted_prompt_accepts_braces():
    assert PromptStore.validate('Return JSON: {"charts": [{"type": "bar"}]}', ()) == ""


def test_corrupt_or_odd_file_falls_back(ps):
    ps.path.write_text("{not json", encoding="utf-8")
    assert ps.get("sql.system", "D") == "D"
    ps.path.write_text('["a list"]', encoding="utf-8")
    assert ps.overrides() == {}
    ps.path.write_text('{"a": 3, "b": "  ", "c": "ok"}', encoding="utf-8")
    assert ps.overrides() == {"c": "ok"}


def test_prompt_format_fallback_on_hand_edited_override(ps):
    ps.path.write_text('{"sql.system": "Write {bogus} SQL"}', encoding="utf-8")
    assert prompt("sql.system", "Write {dialect} SQL", dialect="MySQL") == "Write MySQL SQL"


# ------------------------------------------------------------------ catalog
def test_every_agent_has_prompts_and_every_default_validates():
    assert {s.agent for s in PROMPT_SPECS} == set(AGENTS)
    for s in PROMPT_SPECS:
        assert PromptStore.validate(s.default, s.placeholders) == "", s.key
        if s.placeholders:
            s.default.format(**{p: "x" for p in s.placeholders})
    assert [len(specs_for(a)) for a in AGENTS] == [1, 2, 2, 3, 7, 1]


def test_catalog_matches_call_sites():
    key = re.compile(r'prompt\(\s*f?"([a-z_.{}]+)"')
    used = {m for f in (ROOT / "agent").glob("*.py") for m in key.findall(f.read_text(encoding="utf-8"))}
    used = {u for u in used if "{" not in u} | {f"expert.focus.{t}" for t in TASK_FOCUS}
    assert used == set(SPECS)


def test_default_model_per_agent(monkeypatch):
    monkeypatch.setattr(settings, "model", "sql-m")
    monkeypatch.setattr(settings, "answer_model", "ans-m")
    monkeypatch.setattr(settings, "expert_model", "")
    assert [default_model(a) for a in ("sql", "router", "memory", "answer", "expert")] == \
           ["sql-m", "sql-m", "ans-m", "ans-m", "ans-m"]


# ------------------------------------------------------------------ pipeline
def test_overrides_reach_the_llm(ps, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", False)
    # keep the first line: the test fake answers by prompt prefix
    ps.set("understanding.system", prompts.REQUEST_SYSTEM + "\nSTD_MARKER", prompts.REQUEST_SYSTEM)
    ps.set("sql.system", prompts.SQL_SYSTEM + "\nSQL_MARKER for {dialect}", prompts.SQL_SYSTEM)
    llm = ThinkLLM()
    res = _agent(llm, expert=False, knowledge=KnowledgeBase(tmp_path / "knowledge")).ask("average salary per department")
    assert res.error is None
    systems = [c[0] for c in llm.calls]
    std = next(s for s in systems if s.startswith("You standardise"))
    sql = next(s for s in systems if s.startswith("You are an expert"))
    assert "STD_MARKER" in std
    assert "SQL_MARKER for MySQL" in sql and "{dialect}" not in sql


def test_expert_prompt_overrides(ps):
    ps.set("expert.focus.data_query", "FOCUS_MARKER", TASK_FOCUS["data_query"])
    ps.set("expert.review", "You are the Expert AI: {persona}. ROLE_MARKER", "x")
    system = ExpertAgent(Recorder(), persona="an HR analyst").system_prompt("data_query")
    assert system.startswith("You are the Expert AI: an HR analyst. ROLE_MARKER")
    assert "FOCUS_MARKER" in system and system.endswith(REVIEW_FORMAT)
    assert "FOCUS_MARKER" in ExpertAgent(Recorder()).system_prompt("unknown task")   # falls back to data_query


def test_wrap_agents_after_replacement(tmp_path):
    agent = _agent(ThinkLLM(), knowledge=KnowledgeBase(tmp_path / "knowledge"))
    agent.answer_agent = AnswerAgent(Recorder())          # what the chat page's sidebar does on every rerun
    agent.context_builder.llm = agent.answer_agent.llm
    agent.wrap_agents()
    agent.wrap_agents()                                    # idempotent
    for owner, key in ((agent.answer_agent, "answer"), (agent.context_builder, "memory")):
        assert isinstance(owner.llm, KnowledgeLLM) and owner.llm.agent == key
        assert not isinstance(owner.llm.inner, KnowledgeLLM)


def test_statistics_chooser_uses_the_router_settings(tmp_path):
    import pandas as pd
    from agent.orchestrator import AgentResult
    from agent.trace import Trace
    kb = KnowledgeBase(tmp_path / "knowledge")
    kb.add_document("router", "stats_guide.md", b"Salary differences between two groups need a ttest on salary.")
    kb.build("router")
    llm = ThinkLLM()
    agent = _agent(llm, expert=False, knowledge=kb)
    res = AgentResult("is salary different by gender", Trace("q"), standalone_question="is salary different by gender")
    res.data = pd.DataFrame({"gender": ["M", "F"] * 5, "salary": range(10)})
    agent._statistics(res)                                 # the fake does not answer this prompt: describe fallback
    system = next(c[0] for c in llm.calls if c[0].startswith("You choose the right statistical method"))
    assert "(stats_guide)" in system


def test_wrapper_health_checks_the_override_model(tmp_path, monkeypatch):
    kb = KnowledgeBase(tmp_path / "knowledge")
    monkeypatch.setattr(OllamaLLM, "health", lambda self: (True, self.model))
    llm = KnowledgeLLM(OllamaLLM("base-m"), kb, "memory")
    assert llm.health() == (True, "base-m")
    kb.set_model("memory", "memory-ft")
    assert llm.health() == (True, "memory-ft")


# ------------------------------------------------------------------ page
def test_page_renders_and_saves(ps, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "knowledge_dir", tmp_path / "knowledge")
    monkeypatch.setattr(OllamaLLM, "health", lambda self: (True, "stub"))
    at = AppTest.from_file(str(ROOT / "pages" / "4_Agent_settings.py"), default_timeout=60).run()
    assert not at.exception
    at.text_area(key="prompt_understanding.system").input(prompts.REQUEST_SYSTEM + "\nBe brief.").run()
    at.button(key="save_understanding.system").click().run()
    assert not at.exception
    assert ps.get("understanding.system", "").endswith("Be brief.")
    at.button(key="reset_understanding.system").click().run()
    assert not ps.is_custom("understanding.system")
    assert at.text_area(key="prompt_understanding.system").value == prompts.REQUEST_SYSTEM


def test_page_saves_the_default_model_as_an_explicit_choice(ps, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "knowledge_dir", tmp_path / "knowledge")
    monkeypatch.setattr(OllamaLLM, "health", lambda self: (True, "stub"))
    at = AppTest.from_file(str(ROOT / "pages" / "4_Agent_settings.py"), default_timeout=60).run()
    next(b for b in at.button if b.label == "Save model").click().run()
    assert not at.exception
    assert KnowledgeBase(tmp_path / "knowledge").model_for("understanding") == default_model("understanding")
    next(b for b in at.button if b.label == "Use default").click().run()
    assert KnowledgeBase(tmp_path / "knowledge").model_for("understanding") == ""


def test_page_rejects_a_prompt_without_its_placeholder(ps, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(settings, "knowledge_dir", tmp_path / "knowledge")
    monkeypatch.setattr(OllamaLLM, "health", lambda self: (True, "stub"))
    at = AppTest.from_file(str(ROOT / "pages" / "4_Agent_settings.py"), default_timeout=60)
    at.session_state["as_agent"] = "sql"
    at.run()
    at.text_area(key="prompt_sql.system").input("You write SQL.").run()
    at.button(key="save_sql.system").click().run()
    assert not ps.is_custom("sql.system")
    assert any("Keep the placeholder" in e.value for e in at.error)
