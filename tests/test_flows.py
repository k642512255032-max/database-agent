"""Custom flows and custom agents: the stores, the editor's checks, the flow engine (Standard parity, reordering,
custom steps, agents assigned to built-in steps, explained failures) and a headless run of the Flows page.

    pytest -q tests/test_flows.py        (no Ollama, no database)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import custom_agents, flows  # noqa: E402
from agent.config import settings  # noqa: E402
from agent.custom_agents import AgentStore, CustomAgent, InstructedLLM  # noqa: E402
from agent.flows import STANDARD, Flow, FlowStep, FlowStore, check_flow  # noqa: E402
from agent.knowledge import AGENTS, KnowledgeBase  # noqa: E402
from test_expert_flow import FlowLLM, _agent  # noqa: E402

GLOSSARY = "You add HR glossary notes."
NOTE = "current employee means to_date = 9999-01-01"


@pytest.fixture
def fs(tmp_path, monkeypatch):
    s = FlowStore(tmp_path / "flows.json")
    monkeypatch.setattr(flows, "flow_store", s)
    return s


@pytest.fixture
def ags(tmp_path, monkeypatch):
    s = AgentStore(tmp_path / "custom_agents.json")
    monkeypatch.setattr(custom_agents, "agent_store", s)
    s.save(CustomAgent("hr_glossary", "HR glossary", GLOSSARY))
    return s


class CustomFlowLLM(FlowLLM):
    """FlowLLM that also answers custom agents' own steps (their prompt starts with the agent's instructions)."""

    def __init__(self, fail_custom: bool = False):
        super().__init__()
        self.fail_custom = fail_custom
        self.model = "custom-stub"

    def chat_json(self, system, user, schema):
        if system.startswith(GLOSSARY):
            self.calls.append((system, user))
            if self.fail_custom:
                raise RuntimeError("custom model down")
            return {"reasoning": "r", "text": NOTE}
        if system.startswith("You are a data-visualisation planner"):
            self.calls.append((system, user))
            return {"charts": []}
        if system.startswith("You maintain a compact memory"):
            self.calls.append((system, user))
            return {"reasoning": "", "entities": [], "filters": [], "preferences": [], "facts": [], "tables": []}
        return super().chat_json(system, user, schema)


def _run(flow: Flow | None, ags=None, question="average salary per department", custom=None, **switches):
    agent, llm = _agent(expert=False, **switches)
    agent.agents = ags
    custom = custom or CustomFlowLLM()
    agent.llm_factory = lambda model: custom
    res = agent.ask(question, build_context=False, flow=flow)
    return res, llm, custom, agent


def _flow(*kinds, name="Mine") -> Flow:
    return Flow(name, [FlowStep(f"s{i}", k) if isinstance(k, str) else k for i, k in enumerate(kinds, 1)])


def _names(res) -> list[str]:
    return [s.name for s in res.trace.steps]


# ------------------------------------------------------------------ stores
def test_standard_is_read_only_and_copied(fs):
    copy = fs.get("Standard")
    copy.steps.pop()
    assert len(STANDARD.steps) == 10 and len(fs.get("Standard").steps) == 10
    with pytest.raises(ValueError):
        fs.save(Flow("Standard", []))


def test_store_roundtrip_default_rename_delete(fs):
    fs.save(_flow("standardise", "link_tables", "sql", "answer", name="Fast"))
    assert fs.names() == ["Standard", "Fast"] and [s.kind for s in fs.get("Fast").steps][-1] == "answer"
    fs.set_default("Fast")
    assert fs.default_name() == "Fast"
    fs.rename("Fast", "Faster")
    assert fs.default_name() == "Faster" and "Fast" not in fs.names()
    with pytest.raises(ValueError):
        fs.rename("Faster", "Standard")
    fs.delete("Faster")
    assert fs.names() == ["Standard"] and fs.default_name() == "Standard"
    assert fs.unique_name("Standard") == "Standard 2"


def test_corrupt_files_fall_back(tmp_path, caplog):
    (tmp_path / "flows.json").write_text("{", encoding="utf-8")
    (tmp_path / "agents.json").write_text("[1", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert FlowStore(tmp_path / "flows.json").names() == ["Standard"]
        assert AgentStore(tmp_path / "agents.json").all() == {}
    assert "ignored" in caplog.text


def test_next_step_id_and_new_agent_id():
    flow = _flow("sql", "answer")
    new_ids = {FlowStore.next_step_id(flow) for _ in range(50)}
    assert len(new_ids) == 50 and not new_ids & {"s1", "s2", "s3"}       # never reuses an earlier id like "s3"
    assert AgentStore.new_id("SQL", set()) == "sql_2" and "sql" in AGENTS          # never a built-in key
    assert AgentStore.new_id("HR glossary!", {"hr_glossary"}) == "hr_glossary_2"
    assert AgentStore.new_id("  ", set()) == "agent"


def test_agent_store_validates(ags):
    with pytest.raises(ValueError):
        ags.save(CustomAgent("x", "", "do things"))
    with pytest.raises(ValueError):
        ags.save(CustomAgent("sql", "SQL", "do things"))
    assert ags.get("hr_glossary").name == "HR glossary" and custom_agents.agent_label("hr_glossary") == "HR glossary"
    ags.delete("hr_glossary")
    assert custom_agents.agent_label("hr_glossary") == "hr_glossary"


# ------------------------------------------------------------------ editor checks
def test_check_flow_standard_clean():
    assert check_flow(STANDARD, lambda a: False) == []


def test_check_flow_answer_before_sql():
    issues = check_flow(_flow("answer", "link_tables", "sql"), lambda a: False)
    assert len(issues) == 1 and issues[0].step_id == "s1"
    assert "'Write & run SQL'" in issues[0].text and "Move it after" in issues[0].text


def test_check_flow_missing_sql_duplicates_memory_first():
    texts = [i.text for i in check_flow(_flow("memory", "charts", "charts"), lambda a: False)]
    assert any("never reads the database" in t for t in texts)
    assert any("appears twice" in t for t in texts)
    assert any("always runs after the answer" in t for t in texts)
    assert any("Add that step before it" in t for t in texts)          # charts need data; no SQL step at all


def test_check_flow_custom_step_and_deleted_agent():
    texts = [i.text for i in check_flow(_flow(FlowStep("s1", "custom"), FlowStep("s2", "sql", agent="gone")),
                                        lambda a: False)]
    assert any("Choose the custom agent" in t for t in texts)
    assert any("Choose what this step reads" in t for t in texts)
    assert any("'gone' no longer exists" in t for t in texts)
    assert check_flow(Flow("Empty", []), lambda a: False)[0].text == "The flow has no steps."


# ------------------------------------------------------------------ engine
def test_standard_flow_matches_legacy_trace(tmp_path):
    """The Standard flow runs exactly the steps the fixed pipeline ran (expert off, charts off, no memory)."""
    res_default, _, _, _ = _run(None)
    res_standard, _, _, _ = _run(FlowStore(tmp_path / "none.json").get("Standard"))
    assert _names(res_default) == _names(res_standard) == [
        "Standardise the request", "Understand the request", "Select relevant tables", "Generate SQL",
        "Validate SQL", "Execute SQL", "Summarise the result (answer agent)"]
    assert res_standard.error is None and res_standard.sql == res_default.sql and res_standard.flow == "Standard"


def test_runtime_explains_broken_flow():
    res, _, _, agent = _run(_flow("standardise", "answer", "link_tables", "sql"))
    assert "needs the result table" in res.error and "'Write & run SQL'" in res.error
    assert agent.db.ran == []                                   # stopped before any SQL ran


def test_empty_flow_explains():
    res, *_ = _run(Flow("Empty", []))
    assert "no steps" in res.error


def test_flow_without_standardiser_uses_message_as_typed():
    res, llm, _, _ = _run(_flow("route", "link_tables", "sql", "answer"))
    assert res.error is None
    assert not any(s.startswith("You standardise") for s, _u in llm.calls)
    assert res.request.standalone_question == "average salary per department"


def test_reordered_flow_changes_step_order():
    agent, _llm = _agent(expert=False)
    agent.charts = True                                  # the shared helper builds agents with charts off
    agent.answer_agent.llm = CustomFlowLLM()             # answers the chart planner's prompt
    res = agent.ask("average salary per department", build_context=False,
                    flow=_flow("standardise", "link_tables", "sql", "charts", "answer"))
    names = _names(res)
    assert names.index("Plan charts (answer agent)") < names.index("Summarise the result (answer agent)")


def test_sidebar_switch_still_applies():
    agent, _llm = _agent(expert=False)
    agent.charts = False
    res = agent.ask("average salary per department", build_context=False,
                    flow=_flow("standardise", "link_tables", "sql", "answer", "charts"))
    assert "Plan charts (answer agent)" not in _names(res)


def test_custom_note_reaches_sql_writer_and_answer(ags):
    step = FlowStep("s9", "custom", "hr_glossary", "question", "note")
    res, llm, custom, _ = _run(_flow("standardise", step, "link_tables", "sql", "answer"), ags)
    assert res.error is None
    assert "HR glossary (custom agent)" in _names(res)
    sql_prompt = next(u for s, u in llm.calls if s.startswith("You are an expert"))
    assert "NOTES FROM OTHER AGENTS" in sql_prompt and NOTE in sql_prompt
    answer_prompt = next(u for s, u in llm.calls if s.startswith("You summarise"))
    assert NOTE in answer_prompt
    assert custom.calls[0][1].startswith("Question: average salary")      # it read the question


def test_custom_rewrites_question(ags):
    step = FlowStep("s9", "custom", "hr_glossary", "question", "question")
    res, llm, _, _ = _run(_flow("standardise", step, "link_tables", "sql"), ags)
    assert res.standalone_question == NOTE
    assert any(NOTE in u for s, u in llm.calls if s.startswith("You are an expert"))


def test_custom_answer_section(ags):
    step = FlowStep("s9", "custom", "hr_glossary", "answer", "answer_section")
    res, *_ = _run(_flow("standardise", "link_tables", "sql", "answer", step), ags)
    assert res.answer.endswith(f"#### HR glossary\n{NOTE}")


def test_custom_step_failure_is_a_warning(ags):
    step = FlowStep("s9", "custom", "hr_glossary", "question", "note")
    res, *_ = _run(_flow("standardise", step, "link_tables", "sql", "answer"), ags,
                   custom=CustomFlowLLM(fail_custom=True))
    assert res.error is None and res.data is not None
    s = next(s for s in res.trace.steps if s.name == "HR glossary (custom agent)")
    assert s.status == "warning" and "custom model down" in s.details["note"]


def test_custom_agent_runs_builtin_step(ags):
    res, llm, custom, _ = _run(_flow("standardise", "link_tables", FlowStep("s3", "sql", agent="hr_glossary")), ags)
    assert res.error is None
    sql_calls = [s for s, _u in custom.calls if s.startswith("You are an expert")]
    assert sql_calls and sql_calls[0].rstrip().endswith(GLOSSARY)          # step prompt + the agent's instructions
    assert not any(s.startswith("You are an expert") for s, _u in llm.calls)   # the default SQL model was not used


def test_deleted_agent_falls_back_to_default(ags):
    res, llm, _, _ = _run(_flow("standardise", "link_tables", FlowStep("s3", "sql", agent="gone")), ags)
    assert res.error is None and any(s.startswith("You are an expert") for s, _u in llm.calls)


def test_memory_llm_resolved_before_thread(ags, monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", True)
    agent, _llm = _agent(expert=False)
    agent.agents = ags
    custom = CustomFlowLLM()
    agent.llm_factory = lambda model: custom
    flow = _flow("standardise", "link_tables", "sql", "answer", FlowStep("s5", "memory", agent="hr_glossary"))
    res = agent.ask("average salary per department", flow=flow)
    assert res.wait_context() is not None
    assert any(s.startswith("You maintain a compact memory") and s.rstrip().endswith(GLOSSARY)
               for s, _u in custom.calls)


def test_instructed_llm_keeps_the_wrapper_on_light_steps():
    inner = CustomFlowLLM()
    wrapped = InstructedLLM(inner, "HR glossary", GLOSSARY)
    assert isinstance(wrapped.light(), InstructedLLM) and wrapped.model == "custom-stub"
    assert wrapped._sys("SYSTEM").endswith(GLOSSARY)


def test_custom_agent_documents(tmp_path, ags):
    kb = KnowledgeBase(tmp_path / "knowledge")
    kb.add_document("hr_glossary", "terms.md", b"Current employee: to_date 9999-01-01.")
    assert kb.documents("hr_glossary")
    with pytest.raises(KeyError):
        kb.add_document("nobody", "x.md", b"text")


# ------------------------------------------------------------------ page
def test_flows_page_renders_headless(fs, ags):
    from streamlit.testing.v1 import AppTest
    fs.save(_flow("standardise", FlowStep("s2", "custom", "hr_glossary", "question", "note"), "link_tables", "sql",
                  "answer", name="Fast"))
    fs.set_default("Fast")
    at = AppTest.from_file(str(ROOT / "pages" / "5_Flows.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="fl_name").value == "Fast"
    at.selectbox(key="fl_name").set_value("Standard").run()
    assert not at.exception, [e.value for e in at.exception]


# ------------------------------------------------------------------ review fixes
def test_failed_turn_is_still_remembered(monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", False)
    agent, _llm = _agent(expert=False)
    agent.context_builder.llm = CustomFlowLLM()
    res = agent.ask("average salary per department",
                    flow=_flow("standardise", "answer", "link_tables", "sql", "memory"))   # broken order
    assert res.error and res.context is not None and res.context.turns == 1


def test_new_agent_id_skips_leftover_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "knowledge_dir", tmp_path)
    (tmp_path / "hr_glossary").mkdir()                        # documents of a deleted agent
    assert AgentStore.new_id("HR glossary", set()) == "hr_glossary_2"


def test_renamed_agent_gets_a_new_wrapper(ags):
    agent, _llm = _agent(expert=False)
    agent.agents = ags
    agent.llm_factory = lambda model: CustomFlowLLM()
    step = FlowStep("s1", "sql", agent="hr_glossary")
    assert agent._llm_for(step).name == "HR glossary"
    ags.save(CustomAgent("hr_glossary", "HR terms", GLOSSARY))
    assert agent._llm_for(step).name == "HR terms"


def test_agents_file_not_utf8_means_no_agents(tmp_path, caplog):
    path = tmp_path / "agents.json"
    path.write_bytes('{"x": {"name": "café", "instructions": "i"}}'.encode("cp1252"))
    with caplog.at_level(logging.WARNING):
        assert AgentStore(path).all() == {}
    assert "ignored" in caplog.text
