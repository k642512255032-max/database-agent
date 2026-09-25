"""Fine-tune agents: per-agent document store, BM25 retrieval, the LLM wrapper and its wiring into the pipeline.

    pytest -q tests/test_knowledge.py        (no Ollama, no database)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import settings  # noqa: E402
from agent.knowledge import KnowledgeBase, KnowledgeLLM, chunk, parse_examples  # noqa: E402
from agent.llm import OllamaLLM  # noqa: E402
from agent.trace import Trace  # noqa: E402
from test_speed import ThinkLLM, _agent  # noqa: E402

GLOSSARY = b"""Pay equity

Pay gap between departments means the difference in average current salary per department.
Current salary: salaries.to_date = '9999-01-01'.

Headcount

Headcount is the number of current employees (dept_emp.to_date = '9999-01-01')."""


class Recorder:
    """Minimal fake: records the system prompt of every call."""

    def __init__(self):
        self.systems: list[str] = []
        self.last_thinking, self.last_stats = "", {}

    def chat_json(self, system, user, schema):
        self.systems.append(system)
        return {"examples": [{"input": "pay gap?", "output": "What is the average current salary per department?"}]}


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(tmp_path / "knowledge")


def test_chunk_packs_paragraphs_and_splits_long_ones():
    parts = chunk("a b c\n\nd e f\n\n" + "x. " * 600, size=200)
    assert parts[0] == "a b c\nd e f"
    assert all(len(p) <= 200 for p in parts)


def test_build_and_search_only_the_agents_own_documents(kb):
    kb.add_document("understanding", "hr_glossary.md", GLOSSARY)
    kb.build("understanding")
    hits = kb.search("understanding", "would the pay gap between departments be unfair?")
    assert hits and hits[0].source == "hr_glossary" and "average current salary" in hits[0].text
    assert kb.search("understanding", "zebra giraffe") == []                  # nothing relevant -> nothing added
    assert kb.search("sql", "pay gap between departments") == []             # other agents are untouched
    st = kb.status("understanding")
    assert st["documents"] == 1 and st["passages"] >= 1 and not st["stale"]
    kb.add_document("understanding", "more.txt", b"Attrition means employees who left.")
    assert kb.status("understanding")["stale"]                                 # new upload -> needs re-tuning


def test_jsonl_upload_is_stored_as_examples(kb):
    rows = [{"input": "avg pay by dept", "output": "What is the average salary per department?"},
            {"messages": [{"role": "user", "content": "hc"}, {"role": "assistant", "content": "What is the headcount?"}]}]
    assert kb.add_document("understanding", "ex.jsonl", "\n".join(json.dumps(r) for r in rows).encode()) == "2 examples"
    assert kb.documents("understanding") == []
    assert [e["input"] for e in kb.examples("understanding")] == ["avg pay by dept", "hc"]
    with pytest.raises(ValueError, match="line 1"):
        parse_examples('{"input": "only input"}')


def test_unsupported_and_empty_files_are_rejected(kb):
    with pytest.raises(ValueError, match="Unsupported"):
        kb.add_document("sql", "image.png", b"\x89PNG")
    with pytest.raises(ValueError, match="No text"):
        kb.add_document("sql", "empty.txt", b"   ")


def test_wrapper_passes_through_without_knowledge_and_appends_with_it(kb):
    inner = Recorder()
    llm = KnowledgeLLM(inner, kb, "understanding")
    llm.chat_json("You standardise.", "New message: pay gap between departments", {})
    assert inner.systems[-1] == "You standardise." and llm.last_knowledge == []
    kb.add_document("understanding", "hr_glossary.md", GLOSSARY)
    kb.build("understanding")
    llm.chat_json("You standardise.", "New message: pay gap between departments", {})
    assert inner.systems[-1].startswith("You standardise.")                    # original prompt first, unchanged
    assert "Reference knowledge" in inner.systems[-1] and "(hr_glossary)" in inner.systems[-1]
    trace = Trace("q")
    with trace.step("s", "why") as s:
        s.add_thinking(llm)
    assert trace.steps[0].details["knowledge_used"][0].startswith("hr_glossary")


def test_fine_tuned_model_replaces_the_agents_model(kb):
    base = OllamaLLM("qwen2.5-coder:3b")
    llm = KnowledgeLLM(base, kb, "sql")
    assert llm.model == "qwen2.5-coder:3b"
    kb.set_model("sql", "sql-ft")
    assert llm.model == "sql-ft" and llm._target().host == base.host
    assert llm.light()._target().think is False
    kb.set_model("sql", "")
    assert llm.model == "qwen2.5-coder:3b"
    fake = KnowledgeLLM(Recorder(), kb, "sql")                                 # fakes are never replaced
    kb.set_model("sql", "sql-ft")
    assert isinstance(fake._target(), Recorder)


def test_generate_examples_writes_chat_jsonl(kb):
    kb.add_document("understanding", "hr_glossary.md", GLOSSARY)
    kb.add_document("understanding", "ex.jsonl", b'{"input": "hc", "output": "What is the headcount?"}')
    kb.build("understanding")
    seen = []
    n = kb.generate_examples("understanding", Recorder(), per_passage=1, on_progress=lambda i, t: seen.append((i, t)))
    lines = [json.loads(l) for l in kb.dataset_path("understanding").read_text(encoding="utf-8").splitlines()]
    assert n == len(lines) == 1 + len(seen) and seen[-1][0] == seen[-1][1]
    assert [m["role"] for m in lines[0]["messages"]] == ["system", "user", "assistant"]
    assert lines[0]["messages"][1]["content"] == "hc"                         # hand-written examples come first


def test_pipeline_gives_each_agent_only_its_own_knowledge(kb, monkeypatch):
    monkeypatch.setattr(settings, "defer_memory_update", False)
    kb.add_document("understanding", "hr_glossary.md", GLOSSARY)
    kb.build("understanding")
    kb.add_document("sql", "dictionary.md", b"departments.dept_name is the department name; join via dept_emp.")
    kb.build("sql")
    llm = ThinkLLM()
    agent = _agent(llm, expert=False, knowledge=kb)
    res = agent.ask("average salary per department")
    assert res.error is None
    systems = {s.split("\n", 1)[0][:25]: s for s, *_ in llm.calls}
    std = next(s for k, s in systems.items() if k.startswith("You standardise"))
    sql = next(s for k, s in systems.items() if k.startswith("You are an expert"))
    assert "(hr_glossary)" in std and "(dictionary)" not in std
    assert "(dictionary)" in sql and "(hr_glossary)" not in sql
    steps = {s.name: s for s in res.trace.steps}
    assert steps["Standardise the request"].details["knowledge_used"]
