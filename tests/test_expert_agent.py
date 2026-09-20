"""Hermetic tests for agent/expert_agent.py: task routing, prompts, coercion, failure, orchestrator toggle."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import expert_agent as ea  # noqa: E402
from agent.answer_agent import AnswerAgent  # noqa: E402
from agent.data_quality import Finding  # noqa: E402
from agent.expert_agent import DEFAULT_PERSONA, ExpertAgent, audit_markdown, review_markdown, task_for  # noqa: E402
from agent.orchestrator import AgentResult, DataAgent  # noqa: E402
from agent.trace import Trace  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402

REVIEW = {"verdict": "Usable with care.", "quality_score": 4, "expert_answer": "Dept a pays 60 on average.",
          "data_issues": ["3% missing salary"],
          "insights": ["Engineering pays most"], "advice": ["Break down by hire year"], "confidence": "high"}
AUDIT = {"summary": "Pay history table, healthy.", "quality_score": 4,
         "issues": [{"severity": "LOW", "column": "to_date", "issue": "sentinel", "impact": "averages skew", "fix": "treat as NULL"}],
         "insights": ["salaries cluster 40-80k"], "recommendations": ["add CHECK salary > 0"], "questions_for_owner": ["why 9999?"]}


class StubLLM:
    def __init__(self, out=REVIEW, fail=False, model="stub-expert"):
        self.out, self.fail, self.model, self.calls = out, fail, model, []

    def chat_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        if self.fail:
            raise RuntimeError("ollama down")
        return self.out


def _res(kind="data_query", answer="scripted answer") -> AgentResult:
    df = pd.DataFrame({"dept": ["a", "b", "a"], "salary": [50.0, None, 70.0]})
    res = AgentResult("q", Trace("q"), standalone_question="Average salary by dept", data=df, answer=answer)
    if kind == "statistics":
        res.intent, res.stats = "statistics", {"method": "ttest", "interpretation": "p = 0.3",
                                               "tables": {"groups": pd.DataFrame({"g": ["a"], "mean": [1.0]})}}
    elif kind == "machine_learning":
        res.intent = "machine_learning"
        res.ml = SimpleNamespace(summary={"model": "churn_rf", "algorithm": "RandomForest", "accuracy": 0.91},
                                 sections=[{"title": "Top features", "kind": "text", "content": "tenure, plan"}],
                                 output=pd.DataFrame({"prediction": [1, 0, 1]}))
    return res


# --------------------------------------------------------------- task routing + prompts


def test_task_for():
    assert task_for(_res()) == "data_query"
    assert task_for(_res("statistics")) == "statistics"
    assert task_for(_res("machine_learning")) == "machine_learning"


@pytest.mark.parametrize("kind, marker, material_marker", [
    ("data_query", "plain data query", "First rows:"),
    ("statistics", "effect size versus statistical significance", "Statistical method: ttest"),
    ("machine_learning", "leakage risk", "Model summary:"),
])
def test_review_uses_task_specific_prompt_and_material(kind, marker, material_marker):
    llm = StubLLM()
    out = ExpertAgent(llm, persona="HR analyst").review(_res(kind), Trace("q"))
    system, user, schema = llm.calls[0]
    assert system.startswith("You are the Expert AI: HR analyst")
    assert not system.startswith("You are an expert")            # the SQL prompt prefix must stay unique
    assert marker in system and "Return JSON with:" in system
    assert "QUALITY REPORT:" in user and material_marker in user and "Answer given" not in user   # answer comes after
    assert schema is ea.REVIEW_SCHEMA and out["task"] == kind


def test_review_result_and_trace():
    llm = StubLLM()
    trace = Trace("q")
    out = ExpertAgent(llm).review(_res(), trace)
    assert out["verdict"] == "Usable with care." and out["quality_score"] == 4 and out["confidence"] == "high"
    assert out["model"] == "stub-expert" and out["persona"] == DEFAULT_PERSONA
    step = trace.steps[0]
    assert step.name == "Expert assessment (expert agent)" and step.status == "ok"
    assert step.details["task"] == "data_query" and "33.3% missing" in step.details["quality_report_given_to_model"]
    assert step.details["verdict"] == "Usable with care."


def test_review_coerces_sloppy_output():
    llm = StubLLM(out={"verdict": "  ok  ", "quality_score": "7", "data_issues": "one", "insights": [f"i{i}" for i in range(9)],
                       "advice": None, "confidence": "HIGH!"})
    out = ExpertAgent(llm).review(_res(), Trace("q"))
    assert out["verdict"] == "ok" and out["quality_score"] == 5 and out["data_issues"] == ["one"]
    assert len(out["insights"]) == 5 and out["advice"] == [] and out["confidence"] == "medium"


def test_review_failure_is_warning_not_error():
    trace = Trace("q")
    assert ExpertAgent(StubLLM(fail=True)).review(_res(), trace) is None
    assert trace.steps[0].status == "warning" and "ollama down" in trace.steps[0].details["note"]


def test_persona_default_and_env(monkeypatch):
    assert ExpertAgent(StubLLM(), persona="  ").persona == DEFAULT_PERSONA
    monkeypatch.setattr(ea.settings, "expert_persona", "retail analyst")
    assert ExpertAgent(StubLLM()).persona == "retail analyst"
    assert ExpertAgent(StubLLM(), persona="x").persona == "x"


def test_row_limit_note():
    llm = StubLLM()
    res = _res()
    res.data = pd.DataFrame({"a": range(ea.settings.max_rows)})
    ExpertAgent(llm).review(res, Trace("q"))
    assert "row limit" in llm.calls[0][1]


# --------------------------------------------------------------- audit + markdown


def test_audit_prompt_and_cleaning():
    llm = StubLLM(out=AUDIT)
    trace = Trace("a")
    out = ExpertAgent(llm, "x").audit("SHEET", "salaries", trace)
    system, user, schema = llm.calls[0]
    assert system.startswith("You audit a database table as the Expert AI: x") and user.startswith("AUDIT SHEET for table salaries")
    assert schema is ea.AUDIT_SCHEMA
    assert out["issues"][0]["severity"] == "low" and out["table"] == "salaries"
    assert trace.steps[0].name == "Expert report (expert agent)" and trace.steps[0].details["issues"] == 1
    assert ExpertAgent(StubLLM(fail=True)).audit("SHEET", "t", Trace("a")) is None


def test_markdown_renderers():
    md = review_markdown({**REVIEW, "advice": []})
    assert md.startswith("**Verdict** — Usable with care.") and "**Advice**" not in md and "_confidence: high_" in md
    assert "**Expert answer** — Dept a pays 60 on average." in md
    findings = [Finding("high", "emp_no -> employees", "orphan foreign-key values", "3 rows")]
    rep = audit_markdown({**AUDIT, "issues": [{**AUDIT["issues"][0], "severity": "low"}]}, "salaries", findings, "2026-09-20 10:00")
    assert rep.startswith("# Data-quality audit: salaries") and "| low | to_date | sentinel |" in rep
    assert "## Deterministic findings" in rep and "[HIGH] emp_no -> employees" in rep
    assert "did not produce a report" in audit_markdown(None, "t", [], "now")


# --------------------------------------------------------------- orchestrator toggle


def _agent(expert: bool) -> DataAgent:
    return DataAgent(db=object(), llm=StubLLM(), registry=ModelRegistry(), answer_agent=AnswerAgent(StubLLM()),
                     expert_agent=ExpertAgent(StubLLM()), expert=expert)


def test_dataagent_expert_toggle():
    res = _res()
    _agent(expert=True)._expert(res)
    assert res.expert["task"] == "data_query" and res.trace.steps[-1].name == "Expert assessment (expert agent)"
    res2 = _res()
    _agent(expert=False)._expert(res2)
    assert res2.expert is None and res2.trace.steps == []
    res3 = _res()
    res3.data = None
    _agent(expert=True)._expert(res3)
    assert res3.expert is None


# --------------------------------------------------------------- new flow: plan in the material, answer agent gets the assessment


def test_assess_material_includes_plan_and_sql_and_expert_answer():
    from agent.expert_agent import DataPlan
    llm = StubLLM()
    res = _res()
    res.plan = DataPlan(tables=["employees"], order_for_sql="Join current salaries only.", pitfalls=["history rows"])
    res.sql = "SELECT 1"
    out = ExpertAgent(llm, briefing="BRIEF {x}").assess(res, Trace("q"))
    system, user, _ = llm.calls[0]
    assert "EXPERT'S PLAN" in user and "Join current salaries only." in user and "pitfalls: history rows" in user
    assert "SQL that ran:" in user and "SELECT 1" in user
    assert "You are briefed on the database:" in system and "BRIEF {x}" in system   # braces survive (no str.format)
    assert out["expert_answer"] == "Dept a pays 60 on average."


def test_answer_agent_gets_assessment():
    from agent.answer_agent import fact_sheet
    res = _res()
    assert "EXPERT ASSESSMENT" not in fact_sheet(res)
    res.expert = {**REVIEW, "task": "data_query", "model": "m", "persona": "p"}
    sheet = fact_sheet(res)
    assert "EXPERT ASSESSMENT" in sheet and "Verdict: Usable with care." in sheet
    assert "Expert answer: Dept a pays 60 on average." in sheet and "- 3% missing salary" in sheet
    llm = StubLLM(out={"answer": "final", "key_findings": [], "interpretation": "", "caveats": []})
    AnswerAgent(llm).compose(res, Trace("q"))
    assert "EXPERT ASSESSMENT" in llm.calls[0][1]
