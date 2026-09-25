"""The expert-led pipeline end to end with a stub database and a stub LLM (no MySQL, no Ollama).

Checks the step ORDER (standardise -> route -> expert plan -> tables -> SQL -> run -> expert assessment -> answer),
that the expert's order reaches the SQL prompt, that the assessment reaches the answer agent, and that the
expert-off path is today's pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.answer_agent import AnswerAgent  # noqa: E402
from agent.briefing import Briefing  # noqa: E402
from agent.db import TableInfo  # noqa: E402
from agent.expert_agent import ExpertAgent  # noqa: E402
from agent.orchestrator import DataAgent  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402

SCHEMA = {
    "employees": TableInfo("employees", [{"name": "emp_no", "type": "INT"}, {"name": "gender", "type": "CHAR"}], ["emp_no"], []),
    "salaries": TableInfo("salaries", [{"name": "emp_no", "type": "INT"}, {"name": "salary", "type": "INT"},
                                       {"name": "to_date", "type": "DATE"}], ["emp_no", "from_date"], []),
    "departments": TableInfo("departments", [{"name": "dept_no", "type": "CHAR"}, {"name": "dept_name", "type": "VARCHAR"}], ["dept_no"], []),
}

PLAN = {"reasoning": "briefing says current rows", "tables": ["salaries", "departments"], "columns": ["departments.dept_name", "salary"],
        "filters": ["current rows only"], "grouping": ["department"], "metrics": ["average salary"],
        "one_row_per": "one row per department", "sort": "", "limit": 0,
        "order_for_sql": "Average current salaries per department.", "pitfalls": ["history rows multiply counts"]}
ASSESS = {"verdict": "Trustworthy.", "quality_score": 4, "expert_answer": "Development pays most at 67k.",
          "data_issues": [], "insights": ["gap is small"], "advice": ["compare within titles"], "confidence": "high"}


class FlowLLM:
    """Answers every prompt of the pipeline by its system-prompt prefix; records the prompts."""

    def __init__(self, plan_fails: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.plan_fails = plan_fails
        self.model = "stub"

    def chat_json(self, system, user, schema):
        self.calls.append((system, user))
        if system.startswith("You standardise"):
            q = user.split("New message: ", 1)[1].split("\n", 1)[0]
            return {"reasoning": "", "is_follow_up": False, "standalone_question": q, "task": "aggregate", "entities": [],
                    "filters": [], "metrics": ["average salary"], "grouping": ["department"], "time_range": "", "sort": "",
                    "limit": 0, "ambiguities": []}
        if system.startswith("You classify"):
            return {"intent": "data_query", "model_name": "", "reasoning": "plain query"}
        if system.startswith("You are the Expert AI") and "WHAT DATA is needed" in system:
            if self.plan_fails:
                raise RuntimeError("expert down")
            return PLAN
        if system.startswith("You are the Expert AI"):
            return ASSESS
        if system.startswith("You are an expert"):                       # the SQL writer
            return {"reasoning": "", "sql": "SELECT dept_name, AVG(salary) AS avg_salary FROM salaries GROUP BY dept_name"}
        if system.startswith(("You are a senior data analyst", "You summarise")):
            return {"answer": "final answer", "key_findings": ["kf"], "interpretation": "i", "caveats": []}
        raise AssertionError("unexpected prompt: " + system[:40])

    def health(self):
        return True, "stub"


class StubDB:
    dialect = "mysql"
    url = "mysql+pymysql://u:p@h/employees"

    def __init__(self):
        self.ran: list[str] = []

    def schema(self, refresh=False):
        return SCHEMA

    def link_tables(self, q, k=None):
        return ["employees"], {"employees": 1.0}

    def schema_text(self, tables=None, with_samples=True):
        return "SCHEMA(" + ", ".join(tables or SCHEMA) + ")"

    def validate_sql(self, sql, max_rows=None):
        return sql, []

    def run(self, sql):
        self.ran.append(sql)
        return pd.DataFrame({"dept_name": ["Development", "Sales"], "avg_salary": [67000.0, 60000.0]})


def _agent(expert: bool, plan_fails: bool = False, **switches) -> tuple[DataAgent, FlowLLM]:
    llm = FlowLLM(plan_fails)
    agent = DataAgent(db=StubDB(), llm=llm, registry=ModelRegistry(), answer_agent=AnswerAgent(llm),
                      expert_agent=ExpertAgent(llm), charts=False, summarise_data=True, expert=expert, **switches)
    agent._briefing = Briefing("employees", "file", None, "BRIEFING: to_date 9999-01-01 means current")
    return agent, llm


def _step_names(res):
    return [s.name for s in res.trace.steps]


def test_expert_led_flow_order_and_handoffs():
    agent, llm = _agent(expert=True)
    res = agent.ask("average salary per department", build_context=False)
    assert res.error is None and res.answer.startswith("final answer")
    names = _step_names(res)
    assert names == ["Standardise the request", "Understand the request", "Expert data plan (expert agent)",
                     "Select relevant tables", "Generate SQL", "Validate SQL", "Execute SQL",
                     "Expert assessment (expert agent)", "Summarise the result (answer agent)"]
    # the plan drove table selection
    sel = res.trace.steps[3].details
    assert sel["source"] == "expert plan" and sel["selected_tables"] == ["salaries", "departments"]
    # the expert's order reached the SQL writer, after the briefing reached the expert
    sql_prompt = next(u for s, u in llm.calls if s.startswith("You are an expert"))
    assert "Expert's data order (follow it" in sql_prompt and "Average current salaries per department." in sql_prompt
    assert "- avoid: history rows multiply counts" in sql_prompt and "- tables to use: salaries, departments" in sql_prompt
    plan_user = next(u for s, u in llm.calls if "WHAT DATA is needed" in s)
    assert "DATABASE BRIEFING:\nBRIEFING: to_date 9999-01-01 means current" in plan_user
    assert "SCHEMA(employees, salaries, departments)" in plan_user
    assert agent.expert_agent.briefing.startswith("BRIEFING")
    # the assessment happened before the answer and reached the answer agent
    assess_user = next(u for s, u in llm.calls if s.startswith("You are the Expert AI") and "WHAT DATA" not in s)
    assert "EXPERT'S PLAN" in assess_user and "SQL that ran:" in assess_user and "Answer given" not in assess_user
    answer_user = next(u for s, u in llm.calls if s.startswith("You summarise"))
    assert "EXPERT ASSESSMENT" in answer_user and "Expert answer: Development pays most at 67k." in answer_user
    assert res.expert["expert_answer"] == "Development pays most at 67k." and res.plan.tables == ["salaries", "departments"]


def test_expert_off_is_the_old_pipeline():
    agent, llm = _agent(expert=False)
    res = agent.ask("average salary per department", build_context=False)
    names = _step_names(res)
    assert "Expert data plan (expert agent)" not in names and "Expert assessment (expert agent)" not in names
    sel = res.trace.steps[2].details
    assert sel["source"] == "lexical linking" and sel["selected_tables"] == ["employees"]
    sql_prompt = next(u for s, u in llm.calls if s.startswith("You are an expert"))
    assert "Expert's data order" not in sql_prompt
    answer_user = next(u for s, u in llm.calls if s.startswith("You summarise"))
    assert "EXPERT ASSESSMENT" not in answer_user
    assert res.expert is None and res.plan is None
    assert not any(s.startswith("You are the Expert AI") for s, _ in llm.calls)


def test_plan_failure_falls_back_but_assessment_still_runs():
    agent, llm = _agent(expert=True, plan_fails=True)
    res = agent.ask("average salary per department", build_context=False)
    assert res.error is None and res.plan is None
    plan_step = res.trace.steps[2]
    assert plan_step.name == "Expert data plan (expert agent)" and plan_step.status == "warning"
    sel = res.trace.steps[3].details
    assert sel["source"] == "lexical linking (expert plan unavailable)"
    sql_prompt = next(u for s, u in llm.calls if s.startswith("You are an expert"))
    assert "Expert's data order" not in sql_prompt
    assert res.expert is not None and "Expert assessment (expert agent)" in _step_names(res)


# ------------------------------------------------------------------ per-agent switches
def test_expert_plan_and_assessment_switch_separately():
    # plan on, assessment off
    agent, llm = _agent(expert=True, expert_assess=False)
    res = agent.ask("average salary per department", build_context=False)
    names = _step_names(res)
    assert "Expert data plan (expert agent)" in names and "Expert assessment (expert agent)" not in names
    assert res.plan is not None and res.expert is None
    assert agent.expert is True                    # either step on => the expert counts as on
    # plan off, assessment on
    agent, llm = _agent(expert=False, expert_assess=True)
    res = agent.ask("average salary per department", build_context=False)
    names = _step_names(res)
    assert "Expert data plan (expert agent)" not in names and "Expert assessment (expert agent)" in names
    assert res.plan is None and res.expert is not None
    assert res.trace.steps[2].details["source"] == "lexical linking"     # no "(expert plan unavailable)" when off


def test_expert_setter_flips_both_steps():
    agent, _ = _agent(expert=True)
    agent.expert = False
    assert agent.expert_plan is False and agent.expert_assess is False and agent.expert is False
    agent.expert = True
    assert agent.expert_plan and agent.expert_assess


def test_standardiser_off_uses_message_as_typed_without_llm_call():
    agent, llm = _agent(expert=False, standardise=False)
    res = agent.ask("  average   salary per department ", build_context=False)
    assert res.error is None
    assert "Standardise the request" not in _step_names(res)
    assert not any(s.startswith("You standardise") for s, _ in llm.calls)
    assert res.standalone_question == "average salary per department"      # whitespace collapsed, otherwise as typed
    assert res.request is not None and res.request.question == "average salary per department"
