"""Hermetic tests for the expert's planning task (ExpertAgent.plan / validate_plan)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import expert_agent as ea  # noqa: E402
from agent.db import TableInfo  # noqa: E402
from agent.expert_agent import DataPlan, ExpertAgent, validate_plan  # noqa: E402
from agent.orchestrator import AgentResult  # noqa: E402
from agent.request_agent import StandardRequest  # noqa: E402
from agent.trace import Trace  # noqa: E402

SCHEMA = {
    "employees": TableInfo("employees", [{"name": "emp_no", "type": "INT"}, {"name": "hire_date", "type": "DATE"}], ["emp_no"], []),
    "salaries": TableInfo("salaries", [{"name": "emp_no", "type": "INT"}, {"name": "salary", "type": "INT"},
                                       {"name": "to_date", "type": "DATE"}], ["emp_no", "from_date"], []),
    "departments": TableInfo("departments", [{"name": "dept_no", "type": "CHAR"}, {"name": "dept_name", "type": "VARCHAR"}], ["dept_no"], []),
}

PLAN = {"reasoning": "current rows only", "tables": ["salaries", "Departments", "nope"],
        "columns": ["departments.dept_name", "salary", "salaries.bogus", "x.y"],
        "filters": ["salaries.to_date = '9999-01-01'"], "grouping": ["department"], "metrics": ["average salary"],
        "one_row_per": "one row per department", "sort": "average salary desc", "limit": "5",
        "order_for_sql": "Join salaries to departments via dept_emp; keep current rows only; average per department.",
        "pitfalls": ["history rows multiply counts"]}


class StubLLM:
    def __init__(self, out=PLAN, fail=False):
        self.out, self.fail, self.calls, self.model = out, fail, [], "stub"

    def chat_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        if self.fail:
            raise RuntimeError("down")
        return self.out


def _res() -> AgentResult:
    res = AgentResult("avg salary per dept", Trace("q"), standalone_question="Average current salary per department")
    res.request = StandardRequest(question="q", standalone_question=res.standalone_question, metrics=["average salary"],
                                  grouping=["department"])
    return res


# --------------------------------------------------------------- validate_plan


def test_validate_plan_keeps_known_names_and_reports_dropped():
    plan, notes = validate_plan(PLAN, SCHEMA)
    assert plan.tables == ["salaries", "departments"]
    assert plan.columns == ["departments.dept_name", "salary"]
    assert plan.limit == 5 and plan.one_row_per == "one row per department"
    assert "unknown table 'nope' dropped" in notes and "unknown column 'salaries.bogus' dropped" in notes
    assert "unknown column 'x.y' dropped" in notes


def test_validate_plan_nothing_valid():
    plan, notes = validate_plan({"tables": ["nope"], "columns": ["a"]}, SCHEMA)
    assert plan is None and notes == ["unknown table 'nope' dropped"]
    assert validate_plan({}, SCHEMA) == (None, ["the expert named no table"])


def test_dataplan_text():
    p = DataPlan(tables=["t"], order_for_sql="do it", one_row_per="one row per x", filters=["f"], pitfalls=["p"])
    assert p.text() == "do it\none row per: one row per x\nfilters: f\npitfalls: p"
    assert DataPlan(tables=["t"]).text() == "(no order written)"


# --------------------------------------------------------------- ExpertAgent.plan


def test_plan_prompt_material_and_trace():
    llm = StubLLM()
    trace = Trace("q")
    plan = ExpertAgent(llm, persona="HR analyst", briefing="BRIEFING TEXT").plan(_res(), trace, "SCHEMA TEXT", SCHEMA)
    system, user, schema = llm.calls[0]
    assert system.startswith("You are the Expert AI: HR analyst") and "WHAT DATA is needed" in system
    assert "DATABASE BRIEFING:\nBRIEFING TEXT" in user and "SCHEMA:\nSCHEMA TEXT" in user
    assert "Request: Average current salary per department" in user and "- measures: average salary" in user
    assert "Analysis type: data query." in user
    assert schema is ea.PLAN_SCHEMA
    assert plan.tables == ["salaries", "departments"]
    step = trace.steps[0]
    assert step.name == "Expert data plan (expert agent)" and step.status == "ok"
    assert step.details["briefing_given_to_model"] == "BRIEFING TEXT" and step.details["plan"]["limit"] == 5
    assert "unknown table 'nope' dropped" in step.details["dropped"] and step.reasoning == "current rows only"


def test_plan_ml_note_is_passed():
    llm = StubLLM()
    ExpertAgent(llm).plan(_res(), Trace("q"), "S", SCHEMA, ml_note="must return emp_no, salary")
    assert "must return emp_no, salary" in llm.calls[0][1]


def test_plan_failure_and_no_valid_table_are_warnings():
    trace = Trace("q")
    assert ExpertAgent(StubLLM(fail=True)).plan(_res(), trace, "S", SCHEMA) is None
    assert trace.steps[0].status == "warning" and "falling back to lexical" in trace.steps[0].details["note"]
    trace = Trace("q")
    assert ExpertAgent(StubLLM(out={"tables": ["nope"]})).plan(_res(), trace, "S", SCHEMA) is None
    assert trace.steps[0].status == "warning" and "named no known table" in trace.steps[0].details["note"]
