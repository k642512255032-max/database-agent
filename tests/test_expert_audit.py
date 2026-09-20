"""Hermetic tests for the table-audit runner (run_audit) and Database.run_bounded, with a stub DB and stub LLM."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.db import Database, TableInfo  # noqa: E402
from agent.expert_agent import ExpertAgent, run_audit  # noqa: E402

COLS = [{"name": "emp_no", "type": "INT", "nullable": False}, {"name": "salary", "type": "INT", "nullable": False},
        {"name": "note", "type": "VARCHAR(50)", "nullable": True}, {"name": "to_date", "type": "DATE", "nullable": False}]
FK = {"columns": ["emp_no"], "ref_table": "employees", "ref_columns": ["emp_no"]}

AUDIT = {"summary": "ok", "quality_score": 3, "issues": [], "insights": [], "recommendations": [], "questions_for_owner": []}


class StubLLM:
    def __init__(self):
        self.calls, self.model = [], "stub"

    def chat_json(self, system, user, schema):
        self.calls.append(user)
        assert system.startswith("You audit")
        return AUDIT


class StubDB:
    """Answers the audit SQL by its shape; records every statement in order."""

    def __init__(self, fail_profile_chunk: int | None = None):
        self.sql: list[str] = []
        self.fail_chunk = fail_profile_chunk

    def quote(self, name):
        return f"`{name}`"

    def run_bounded(self, sql, timeout_ms):
        self.sql.append(sql)
        assert sql.startswith("SELECT") and timeout_ms > 0
        if "COUNT(*) AS `rows`" in sql:
            chunk = sum(1 for s in self.sql if "COUNT(*) AS `rows`" in s)
            if chunk == self.fail_chunk:
                raise TimeoutError("MAX_EXECUTION_TIME exceeded")
            return pd.DataFrame([{"rows": 100, "emp_no__nonnull": 100, "emp_no__distinct": 100, "emp_no__min": 1,
                                  "emp_no__max": 100, "emp_no__negatives": 0, "emp_no__avg": 50.5,
                                  "salary__nonnull": 90, "salary__distinct": 40, "salary__min": -5, "salary__max": 9000,
                                  "salary__negatives": 2, "salary__avg": 4000.0,
                                  "note__nonnull": 10, "note__distinct": 9, "note__min": "a", "note__max": "z", "note__blank": 3,
                                  "to_date__nonnull": 100, "to_date__distinct": 5, "to_date__min": "2000-01-01",
                                  "to_date__max": "9999-01-01", "to_date__future": 60}])
        if "AS `orphans`" in sql:
            return pd.DataFrame([{"orphans": 4}])
        if "AS `duplicates`" in sql:
            return pd.DataFrame([{"duplicates": 2}])
        if sql.startswith("SELECT * FROM"):
            return pd.DataFrame({"emp_no": range(30), "salary": list(range(100, 129)) + [100000], "note": ["Ok", "ok "] * 15,
                                 "to_date": ["2020-01-01"] * 30})
        raise AssertionError(sql)


def _info(pk=None, comment=None) -> TableInfo:
    return TableInfo("salaries", COLS, pk or [], [FK], row_count=100, comment=comment)


def test_run_audit_end_to_end():
    db, llm = StubDB(), StubLLM()
    expert = ExpertAgent(llm)
    res = run_audit(db, _info(), expert)
    assert res.error is None and res.report == {**AUDIT, "table": "salaries", "model": "stub", "persona": expert.persona}
    kinds = ["profile" if "AS `rows`" in s else "orphans" if "orphans" in s else "dups" if "duplicates" in s else "sample"
             for s in db.sql]
    assert kinds == ["profile", "orphans", "dups", "sample"]
    names = [s.name for s in res.trace.steps]
    assert names == ["Profile columns (SQL)", "Check foreign keys", "Sample checks", "Expert report (expert agent)"]
    issues = {(f.column, f.issue): f.severity for f in res.findings}
    assert issues[("emp_no -> employees", "orphan foreign-key values")] == "high"
    assert issues[("(table)", "duplicate rows (no primary key)")] == "high"
    assert issues[("salary", "10.0% missing")] == "medium" and issues[("salary", "negative values")] == "medium"
    assert issues[("note", "90.0% missing")] == "high" and issues[("note", "blank strings")] == "low"
    assert issues[("to_date", "open-ended sentinel date")] == "low"
    assert issues[("salary", "outliers (sample)")] == "low" and issues[("note", "inconsistent spelling / casing (sample)")] == "medium"
    assert res.sample_rows == 30 and res.profile["rows"] == 100
    sheet = llm.calls[0]
    assert "Table salaries (100 rows)" in sheet and "Sample checks (30 rows)" in sheet and "[HIGH] emp_no -> employees" in sheet


def test_run_audit_skips_duplicates_with_pk_or_view():
    db = StubDB()
    run_audit(db, _info(pk=["emp_no"]), ExpertAgent(StubLLM()))
    assert not any("duplicates" in s for s in db.sql)
    db = StubDB()
    run_audit(db, _info(comment="VIEW"), ExpertAgent(StubLLM()))
    assert not any("duplicates" in s for s in db.sql)


def test_run_audit_survives_failed_profile_chunk():
    wide = TableInfo("w", [{"name": f"c{i}", "type": "INT", "nullable": True} for i in range(20)], ["c0"], [], row_count=5)
    db = StubDB(fail_profile_chunk=2)
    llm = StubLLM()
    res = run_audit(db, wide, ExpertAgent(llm))
    assert res.error is None and res.report is not None
    step = res.trace.steps[0]
    assert step.status == "warning" and "group 2: TimeoutError" in step.details["note"]
    assert res.profile["columns"]["c15"] == {"error": "query timed out or failed"}
    assert "- c15 (INT, null ok): not profiled: query timed out or failed" in llm.calls[0]


def test_run_audit_without_expert_report():
    class Failing(StubLLM):
        def chat_json(self, *a):
            raise RuntimeError("no model")
    res = run_audit(StubDB(), _info(), ExpertAgent(Failing()))
    assert res.report is None and res.error is None and res.findings
    assert res.trace.steps[-1].status == "warning"


# --------------------------------------------------------------- run_bounded


def test_run_bounded_adds_hint_for_mysql_only(monkeypatch):
    db = Database.__new__(Database)
    seen = []
    monkeypatch.setattr(Database, "run", lambda self, sql: seen.append(sql) or pd.DataFrame())
    db.dialect = "mysql"
    db.run_bounded("SELECT COUNT(*) FROM t", 1500)
    assert seen[-1] == "SELECT /*+ MAX_EXECUTION_TIME(1500) */ COUNT(*) FROM t"
    db.dialect = "sqlite"
    db.run_bounded("SELECT 1", 10)
    assert seen[-1] == "SELECT 1"
    with pytest.raises(ValueError):
        db.run_bounded("DELETE FROM t", 10)
