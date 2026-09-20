"""Hermetic tests for dashboard/builder.py and dashboard/store.py (stub LLM + stub agent, no MySQL/Ollama)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import settings  # noqa: E402
from agent.db import UnsafeSQLError  # noqa: E402
from agent.orchestrator import AgentResult  # noqa: E402
from agent.trace import Trace  # noqa: E402
from dashboard import store  # noqa: E402
from dashboard.builder import DashboardBuilder  # noqa: E402
from dashboard.spec import DashboardSpec, Widget  # noqa: E402

DESIGN = {
    "reasoning": "scripted", "title": "Sales overview", "subtitle": "demo", "accent": "#123456",
    "widgets": [
        {"title": "Revenue", "kind": "kpi", "question": "Total revenue", "x": "", "y": "", "width": "small", "note": ""},
        {"title": "By city", "kind": "bar", "question": "Revenue per city", "x": "city", "y": "revenue", "width": "medium", "note": ""},
        {"title": "Broken", "kind": "line", "question": "Something impossible", "x": "", "y": "", "width": "full", "note": ""},
    ],
}


class StubLLM:
    def __init__(self, out=DESIGN, fail=False):
        self.out, self.fail, self.calls = out, fail, []
        self.model = "stub"

    def chat_json(self, system, user, schema):
        self.calls.append((system[:20], user))
        if self.fail:
            raise RuntimeError("ollama down")
        assert system.startswith("You design a dashboard")
        return self.out


class StubDB:
    def __init__(self):
        self.ran: list[str] = []

    def link_tables(self, q, k=None):
        return ["orders"], {"orders": 1.0}

    def schema_text(self, tables=None, with_samples=True):
        assert with_samples is False
        return "TABLE orders(city, revenue)"

    def validate_sql(self, sql, max_rows=None):
        if "DROP" in sql:
            raise UnsafeSQLError("nope")
        return sql + f" LIMIT {max_rows}", []

    def run(self, sql):
        self.ran.append(sql)
        return pd.DataFrame({"city": ["a", "b"], "revenue": [1.0, 2.0]})


class StubAgent:
    """Answers questions from a table; 'impossible' questions fail like the real pipeline would."""

    def __init__(self, llm):
        self.db, self.asked = StubDB(), []
        self.answer_agent = type("AA", (), {"llm": llm})()

    def ask(self, question, force_intent=None, build_context=True, **kw):
        assert force_intent == "data_query" and build_context is False
        self.asked.append(question)
        res = AgentResult(question, Trace(question))
        if "impossible" in question:
            res.error = "UnsafeSQLError: Unknown table 'nothing'"
        elif question == "Total revenue":
            res.sql, res.data = "SELECT SUM(revenue) AS total FROM orders", pd.DataFrame({"total": [3.0]})
        else:
            res.sql, res.data = "SELECT city, SUM(revenue) AS revenue FROM orders GROUP BY city", \
                pd.DataFrame({"city": ["a", "b"], "revenue": [1.0, 2.0]})
        return res


@pytest.fixture
def builder():
    llm = StubLLM()
    return DashboardBuilder(StubAgent(llm), llm=llm)


def test_build_end_to_end(builder):
    b = builder.build("sales dashboard")
    assert b.error is None and b.spec.title == "Sales overview"
    assert [r.error is None for r in b.results] == [True, True, False]
    assert b.ok_widgets() == 2 and set(b.sqls()) == {"w1", "w2"}
    names = [s.name for s in b.trace.steps]
    assert names[0] == "Design dashboard" and names[-1] == "Render dashboard files"
    assert names[1:4] == ["Widget 1/3: Revenue", "Widget 2/3: By city", "Widget 3/3: Broken"]
    assert b.trace.steps[3].status == "warning"
    assert set(b.bundle) == {"index.html", "style.css", "chart.umd.js", "data.js", "app.js"}
    assert '"value": 3.0' in b.bundle["data.js"] and "Unknown table" in b.bundle["data.js"]


def test_build_reuses_unchanged_questions(builder):
    first = builder.build("sales dashboard")
    builder.agent.asked.clear()
    second = builder.build("sales dashboard", previous=first, change="make it blue")
    assert second.error is None
    assert [r.reused for r in second.results] == [True, True, False]   # the failed one is retried
    assert builder.agent.asked == ["Something impossible"]
    assert second.trace.steps[0].name == "Refine dashboard"
    assert "Current dashboard spec" in builder.llm.calls[-1][1] and "make it blue" in builder.llm.calls[-1][1]


def test_design_failure_is_fatal_and_fetches_nothing():
    llm = StubLLM(fail=True)
    b = DashboardBuilder(StubAgent(llm), llm=llm).build("x")
    assert b.error.startswith("RuntimeError: ollama down") and b.results == [] and b.bundle == {}
    assert b.trace.steps[0].status == "error"


def test_design_with_no_usable_widgets_is_fatal():
    llm = StubLLM(out={"reasoning": "", "title": "T", "subtitle": "", "accent": "", "widgets": []})
    b = DashboardBuilder(StubAgent(llm), llm=llm).build("x")
    assert b.error.startswith("LLMError: The model proposed no usable widgets")


def test_refresh_reruns_stored_sql_without_llm(builder):
    spec = DashboardSpec("T", widgets=[Widget("w1", "A", "bar", "q"), Widget("w2", "B", "kpi", "q2"),
                                       Widget("w3", "C", "table", "q3")])
    record = {"spec": spec.to_dict(), "sqls": {"w1": "SELECT city, revenue FROM orders", "w2": "DROP TABLE x"},
              "description": "d"}
    b = builder.refresh(record)
    assert builder.llm.calls == [] and builder.agent.asked == []
    assert builder.agent.db.ran == [f"SELECT city, revenue FROM orders LIMIT {settings.dashboard_rows_per_widget}"]
    assert b.results[0].error is None and b.results[1].error.startswith("UnsafeSQLError") \
        and b.results[2].error == "No stored SQL for this widget"
    assert b.bundle and b.error is None


def test_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "dashboards_dir", tmp_path / "dash")
    spec = DashboardSpec("Sales overview", widgets=[Widget("w1", "A", "bar", "q")])
    rec = store.new_record("desc", spec, {"w1": "SELECT 1"})
    assert rec["slug"] == "sales-overview" and rec["site_id"] is None
    rec["published_at"] = "2026-09-20T10:00:00"
    store.save(rec)
    other = store.save({**store.new_record("d2", DashboardSpec("Older"), {}), "published_at": "2026-01-01T00:00:00"})
    assert store.load("sales-overview") == rec
    assert [r["slug"] for r in store.list_all()] == ["sales-overview", "older"]
    store.delete("sales-overview")
    assert store.load("sales-overview") is None and [r["slug"] for r in store.list_all()] == [other["slug"]]
    (tmp_path / "dash" / "bad.json").write_text("{not json", encoding="utf-8")
    assert [r["slug"] for r in store.list_all()] == ["older"]      # corrupt file is skipped, not fatal
