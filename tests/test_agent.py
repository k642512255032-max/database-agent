"""End-to-end tests with a scripted fake LLM (no Ollama needed) against the real MySQL demo DB.

    pytest -q tests/            (requires the seeded `shop` database and trained models)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.db import Database, UnsafeSQLError  # noqa: E402
from agent.orchestrator import DataAgent  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402


class ScriptedLLM:
    """Returns pre-written responses so the pipeline can be tested deterministically."""

    def __init__(self, route: dict, sqls: list[str], stats: dict | None = None, rewrite: dict | None = None):
        self.route, self.sqls, self.stats, self.rewrite = route, list(sqls), stats, rewrite
        self.calls: list[str] = []
        self.prompts: list[str] = []

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        self.prompts.append(user)
        if system.startswith("You turn a follow-up"):
            self.calls.append("rewrite")
            return self.rewrite
        if system.startswith("You classify"):
            self.calls.append("route")
            return self.route
        if system.startswith("You are an expert") or system.startswith("You fix"):
            self.calls.append("sql" if system.startswith("You are") else "fix")
            return {"reasoning": "scripted", "sql": self.sqls.pop(0)}
        if system.startswith("You choose"):
            self.calls.append("stats")
            return self.stats
        if system.startswith("You are a senior data analyst"):      # answer agent
            self.calls.append("answer")
            return {"answer": "scripted answer", "key_findings": ["f1"], "interpretation": "i", "caveats": []}
        if system.startswith("You are a data-visualisation planner"):  # chart planner -> heuristic fallback
            self.calls.append("charts")
            return {"charts": []}
        raise AssertionError(system[:40])

    def chat(self, system: str, user: str, schema=None) -> str:
        self.calls.append("answer")
        return "scripted answer"

    def health(self):
        return True, "fake"


@pytest.fixture(scope="module")
def db():
    return Database(os.getenv("DATABASE_URL", "mysql+pymysql://agent_ro:agent_ro_pw@127.0.0.1:3306/shop"))


def agent(db, llm):
    return DataAgent(db=db, llm=llm, registry=ModelRegistry())


# ------------------------------------------------------------------ guard
@pytest.mark.parametrize("sql", [
    "DELETE FROM customers",
    "SELECT 1; DROP TABLE customers",
    "UPDATE customers SET plan='x'",
    "SELECT * FROM secrets",
    "SELECT nonexistent_col FROM customers",
    "SELECT * FROM customers INTO OUTFILE '/tmp/x'",
])
def test_guard_blocks(db, sql):
    with pytest.raises(UnsafeSQLError):
        db.validate_sql(sql)


def test_guard_adds_limit(db):
    sql, notes = db.validate_sql("SELECT city, COUNT(*) AS n FROM customers GROUP BY city")
    assert "LIMIT" in sql.upper() and notes


def test_read_only_session(db):
    from sqlalchemy import text
    with pytest.raises(Exception):
        with db.engine.begin() as c:
            c.execute(text("CREATE TEMPORARY TABLE t (x INT)"))
            c.execute(text("INSERT INTO customers (customer_id, full_name) VALUES (99999, 'x')"))


def test_schema_probe_failures_are_logged(caplog, monkeypatch):
    import logging
    from sqlalchemy import text
    d = Database("sqlite://")
    with d.engine.begin() as c:
        c.execute(text("CREATE TABLE t (id INTEGER PRIMARY KEY, d DATE)"))
    # make every probe hit a non-existent table so they all raise
    monkeypatch.setattr(d, "quote", lambda name: '"no_such_table"')
    with caplog.at_level(logging.WARNING, logger="agent.db"):
        schema = d.schema(refresh=True)
    assert "t" in schema and schema["t"].row_count is None      # still degrades gracefully
    msgs = [r.getMessage() for r in caplog.records if r.name == "agent.db"]
    assert any("row-count probe failed for t:" in m for m in msgs), msgs
    assert any("sample" in m for m in msgs), msgs
    assert any("sentinel" in m or "date" in m for m in msgs), msgs


# ------------------------------------------------------------- pipelines
def test_data_query_with_repair(db):
    llm = ScriptedLLM(
        {"reasoning": "list", "intent": "data_query", "model_name": ""},
        ["SELECT c.city, SUM(o.amount) AS revenue FROM orders o JOIN customers c ON c.customer_id=o.customer_id GROUP BY c.city",
         "SELECT c.city, SUM(o.total_amount) AS revenue FROM orders o JOIN customers c ON c.customer_id=o.customer_id "
         "WHERE o.status='completed' GROUP BY c.city ORDER BY revenue DESC LIMIT 5"],
    )
    r = agent(db, llm).ask("Top 5 cities by revenue")
    assert r.error is None, r.trace.as_text()
    assert len(r.data) == 5 and "fix" in llm.calls
    assert any(s.name.startswith("Repair SQL") for s in r.trace.steps)
    print(r.trace.as_text())


def test_statistics_ttest(db):
    llm = ScriptedLLM(
        {"reasoning": "significance", "intent": "statistics", "model_name": ""},
        ["SELECT support_tickets, churned FROM customers"],
        {"reasoning": "2 groups", "method": "ttest", "target": "support_tickets", "group": "churned", "columns": []},
    )
    r = agent(db, llm).ask("Do churned customers open more support tickets?")
    assert r.error is None, r.trace.as_text()
    assert r.stats["method"] == "ttest" and r.stats["facts"]["p"] < 0.05


def test_statistics_regression(db):
    llm = ScriptedLLM(
        {"reasoning": "regression", "intent": "statistics", "model_name": ""},
        ["SELECT total_spent, plan, tenure_months, support_tickets, age FROM customer_features"],
        {"reasoning": "ols", "method": "linear_regression", "target": "total_spent", "group": "",
         "columns": ["plan", "tenure_months", "support_ticket", "age"]},   # typo on purpose -> fuzzy match
    )
    r = agent(db, llm).ask("What explains total spent?")
    assert r.error is None and r.stats["method"] == "linear_regression", r.trace.as_text()
    assert "tenure_months" in r.stats["facts"]["significant_terms"]


@pytest.mark.parametrize("model,where", [
    ("churn_decision_tree", "WHERE plan='premium' AND city='Hanoi'"),
    ("churn_random_forest", "WHERE city='Hanoi'"),
    ("churn_knn", "WHERE age > 50"),
    ("spend_random_forest", "WHERE plan='basic'"),
    ("customer_segments_kmeans", "WHERE city='Da Nang'"),
    ("customer_pca", ""),
    ("customer_density_dbscan", "WHERE tenure_months > 12"),
])
def test_ml_models(db, model, where):
    base = ModelRegistry().get(model).base_sql
    llm = ScriptedLLM({"reasoning": "ml", "intent": "machine_learning", "model_name": model},
                      [f"{base} {where}"])
    r = agent(db, llm).ask(f"run {model}")
    assert r.error is None, r.trace.as_text()
    assert len(r.ml.output) == len(r.data) > 0
    assert r.ml.sections
    print(model, r.ml.summary)


def test_ml_anomaly_router_fallback_and_bad_model_name(db):
    llm = ScriptedLLM({"reasoning": "fraud", "intent": "machine_learning", "model_name": "fraud_detector"},
                      ["SELECT order_id, total_amount FROM order_features",           # missing columns
                       "SELECT * FROM order_features WHERE payment_method='bank_transfer'"])
    r = agent(db, llm).ask("Find suspicious orders paid by bank transfer")
    assert r.error is None, r.trace.as_text()
    assert r.model_name == "order_anomaly_isoforest"
    assert r.ml.summary["anomalies"] > 0
    print(r.trace.as_text())


def test_follow_up_question_uses_previous_answer(db):
    first = agent(db, ScriptedLLM(
        {"reasoning": "lookup", "intent": "data_query", "model_name": ""},
        ["SELECT customer_id, full_name, age FROM customers ORDER BY age DESC, customer_id LIMIT 1"],
    )).ask("Who is the oldest customer?")
    assert first.error is None and len(first.data) == 1
    cid = int(first.data.iloc[0]["customer_id"])

    llm = ScriptedLLM(
        {"reasoning": "lookup", "intent": "data_query", "model_name": ""},
        [f"SELECT full_name, age FROM customers WHERE customer_id = {cid}"],
        rewrite={"reasoning": "he = the oldest customer", "is_follow_up": True,
                 "standalone_question": f"How old is customer {cid}?"},
    )
    second = agent(db, llm).ask("How old is he?", history=[first])
    assert second.error is None, second.trace.as_text()
    assert second.standalone_question == f"How old is customer {cid}?"
    assert second.trace.steps[0].name == "Resolve follow-up question"
    rewrite_prompt = llm.prompts[0]
    assert str(cid) in rewrite_prompt and "Who is the oldest customer?" in rewrite_prompt
    sql_prompt = next(p for p in llm.prompts if "Database schema" in p)
    assert "Earlier in this conversation" in sql_prompt and f"How old is customer {cid}?" in sql_prompt
    assert len(second.data) == 1


def test_independent_question_is_not_rewritten(db):
    first = agent(db, ScriptedLLM({"reasoning": "x", "intent": "data_query", "model_name": ""},
                                  ["SELECT COUNT(*) AS n FROM customers"])).ask("How many customers are there?")
    llm = ScriptedLLM({"reasoning": "x", "intent": "data_query", "model_name": ""},
                      ["SELECT city, COUNT(*) AS n FROM customers GROUP BY city"])
    r = agent(db, llm).ask("Count customers per city in the whole database please", history=[first])
    assert r.error is None and "rewrite" not in llm.calls
    assert r.standalone_question == r.question
