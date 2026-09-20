"""The agent: natural language -> SQL -> (statistics | ML inference) -> explained answer.

Pipeline (every box is a recorded, visible Step):

  1 Understand request ----> intent: data_query | statistics | machine_learning (+ model)
  2 Select relevant tables    (lexical schema linking, scores shown)
  3 Generate SQL              (LLM, with its reasoning)
  4 Validate SQL              (sqlglot: read-only, known tables/columns, LIMIT)
  5 Execute SQL               (on failure -> LLM repairs using the DB error, up to N retries)
  6a statistics:  choose method -> run test -> interpretation
  6b ML:          check columns -> feature engineering -> inference -> explanations
  7 Write answer              (LLM summary grounded in computed facts)
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from ml.inference import InferenceResult, check_columns, run_inference
from ml.registry import ModelRegistry

from . import prompts, stats_tools
from .answer_agent import AnswerAgent
from .config import settings
from .db import Database, UnsafeSQLError
from .llm import OllamaLLM
from .trace import Step, Trace

ML_WORDS = r"predict|forecast|classif|cluster|segment|group similar|anomal|outlier|unusual|suspicious|fraud|pca|principal component|churn risk|likely to"
FOLLOW_UP_WORDS = (r"\b(he|she|him|his|her|hers|it|its|they|them|their|that|those|this|these|same|what about|how about|"
                   r"and for|instead|also|only|just|more|less|previous|above|first one|second one|last one|the one|"
                   r"same|again|why|sort|order them|top)\b")
MAX_HISTORY_TURNS = 3
STATS_WORDS = r"correlat|significan|t-test|ttest|anova|chi|regression|relationship|distribution|normal|variance|hypothesis|statistic|describe"


@dataclass
class AgentResult:
    question: str
    trace: Trace
    standalone_question: Optional[str] = None   # the question after resolving follow-up references
    intent: str = "data_query"
    sql: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    stats: Optional[dict] = None
    ml: Optional[InferenceResult] = None
    model_name: Optional[str] = None
    answer: str = ""
    charts: list[dict] = field(default_factory=list)   # chart specs planned by the answer agent
    error: Optional[str] = None
    extras: dict = field(default_factory=dict)

    def frame(self) -> Optional[pd.DataFrame]:
        """The table the user sees: query rows, with model outputs prepended for ML."""
        if self.data is None:
            return None
        if self.ml is None:
            return self.data
        return pd.concat([self.ml.output.reset_index(drop=True),
                          self.data.drop(columns=[c for c in self.ml.output.columns if c in self.data.columns])
                          .reset_index(drop=True)], axis=1)


class DataAgent:
    def __init__(self, db: Database | None = None, llm: Any | None = None,
                 registry: ModelRegistry | None = None, answer_agent: AnswerAgent | None = None,
                 charts: bool = True, summarise_data: bool | None = None):
        self.db = db or Database()
        self.llm = llm or OllamaLLM()
        self.registry = registry or ModelRegistry()
        # the answer agent defaults to the same LLM in tests / when no separate model is configured
        self.answer_agent = answer_agent or AnswerAgent(llm if llm is not None and not settings.answer_model else None)
        self.charts = charts
        # plain-English summary for data queries; costs one answer-model call per query
        self.summarise_data = settings.summarise_data_queries if summarise_data is None else summarise_data

    # ================================================================ public
    def ask(self, question: str, on_step: Callable[[Step], None] | None = None,
            force_intent: str | None = None, force_model: str | None = None,
            history: list["AgentResult"] | None = None) -> AgentResult:
        """history = previous AgentResults of this chat (oldest first) for follow-up questions."""
        trace = Trace(question, on_step=on_step)
        res = AgentResult(question, trace)
        res.standalone_question = question
        self._conversation = ""
        try:
            self._resolve_follow_up(res, history or [])
            self._route(res, force_intent, force_model)
            tables = self._link_tables(res)
            self._generate_and_run_sql(res, tables)
            if res.intent == "statistics":
                self._statistics(res)
            elif res.intent == "machine_learning":
                self._machine_learning(res)
            self._answer(res)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.answer = f"I could not complete this request: {exc}"
        return res

    # ================================================== 0. follow-up questions
    _conversation_tables_hint = ""

    def _resolve_follow_up(self, res: AgentResult, history: list[AgentResult]) -> None:
        turns = [h for h in history if h.sql or h.answer][-MAX_HISTORY_TURNS:]
        self._conversation_tables_hint = ""
        if not turns:
            return
        conversation = "\n\n".join(_turn_context(h) for h in turns)
        q = res.question.strip()
        looks_like_follow_up = bool(re.search(FOLLOW_UP_WORDS, q.lower())) or len(q.split()) <= 5
        with res.trace.step("Resolve follow-up question",
                            "Check whether the question refers to earlier answers (e.g. 'he', 'those') and "
                            "rewrite it into a standalone question.") as s:
            s.add(previous_turns_used=len(turns))
            if not looks_like_follow_up:
                s.add(decision="No reference words found - treated as a new, independent question.")
                return
            try:
                out = self.llm.chat_json(prompts.REWRITE_SYSTEM, prompts.rewrite_user(conversation, q),
                                         prompts.REWRITE_SCHEMA)
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Rewrite failed ({exc}); the earlier conversation is still given to the SQL step.")
                self._conversation = conversation
                return
            s.reasoning = out.get("reasoning")
            standalone = (out.get("standalone_question") or "").strip()
            if out.get("is_follow_up") and standalone:
                res.standalone_question = standalone
                self._conversation = conversation
                # keep the tables of the previous query in scope
                last_sql = next((h.sql for h in reversed(turns) if h.sql), "")
                self._conversation_tables_hint = " ".join(re.findall(r"(?:from|join)\s+`?(\w+)", last_sql, re.I))
                s.add(decision="Follow-up question", original_question=q, standalone_question=standalone,
                      context_given_to_model=conversation)
            else:
                s.add(decision="Independent question - no rewrite needed.")

    # ============================================================ 1. router
    def _route(self, res: AgentResult, force_intent: str | None, force_model: str | None) -> None:
        with res.trace.step("Understand the request",
                            "Decide whether this needs plain SQL, a statistical test, or a trained ML model.") as s:
            cards = {c["name"]: c for c in self.registry.cards()}
            s.add(available_models=list(cards) or "none")
            if force_intent:
                intent, model, reasoning = force_intent, force_model or "", "Intent chosen manually in the UI."
            else:
                try:
                    out = self.llm.chat_json(prompts.ROUTER_SYSTEM,
                                             prompts.router_user(res.standalone_question, self.registry.manifest_text()),
                                             prompts.ROUTER_SCHEMA)
                    intent, model, reasoning = out.get("intent"), out.get("model_name", ""), out.get("reasoning")
                except Exception as exc:
                    intent, model, reasoning = None, "", f"LLM router failed ({exc}); used keyword rules."
                # keyword safety net
                q = res.standalone_question.lower()
                if intent not in {"data_query", "statistics", "machine_learning"}:
                    intent = ("machine_learning" if re.search(ML_WORDS, q) and cards
                              else "statistics" if re.search(STATS_WORDS, q) else "data_query")
            s.reasoning = reasoning

            if intent == "machine_learning":
                model = self._resolve_model(res.standalone_question, model, cards)
                if model is None:
                    s.status = "warning"
                    s.add(note="No suitable trained model found - falling back to a plain data query.")
                    intent = "data_query"
            res.intent, res.model_name = intent, (model if intent == "machine_learning" else None)
            s.add(intent=intent, model=res.model_name)

    def _resolve_model(self, question: str, proposed: str, cards: dict) -> str | None:
        if not cards:
            return None
        if proposed in cards:
            return proposed
        close = difflib.get_close_matches(proposed or "", list(cards), n=1, cutoff=0.6)
        if close:
            return close[0]
        q = question.lower()
        task_hint = ("anomaly_detection" if re.search(r"anomal|outlier|unusual|suspicious|fraud", q) else
                     "clustering" if re.search(r"cluster|segment|group similar", q) else
                     "dimensionality_reduction" if re.search(r"pca|principal|dimension", q) else None)
        algo_hint = next((a for a, pat in {"random_forest": "random forest", "decision_tree": "decision tree|tree",
                                           "knn": r"knn|nearest", "kmeans": "k-?means", "dbscan": "dbscan",
                                           "isolation_forest": "isolation", "pca": "pca"}.items()
                          if re.search(pat, q)), None)
        best, best_score = None, 0
        for name, c in cards.items():
            score = 0
            score += 3 if algo_hint and c["algorithm"] == algo_hint else 0
            score += 2 if task_hint and c["task"] == task_hint else 0
            score += 2 if c.get("target") and c["target"].lower() in q else 0
            score += sum(1 for w in re.findall(r"[a-z]+", c["description"].lower()) if len(w) > 3 and w in q)
            if score > best_score:
                best, best_score = name, score
        return best

    # ====================================================== 2. schema linking
    def _link_tables(self, res: AgentResult) -> list[str]:
        with res.trace.step("Select relevant tables",
                            "Only relevant tables go into the prompt, which keeps a small model accurate.") as s:
            tables, scores = self.db.link_tables(res.standalone_question + " " + self._conversation_tables_hint)
            if res.model_name:
                card = self.registry.get(res.model_name).card()
                for t in self.db.schema():
                    if re.search(rf"\b{re.escape(t)}\b", card["base_sql"], re.I) and t not in tables:
                        tables.append(t)
            s.add(selected_tables=tables, relevance_scores={k: v for k, v in scores.items() if v})
            return tables

    # ================================================= 3-5. SQL gen / run
    def _generate_and_run_sql(self, res: AgentResult, tables: list[str]) -> None:
        schema_text = self.db.schema_text(tables)
        dialect = "MySQL" if self.db.dialect == "mysql" else self.db.dialect
        extra = ""
        card = None
        if res.intent == "machine_learning":
            card = self.registry.get(res.model_name).card()
            extra = prompts.ml_sql_extra(card)
        elif res.intent == "statistics":
            extra = prompts.stats_sql_extra()
        if self._conversation:
            extra = prompts.conversation_extra(self._conversation) + "\n" + extra

        with res.trace.step("Generate SQL", "Translate the English request into a SQL query using the schema.") as s:
            out = self.llm.chat_json(prompts.SQL_SYSTEM.format(dialect=dialect),
                                     prompts.sql_user(res.standalone_question, schema_text, extra), prompts.SQL_SCHEMA)
            sql = _clean_sql(out.get("sql", ""))
            s.reasoning = out.get("reasoning")
            s.add(sql=sql, schema_given_to_model=schema_text)

        last_error = None
        for attempt in range(1, settings.max_sql_retries + 2):
            if last_error is not None:
                with res.trace.step(f"Repair SQL (attempt {attempt - 1})",
                                    "The previous query failed; the error message is sent back to the model to fix it.") as s:
                    out = self.llm.chat_json(prompts.FIX_SYSTEM.format(dialect=dialect),
                                             prompts.fix_user(res.standalone_question, schema_text, sql, last_error, extra),
                                             prompts.SQL_SCHEMA)
                    sql = _clean_sql(out.get("sql", ""))
                    s.reasoning = out.get("reasoning")
                    s.add(previous_error=last_error, sql=sql)
            try:
                with res.trace.step("Validate SQL",
                                    "Check the query is a single read-only SELECT on real tables/columns, and cap rows.") as s:
                    limit = settings.analysis_max_rows if res.intent != "data_query" else settings.max_rows
                    safe_sql, notes = self.db.validate_sql(sql, max_rows=limit)
                    s.add(validated_sql=safe_sql, notes=notes or "passed all checks")
                with res.trace.step("Execute SQL", "Run the validated query against the database (read-only session).") as s:
                    df = self.db.run(safe_sql)
                    s.add(rows=len(df), columns=list(df.columns))
                    if card is not None:
                        missing = check_columns(self.registry.get(res.model_name), df)
                        if missing:
                            raise UnsafeSQLError(f"Result is missing columns required by the model: {missing}")
                    if df.empty:
                        s.status = "warning"
                        s.add(note="Query returned 0 rows.")
                res.sql, res.data = safe_sql, df
                return
            except Exception as exc:
                last_error = str(exc).split("\n[SQL:")[0][:800]

        if card is not None:  # deterministic fallback for ML: the model's own training query
            with res.trace.step("Fallback to the model's base query",
                                "The model could not produce valid SQL with the required columns; use the query the "
                                "model was trained on so inference can still run (request filters are NOT applied).") as s:
                s.status = "warning"
                safe_sql, notes = self.db.validate_sql(card["base_sql"], max_rows=settings.analysis_max_rows)
                res.data, res.sql = self.db.run(safe_sql), safe_sql
                s.add(sql=safe_sql, rows=len(res.data))
                return
        raise RuntimeError(f"SQL failed after {settings.max_sql_retries} repairs: {last_error}")

    # ========================================================= 6a. statistics
    def _statistics(self, res: AgentResult) -> None:
        df = stats_tools.numericize(res.data)
        with res.trace.step("Choose statistical method",
                            "Pick the test that matches the question and the column types in the result.") as s:
            profile = "\n".join(f"- {c}: {df[c].dtype}, {df[c].nunique()} distinct" for c in df.columns)
            try:
                out = self.llm.chat_json(prompts.STATS_SYSTEM, prompts.stats_user(res.standalone_question, profile),
                                         prompts.STATS_SCHEMA)
            except Exception as exc:
                out = {"method": "describe", "target": "", "group": "", "columns": [],
                       "reasoning": f"LLM failed ({exc}); defaulting to descriptive statistics."}
            s.reasoning = out.get("reasoning")
            cols = list(df.columns)
            fix = lambda c: _match_column(c, cols)  # noqa: E731
            params = {"target": fix(out.get("target")), "group": fix(out.get("group")),
                      "columns": [x for x in (fix(c) for c in out.get("columns", [])) if x]}
            method = out.get("method", "describe")
            s.add(method=method, description=stats_tools.METHODS.get(method), **params)

        with res.trace.step(f"Run statistical model: {method}",
                            "Compute the statistic with scipy/statsmodels; the interpretation is rule-based, not guessed.") as s:
            try:
                result = stats_tools.run(method, df, **params)
            except Exception as exc:
                s.status = "warning"
                s.add(failed=str(exc), fallback="describe")
                result = stats_tools.run("describe", df)
            s.add(facts=result["facts"], interpretation=result["interpretation"])
            res.stats = result

    # ================================================================ 6b. ML
    def _machine_learning(self, res: AgentResult) -> None:
        bundle = self.registry.get(res.model_name)
        with res.trace.step("Load trained model",
                            "Use the already-trained model (inference only - nothing is retrained).") as s:
            s.add(model=bundle.name, algorithm=bundle.algorithm, task=bundle.task, target=bundle.target,
                  trained_at=bundle.trained_at, training_metrics=bundle.metrics,
                  required_columns=bundle.feature_columns)

        with res.trace.step("Automatic feature engineering",
                            "Replay the exact transformations learned at training time so the model sees the "
                            "same kind of features.") as s:
            fe = bundle.feature_engineer
            plan = fe.plan_table()
            s.add(plan=plan.to_dict("records"), raw_columns=len(bundle.feature_columns),
                  model_features=len(fe.feature_names_out_), feature_names=fe.feature_names_out_)

        with res.trace.step("Run inference", "Score every returned row with the model.") as s:
            ml = run_inference(bundle, res.data)
            s.add(rows_scored=len(ml.output), summary=ml.summary)
            res.ml = ml

        with res.trace.step("Explain predictions",
                            "Show why the model produced its outputs (paths, neighbours, importances, profiles).") as s:
            s.add(explanations=[sec["title"] for sec in ml.sections])

    # ============================================================= 7. answer
    def _answer(self, res: AgentResult) -> None:
        if res.intent == "data_query":
            n = len(res.data) if res.data is not None else 0
            if self.summarise_data and n > 0:
                # the SQL and table are still shown; the answer model adds a short plain-English summary on top
                res.answer = self.answer_agent.compose(res, res.trace, brief=True)
                res.extras["summarised"] = True
            else:
                with res.trace.step("Write the answer", "Return the SQL and the result table directly.") as s:
                    res.answer = f"{n:,} row{'s' if n != 1 else ''} returned."
                    s.add(decision="Data query - the SQL and its result table are shown as-is, without an LLM summary.")
        else:
            res.answer = self.answer_agent.compose(res, res.trace)
        if self.charts:
            res.charts = self.answer_agent.plan_charts(res, res.trace, res.frame())


# ---------------------------------------------------------------- helpers
def _turn_context(r: AgentResult, max_rows: int = 5) -> str:
    lines = [f"Question: {r.standalone_question or r.question}"]
    if r.sql:
        lines.append(f"SQL: {r.sql}")
    if r.data is not None:
        preview = r.data.head(max_rows)
        if r.ml is not None:
            preview = pd.concat([r.ml.output.head(max_rows).reset_index(drop=True),
                                 r.data.drop(columns=[c for c in r.ml.output.columns if c in r.data.columns])
                                 .head(max_rows).reset_index(drop=True)], axis=1)
        lines.append(f"Result: {len(r.data)} rows. First rows:\n{preview.to_string(index=False, max_colwidth=40)}")
    if r.answer:
        lines.append(f"Answer: {r.answer[:400]}")
    return "\n".join(lines)


def _clean_sql(sql: str) -> str:
    sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M).strip()
    return sql


def _match_column(name: str | None, columns: list[str]) -> str | None:
    if not name:
        return None
    if name in columns:
        return name
    low = {c.lower(): c for c in columns}
    if name.lower() in low:
        return low[name.lower()]
    close = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.6)
    return low[close[0]] if close else None
