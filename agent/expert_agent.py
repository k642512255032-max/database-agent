"""Expert AI agent: a specialist that works alongside the answer agent.

The answer agent explains what the numbers say. The expert agent judges whether the data and the
analysis can be trusted, adds the insights a domain expert would notice, and gives concrete advice.
It has one task-specific prompt per kind of result, plus a planning task that runs BEFORE the SQL:

  * plan              (briefed) decide which tables / columns / filters answer the request; write the order for SQL
  * data_query        completeness, suspicious values, whether the rows really answer the question
  * statistics        test fit, sample size, effect size vs significance, assumptions
  * machine_learning  metrics vs trust, class balance, leakage, plausibility of explanations
  * audit             a whole table (from the Expert audit page): structure, per-column stats, findings

Every number the expert may cite comes from agent/data_quality.py (pandas / bounded SQL); the model
only judges, prioritises, explains and advises, in the persona the user set ("Domain & goals").
Failures are recorded as a warning step and return None - an expert failure never breaks an answer.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from agent import data_quality as dq
from agent.answer_agent import _table_text
from agent.config import settings
from agent.llm import OllamaLLM
from agent.trace import Step, Trace

if TYPE_CHECKING:
    from agent.db import Database, TableInfo
    from agent.orchestrator import AgentResult

log = logging.getLogger(__name__)

DEFAULT_PERSONA = "a senior data analyst and data-quality engineer advising a business team"
TASKS = ("data_query", "statistics", "machine_learning", "audit")
MAX_LIST = 5
CONFIDENCE = ("low", "medium", "high")

# ------------------------------------------------------------------ prompts (prefixes must stay unique, see tests)
_COMMON = """You are the Expert AI: {persona}.
You work alongside an answer-writing assistant that will write the final reply AFTER you. Your job: judge whether
the data and the analysis can be trusted, give your own expert answer, add the insights an expert would notice, and
give concrete advice. You get the question, the data plan you ordered, a QUALITY REPORT computed by tools, and material
about the analysis. Never invent numbers; cite only what is in the material. Be specific and practical; no generic
platitudes."""

TASK_FOCUS = {
    "data_query": """This was a plain data query (a SQL result table). Focus on: completeness (missing values, blanks,
duplicates), suspicious values (negatives, outliers, future dates, sentinel dates), whether the rows answer the question
or hide detail (aggregation, filters, a row limit that was reached), and which breakdown or filter would make the result
decision-ready.""",
    "statistics": """This was a statistical analysis (method, tables and p-values are in the material). Focus on: whether the
test fits the question and the data types, sample size per group, effect size versus statistical significance,
assumptions (normality, equal variance, independence, aggregated rows instead of one row per observation), multiple
comparisons, and what a careful analyst would run next.""",
    "machine_learning": """This was a machine-learning inference (model summary, metrics and explanations are in the material).
Focus on: whether the training metrics justify trusting these predictions, class balance and base rates, leakage risk
from the features listed, how many rows were scored and how many predictions look uncertain, whether the explanations
are plausible for the domain, and how to validate before acting on them.""",
}

REVIEW_FORMAT = """Return JSON with:
- verdict: one sentence - can this be trusted and used as is?
- quality_score: integer 1 (unusable) .. 5 (clean and sufficient).
- expert_answer: 2-4 sentences: your own answer to the question from this data, with the key numbers, or why it
  cannot be answered from this data.
- data_issues: 0-5 bullets, each naming one concrete problem from the quality report or the material; empty if none.
- insights: 2-4 bullets a domain expert would notice (patterns, surprises, what is missing to conclude).
- advice: 2-4 bullets of concrete next actions (what to check, which breakdown or filter to add, what to decide).
- confidence: low | medium | high."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "quality_score": {"type": "integer"},
        "expert_answer": {"type": "string"},
        "data_issues": {"type": "array", "items": {"type": "string"}},
        "insights": {"type": "array", "items": {"type": "string"}},
        "advice": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
    },
    "required": ["verdict", "quality_score", "expert_answer", "data_issues", "insights", "advice", "confidence"],
}

AUDIT_SYSTEM = """You audit a database table as the Expert AI: {persona}.
You get an AUDIT SHEET computed by tools: the table's structure, per-column statistics, deterministic findings with a
severity, and checks on a sample of rows. Return JSON with:
- summary: 2-3 sentences: what the table holds, how healthy it is, the single biggest concern.
- quality_score: integer 1 (unusable) .. 5 (clean and reliable).
- issues: list of {{severity: high|medium|low, column, issue, impact, fix}} - one per real problem (merge related ones);
  impact says what goes wrong downstream, fix is a concrete remediation.
- insights: 2-5 bullets about the data itself (ranges, distributions, relationships, what they imply for the domain).
- recommendations: 3-6 bullets: constraints or monitoring rules to add, cleaning steps, checks to run first.
- questions_for_owner: 1-4 questions to ask the data owner before trusting this table.
Rules: every issue traces back to a finding or a statistic in the sheet; never invent; be specific."""

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "quality_score": {"type": "integer"},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": list(dq.SEVERITIES)}, "column": {"type": "string"},
            "issue": {"type": "string"}, "impact": {"type": "string"}, "fix": {"type": "string"}},
            "required": ["severity", "column", "issue", "impact", "fix"]}},
        "insights": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "questions_for_owner": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "quality_score", "issues", "insights", "recommendations", "questions_for_owner"],
}


PLAN_SYSTEM = """You are the Expert AI: {persona}. You decide WHAT DATA is needed to answer a question, before any
SQL is written. You get a DATABASE BRIEFING (what the tables mean, their grain, keys, joins and pitfalls), the
schema, the standardised request and its extracted details, and the analysis type (data query / statistics /
machine learning). Return JSON with:
- reasoning: one or two sentences on how the briefing shapes the plan.
- tables: the tables or views to read, fewest possible, exact names from the schema.
- columns: the columns to output, as table.column, exact names.
- filters: conditions in plain words (e.g. "current rows only: salaries.to_date = '9999-01-01'").
- grouping: what each output row represents when aggregating (e.g. ["department"]); [] for row-level output.
- metrics: measures to compute (e.g. ["average salary", "number of employees"]).
- one_row_per: what one output row is (e.g. "one row per department", "one row per employee").
- sort / limit: as the request asks; "" / 0 if none.
- order_for_sql: 2-5 sentences telling the SQL writer exactly what to build: which joins, which date filter,
  which aggregation, what to avoid. Written as an instruction.
- pitfalls: 1-3 traps for this specific question, taken from the briefing (e.g. history rows multiplying counts).
For statistics, request one row per observation (not aggregated). For machine learning, the tables must include the
model's feature view and the columns it needs (given). Use only names in the schema; never invent."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tables": {"type": "array", "items": {"type": "string"}},
        "columns": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "string"}},
        "grouping": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "one_row_per": {"type": "string"},
        "sort": {"type": "string"},
        "limit": {"type": "integer"},
        "order_for_sql": {"type": "string"},
        "pitfalls": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "tables", "columns", "filters", "grouping", "metrics", "one_row_per", "sort", "limit",
                 "order_for_sql", "pitfalls"],
}


@dataclass
class DataPlan:
    """What the expert ordered for the SQL writer."""
    tables: list[str]
    columns: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    grouping: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    one_row_per: str = ""
    sort: str = ""
    limit: int = 0
    order_for_sql: str = ""
    pitfalls: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        """Compact description for the assessment prompt and the trace."""
        lines = [self.order_for_sql.strip() or "(no order written)"]
        if self.one_row_per:
            lines.append(f"one row per: {self.one_row_per}")
        if self.filters:
            lines.append("filters: " + "; ".join(self.filters))
        if self.pitfalls:
            lines.append("pitfalls: " + "; ".join(self.pitfalls))
        return "\n".join(lines)


def plan_material(request: str, details: str, intent: str, schema_text: str, briefing: str, ml_note: str = "") -> str:
    parts = [f"DATABASE BRIEFING:\n{briefing.strip() or '(none)'}", f"SCHEMA:\n{schema_text}",
             f"Analysis type: {intent.replace('_', ' ')}." + (f"\n{ml_note}" if ml_note else ""),
             f"Request: {request}" + (f"\n{details}" if details else "")]
    return "\n\n".join(parts)


def validate_plan(raw: dict, schema: dict) -> tuple[DataPlan | None, list[str]]:
    """Keep only tables that exist (case-insensitive) and columns that exist in the kept tables ("t.c" or "c").
    Returns (plan, notes about what was dropped); (None, notes) when no valid table remains."""
    raw = raw or {}
    canon = {name.lower(): name for name in schema}
    notes: list[str] = []
    tables: list[str] = []
    for t in _str_list(raw.get("tables"), 12):
        key = t.strip().strip("`").lower()
        if key in canon and canon[key] not in tables:
            tables.append(canon[key])
        else:
            notes.append(f"unknown table '{t}' dropped")
    if not tables:
        return None, notes or ["the expert named no table"]
    known: dict[str, set[str]] = {t: {c.lower() for c in schema[t].column_names} for t in tables}
    columns: list[str] = []
    for c in _str_list(raw.get("columns"), 40):
        col = c.strip().strip("`")
        if "." in col:
            t, name = col.rsplit(".", 1)
            tk = canon.get(t.strip().lower())
            ok = tk in known and name.strip().lower() in known[tk]
            col = f"{tk}.{name.strip()}" if ok else col
        else:
            ok = any(col.lower() in cols for cols in known.values())
        if ok:
            if col not in columns:
                columns.append(col)
        else:
            notes.append(f"unknown column '{c}' dropped")
    try:
        limit = max(0, int(raw.get("limit") or 0))
    except (TypeError, ValueError):
        limit = 0
    plan = DataPlan(tables=tables, columns=columns, filters=_str_list(raw.get("filters"), 8),
                    grouping=_str_list(raw.get("grouping"), 6), metrics=_str_list(raw.get("metrics"), 6),
                    one_row_per=" ".join(str(raw.get("one_row_per") or "").split()),
                    sort=" ".join(str(raw.get("sort") or "").split()), limit=limit,
                    order_for_sql=" ".join(str(raw.get("order_for_sql") or "").split()),
                    pitfalls=_str_list(raw.get("pitfalls"), 3), reasoning=str(raw.get("reasoning") or ""))
    return plan, notes


# ------------------------------------------------------------------ material
def task_for(res: "AgentResult") -> str:
    if res.ml is not None:
        return "machine_learning"
    if res.stats:
        return "statistics"
    return "data_query"


def review_material(res: "AgentResult", task: str) -> str:
    df = res.frame()
    q = dq.frame_quality(df)
    parts = [f"Question: {res.standalone_question or res.question}"]
    plan = getattr(res, "plan", None)
    if plan is not None:
        parts.append("EXPERT'S PLAN (what you ordered for the SQL writer):\n" + plan.text())
    if res.sql:
        parts.append(f"SQL that ran:\n{res.sql[:1200]}")
    parts.append("QUALITY REPORT:\n" + dq.quality_text(q, dq.findings_from_frame(q)))
    if df is not None and len(df) >= settings.max_rows and task == "data_query":
        parts.append(f"Note: the row limit of {settings.max_rows:,} was reached - the table may be incomplete.")
    if task == "statistics" and res.stats:
        parts.append(f"Statistical method: {res.stats.get('method')}.\nInterpretation: {res.stats.get('interpretation', '')}")
        for name, t in (res.stats.get("tables") or {}).items():
            parts.append(f"{name}:\n{_table_text(t)}")
    elif task == "machine_learning" and res.ml is not None:
        parts.append("Model summary: " + json.dumps(res.ml.summary, default=str)[:1500])
        for sec in res.ml.sections[:4]:
            body = _table_text(sec["content"]) if sec["kind"] == "table" else str(sec["content"])[:800]
            parts.append(f"{sec['title']}:\n{body}")
    if df is not None and not df.empty:
        parts.append(f"First rows:\n{_table_text(df, 10)}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ cleaning
def _int_score(v, default: int = 3) -> int:
    try:
        return max(1, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def _str_list(v, limit: int = MAX_LIST) -> list[str]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [" ".join(str(x).split()) for x in v if str(x).strip()][:limit]


def _clean_review(out: dict) -> dict:
    out = out or {}
    conf = str(out.get("confidence") or "medium").lower().strip()
    return {"verdict": " ".join(str(out.get("verdict") or "").split()) or "No verdict given.",
            "quality_score": _int_score(out.get("quality_score")),
            "expert_answer": " ".join(str(out.get("expert_answer") or "").split()),
            "data_issues": _str_list(out.get("data_issues")), "insights": _str_list(out.get("insights")),
            "advice": _str_list(out.get("advice")), "confidence": conf if conf in CONFIDENCE else "medium"}


def _clean_audit(out: dict) -> dict:
    out = out or {}
    issues = []
    for it in out.get("issues") or []:
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity") or "medium").lower().strip()
        issues.append({"severity": sev if sev in dq.SEVERITIES else "medium",
                       **{k: " ".join(str(it.get(k) or "").split()) for k in ("column", "issue", "impact", "fix")}})
    return {"summary": " ".join(str(out.get("summary") or "").split()) or "No summary given.",
            "quality_score": _int_score(out.get("quality_score")), "issues": issues[:12],
            "insights": _str_list(out.get("insights")), "recommendations": _str_list(out.get("recommendations"), 6),
            "questions_for_owner": _str_list(out.get("questions_for_owner"), 4)}


# ------------------------------------------------------------------ the agent
class ExpertAgent:
    def __init__(self, llm: Any | None = None, persona: str = "", briefing: str = ""):
        self.llm = llm or OllamaLLM(model=settings.expert_model or settings.answer_model or settings.model)
        self.persona = (persona or "").strip() or settings.expert_persona.strip() or DEFAULT_PERSONA
        self.briefing = (briefing or "").strip()      # database briefing text (agent/briefing.py); set by the caller

    def _briefed(self, system: str) -> str:
        # concatenation, never str.format: the briefing is free text and may contain braces
        return system + ("\n\nYou are briefed on the database:\n" + self.briefing if self.briefing else "")

    def system_prompt(self, task: str) -> str:
        focus = TASK_FOCUS.get(task, TASK_FOCUS["data_query"])
        return self._briefed(_COMMON.format(persona=self.persona)) + "\n\n" + focus + "\n\n" + REVIEW_FORMAT

    # ------------------------------------------------------------ plan (before SQL)
    def plan(self, res: "AgentResult", trace: Trace, schema_text: str, schema: dict, ml_note: str = "") -> DataPlan | None:
        """Decide which tables / columns / filters answer the request and write the order for the SQL writer.
        Returns None (warning step) when the model fails or names no valid table."""
        with trace.step("Expert data plan (expert agent)",
                        "The briefed expert decides which tables, columns and filters answer the request and "
                        "writes the order the SQL writer must follow.") as s:
            req = getattr(res, "request", None)
            details = req.details_text() if req is not None else ""
            material = plan_material(res.standalone_question or res.question, details, res.intent, schema_text,
                                     self.briefing, ml_note)
            s.add(model=getattr(self.llm, "model", "?"), persona=self.persona,
                  briefing_given_to_model=self.briefing or "(none)", schema_given_to_model=schema_text)
            try:
                out = self.llm.chat_json(PLAN_SYSTEM.format(persona=self.persona), material, PLAN_SCHEMA)
                s.add_thinking(self.llm)
                s.reasoning = out.get("reasoning")
                plan, notes = validate_plan(out, schema)
                if notes:
                    s.add(dropped=notes)
                if plan is None:
                    s.status = "warning"
                    s.add(note="The expert's plan named no known table; falling back to lexical table linking.")
                    return None
                s.add(plan=plan.to_dict())
                return plan
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Expert agent failed ({exc}); falling back to lexical table linking.")
                return None

    # ------------------------------------------------------------ assess (after the data)
    def assess(self, res: "AgentResult", trace: Trace) -> dict | None:
        """Expert assessment of the data before the answer is written: {task, verdict, quality_score, expert_answer,
        data_issues, insights, advice, confidence, model, persona} or None when the model failed (warning step)."""
        task = task_for(res)
        with trace.step("Expert assessment (expert agent)",
                        f"The briefed expert judges the {task.replace('_', ' ')} result - data quality, its own "
                        "answer, insights and advice - which the answer agent then builds on.") as s:
            material = review_material(res, task)
            s.add(model=getattr(self.llm, "model", "?"), task=task, persona=self.persona,
                  quality_report_given_to_model=material)
            try:
                out = _clean_review(self.llm.chat_json(self.system_prompt(task), material, REVIEW_SCHEMA))
                s.add_thinking(self.llm)
                s.add(verdict=out["verdict"], quality_score=out["quality_score"], expert_answer=out["expert_answer"])
                return {**out, "task": task, "model": getattr(self.llm, "model", "?"), "persona": self.persona}
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Expert agent failed ({exc}); the answer is written from the facts alone.")
                return None

    review = assess     # backwards-compatible name

    def audit(self, sheet: str, table: str, trace: Trace) -> dict | None:
        with trace.step("Expert report (expert agent)",
                        "The expert reads the audit sheet and writes the data-quality report: issues with impact "
                        "and fix, insights, recommendations, questions for the data owner.") as s:
            s.add(model=getattr(self.llm, "model", "?"), persona=self.persona, quality_report_given_to_model=sheet)
            try:
                out = _clean_audit(self.llm.chat_json(self._briefed(AUDIT_SYSTEM.format(persona=self.persona)),
                                                      f"AUDIT SHEET for table {table}:\n{sheet}", AUDIT_SCHEMA))
                s.add_thinking(self.llm)
                s.add(quality_score=out["quality_score"], issues=len(out["issues"]))
                return {**out, "table": table, "model": getattr(self.llm, "model", "?"), "persona": self.persona}
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Expert agent failed ({exc}); only the deterministic findings are available.")
                return None


# ------------------------------------------------------------------ rendering
def _bullets(title: str, items: list[str]) -> str | None:
    return f"**{title}**\n" + "\n".join(f"- {i}" for i in items if i) if items else None


def review_markdown(out: dict) -> str:
    parts = [f"**Verdict** — {out.get('verdict', '').strip()}",
             f"**Expert answer** — {out['expert_answer'].strip()}" if out.get("expert_answer") else None,
             _bullets("Data issues", out.get("data_issues") or []),
             _bullets("Insights", out.get("insights") or []),
             _bullets("Advice", out.get("advice") or []),
             f"_confidence: {out.get('confidence', 'medium')}_"]
    return "\n\n".join(p for p in parts if p)


def audit_markdown(out: dict | None, table: str, findings: list[dq.Finding], generated_at: str) -> str:
    lines = [f"# Data-quality audit: {table}", f"_Generated {generated_at} by Local Data Agent - Expert AI_", ""]
    if out:
        lines += [f"**Quality score: {out['quality_score']}/5** - {out['summary']}", ""]
        if out.get("issues"):
            lines += ["## Issues", "", "| Severity | Column | Issue | Impact | Fix |", "|---|---|---|---|---|"]
            lines += [f"| {i['severity']} | {i['column']} | {i['issue']} | {i['impact']} | {i['fix']} |" for i in out["issues"]]
            lines.append("")
        for title, key in (("Insights", "insights"), ("Recommendations", "recommendations"),
                           ("Questions for the data owner", "questions_for_owner")):
            if out.get(key):
                lines += [f"## {title}", ""] + [f"- {x}" for x in out[key]] + [""]
    else:
        lines += ["_The expert model did not produce a report; the deterministic findings are listed below._", ""]
    lines += ["## Deterministic findings", ""]
    lines += [f"- {f.line()}" for f in findings] if findings else ["- none"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ table audit runner
@dataclass
class AuditResult:
    table: str
    trace: Trace
    profile: dict = field(default_factory=dict)
    findings: list[dq.Finding] = field(default_factory=list)
    sample_quality: dict | None = None
    sample_rows: int = 0
    report: dict | None = None
    generated_at: str = ""
    error: str | None = None

    def findings_dicts(self) -> list[dict]:
        return [asdict(f) for f in self.findings]


def run_audit(db: "Database", info: "TableInfo", expert: ExpertAgent,
              on_step: Callable[[Step], None] | None = None) -> AuditResult:
    """Profile a table with bounded SQL, derive findings, then ask the expert for the report. Never raises."""
    trace = Trace(f"Audit {info.name}", on_step=on_step)
    res = AuditResult(info.name, trace, generated_at=dq.now_text())
    timeout = settings.expert_audit_timeout_ms
    try:
        with trace.step("Profile columns (SQL)",
                        "Per-column counts, distinct values, ranges and suspicious-value counts, one bounded query "
                        "per group of columns.") as s:
            rows: list[dict] = []
            failed: dict[int, str] = {}
            chunks = dq.column_profile_sql(info.name, info.columns, db.quote)
            for i, sql in enumerate(chunks, 1):
                try:
                    rows.append(db.run_bounded(sql, timeout).iloc[0].to_dict())
                except Exception as exc:
                    failed[i] = f"{type(exc).__name__}: {str(exc)[:160]}"
            res.profile = dq.profile_from_rows(info.columns, rows)
            for i in failed:                              # mark the columns of a failed group
                for col in info.columns[(i - 1) * dq.COLS_PER_QUERY: i * dq.COLS_PER_QUERY]:
                    res.profile["columns"][col["name"]] = {"error": "query timed out or failed"}
            s.add(sql=chunks, columns=len(res.profile["columns"]), rows=res.profile.get("rows"))
            if failed:
                s.status = "warning"
                s.add(note="Some column groups could not be profiled: "
                           + "; ".join(f"group {i}: {msg}" for i, msg in failed.items()))

        orphans: dict[str, int] = {}
        duplicates: int | None = None
        with trace.step("Check foreign keys",
                        "Count rows whose foreign key points to a missing parent; count duplicate rows when the "
                        "table has no primary key.") as s:
            notes = []
            for fk in info.foreign_keys:
                label = f"{', '.join(fk['columns'])} -> {fk['ref_table']}"
                try:
                    orphans[label] = int(db.run_bounded(dq.fk_orphan_sql(info.name, fk, db.quote), timeout).iloc[0, 0] or 0)
                except Exception as exc:
                    notes.append(f"{label}: {type(exc).__name__}")
            dup_sql = None if info.primary_key or info.comment == "VIEW" else \
                dq.duplicate_rows_sql(info.name, info.column_names, db.quote)
            if dup_sql:
                try:
                    duplicates = int(db.run_bounded(dup_sql, timeout).iloc[0, 0] or 0)
                except Exception as exc:
                    notes.append(f"duplicates: {type(exc).__name__}")
            s.add(orphans=orphans or "no foreign keys", duplicates="not checked" if duplicates is None else duplicates)
            if notes:
                s.status = "warning"
                s.add(note="Some checks failed: " + "; ".join(notes))

        with trace.step("Sample checks", "Outliers and inconsistent spelling on a sample of rows (pandas).") as s:
            try:
                sample = db.run_bounded(dq.sample_sql(info.name, db.quote, settings.expert_audit_sample_rows), timeout)
                res.sample_rows = len(sample)
                res.sample_quality = dq.frame_quality(sample)
                s.add(rows=res.sample_rows)
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Sample query failed: {type(exc).__name__}: {str(exc)[:160]}")

        res.findings = dq.findings_from_profile(res.profile, orphans, duplicates, info.columns)
        if res.sample_quality:
            for f in dq.findings_from_frame(res.sample_quality):
                if f.issue in ("outliers", "inconsistent spelling / casing"):
                    res.findings.append(dq.Finding(f.severity, f.column, f.issue + " (sample)", f.evidence))
        sheet = dq.profile_table_text(info, res.profile, res.findings, res.sample_quality, res.sample_rows)
        res.report = expert.audit(sheet, info.name, trace)
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    return res
