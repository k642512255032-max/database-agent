"""Expert AI agent: a specialist that works alongside the answer agent.

The answer agent explains what the numbers say. The expert agent judges whether the data and the
analysis can be trusted, adds the insights a domain expert would notice, and gives concrete advice.
It has one task-specific prompt per kind of result:

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
You work alongside an answer-writing assistant. Your job is different: judge whether the data and the analysis can be
trusted, add the insights an expert would notice, and give concrete advice. You get the question, the answer already
given, a QUALITY REPORT computed by tools, and material about the analysis. Never invent numbers; cite only what is in
the material. Do not repeat the answer. Be specific and practical; no generic platitudes."""

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
- data_issues: 0-5 bullets, each naming one concrete problem from the quality report or the material; empty if none.
- insights: 2-4 bullets a domain expert would notice (patterns, surprises, what is missing to conclude).
- advice: 2-4 bullets of concrete next actions (what to check, which breakdown or filter to add, what to decide).
- confidence: low | medium | high."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string"},
        "quality_score": {"type": "integer"},
        "data_issues": {"type": "array", "items": {"type": "string"}},
        "insights": {"type": "array", "items": {"type": "string"}},
        "advice": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
    },
    "required": ["verdict", "quality_score", "data_issues", "insights", "advice", "confidence"],
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
    parts = [f"Question: {res.standalone_question or res.question}",
             f"Answer given:\n{(res.answer or '(none)')[:1500]}",
             "QUALITY REPORT:\n" + dq.quality_text(q, dq.findings_from_frame(q))]
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
    def __init__(self, llm: Any | None = None, persona: str = ""):
        self.llm = llm or OllamaLLM(model=settings.expert_model or settings.answer_model or settings.model)
        self.persona = (persona or "").strip() or settings.expert_persona.strip() or DEFAULT_PERSONA

    def system_prompt(self, task: str) -> str:
        focus = TASK_FOCUS.get(task, TASK_FOCUS["data_query"])
        return _COMMON.format(persona=self.persona) + "\n\n" + focus + "\n\n" + REVIEW_FORMAT

    def review(self, res: "AgentResult", trace: Trace) -> dict | None:
        """Expert review of one chat result: {task, verdict, quality_score, data_issues, insights, advice, confidence,
        model, persona} or None when the model failed (recorded as a warning step)."""
        task = task_for(res)
        with trace.step("Expert review (expert agent)",
                        f"A specialist model reviews the {task.replace('_', ' ')} result: data quality, expert "
                        "insights and advice, in the persona set in the sidebar.") as s:
            material = review_material(res, task)
            s.add(model=getattr(self.llm, "model", "?"), task=task, persona=self.persona,
                  quality_report_given_to_model=material)
            try:
                out = _clean_review(self.llm.chat_json(self.system_prompt(task), material, REVIEW_SCHEMA))
                s.add(verdict=out["verdict"], quality_score=out["quality_score"])
                return {**out, "task": task, "model": getattr(self.llm, "model", "?"), "persona": self.persona}
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Expert agent failed ({exc}); no expert review for this answer.")
                return None

    def audit(self, sheet: str, table: str, trace: Trace) -> dict | None:
        with trace.step("Expert report (expert agent)",
                        "The expert reads the audit sheet and writes the data-quality report: issues with impact "
                        "and fix, insights, recommendations, questions for the data owner.") as s:
            s.add(model=getattr(self.llm, "model", "?"), persona=self.persona, quality_report_given_to_model=sheet)
            try:
                out = _clean_audit(self.llm.chat_json(AUDIT_SYSTEM.format(persona=self.persona),
                                                      f"AUDIT SHEET for table {table}:\n{sheet}", AUDIT_SCHEMA))
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
