"""Answer agent: turns computed results into a coherent explanation and a chart plan.

It runs on its own Ollama model (settings.answer_model) because a general instruct model
writes far better prose than the coder model that generates SQL. It never computes numbers:
it receives a fact sheet (column profile, statistics tables, ML summary, sample rows) and
may only restate what is in it.

Two responsibilities, two LLM calls, both recorded as trace steps:
  * compose()      -> markdown answer (headline, findings, interpretation, caveats)
  * plan_charts()  -> list of chart specs, validated against the real columns; if the model
                      fails or proposes nonsense, a deterministic heuristic picks the chart.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from .config import settings
from .llm import OllamaLLM
from .trace import Trace

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .orchestrator import AgentResult

CHART_TYPES = ("bar", "line", "scatter", "histogram", "box")
MAX_CATEGORIES = 30      # bars/boxes beyond this are unreadable; the renderer keeps the top-N
MAX_SERIES = 8           # categorical colour slots (see app.py PALETTE)

# ------------------------------------------------------------------ prompts
ANSWER_SYSTEM = """You are a senior data analyst explaining results to a business user.
You get the question and a FACT SHEET computed by tools (SQL, statistics, machine learning). Return JSON with:
- answer: 1-2 sentences that directly answer the question, with the key number(s).
- key_findings: 3-5 bullet strings, each stating one concrete number or comparison from the fact sheet
  (e.g. "Women's average is 76,720 vs 76,410 for men, a 0.4% gap").
- interpretation: 2-3 sentences on what this means in practice: direction, size of the effect (is it large or
  negligible?), which group / segment / feature matters most, and what a manager could do with it.
- caveats: 1-3 bullet strings on limits: sample size, borderline p-value, missing data, model accuracy, filters applied.
Rules: use ONLY numbers in the fact sheet; never invent or extrapolate. Round sensibly (2 decimals, thousands
separators). Do not mention SQL, column types or the tools. Do not repeat the question. Plain English; explain any
statistic in a few words (e.g. "p = 0.28, i.e. a difference this small could easily be chance").
If a statistic is nan / missing or there are too few rows (e.g. a test on 2 rows), the answer must say the analysis
could not be computed and why - never conclude anything from a nan.
If the fact sheet contains an EXPERT ASSESSMENT, build on it: state the expert answer's key numbers, carry its data
issues into the caveats, use its insights in the interpretation, and never contradict its verdict. If the verdict says
the data cannot answer the question, say so instead of answering."""

DATA_SUMMARY_SYSTEM = """You summarise the result of a database query for a business user, in plain English.
You get the question and a FACT SHEET (row count, columns, numeric summary, top values, sample rows). Return JSON with:
- answer: 2-4 sentences saying what the table shows and the key numbers (the largest / smallest values, totals,
  the leader and the runner-up, a notable gap or outlier). Lead with the direct answer to the question.
- key_findings: 0-3 short bullet strings with one concrete number each; leave empty if the answer already says it all.
Rules: use ONLY numbers in the fact sheet; never invent, extrapolate or explain causes. Round sensibly (2 decimals,
thousands separators). Do not mention SQL, column types or tools. Do not repeat the question. If the sample rows are
only part of the result, describe the whole table (from the row count and summary), not just the sample.
If there is an EXPERT ASSESSMENT, reflect the expert answer and mention its data issues in one clause."""

DATA_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "key_findings"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "interpretation": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "key_findings", "interpretation", "caveats"],
}


def answer_markdown(out: dict, brief: bool = False) -> str:
    # a brief data-query summary sits above the table and needs no 'Answer' heading
    parts = [out.get('answer', '').strip() if brief else f"**Answer** — {out.get('answer', '').strip()}"]
    if out.get("key_findings"):
        parts.append("**Key findings**\n" + "\n".join(f"- {f}" for f in out["key_findings"] if f))
    if out.get("interpretation"):
        parts.append(f"**Interpretation** — {out['interpretation'].strip()}")
    if out.get("caveats"):
        parts.append("**Caveats**\n" + "\n".join(f"- {c}" for c in out["caveats"] if c))
    return "\n\n".join(parts)

CHART_SYSTEM = """You are a data-visualisation planner. Given a question and a profile of the result table,
propose the chart(s) that best show the finding. Return JSON only.
Chart types and when to use them:
- bar: compare a numeric value across categories (x = category column, y = numeric column). Best default.
  To count rows per category (e.g. how many per predicted class / cluster), set y = "count", aggregate = "count".
- line: a numeric value over time or an ordered axis (x = date/period/ordered column, y = numeric).
- scatter: relationship between two numeric columns (x numeric, y numeric); color = optional category.
- histogram: distribution of one numeric column (x = numeric column, no y).
- box: distribution of a numeric column per group (x = category column, y = numeric column).
Rules:
- Use ONLY column names from the profile, exactly as written.
- x, y and color must be different columns. color must be a category column with <= 8 distinct values, or "".
- aggregate applies when several rows share the same x: "none" if x is unique per row, else "sum" | "mean" | "count".
- Prefer the columns the question is about (the measure and the grouping). One chart is enough; two only if they
  show different things. If nothing is worth plotting (single value, only IDs or free text), return an empty list.
Return JSON: {"charts": [{"type": "...", "x": "...", "y": "...", "color": "", "aggregate": "none",
                            "title": "<short title>", "why": "<one sentence>"}]}"""

CHART_SCHEMA = {
    "type": "object",
    "properties": {
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(CHART_TYPES)},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "color": {"type": "string"},
                    "aggregate": {"type": "string", "enum": ["none", "sum", "mean", "count"]},
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["type", "x", "y", "color", "aggregate", "title", "why"],
            },
        }
    },
    "required": ["charts"],
}


class AnswerAgent:
    def __init__(self, llm: Any | None = None):
        self.llm = llm or OllamaLLM(model=settings.answer_model or settings.model)

    # =============================================================== answer
    def compose(self, res: "AgentResult", trace: Trace, brief: bool = False) -> str:
        """Full analyst answer, or with brief=True a short plain-English summary of a data query."""
        system, schema = (DATA_SUMMARY_SYSTEM, DATA_SUMMARY_SCHEMA) if brief else (ANSWER_SYSTEM, ANSWER_SCHEMA)
        with trace.step("Summarise the result (answer agent)" if brief else "Compose the answer (answer agent)",
                        "A second model describes what the result table shows, in plain English." if brief else
                        "A second model turns the computed facts into a coherent explanation: answer, "
                        "key findings, interpretation and caveats.") as s:
            facts = fact_sheet(res)
            s.add(model=getattr(self.llm, "model", "?"), facts_given_to_model=facts)
            try:
                out = self.llm.chat_json(system, f"Question: {res.standalone_question}\n\nFACT SHEET:\n{facts}", schema)
                s.add_thinking(self.llm)
                out = {k: out.get(k) for k in schema["properties"]}     # brief mode renders only its own fields
                if not (out.get("answer") or "").strip():
                    raise ValueError("empty answer")
                return answer_markdown(out, brief=brief)
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Answer agent failed ({exc}); showing the raw facts instead.")
                return facts

    # =============================================================== charts
    def plan_charts(self, res: "AgentResult", trace: Trace, df: pd.DataFrame | None) -> list[dict]:
        if df is None or df.empty or len(df) < 2:
            return []
        prof = profile(df)
        if not prof["numeric"] and not prof["category"]:
            return []
        with trace.step("Plan charts (answer agent)",
                        "Choose chart type and columns that visualise the finding; the choice is validated "
                        "against the real columns before drawing.") as s:
            s.add(model=getattr(self.llm, "model", "?"), column_profile=profile_text(prof))
            specs: list[dict] = []
            try:
                out = self.llm.chat_json(CHART_SYSTEM, f"Question: {res.standalone_question}\n\n"
                                         f"Result table profile ({len(df)} rows):\n{profile_text(prof)}\nJSON:",
                                         CHART_SCHEMA)
                proposed = out.get("charts") or []
                rejected = []
                for spec in proposed:
                    ok, reason = validate_spec(spec, prof)
                    (specs if ok else rejected).append(spec if ok else f"{spec.get('type')} {spec.get('x')}/{spec.get('y')}: {reason}")
                if rejected:
                    s.add(rejected=rejected)
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Chart planner failed ({exc}); using the heuristic chart.")
            if not specs:
                fallback = heuristic_chart(prof)
                if fallback:
                    specs = [fallback]
                    s.add(decision="Model proposed nothing usable - heuristic chart used.")
            specs = specs[: settings.max_charts]
            s.add(charts=[f"{c['type']}: {c['x']}" + (f" vs {c['y']}" if c.get("y") else "") for c in specs] or "none")
            return specs


# ------------------------------------------------------------------ fact sheet
def fact_sheet(res: "AgentResult", max_rows: int = 10) -> str:
    """Everything the answer agent may talk about - computed by tools, not by an LLM."""
    df = res.data
    parts = [f"Analysis type: {res.intent.replace('_', ' ')}."]
    if res.expert:
        parts.append(expert_assessment_text(res.expert))
    if df is not None:
        parts.append(f"Rows returned: {len(df)}. Columns: {', '.join(map(str, df.columns))}.")
        prof = profile(df)
        if prof["numeric"]:
            desc = df[prof["numeric"][:8]].apply(pd.to_numeric, errors="coerce").describe().T[["count", "mean", "min", "max"]]
            parts.append("Numeric column summary:\n" + desc.round(2).to_string())
        for c in prof["category"][:4]:
            vc = df[c].astype(str).value_counts().head(5)
            parts.append(f"Top values of {c}: " + ", ".join(f"{k} ({v})" for k, v in vc.items()))
    if res.stats:
        parts.append(f"Statistical method: {res.stats['method']}.\nInterpretation: {res.stats['interpretation']}")
        for name, t in res.stats.get("tables", {}).items():
            parts.append(f"{name}:\n{_table_text(t)}")
        if any(isinstance(t, pd.DataFrame) and t.isna().any().any() for t in res.stats.get("tables", {}).values()):
            parts.append(f"WARNING: some statistics are nan - the test could not be computed "
                         f"(only {len(df) if df is not None else 0} rows, likely aggregated instead of one row per observation).")
    elif res.ml is not None:
        parts.append(f"Model: {res.ml.summary.get('model')} ({res.ml.summary.get('algorithm')}). "
                     f"Summary: {json.dumps(res.ml.summary, default=str)[:1500]}")
        for sec in res.ml.sections[:4]:
            body = _table_text(sec["content"]) if sec["kind"] == "table" else str(sec["content"])[:800]
            parts.append(f"{sec['title']}:\n{body}")
        if df is not None:
            parts.append("Predictions (first rows):\n" + _table_text(
                pd.concat([res.ml.output.reset_index(drop=True),
                           df.drop(columns=[c for c in res.ml.output.columns if c in df.columns]).reset_index(drop=True)],
                          axis=1), max_rows))
    if df is not None and res.ml is None:
        parts.append(f"First rows:\n{_table_text(df, max_rows)}")
    return "\n\n".join(parts)


def expert_assessment_text(exp: dict) -> str:
    """The expert's assessment as a fact-sheet section (the answer agent must build on it)."""
    def bullets(items):
        return "\n".join(f"- {i}" for i in items) if items else "- none"
    return ("EXPERT ASSESSMENT (a briefed domain expert reviewed this data before you):\n"
            f"Verdict: {exp.get('verdict', '')}\n"
            f"Expert answer: {exp.get('expert_answer') or '(none)'}\n"
            f"Data issues:\n{bullets(exp.get('data_issues'))}\n"
            f"Insights:\n{bullets(exp.get('insights'))}\n"
            f"Advice:\n{bullets(exp.get('advice'))}\n"
            f"Confidence: {exp.get('confidence', 'medium')}")


def _table_text(t, max_rows: int = 12) -> str:
    if isinstance(t, pd.DataFrame):
        return t.head(max_rows).round(3).to_string(index=False, max_colwidth=32)
    return str(t)[:800]


# ------------------------------------------------------------------ profiling
def profile(df: pd.DataFrame) -> dict:
    """Classify columns: numeric / temporal / category (<= MAX_CATEGORIES distinct) / other (ids, text)."""
    out: dict[str, Any] = {"numeric": [], "temporal": [], "category": [], "other": [], "nunique": {}, "rows": len(df)}
    for c in df.columns:
        col = df[c]
        n = int(col.nunique(dropna=True))
        out["nunique"][c] = n
        name = str(c).lower()
        if pd.api.types.is_datetime64_any_dtype(col) or (
                col.dtype == object and any(k in name for k in ("date", "time", "month", "year", "period"))
                and pd.to_datetime(col, errors="coerce").notna().mean() > 0.9):
            out["temporal"].append(c)
        elif pd.api.types.is_bool_dtype(col) or (
                pd.api.types.is_numeric_dtype(col) and set(col.dropna().unique()) <= {0, 1}):
            out["category"].append(c)          # 0/1 flag
        elif pd.api.types.is_numeric_dtype(col) or pd.to_numeric(col, errors="coerce").notna().mean() > 0.95:
            if name.endswith(("_id", "_no", "id")) and n == len(df):
                out["other"].append(c)          # identifier
            else:
                out["numeric"].append(c)
        elif n <= MAX_CATEGORIES:
            out["category"].append(c)
        else:
            out["other"].append(c)
    return out


def profile_text(prof: dict) -> str:
    lines = []
    for kind in ("numeric", "temporal", "category"):
        for c in prof[kind]:
            lines.append(f"- {c}: {kind}, {prof['nunique'][c]} distinct values")
    if prof["other"]:
        lines.append("- not plottable (ids / free text): " + ", ".join(map(str, prof["other"])))
    return "\n".join(lines)


def validate_spec(spec: dict, prof: dict) -> tuple[bool, str]:
    numeric, temporal, category = prof["numeric"], prof["temporal"], prof["category"]
    t, x, y, color = spec.get("type"), spec.get("x"), spec.get("y") or "", spec.get("color") or ""
    known = set(numeric) | set(temporal) | set(category)
    if t not in CHART_TYPES:
        return False, "unknown chart type"
    if x not in known:
        return False, f"x '{x}' is not a plottable column"
    if t == "histogram":
        if x not in numeric:
            return False, "histogram needs a numeric x"
        spec["y"] = ""
    elif t in ("bar", "line") and (spec.get("aggregate") == "count" or y.lower() in ("", "count", "count()")):
        spec["y"], spec["aggregate"] = "count", "count"      # a count chart needs no y column
    else:
        if y not in numeric:
            return False, f"y '{y}' must be a numeric column"
        if x == y:
            return False, "x and y are the same column"
    if t == "scatter" and x not in numeric:
        return False, "scatter needs a numeric x"
    if t in ("bar", "box") and x in numeric and prof["nunique"][x] > MAX_CATEGORIES:
        return False, "x has too many distinct values for a bar/box chart"
    if color:
        if color not in category or prof["nunique"][color] > MAX_SERIES or color in (x, y):
            spec["color"] = ""      # drop a bad colour rather than reject the chart
    if spec.get("aggregate") not in ("none", "sum", "mean", "count"):
        spec["aggregate"] = "none"
    if t in ("bar", "line") and spec["aggregate"] == "none" and prof["nunique"][x] < prof["rows"] and spec["y"] != "count":
        spec["aggregate"] = "sum" if t == "bar" else "mean"     # repeated x needs an aggregate
    if t in ("scatter", "box", "histogram"):
        spec["aggregate"] = "none"
    spec.setdefault("title", f"{y or x} by {x}" if y else str(x))
    return True, ""


def heuristic_chart(prof: dict) -> dict | None:
    """Deterministic fallback when the planner fails: the same rules as the old auto_chart."""
    numeric, temporal, category = prof["numeric"], prof["temporal"], prof["category"]
    labels = [c for c in category if str(c).lower().startswith(("predicted", "cluster", "anomaly", "is_anomaly"))]
    if labels:   # ML output: how many rows fell in each predicted class / cluster
        return {"type": "bar", "x": labels[0], "y": "count", "color": "", "aggregate": "count",
                "title": f"Rows per {labels[0]}", "why": "number of rows in each predicted class"}
    if temporal and numeric:
        return {"type": "line", "x": temporal[0], "y": numeric[-1], "color": "", "aggregate": "mean",
                "title": f"{numeric[-1]} over {temporal[0]}", "why": "numeric value over time"}
    if category and numeric:
        return {"type": "bar", "x": category[0], "y": numeric[-1], "color": "", "aggregate": "sum",
                "title": f"{numeric[-1]} by {category[0]}", "why": "numeric value per category"}
    if len(numeric) >= 2:
        return {"type": "scatter", "x": numeric[0], "y": numeric[1], "color": "", "aggregate": "none",
                "title": f"{numeric[1]} vs {numeric[0]}", "why": "relationship between two measures"}
    if numeric:
        return {"type": "histogram", "x": numeric[0], "y": "", "color": "", "aggregate": "none",
                "title": f"Distribution of {numeric[0]}", "why": "distribution of the measure"}
    return None
