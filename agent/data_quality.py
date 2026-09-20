"""Data-quality toolkit for the Expert AI: every number the expert may cite is computed here.

Two halves, both pure (no LLM, no Streamlit):
  * result-set checks on a DataFrame (frame_quality / findings_from_frame / quality_text) - used for the
    chat review and for the sample step of a table audit;
  * table-audit SQL builders (column_profile_sql / fk_orphan_sql / duplicate_rows_sql / sample_sql) whose
    rows are merged by profile_from_rows and judged by findings_from_profile; profile_table_text writes the
    AUDIT SHEET the expert reads. All identifiers are quoted; every statement starts with SELECT so
    Database.run_bounded can add the MySQL time limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import pandas as pd

from agent.stats_tools import numericize
from dashboard.render import _json_safe

if TYPE_CHECKING:
    from agent.db import TableInfo

OUTLIER_MIN_ROWS = 20
MAX_PROFILE_COLUMNS = 40
COLS_PER_QUERY = 12
MAX_DUPLICATE_COLUMNS = 10
AMOUNT_WORDS = ("amount", "price", "salary", "revenue", "total", "cost", "qty", "quantity", "count", "spent")
SEVERITIES = ("high", "medium", "low")
SENTINEL_YEAR = 9000            # open-ended dates such as 9999-01-01 (see agent/db.py schema probes)


@dataclass
class Finding:
    severity: str           # high | medium | low
    column: str
    issue: str
    evidence: str

    def line(self) -> str:
        return f"[{self.severity.upper()}] {self.column}: {self.issue} ({self.evidence})"


# ====================================================================== result-set checks
def _is_amount(name: str) -> bool:
    n = str(name).lower()
    return any(w in n for w in AMOUNT_WORDS)


def frame_quality(df: pd.DataFrame | None, max_columns: int = MAX_PROFILE_COLUMNS) -> dict:
    """Column-level quality facts of a result table. Keys are present only when they apply."""
    out: dict = {"rows": 0, "duplicate_rows": 0, "columns": {}}
    if df is None or df.empty:
        return out
    total_cols = df.shape[1]
    df = numericize(df.iloc[:, :max_columns])
    n = len(df)
    out["rows"] = int(n)
    try:
        out["duplicate_rows"] = int(df.duplicated().sum())
    except TypeError:               # unhashable cells (lists / dicts)
        out["duplicate_rows"] = int(df.astype(str).duplicated().sum())
    if total_cols > max_columns:
        out["columns_skipped"] = total_cols - max_columns
    now = pd.Timestamp.now()
    for c in df.columns:
        s = df[c]
        info: dict = {"missing_pct": round(float(s.isna().mean() * 100), 1), "distinct": int(s.nunique(dropna=True))}
        info["constant"] = bool(n > 1 and info["distinct"] <= 1)
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if len(vals) >= OUTLIER_MIN_ROWS:
                q1, q3 = vals.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    info["outliers"] = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            if _is_amount(c):
                info["negatives"] = int((vals < 0).sum())
        elif pd.api.types.is_datetime64_any_dtype(s):
            ts = s.dropna()
            if len(ts):
                if getattr(ts.dt, "tz", None) is not None:
                    ts = ts.dt.tz_localize(None)
                info["future_dates"] = int((ts > now).sum())
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            strs = pd.Series([v for v in s.dropna() if isinstance(v, str)], dtype="object")
            if len(strs):
                info["blank_strings"] = int((strs.str.strip() == "").sum())
                folded = strs.str.strip().str.lower()
                info["case_variants"] = int(max(0, strs.nunique() - folded.nunique()))
        out["columns"][str(c)] = info
    return out


def _severity_for_missing(pct: float) -> str | None:
    if pct > 20:
        return "high"
    if pct > 5:
        return "medium"
    if pct > 0:
        return "low"
    return None


def findings_from_frame(q: dict) -> list[Finding]:
    rows = q.get("rows", 0)
    out: list[Finding] = []
    if q.get("duplicate_rows"):
        out.append(Finding("medium", "(table)", "duplicate rows", f"{q['duplicate_rows']:,} of {rows:,} rows repeat"))
    for c, info in q.get("columns", {}).items():
        sev = _severity_for_missing(info.get("missing_pct", 0))
        if sev:
            miss = round(rows * info["missing_pct"] / 100)
            out.append(Finding(sev, c, f"{info['missing_pct']}% missing", f"about {miss:,} of {rows:,} rows are NULL"))
        if info.get("constant"):
            out.append(Finding("low", c, "constant column", "one distinct value across all rows"))
        if info.get("outliers"):
            out.append(Finding("low", c, "outliers", f"{info['outliers']:,} values outside 1.5 x IQR"))
        if info.get("negatives"):
            out.append(Finding("medium", c, "negative values in an amount-like column", f"{info['negatives']:,} rows < 0"))
        if info.get("future_dates"):
            out.append(Finding("medium", c, "dates in the future", f"{info['future_dates']:,} rows after today"))
        if info.get("blank_strings"):
            out.append(Finding("low", c, "blank strings", f"{info['blank_strings']:,} empty / whitespace-only values"))
        if info.get("case_variants"):
            out.append(Finding("medium", c, "inconsistent spelling / casing",
                               f"{info['case_variants']:,} values differ only by case or spaces"))
    return out


def quality_text(q: dict, findings: list[Finding], sample_based: bool = False) -> str:
    """Compact text for the model: one line per column, then the findings."""
    rows = q.get("rows", 0)
    head = f"Rows: {rows:,}" + (f" ({q['duplicate_rows']:,} duplicate rows)" if q.get("duplicate_rows") else "")
    if sample_based:
        head += " - checks below are based on this sample only"
    lines = [head]
    for c, info in q.get("columns", {}).items():
        bits = [f"{info.get('missing_pct', 0)}% missing", f"{info.get('distinct', 0):,} distinct"]
        for k in ("outliers", "negatives", "future_dates", "blank_strings", "case_variants"):
            if info.get(k):
                bits.append(f"{info[k]:,} {k.replace('_', ' ')}")
        if info.get("constant"):
            bits.append("constant")
        lines.append(f"- {c}: " + ", ".join(bits))
    if q.get("columns_skipped"):
        lines.append(f"({q['columns_skipped']} more columns not profiled)")
    lines.append("Findings:" if findings else "Findings: none - no quality problem detected by the checks.")
    lines += [f"- {f.line()}" for f in findings]
    return "\n".join(lines)


# ====================================================================== table-audit SQL
def column_kind(sql_type: str) -> str:
    t = (sql_type or "").upper()
    if any(k in t for k in ("INT", "DEC", "FLOAT", "DOUBLE", "NUM", "REAL", "BIT")):
        return "numeric"
    if any(k in t for k in ("DATE", "TIME")):
        return "date"
    if any(k in t for k in ("CHAR", "TEXT", "ENUM", "SET", "STRING")):
        return "text"
    return "other"


def column_profile_sql(table: str, columns: list[dict], quote: Callable[[str], str]) -> list[str]:
    """One SELECT per chunk of COLS_PER_QUERY columns with per-column aggregates. Aliases are `<col>__<stat>`."""
    q = quote(table)
    cols = columns[:MAX_PROFILE_COLUMNS]
    out = []
    for i in range(0, len(cols), COLS_PER_QUERY):
        parts = ["COUNT(*) AS " + quote("rows")]
        for col in cols[i:i + COLS_PER_QUERY]:
            name, qc, kind = col["name"], quote(col["name"]), column_kind(col.get("type", ""))
            parts += [f"COUNT({qc}) AS {quote(name + '__nonnull')}",
                      f"COUNT(DISTINCT {qc}) AS {quote(name + '__distinct')}",
                      f"MIN({qc}) AS {quote(name + '__min')}", f"MAX({qc}) AS {quote(name + '__max')}"]
            if kind == "numeric":
                parts += [f"SUM({qc} < 0) AS {quote(name + '__negatives')}", f"AVG({qc}) AS {quote(name + '__avg')}"]
            elif kind == "text":
                parts.append(f"SUM(TRIM({qc}) = '') AS {quote(name + '__blank')}")
            elif kind == "date":
                parts.append(f"SUM({qc} > CURRENT_DATE()) AS {quote(name + '__future')}")
        out.append("SELECT " + ", ".join(parts) + f" FROM {q}")
    return out


def fk_orphan_sql(table: str, fk: dict, quote: Callable[[str], str]) -> str:
    t, r = quote(table), quote(fk["ref_table"])
    pairs = list(zip(fk["columns"], fk["ref_columns"]))
    on = " AND ".join(f"{t}.{quote(c)} = r.{quote(rc)}" for c, rc in pairs)
    not_null = " AND ".join(f"{t}.{quote(c)} IS NOT NULL" for c, _ in pairs)
    return (f"SELECT COUNT(*) AS {quote('orphans')} FROM {t} LEFT JOIN {r} r ON {on} "
            f"WHERE {not_null} AND r.{quote(pairs[0][1])} IS NULL")


def duplicate_rows_sql(table: str, columns: list[str], quote: Callable[[str], str]) -> str | None:
    """Only for tables without a primary key and few columns (the expression grows with the width)."""
    if not columns or len(columns) > MAX_DUPLICATE_COLUMNS:
        return None
    key = ", ".join(f"COALESCE({quote(c)}, '')" for c in columns)
    return f"SELECT COUNT(*) - COUNT(DISTINCT CONCAT_WS('|', {key})) AS {quote('duplicates')} FROM {quote(table)}"


def sample_sql(table: str, quote: Callable[[str], str], n: int) -> str:
    return f"SELECT * FROM {quote(table)} LIMIT {int(n)}"


def profile_from_rows(columns: list[dict], rows: list[dict]) -> dict:
    """Merge the chunk results (one dict per chunk) into {"rows": n, "columns": {name: {...}}}."""
    merged: dict = {}
    for r in rows:
        merged.update({k: _json_safe(v) for k, v in r.items()})
    n = int(merged.get("rows") or 0)
    prof: dict = {"rows": n, "columns": {}}
    for col in columns[:MAX_PROFILE_COLUMNS]:
        name = col["name"]
        if f"{name}__nonnull" not in merged:
            continue
        nonnull = int(merged.get(f"{name}__nonnull") or 0)
        info = {"nonnull": nonnull, "missing_pct": round(100 * (n - nonnull) / n, 1) if n else 0.0,
                "distinct": int(merged.get(f"{name}__distinct") or 0),
                "min": merged.get(f"{name}__min"), "max": merged.get(f"{name}__max")}
        for stat in ("avg", "negatives", "blank", "future"):
            if f"{name}__{stat}" in merged:
                v = merged[f"{name}__{stat}"]
                info[stat] = (round(float(v), 3) if stat == "avg" else int(v or 0)) if v is not None else None
        prof["columns"][name] = info
    return prof


def _is_sentinel_date(v) -> bool:
    if v is None:
        return False
    year = getattr(v, "year", None)
    if year is None:                       # ISO string such as '9999-01-01' (pandas cannot hold year 9999 as ns)
        head = str(v).strip()[:4]
        year = int(head) if head.isdigit() else 0
    return year >= SENTINEL_YEAR


def findings_from_profile(prof: dict, orphans: dict[str, int], duplicates: int | None, columns: list[dict]) -> list[Finding]:
    rows = prof.get("rows", 0)
    kinds = {c["name"]: column_kind(c.get("type", "")) for c in columns}
    out: list[Finding] = []
    for fk_label, n in orphans.items():
        if n:
            out.append(Finding("high", fk_label, "orphan foreign-key values", f"{n:,} rows point to a missing parent row"))
    if duplicates:
        out.append(Finding("high", "(table)", "duplicate rows (no primary key)", f"{duplicates:,} rows repeat"))
    for c, info in prof.get("columns", {}).items():
        sev = _severity_for_missing(info.get("missing_pct", 0))
        if sev:
            out.append(Finding(sev, c, f"{info['missing_pct']}% missing",
                               f"{rows - info['nonnull']:,} of {rows:,} rows are NULL"))
        if rows > 1 and info.get("distinct") == 1:
            out.append(Finding("low", c, "constant column", f"one distinct value: {info.get('min')}"))
        if kinds.get(c) == "text" and rows > 1 and info.get("distinct") == info.get("nonnull") == rows:
            out.append(Finding("low", c, "looks like an identifier", "every row has a different value"))
        if info.get("negatives"):
            out.append(Finding("medium" if _is_amount(c) else "low", c, "negative values", f"{info['negatives']:,} rows < 0"))
        if info.get("blank"):
            out.append(Finding("low", c, "blank strings", f"{info['blank']:,} empty / whitespace-only values"))
        if info.get("future"):
            if kinds.get(c) == "date" and _is_sentinel_date(info.get("max")):
                out.append(Finding("low", c, "open-ended sentinel date", f"max {info['max']} means 'still current'"))
            else:
                out.append(Finding("medium", c, "dates in the future", f"{info['future']:,} rows after today"))
    return out


def profile_table_text(info: "TableInfo", prof: dict, findings: list[Finding], sample_q: dict | None,
                       sample_rows: int = 0) -> str:
    """The AUDIT SHEET: structure, per-column statistics, findings, sample checks."""
    n = prof.get("rows")
    lines = [f"Table {info.name} ({n:,} rows)." if n is not None else f"Table {info.name}."]
    if info.comment:
        lines.append(f"Comment: {info.comment}")
    lines.append("Primary key: " + (", ".join(info.primary_key) if info.primary_key else "none"))
    if info.foreign_keys:
        lines.append("Foreign keys: " + "; ".join(
            f"{', '.join(fk['columns'])} -> {fk['ref_table']}({', '.join(fk['ref_columns'])})" for fk in info.foreign_keys))
    types = {c["name"]: c for c in info.columns}
    lines.append("Columns:")
    for c, p in prof.get("columns", {}).items():
        col = types.get(c, {})
        if p.get("error"):
            bits = [f"not profiled: {p['error']}"]
        else:
            bits = [f"{p.get('nonnull', 0):,} non-null ({p.get('missing_pct', 0)}% missing)",
                    f"{p.get('distinct', 0):,} distinct"]
            if p.get("min") is not None:
                bits.append(f"min {p['min']}, max {p['max']}")
            if p.get("avg") is not None:
                bits.append(f"avg {p['avg']}")
            for k in ("negatives", "blank", "future"):
                if p.get(k):
                    bits.append(f"{p[k]:,} {k}")
        nullable = "null ok" if col.get("nullable", True) else "not null"
        lines.append(f"- {c} ({col.get('type', '?')}, {nullable}): " + ", ".join(bits))
    if len(info.columns) > MAX_PROFILE_COLUMNS:
        lines.append(f"({len(info.columns) - MAX_PROFILE_COLUMNS} more columns not profiled)")
    lines.append("Findings:" if findings else "Findings: none - no quality problem detected by the checks.")
    lines += [f"- {f.line()}" for f in findings]
    if sample_q and sample_q.get("columns"):
        extra = []
        for c, q in sample_q["columns"].items():
            hits = [f"{q[k]:,} {k.replace('_', ' ')}" for k in ("outliers", "case_variants") if q.get(k)]
            if hits:
                extra.append(f"- {c}: " + ", ".join(hits))
        lines.append(f"Sample checks ({sample_rows:,} rows): " + ("" if extra else "nothing notable"))
        lines += extra
    return "\n".join(lines)


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
