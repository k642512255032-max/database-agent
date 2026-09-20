"""Database layer: MySQL connection, schema introspection, schema linking and a
read-only SQL guard.

Safety is enforced in three layers:
  1. sqlglot parses the SQL - only a single SELECT/UNION/WITH query is allowed.
  2. Every MySQL session is set to READ ONLY.
  3. (Recommended) connect with a MySQL user that only has SELECT privileges.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import sqlglot
from sqlalchemy import create_engine, event, inspect, text
from sqlglot import exp

from .config import settings

log = logging.getLogger(__name__)

FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "Command",
        "Merge", "TruncateTable", "Into", "Grant", "LoadData", "Use", "Set",
    )
    if hasattr(exp, n)
)


class UnsafeSQLError(ValueError):
    pass


@dataclass
class TableInfo:
    name: str
    columns: list[dict]                   # {name, type, nullable}
    primary_key: list[str]
    foreign_keys: list[dict]              # {columns, ref_table, ref_columns}
    row_count: Optional[int] = None
    sample: list[dict] = field(default_factory=list)
    comment: Optional[str] = None

    @property
    def column_names(self) -> list[str]:
        return [c["name"] for c in self.columns]


class Database:
    def __init__(self, url: str | None = None):
        self.url = url or settings.database_url
        connect_args = {}
        if self.url.startswith("mysql"):
            connect_args = {"read_timeout": settings.query_timeout_s, "connect_timeout": 10}
        self.engine = create_engine(self.url, pool_pre_ping=True, connect_args=connect_args)
        self.dialect = "mysql" if self.url.startswith("mysql") else self.engine.dialect.name

        if self.dialect == "mysql":
            @event.listens_for(self.engine, "connect")
            def _read_only(dbapi_conn, _record):  # noqa: ANN001
                cur = dbapi_conn.cursor()
                cur.execute("SET SESSION TRANSACTION READ ONLY")
                cur.close()

        self._schema: Optional[dict[str, TableInfo]] = None

    # ----------------------------------------------------------- introspection
    def ping(self) -> tuple[bool, str]:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True, f"Connected ({self.engine.url.render_as_string(hide_password=True)})"
        except Exception as exc:
            return False, f"DB connection failed: {exc}"

    def schema(self, refresh: bool = False) -> dict[str, TableInfo]:
        if self._schema is not None and not refresh:
            return self._schema
        insp = inspect(self.engine)
        out: dict[str, TableInfo] = {}
        views = set(insp.get_view_names())
        for t in insp.get_table_names() + sorted(views):
            cols = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in insp.get_columns(t)
            ]
            if t in views:
                pk, fks, comment = [], [], "VIEW"
            else:
                pk = insp.get_pk_constraint(t).get("constrained_columns", []) or []
                fks = [
                    {"columns": fk["constrained_columns"], "ref_table": fk["referred_table"],
                     "ref_columns": fk["referred_columns"]}
                    for fk in insp.get_foreign_keys(t)
                ]
                try:
                    comment = insp.get_table_comment(t).get("text")
                except Exception as exc:
                    log.warning("table-comment probe failed for %s: %s", t, exc)
                    comment = None
            info = TableInfo(t, cols, pk, fks, comment=comment)
            q = self.quote(t)
            # bound every probe: a heavy view (GROUP BY over millions of rows) must not stall the schema load
            hint = f"/*+ MAX_EXECUTION_TIME({settings.schema_probe_ms}) */ " if self.dialect == "mysql" else ""
            with self.engine.connect() as c:
                try:
                    info.row_count = int(c.execute(text(f"SELECT {hint}COUNT(*) FROM {q}")).scalar())
                except Exception as exc:
                    log.warning("row-count probe failed for %s: %s", t, exc)
                try:
                    rows = c.execute(text(f"SELECT {hint}* FROM {q} LIMIT {settings.sample_rows_per_table}"))
                    info.sample = [dict(r._mapping) for r in rows]
                except Exception as exc:
                    # this probe fetches real rows: keep data out of the default log, full text only at DEBUG
                    log.warning("sample-rows probe failed for %s: %s", t, type(exc).__name__)
                    log.debug("sample-rows probe failed for %s", t, exc_info=True)
                # open-ended date sentinel (e.g. to_date = 9999-01-01 means "still current")
                for col in cols:
                    if not col["type"].upper().startswith(("DATE", "TIMESTAMP")):
                        continue
                    try:   # cheap probe: stops at the first sentinel row instead of scanning for MAX()
                        qc = self.quote(col["name"])
                        hit = c.execute(text(f"SELECT {hint}{qc} FROM {q} WHERE {qc} >= '9000-01-01' LIMIT 1")).scalar()
                        if hit is not None:
                            col["note"] = f"{hit} means still current / open-ended"
                    except Exception as exc:
                        log.warning("date-sentinel probe failed for %s.%s: %s", t, col["name"], exc)
            out[t] = info
        self._schema = out
        return out

    def quote(self, name: str) -> str:
        return f"`{name}`" if self.dialect == "mysql" else f'"{name}"'

    # ------------------------------------------------------------ schema text
    def schema_text(self, tables: list[str] | None = None, with_samples: bool = True) -> str:
        """Compact, LLM-friendly description of the schema."""
        sch = self.schema()
        parts = []
        for name in tables or list(sch):
            t = sch[name]
            cols = []
            for c in t.columns:
                flag = " PK" if c["name"] in t.primary_key else ""
                note = f"  -- {c['note']}" if c.get("note") else ""
                cols.append(f"  {c['name']} {c['type']}{flag}{note}")
            fk_lines = [
                f"  FOREIGN KEY ({', '.join(fk['columns'])}) -> {fk['ref_table']}({', '.join(fk['ref_columns'])})"
                for fk in t.foreign_keys
            ]
            header = f"TABLE {t.name}"
            if t.row_count is not None:
                header += f"  -- {t.row_count} rows"
            if t.comment:
                header += f"  -- {t.comment}"
            block = header + "\n" + "\n".join(cols + fk_lines)
            if with_samples and t.sample:
                sample = "; ".join(
                    ", ".join(f"{k}={_short(v)}" for k, v in row.items()) for row in t.sample[:2]
                )
                block += f"\n  sample: {sample}"
            parts.append(block)
        return "\n\n".join(parts)

    # ---------------------------------------------------------- schema linking
    def link_tables(self, question: str, k: int | None = None) -> tuple[list[str], dict[str, float]]:
        """Pick the tables most relevant to a question (lexical match + FK expansion).

        Returns (selected_tables, scores) so the choice is explainable.
        """
        k = k or settings.max_tables_in_prompt
        sch = self.schema()
        if len(sch) <= k:
            return list(sch), {t: 1.0 for t in sch}

        q_tokens = {_stem(w) for w in re.findall(r"[a-zA-Z]+", question.lower())}
        scores: dict[str, float] = {}
        for name, t in sch.items():
            score = 0.0
            for tok in _split_ident(name):
                if tok in q_tokens:
                    score += 3
            for col in t.column_names:
                for tok in _split_ident(col):
                    if tok in q_tokens and tok not in {"id", "name"}:
                        score += 1
            scores[name] = score
        ranked = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]
        selected = ranked[:k]
        # add FK neighbours so joins are possible
        for t in list(selected):
            for fk in sch[t].foreign_keys:
                if fk["ref_table"] not in selected and len(selected) < k:
                    selected.append(fk["ref_table"])
        # add "bridge" tables that link selected tables (e.g. dept_emp between employees and departments)
        refs = {t: len({fk["ref_table"] for fk in info.foreign_keys} & set(selected))
                for t, info in sch.items() if t not in selected}
        for t, n in sorted(refs.items(), key=lambda kv: -kv[1]):
            if n >= 1 and len(selected) < k + 2:
                selected.append(t)
        if not selected:
            selected = list(sch)[:k]
        return selected, scores

    # --------------------------------------------------------------- SQL guard
    def validate_sql(self, sql: str, max_rows: int | None = None) -> tuple[str, list[str]]:
        """Return (safe_sql, notes). Raises UnsafeSQLError with a helpful message."""
        notes: list[str] = []
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise UnsafeSQLError("Empty SQL.")
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except sqlglot.errors.ParseError as exc:
            raise UnsafeSQLError(f"SQL syntax error: {exc}") from exc
        if len(statements) != 1:
            raise UnsafeSQLError("Exactly one SQL statement is allowed.")
        root = statements[0]
        if not isinstance(root, exp.Query):
            raise UnsafeSQLError(f"Only read-only SELECT queries are allowed (got {type(root).__name__}).")
        for node in root.walk():
            if isinstance(node, FORBIDDEN_NODES):
                raise UnsafeSQLError(f"Forbidden operation in query: {type(node).__name__}.")

        # table existence check (ignoring CTE names)
        sch = self.schema()
        lower_tables = {t.lower(): t for t in sch}
        cte_names = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
        for tbl in root.find_all(exp.Table):
            name = tbl.name.lower()
            if name and name not in lower_tables and name not in cte_names:
                raise UnsafeSQLError(
                    f"Unknown table '{tbl.name}'. Available tables: {', '.join(sch)}."
                )

        # column existence check (only for simple queries without subqueries/CTEs)
        if not list(root.find_all(exp.Subquery)) and not cte_names:
            used_tables = {lower_tables[t.name.lower()] for t in root.find_all(exp.Table) if t.name.lower() in lower_tables}
            known_cols = {c.lower() for t in used_tables for c in sch[t].column_names}
            aliases = {a.alias.lower() for a in root.find_all(exp.Alias)}
            for col in root.find_all(exp.Column):
                cname = col.name.lower()
                if cname and cname != "*" and cname not in known_cols and cname not in aliases:
                    cols_hint = "; ".join(f"{t}({', '.join(sch[t].column_names)})" for t in used_tables)
                    suggestion = self._closest_columns(cname)
                    raise UnsafeSQLError(f"Unknown column '{col.name}'. Columns available: {cols_hint}."
                                         + (f" Did you mean: {suggestion}?" if suggestion else ""))

        # enforce a row limit
        limit_node = root.args.get("limit")
        current = None
        if limit_node is not None:
            try:
                current = int(limit_node.expression.name)
            except Exception:
                current = None
        max_rows = max_rows or settings.max_rows
        if current is None or current > max_rows:
            root = root.limit(max_rows)
            notes.append(f"Applied LIMIT {max_rows} to protect memory.")
        return root.sql(dialect=self.dialect), notes

    def _closest_columns(self, name: str, n: int = 4) -> str:
        """'department' -> 'employee_features.department, departments.dept_name' (whole schema)."""
        cands = {f"{t}.{c}": c.lower() for t, info in self.schema().items() for c in info.column_names}
        toks = set(_split_ident(name))
        scored = []
        for full, col in cands.items():
            sim = difflib.SequenceMatcher(None, name, col).ratio()
            overlap = len(toks & set(_split_ident(col)))
            if sim >= 0.6 or overlap:
                scored.append((-(sim + overlap), full))
        return ", ".join(f for _, f in sorted(scored)[:n])

    def run(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as c:
            result = c.execute(text(sql))
            # MySQL allows duplicate names (e.g. `SELECT e.*, f.*` on a join); pandas/pyarrow do not.
            return pd.DataFrame(result.fetchall(), columns=_unique_columns(list(result.keys())))

    def run_bounded(self, sql: str, timeout_ms: int) -> pd.DataFrame:
        """Run a SELECT built by our own code (never by the LLM) with a server-side time limit, like the
        schema probes. Used by the Expert AI table audit."""
        if not sql.lstrip().upper().startswith("SELECT"):
            raise ValueError("run_bounded is for SELECT statements only")
        if self.dialect != "mysql":
            return self.run(sql)
        head, rest = sql.lstrip().split(" ", 1)          # "SELECT", "<rest of the statement>"
        return self.run(f"{head} /*+ MAX_EXECUTION_TIME({int(timeout_ms)}) */ {rest}")


# ---------------------------------------------------------------- helpers
def _unique_columns(names: list[str]) -> list[str]:
    """['emp_no', 'gender', 'emp_no'] -> ['emp_no', 'gender', 'emp_no_2']"""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def _short(v, n: int = 30) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _split_ident(name: str) -> list[str]:
    parts = re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", name)
    return [_stem(p.lower()) for p in parts if p]
