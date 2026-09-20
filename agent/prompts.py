"""Prompts + JSON schemas. Kept short and explicit - small coder models do best
with a tight instruction, the schema, and one example."""
from __future__ import annotations

from .stats_tools import METHODS

# ------------------------------------------------------------------ router
ROUTER_SYSTEM = """You classify a user's request about a database into ONE intent.
Intents:
- data_query: the user wants rows, counts, totals, rankings, lists, trends (plain SQL answers it).
- statistics: the user asks for a statistical analysis: correlation, significance test, t-test, ANOVA,
  chi-square, regression, distribution, "is there a relationship/difference", descriptive statistics.
- machine_learning: the user wants to PREDICT, CLASSIFY, SEGMENT/CLUSTER, detect ANOMALIES/outliers,
  or reduce dimensions (PCA) using one of the trained models listed below.
For machine_learning choose model_name from the list (exact name), otherwise model_name="".
Return JSON only."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "intent": {"type": "string", "enum": ["data_query", "statistics", "machine_learning"]},
        "model_name": {"type": "string"},
    },
    "required": ["reasoning", "intent", "model_name"],
}


def router_user(question: str, models_manifest: str) -> str:
    return f"""Trained models:
{models_manifest}

Request: {question}
Answer as JSON with keys reasoning (one sentence), intent, model_name."""


# ------------------------------------------------- request standardiser
REQUEST_SYSTEM = """You standardise a user's message about a database before it is turned into SQL.
You get the conversation memory (what was discussed, entities and filters in play, facts already found),
the previous query with its first rows, and the new message. Return JSON:
- is_follow_up: true if the message refers to earlier turns (he, she, it, they, them, that, those, the same,
  what about, only, instead, also, "and for X", a bare value like "2001" or "Marketing") or cannot be
  answered on its own.
- standalone_question: ONE clear, complete English question that can be answered WITHOUT the conversation.
  Replace every reference with the concrete values from the memory / previous rows (IDs such as emp_no = 10001,
  names, dates, departments). Keep the standing filters and user preferences from the memory that still apply.
  Fix typos, expand abbreviations (dept -> department, avg -> average), name the measure explicitly
  ("average salary", not "how much"). If the message is already clear, keep its wording.
- task: lookup | list | count | aggregate | rank | compare | trend | analysis | prediction | other
- entities: concrete things the request is about, e.g. ["employee emp_no 10001 (Georgi Facello)", "department Marketing"]
- filters: conditions in plain words, e.g. ["hired after 2000-01-01", "current employees only"]
- metrics: measures asked for, e.g. ["average salary", "number of employees"]
- grouping: what to break figures down by, e.g. ["department", "hire year"]
- time_range: e.g. "1995-2000"; "" if none
- sort: e.g. "salary descending"; "" if none
- limit: rows asked for (top 10 -> 10); 0 if none
- ambiguities: assumptions you had to make, e.g. ["'best paid' taken as highest current salary"]
Never answer the question. Never invent IDs, names or values that are not in the message or the memory."""

REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "is_follow_up": {"type": "boolean"},
        "standalone_question": {"type": "string"},
        "task": {"type": "string", "enum": ["lookup", "list", "count", "aggregate", "rank", "compare", "trend",
                                             "analysis", "prediction", "other"]},
        "entities": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "grouping": {"type": "array", "items": {"type": "string"}},
        "time_range": {"type": "string"},
        "sort": {"type": "string"},
        "limit": {"type": "integer"},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "is_follow_up", "standalone_question", "task", "entities", "filters", "metrics",
                 "grouping", "time_range", "sort", "limit", "ambiguities"],
}


def request_user(memory: str, previous: str, question: str) -> str:
    return f"""Conversation memory:
{memory or "(empty - this is the first message)"}

Previous turn:
{previous or "(none)"}

New message: {question}

Example: memory says employee = emp_no 10001 (Georgi Facello); new message "how old is he?" ->
is_follow_up true, standalone_question "How old is employee emp_no 10001 (Georgi Facello)?", task lookup,
entities ["employee emp_no 10001 (Georgi Facello)"], metrics ["age"].
JSON:"""


# ------------------------------------------------------ context builder
CONTEXT_SYSTEM = """You maintain a compact memory of a conversation between a user and a database assistant.
You get the current memory and the latest turn (message, standardised request, SQL, result rows, answer).
Return the UPDATED memory as JSON:
- summary: 1-3 sentences on what the user has been exploring, most recent focus last.
- entities: object {name: concrete value} for things still in play, with IDs, e.g.
  {"employee": "emp_no 10001 (Georgi Facello)", "department": "d001 Marketing"}. Drop what the user moved away from.
- filters: conditions the user keeps applying (e.g. "current employees only"). Drop one-off conditions.
- metrics: measures the user cares about (e.g. "average salary").
- preferences: standing instructions the user gave ("always show the top 10", "sort by salary descending").
  Keep earlier ones unless the user changed them.
- findings: up to 8 short facts established so far, newest last, each with its concrete numbers / IDs copied
  exactly from the result rows or answer (e.g. "Oldest employee: emp_no 10001 Georgi Facello, born 1952-04-19").
- tables: leave as given.
- turns: leave as given.
Keep it short. Never invent values that are not in the turn or the current memory. If the turn failed, keep the
memory as it is and note the failure in the summary."""

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
        "entities": {"type": "object", "additionalProperties": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "preferences": {"type": "array", "items": {"type": "string"}},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "summary", "entities", "filters", "metrics", "preferences", "findings"],
}


def context_user(memory_json: str, turn: str) -> str:
    return f"""Current memory (JSON):
{memory_json}

Latest turn:
{turn}

Return the updated memory as JSON:"""


def expert_order_extra(order: str, pitfalls: list[str], tables: list[str], one_row_per: str) -> str:
    """Block for the SQL prompt: the expert's data order (section + bullets, like context_extra)."""
    lines = ["Expert's data order (follow it; it already resolves the domain pitfalls):", order.strip() or "(none)"]
    if one_row_per:
        lines.append(f"- one output row per: {one_row_per}")
    if tables:
        lines.append(f"- tables to use: {', '.join(tables)}")
    lines += [f"- avoid: {p}" for p in pitfalls]
    return "\n".join(lines) + "\n"


def context_extra(memory: str, request_details: str) -> str:
    """Block for the SQL prompt: conversation memory + what the standardiser extracted."""
    parts = []
    if memory:
        parts.append(f"Conversation memory (use it to resolve references and reuse IDs / filters):\n{memory}")
    if request_details:
        parts.append(f"Request details (already extracted from the question):\n{request_details}")
    return "\n\n".join(parts) + ("\n" if parts else "")


# --------------------------------------------------------------------- SQL
SQL_SYSTEM = """You are an expert {dialect} SQL writer. Convert the request into ONE read-only SELECT query.
Rules:
- Use ONLY tables and columns that appear in the schema. Never invent columns.
- {dialect} syntax (use LIMIT, backticks for odd names, DATE_FORMAT/YEAR()/MONTH() for dates).
- Use explicit JOIN ... ON with the foreign keys shown.
- Give readable aliases to computed columns (e.g. SUM(total_amount) AS revenue).
- "The oldest / youngest / highest / lowest / earliest / latest ..." (one row) = ORDER BY <column> ASC|DESC LIMIT 1.
  Never put MIN()/MAX() next to ordinary columns to find that row.
- Use aggregates (COUNT/SUM/AVG/MIN/MAX) only when the request asks for a total, average, count or per-group figure.
  Then every non-aggregated column in SELECT must also be in GROUP BY (MySQL only_full_group_by).
  Do not add GROUP BY to a query that has no aggregate.
- A date column marked "9999-01-01 means still current" is an END date: use `col = '9999-01-01'` for current
  rows, `col <> '9999-01-01'` for ended rows, and LEAST(col, CURRENT_DATE) when computing durations.
  Tenure / years of service / "worked for N years" = from the START date (hire_date or from_date) to today.
- Only JOIN a table when the request needs its columns or its filter (e.g. join dept_manager only for managers).
- No INSERT/UPDATE/DELETE/DDL. No comments. One statement.
Return JSON: {{"reasoning": "<short plan: tables, joins, filters, aggregation>", "sql": "<query>"}}"""

SQL_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "sql": {"type": "string"}},
    "required": ["reasoning", "sql"],
}


def sql_user(question: str, schema_text: str, extra: str = "") -> str:
    return f"""Database schema:
{schema_text}

{extra}
Request: {question}
JSON:"""


def ml_sql_extra(card: dict) -> str:
    return f"""This query feeds the trained model '{card['name']}' ({card['algorithm']}, {card['task']}).
The result MUST contain exactly these columns (same names): {', '.join(([card['id_column']] if card.get('id_column') else []) + card['feature_columns'])}
The model was trained on data from this query - start from it and only add WHERE filters the request asks for:
{card['base_sql']}
Return individual rows (no GROUP BY)."""


def stats_sql_extra() -> str:
    return ("This query feeds a statistical analysis. Return one row per observation (not aggregated) with every "
            "column the analysis needs (the measure and the grouping / explanatory columns).")


FIX_SYSTEM = """You fix {dialect} SQL queries. You get the schema, the request, a failing query and the error.
Rewrite the query so the error cannot happen again - do not just patch it (e.g. adding GROUP BY to a query
that should not aggregate). Follow the hint if one is given.
Return a corrected single SELECT query as JSON: {{"reasoning": "<what was wrong>", "sql": "<fixed query>"}}"""

# Concrete advice for errors a small model tends to "fix" the wrong way. Keyed by MySQL error code.
ERROR_HINTS = {
    "1140": "You mixed MIN()/MAX()/COUNT() with ordinary columns. If the request wants ONE row (the oldest, "
            "youngest, highest, lowest, first, last ...), drop the aggregate and the subquery and write: "
            "SELECT <columns> FROM <table> [JOIN ...] [WHERE ...] ORDER BY <column> ASC|DESC LIMIT 1. "
            "If the request really wants per-group figures, put every non-aggregated SELECT column in GROUP BY.",
    "1055": "Every column in SELECT that is not inside an aggregate must appear in GROUP BY, or be removed. "
            "If the request wants one extreme row, use ORDER BY <column> LIMIT 1 instead of aggregates.",
    "1054": "That column does not exist. Use only the column names listed in the schema, and qualify them with "
            "the right table alias.",
    "1146": "That table does not exist. Use only the table names listed in the schema.",
    "1052": "The column name exists in more than one joined table. Prefix it with the table alias.",
    "1064": "Syntax error. Write plain MySQL: no trailing semicolon, no comments, balanced parentheses, "
            "LIMIT at the very end.",
}


# The agent's own validator raises these before the query reaches MySQL.
VALIDATOR_HINTS = {
    "Unknown column": "That column is not in the schema. Use the 'Did you mean' column if one is given, "
                      "otherwise pick the closest column from the schema and JOIN its table if needed. "
                      "Never invent column names.",
    "Unknown table": "That table is not in the schema. Use only the listed tables.",
    "SQL syntax error": "Syntax error. Write plain MySQL: no trailing semicolon, no comments, balanced "
                        "parentheses, LIMIT at the very end.",
    "missing columns required by the model": "Select from the model's base query (SELECT * FROM <view>) and "
                                             "only add WHERE filters; do not rename or drop columns.",
}


def error_hint(error: str) -> str:
    for code, hint in ERROR_HINTS.items():
        if f"({code}," in error or f"{code}," in error[:40]:
            return hint
    for key, hint in VALIDATOR_HINTS.items():
        if key in error:
            return hint
    return ""


def fix_user(question: str, schema_text: str, sql: str, error: str, extra: str = "") -> str:
    hint = error_hint(error)
    hint_block = f"\nHint:\n{hint}\n" if hint else ""
    return f"""Database schema:
{schema_text}
{extra}
Request: {question}
Failing query:
{sql}
Error:
{error}
{hint_block}JSON:"""


# ------------------------------------------------------------------ stats
STATS_SYSTEM = "You choose the right statistical method and columns for a request. Return JSON only.\nMethods:\n" + "\n".join(
    f"- {k}: {v}" for k, v in METHODS.items()
) + """
Parameter meaning:
- target: the numeric measure (or the dependent variable / binary outcome / 2nd categorical for chi_square)
- group: the grouping categorical column (ttest, anova, group_summary, chi_square)
- columns: columns for describe/correlation, or explanatory features for regressions
Use only column names from the data."""

STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "method": {"type": "string", "enum": list(METHODS)},
        "target": {"type": "string"},
        "group": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "method", "target", "group", "columns"],
}


def stats_user(question: str, profile: str) -> str:
    return f"""Data columns (name: dtype, distinct values):
{profile}

Request: {question}
JSON:"""
