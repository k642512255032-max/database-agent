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


# ------------------------------------------------------------ follow-ups
REWRITE_SYSTEM = """You turn a follow-up question into a STANDALONE question about a database.
You get the previous conversation (questions, SQL and result rows) and a new question.
- If the new question refers to earlier results (he, she, it, they, them, his, her, that, those, this one,
  the same, what about, only, instead, also, ...), rewrite it so it can be answered WITHOUT the conversation.
  Copy the concrete values it refers to from the previous results: IDs (e.g. emp_no = 10001), names, dates,
  filters, departments.
- If the new question is already self-contained, return it unchanged and set is_follow_up to false.
- Do not answer the question. Do not invent values that are not in the conversation.
Return JSON only."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "is_follow_up": {"type": "boolean"},
        "standalone_question": {"type": "string"},
    },
    "required": ["reasoning", "is_follow_up", "standalone_question"],
}


def rewrite_user(conversation: str, question: str) -> str:
    return f"""Previous conversation:
{conversation}

New question: {question}

Example: previous question "Who is the oldest employee?" returned emp_no=10001, first_name=Georgi,
last_name=Facello. New question "How old is he?" -> standalone_question
"How old is employee emp_no 10001 (Georgi Facello)?"
JSON:"""


def conversation_extra(conversation: str) -> str:
    return f"""Earlier in this conversation (use it to resolve references and reuse IDs / filters):
{conversation}
"""


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
