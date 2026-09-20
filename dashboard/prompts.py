"""Prompts for the dashboard design agent (answer model, JSON output)."""
from __future__ import annotations

from .spec import KINDS, WIDTHS

DESIGN_SYSTEM = """You design a dashboard for a business user from a plain-English description and a database schema.
Return JSON only, describing 3-8 widgets. Widget kinds and when to use them:
- kpi: one headline number (total, count, average). Its question must return a single value.
- bar: compare a numeric value across categories (top N, per department, per status).
- line: a numeric value over time (per month / year / day).
- pie: share of a whole across a few categories (<= 8).
- table: a list of rows (latest orders, top employees) - keep it to the columns worth reading.
Rules:
- Each widget's "question" is a complete, standalone data request that a SQL writer can answer from the schema
  alone: name the measure, the grouping, the filters and the time range ("Total revenue from completed orders
  in 2025", "Number of orders per month in 2025", "Top 10 cities by revenue"). One question = one result table.
- "x" is the column (or concept) for categories / time, "y" the numeric measure; leave "" if not applicable.
- "width": small | medium | full. kpi -> small; tables and time series -> full; others -> medium.
- "accent": a hex colour like "#2F5BEA" fitting the theme the user asked for, else "".
- Use only tables and columns that exist in the schema. Never invent data. Never answer the questions yourself."""

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "accent": {"type": "string"},
        "widgets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "question": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "width": {"type": "string", "enum": list(WIDTHS)},
                    "note": {"type": "string"},
                },
                "required": ["title", "kind", "question", "x", "y", "width", "note"],
            },
        },
    },
    "required": ["reasoning", "title", "subtitle", "accent", "widgets"],
}


def design_user(description: str, schema_text: str) -> str:
    return f"""Database schema:
{schema_text}

The user wants this dashboard:
{description}

Return the dashboard spec as JSON:"""


def refine_user(description: str, spec_json: str, change_request: str, schema_text: str) -> str:
    return f"""Database schema:
{schema_text}

The user originally asked for:
{description}

Current dashboard spec (JSON):
{spec_json}

The user now wants these changes:
{change_request}

Apply only the requested changes. Keep every other widget identical (same title, same question, same kind).
Return the full updated dashboard spec as JSON:"""
