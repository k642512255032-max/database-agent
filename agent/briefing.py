"""Database briefing for the Expert AI: what the tables mean, their grain, keys, joins and pitfalls.

A hand-written briefing lives in briefings/<database>.md (bundled: employees.md, shop.md; editable). For any
other database a deterministic briefing is derived from the introspected schema - no LLM involved. The text is
prepended to the expert's prompts so it plans the data and judges results with domain knowledge, not just
column names.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.engine import make_url

from agent.config import ROOT

if TYPE_CHECKING:
    from agent.db import Database

log = logging.getLogger(__name__)

BRIEFINGS_DIR = ROOT / "briefings"
MAX_BRIEFING_CHARS = 6000
AUTO_RULES = ("Rules: use only the listed tables and columns; an end-date column whose note says '9999' means the row is "
              "still current - filter on it for current rows and avoid multiplying rows through history tables; "
              "prefer views that already select current rows; say which filters a figure includes.")


@dataclass
class Briefing:
    name: str               # database name from the connection URL ("?" when unknown)
    source: str             # "file" | "auto"
    path: Path | None
    text: str


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


def database_name(db: "Database") -> str:
    """Database name from the connection URL, or "?" when missing or unsafe as a file name (no path parts)."""
    try:
        name = make_url(db.url).database or ""
    except Exception:
        return "?"
    return name if _SAFE_NAME.match(name) and ".." not in name else "?"


def load_briefing(db: "Database") -> Briefing:
    """briefings/<database>.md when present (clipped), else a deterministic briefing from the schema."""
    name = database_name(db)
    path = BRIEFINGS_DIR / f"{name}.md"
    if name != "?" and path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if len(text) > MAX_BRIEFING_CHARS:
            text = text[:MAX_BRIEFING_CHARS].rstrip() + "\n(briefing clipped)"
        return Briefing(name, "file", path, text)
    return Briefing(name, "auto", None, auto_briefing(db))


def auto_briefing(db: "Database") -> str:
    """One line per table from the introspected schema: kind, row count, keys, foreign keys, date sentinels."""
    try:
        schema = db.schema()
    except Exception as exc:
        log.warning("auto briefing: schema unavailable: %s", exc)
        return "(schema unavailable)"
    lines = [f"# Database briefing: {database_name(db)} (generated from the schema)", "", "## Tables"]
    for name, info in schema.items():
        kind = "view" if info.comment == "VIEW" else "table"
        bits = [f"{info.row_count:,} rows" if info.row_count is not None else "row count unknown"]
        if info.primary_key:
            bits.append("PK " + ", ".join(info.primary_key))
        for fk in info.foreign_keys:
            bits.append(f"{', '.join(fk['columns'])} -> {fk['ref_table']}({', '.join(fk['ref_columns'])})")
        notes = [f"{c['name']}: {c['note']}" for c in info.columns if c.get("note")]
        line = f"- **{name}** ({kind}; {'; '.join(bits)}): " + ", ".join(c["name"] for c in info.columns)
        if info.comment and info.comment != "VIEW":
            line += f". {info.comment}"
        if notes:
            line += ". Note: " + "; ".join(notes)
        lines.append(line)
    lines += ["", "## Rules of thumb", AUTO_RULES]
    return "\n".join(lines)
