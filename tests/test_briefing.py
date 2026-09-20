"""Hermetic tests for agent/briefing.py (no MySQL)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import briefing as br  # noqa: E402
from agent.briefing import MAX_BRIEFING_CHARS, auto_briefing, load_briefing  # noqa: E402
from agent.db import TableInfo  # noqa: E402


def _db(name: str, schema=None):
    return SimpleNamespace(url=f"mysql+pymysql://u:p@h:3306/{name}", schema=lambda: schema or {})


def test_bundled_employees_briefing_is_loaded_from_file():
    b = load_briefing(_db("employees"))
    assert b.source == "file" and b.name == "employees" and b.path.name == "employees.md"
    assert b.text.startswith("# Database briefing: employees")
    assert "9999-01-01" in b.text and "2002" in b.text and len(b.text) < 4500


def test_bundled_shop_briefing_exists():
    b = load_briefing(_db("shop"))
    assert b.source == "file" and "completed" in b.text


def test_auto_briefing_from_schema():
    schema = {
        "t": TableInfo("t", [{"name": "id", "type": "INT", "nullable": False},
                             {"name": "to_date", "type": "DATE", "nullable": True, "note": "9999-01-01 means still current"}],
                       ["id"], [{"columns": ["id"], "ref_table": "r", "ref_columns": ["id"]}], row_count=12, comment="things"),
        "v": TableInfo("v", [{"name": "x", "type": "INT", "nullable": True}], [], [], comment="VIEW"),
    }
    b = load_briefing(_db("other", schema))
    assert b.source == "auto" and b.path is None
    assert "- **t** (table; 12 rows; PK id; id -> r(id)): id, to_date. things. Note: to_date: 9999-01-01 means still current" in b.text
    assert "- **v** (view; row count unknown): x" in b.text
    assert "## Rules of thumb" in b.text


def test_auto_briefing_schema_failure_is_soft():
    def boom():
        raise RuntimeError("down")
    db = SimpleNamespace(url="mysql+pymysql://u:p@h/x", schema=boom)
    assert auto_briefing(db) == "(schema unavailable)"


def test_file_briefing_is_clipped(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "BRIEFINGS_DIR", tmp_path)
    (tmp_path / "big.md").write_text("x" * 20000, encoding="utf-8")
    b = load_briefing(_db("big"))
    assert b.source == "file" and len(b.text) <= MAX_BRIEFING_CHARS + 30 and b.text.endswith("(briefing clipped)")


def test_unknown_database_name_uses_auto():
    b = load_briefing(SimpleNamespace(url="not a url", schema=lambda: {}))
    assert b.name == "?" and b.source == "auto"


def test_database_name_cannot_traverse_paths():
    for bad in ("../../.env", "..", "a/b", "a%5Cb", "a b"):
        b = load_briefing(SimpleNamespace(url=f"mysql+pymysql://u:p@h/{bad}", schema=lambda: {}))
        assert b.name == "?" and b.source == "auto", bad
    assert load_briefing(_db("my_db-2.0")).name == "my_db-2.0"
