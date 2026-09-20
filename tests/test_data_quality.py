"""Hermetic tests for agent/data_quality.py (no MySQL/Ollama)."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.data_quality import (  # noqa: E402
    COLS_PER_QUERY,
    Finding,
    column_profile_sql,
    duplicate_rows_sql,
    findings_from_frame,
    findings_from_profile,
    fk_orphan_sql,
    frame_quality,
    profile_from_rows,
    profile_table_text,
    quality_text,
    sample_sql,
)
from agent.db import TableInfo  # noqa: E402

Q = lambda n: f"`{n}`"  # noqa: E731


# --------------------------------------------------------------- frame_quality


def test_missing_duplicates_constant():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 3.0], "k": ["x"] * 4})
    df.loc[3] = df.loc[2]
    q = frame_quality(df)
    assert q["rows"] == 4 and q["duplicate_rows"] == 1
    assert q["columns"]["a"]["missing_pct"] == 25.0 and q["columns"]["k"]["constant"] is True
    assert q["columns"]["a"]["constant"] is False


def test_outliers_need_enough_rows():
    small = pd.DataFrame({"v": [1.0] * 5 + [100.0]})
    assert "outliers" not in frame_quality(small)["columns"]["v"]
    big = pd.DataFrame({"v": list(range(30)) + [1000]})
    assert frame_quality(big)["columns"]["v"]["outliers"] == 1


def test_negatives_only_for_amount_like_columns():
    df = pd.DataFrame({"salary": [1, -1], "delta": [1, -1]})
    q = frame_quality(df)["columns"]
    assert q["salary"]["negatives"] == 1 and "negatives" not in q["delta"]


def test_future_blank_and_case_variants():
    df = pd.DataFrame({"d": pd.to_datetime(["2999-01-01", "2020-01-01"]), "city": ["Hanoi", "hanoi "], "s": [" ", "x"]})
    q = frame_quality(df)["columns"]
    assert q["d"]["future_dates"] == 1 and q["city"]["case_variants"] == 1 and q["s"]["blank_strings"] == 1


def test_empty_frame_and_column_cap():
    assert frame_quality(pd.DataFrame()) == {"rows": 0, "duplicate_rows": 0, "columns": {}}
    assert frame_quality(None)["rows"] == 0
    wide = pd.DataFrame({f"c{i}": [1] for i in range(45)})
    q = frame_quality(wide)
    assert len(q["columns"]) == 40 and q["columns_skipped"] == 5


def test_findings_thresholds_and_text():
    q = {"rows": 200, "duplicate_rows": 0, "columns": {
        "a": {"missing_pct": 30.0, "distinct": 3, "constant": False},
        "b": {"missing_pct": 6.0, "distinct": 3, "constant": False},
        "c": {"missing_pct": 0.5, "distinct": 3, "constant": False},
        "d": {"missing_pct": 0.0, "distinct": 1, "constant": True, "outliers": 2}}}
    f = findings_from_frame(q)
    assert [(x.column, x.severity) for x in f] == [("a", "high"), ("b", "medium"), ("c", "low"), ("d", "low"), ("d", "low")]
    text = quality_text(q, f, sample_based=True)
    assert text.startswith("Rows: 200 - checks below are based on this sample only")
    assert "[HIGH] a: 30.0% missing" in text and "- d: 0.0% missing, 1 distinct, 2 outliers, constant" in text
    assert "Findings: none" in quality_text({"rows": 1, "columns": {}}, [])


# --------------------------------------------------------------- SQL builders


def test_column_profile_sql_kinds_and_chunks():
    cols = [{"name": "emp_no", "type": "INTEGER"}, {"name": "name", "type": "VARCHAR(20)"}, {"name": "to_date", "type": "DATE"}]
    (sql,) = column_profile_sql("t", cols, Q)
    assert sql.startswith("SELECT COUNT(*) AS `rows`") and sql.endswith("FROM `t`")
    assert "SUM(`emp_no` < 0) AS `emp_no__negatives`" in sql and "AVG(`emp_no`) AS `emp_no__avg`" in sql
    assert "SUM(TRIM(`name`) = '') AS `name__blank`" in sql
    assert "SUM(`to_date` > CURRENT_DATE()) AS `to_date__future`" in sql
    many = [{"name": f"c{i}", "type": "INT"} for i in range(30)]
    chunks = column_profile_sql("t", many, Q)
    assert len(chunks) == 3 and all(c.startswith("SELECT") for c in chunks)
    assert "`c11__nonnull`" in chunks[0] and "`c12__nonnull`" in chunks[1] and COLS_PER_QUERY == 12


def test_fk_orphan_sql_composite():
    sql = fk_orphan_sql("t", {"columns": ["a", "b"], "ref_table": "r", "ref_columns": ["x", "y"]}, Q)
    assert sql == ("SELECT COUNT(*) AS `orphans` FROM `t` LEFT JOIN `r` r ON `t`.`a` = r.`x` AND `t`.`b` = r.`y` "
                   "WHERE `t`.`a` IS NOT NULL AND `t`.`b` IS NOT NULL AND r.`x` IS NULL")


def test_duplicate_rows_sql_limits():
    assert duplicate_rows_sql("t", ["a", "b"], Q) == \
        "SELECT COUNT(*) - COUNT(DISTINCT CONCAT_WS('|', COALESCE(`a`, ''), COALESCE(`b`, ''))) AS `duplicates` FROM `t`"
    assert duplicate_rows_sql("t", [f"c{i}" for i in range(11)], Q) is None
    assert duplicate_rows_sql("t", [], Q) is None
    assert sample_sql("t", Q, 7) == "SELECT * FROM `t` LIMIT 7"


# --------------------------------------------------------------- profile -> findings


COLS = [{"name": "emp_no", "type": "INT", "nullable": False}, {"name": "name", "type": "VARCHAR(20)", "nullable": True},
        {"name": "to_date", "type": "DATE", "nullable": True}]


def _rows():
    return [{"rows": 10, "emp_no__nonnull": 10, "emp_no__distinct": 10, "emp_no__min": 1, "emp_no__max": 10,
             "emp_no__negatives": Decimal("0"), "emp_no__avg": Decimal("5.5")},
            {"rows": 10, "name__nonnull": 8, "name__distinct": 8, "name__min": "a", "name__max": "z", "name__blank": 1,
             "to_date__nonnull": 10, "to_date__distinct": 2, "to_date__min": "2020-01-01", "to_date__max": "9999-01-01",
             "to_date__future": 7}]


def test_profile_from_rows_merges_chunks():
    prof = profile_from_rows(COLS, _rows())
    assert prof["rows"] == 10
    assert prof["columns"]["emp_no"] == {"nonnull": 10, "missing_pct": 0.0, "distinct": 10, "min": 1, "max": 10,
                                         "avg": 5.5, "negatives": 0}
    assert prof["columns"]["name"]["missing_pct"] == 20.0 and prof["columns"]["name"]["blank"] == 1
    assert profile_from_rows(COLS, [])["rows"] == 0


def test_findings_from_profile_rules():
    prof = profile_from_rows(COLS, _rows())
    f = findings_from_profile(prof, {"emp_no -> employees": 3}, 2, COLS)
    got = {(x.column, x.issue): x.severity for x in f}
    assert got[("emp_no -> employees", "orphan foreign-key values")] == "high"
    assert got[("(table)", "duplicate rows (no primary key)")] == "high"
    assert got[("name", "20.0% missing")] == "medium" and got[("name", "blank strings")] == "low"
    assert got[("to_date", "open-ended sentinel date")] == "low"
    assert ("to_date", "dates in the future") not in got
    # a real future date (not a sentinel) is medium
    prof["columns"]["to_date"]["max"] = "2031-01-01"
    assert {(x.column, x.issue): x.severity for x in findings_from_profile(prof, {}, None, COLS)}[
        ("to_date", "dates in the future")] == "medium"


def test_profile_table_text_sheet():
    info = TableInfo("salaries", COLS, ["emp_no"], [{"columns": ["emp_no"], "ref_table": "employees", "ref_columns": ["emp_no"]}],
                     row_count=10, comment="pay history")
    prof = profile_from_rows(COLS, _rows())
    findings = [Finding("low", "to_date", "open-ended sentinel date", "max 9999-01-01")]
    sample = {"rows": 5, "columns": {"emp_no": {"outliers": 1}, "name": {}}}
    text = profile_table_text(info, prof, findings, sample, sample_rows=5)
    assert text.startswith("Table salaries (10 rows).") and "Comment: pay history" in text
    assert "Primary key: emp_no" in text and "Foreign keys: emp_no -> employees(emp_no)" in text
    assert "- emp_no (INT, not null): 10 non-null (0.0% missing), 10 distinct, min 1, max 10, avg 5.5" in text
    assert "- [LOW] to_date: open-ended sentinel date (max 9999-01-01)" in text
    assert "Sample checks (5 rows):" in text and "- emp_no: 1 outliers" in text
