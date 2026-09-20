"""Hermetic tests for agent/powerbi.py (no MySQL/Ollama/Power BI)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import powerbi  # noqa: E402
from agent.export import export_filenames  # noqa: E402
from agent.powerbi import (  # noqa: E402
    Column,
    chart_visual,
    columns_of,
    dax_value,
    effective_source,
    m_str,
    mysql_source,
    nice_bin_size,
    tmdl_name,
    to_pbip_bytes,
)

URL = "mysql+pymysql://agent_ro:s3cret-pw@db.local:3307/shop"


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "city": ["Hanoi", "Da Nang", "HCM", "Hue"],
        "revenue": [10.5, 3.0, None, 7.25],
        "orders": [3, 1, 2, 5],
        "day": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]),
        "vip": [True, False, True, False],
    })


def _spec(**kw) -> dict:
    base = {"type": "bar", "x": "city", "y": "revenue", "color": "", "aggregate": "none", "title": "t", "why": ""}
    base.update(kw)
    return base


def _unzip(b: bytes) -> dict[str, str]:
    zf = zipfile.ZipFile(io.BytesIO(b))
    return {n: zf.read(n).decode("utf-8") for n in zf.namelist()}


# --------------------------------------------------------------- helpers


def test_export_filenames_has_pbip():
    assert export_filenames("Doanh thu theo tháng?")["pbip"] == "doanh-thu-theo-thang-powerbi.zip"


def test_mysql_source_drops_credentials():
    assert mysql_source(URL) == ("db.local:3307", "shop")
    assert mysql_source("mysql+pymysql://u:p@localhost/x") == ("localhost", "x")


def test_column_kinds():
    df = _df()
    df["dec"] = [Decimal("1.5"), Decimal("2"), None, Decimal("0")]
    kinds = {c.name: c.kind for c in columns_of(df)}
    assert kinds == {"city": "string", "revenue": "double", "orders": "int", "day": "dateTime",
                     "vip": "boolean", "dec": "double"}


def test_columns_of_dedupes_and_reserves_row_id():
    df = pd.DataFrame([[1, 2, 3]], columns=["row_id", "a", "A"])
    assert [c.name for c in columns_of(df)] == ["row_id_2", "a", "A_2"]


@pytest.mark.parametrize("name, quoted", [("city", "city"), ("total spent", "'total spent'"),
                                          ("it's", "'it''s'"), ("1st", "'1st'")])
def test_tmdl_name_quoting(name, quoted):
    assert tmdl_name(name) == quoted


def test_m_str_escapes_quotes_and_newlines():
    assert m_str('SELECT "a"\nFROM t\t-- #(x)') == '"SELECT ""a""#(lf)FROM t#(tab)-- #(#)(x)"'


def test_dax_values():
    assert dax_value(None, Column("a", "double")) == "BLANK()"
    assert dax_value(float("nan"), Column("a", "double")) == "BLANK()"
    assert dax_value(3, Column("a", "int")) == "3"
    assert dax_value(True, Column("a", "boolean")) == "TRUE"
    assert dax_value('say "hi"', Column("a", "string")) == '"say ""hi"""'
    assert dax_value(pd.Timestamp("2025-01-02 03:04:05"), Column("a", "dateTime")) == '"2025-01-02 03:04:05"'


def test_dax_value_stray_string_in_numeric_column_is_blank():
    assert dax_value("n/a", Column("a", "double")) == "BLANK()"
    assert dax_value(float("inf"), Column("a", "double")) == "BLANK()"
    assert dax_value(2.0, Column("a", "int")) == "2"


def test_nice_bin_size():
    assert nice_bin_size(0, 30) == 1
    assert nice_bin_size(0, 100) == 5
    assert nice_bin_size(0, 0.6) == 0.02
    assert nice_bin_size(5, 5) is None


def test_effective_source_forces_inline_for_ml():
    assert effective_source("live", has_ml=False) == ("live", None)
    assert effective_source("live", has_ml=True) == ("inline", powerbi.PBIP_ML_INLINE)
    assert effective_source("inline", has_ml=True) == ("inline", None)
    assert effective_source("bogus", has_ml=False) == ("live", None)


# --------------------------------------------------------------- spec -> visual


def _cols():
    df = _df()
    cols = {c.name: c for c in columns_of(df)}
    cols[powerbi.ROW_ID] = Column(powerbi.ROW_ID, "int", hidden=True)
    return df, cols


def _roles(v: dict) -> dict:
    return v["visual"]["query"]["queryState"]


def test_bar_maps_to_column_chart_with_sum_and_sort():
    df, cols = _cols()
    v = chart_visual(_spec(), cols, [], df, {})
    assert v["visual"]["visualType"] == "clusteredColumnChart"
    r = _roles(v)
    assert r["Category"]["projections"][0]["field"]["Column"]["Property"] == "city"
    y = r["Y"]["projections"][0]
    assert y["field"]["Aggregation"]["Function"] == powerbi.AGG_SUM
    assert y["queryRef"] == "Sum(Result.revenue)"
    assert v["visual"]["query"]["sortDefinition"]["sort"][0]["direction"] == "Descending"
    assert "'t'" in json.dumps(v["visual"]["visualContainerObjects"])


def test_bar_many_categories_is_horizontal():
    df = pd.DataFrame({"city": [f"c{i}" for i in range(12)], "n": range(12)})
    cols = {c.name: c for c in columns_of(df)}
    v = chart_visual(_spec(x="city", y="n"), cols, [], df, {})
    assert v["visual"]["visualType"] == "clusteredBarChart"


def test_bar_count_uses_row_id_count():
    df, cols = _cols()
    v = chart_visual(_spec(y="count", aggregate="count"), cols, [], df, {})
    y = _roles(v)["Y"]["projections"][0]
    assert y["field"]["Aggregation"]["Function"] == powerbi.AGG_COUNT
    assert y["field"]["Aggregation"]["Expression"]["Column"]["Property"] == powerbi.ROW_ID


def test_line_mean_with_color_series():
    df, cols = _cols()
    v = chart_visual(_spec(type="line", x="day", y="revenue", color="vip", aggregate="mean"), cols, [], df, {})
    assert v["visual"]["visualType"] == "lineChart"
    r = _roles(v)
    assert r["Y"]["projections"][0]["field"]["Aggregation"]["Function"] == powerbi.AGG_AVG
    assert r["Series"]["projections"][0]["field"]["Column"]["Property"] == "vip"


def test_scatter_uses_row_id_as_detail():
    df, cols = _cols()
    v = chart_visual(_spec(type="scatter", x="orders", y="revenue"), cols, [], df, {})
    r = _roles(v)
    assert v["visual"]["visualType"] == "scatterChart"
    assert r["Category"]["projections"][0]["field"]["Column"]["Property"] == powerbi.ROW_ID
    assert set(r) == {"Category", "X", "Y"}


def test_histogram_adds_bin_column_once():
    df, cols = _cols()
    calc: list = []
    v1 = chart_visual(_spec(type="histogram", x="revenue", y=""), cols, calc, df, {})
    v2 = chart_visual(_spec(type="histogram", x="revenue", y=""), cols, calc, df, {})
    assert v1 and v2 and len(calc) == 1
    name, dax, kind = calc[0]
    assert name == "revenue (bin)" and dax.startswith("INT([revenue] / ") and kind == "double"
    assert _roles(v1)["Category"]["projections"][0]["field"]["Column"]["Property"] == name


def test_histogram_constant_column_is_skipped():
    df = pd.DataFrame({"v": [2.0, 2.0]})
    cols = {c.name: c for c in columns_of(df)}
    assert chart_visual(_spec(type="histogram", x="v", y=""), cols, [], df, {}) is None


def test_box_falls_back_to_avg_min_max():
    df, cols = _cols()
    v = chart_visual(_spec(type="box", x="city", y="revenue"), cols, [], df, {})
    fns = [p["field"]["Aggregation"]["Function"] for p in _roles(v)["Y"]["projections"]]
    assert fns == [powerbi.AGG_AVG, powerbi.AGG_MIN, powerbi.AGG_MAX]


def test_unknown_column_or_type_returns_none():
    df, cols = _cols()
    assert chart_visual(_spec(x="nope"), cols, [], df, {}) is None
    assert chart_visual(_spec(type="pie"), cols, [], df, {}) is None


# --------------------------------------------------------------- whole project


def test_to_pbip_bytes_live_layout_and_no_secrets():
    b, note = to_pbip_bytes("Top cities by revenue", "**Hanoi** leads.", 'SELECT * FROM t WHERE a = "x"', _df(),
                            [_spec(), _spec(type="histogram", x="revenue", y="")], source="live", database_url=URL)
    assert note is None
    files = _unzip(b)
    root = "top-cities-by-revenue"
    assert f"{root}/{root}.pbip" in files
    assert f"{root}/{root}.Report/definition.pbir" in files
    assert f"{root}/{root}.SemanticModel/definition/model.tmdl" in files
    assert f"{root}/README.txt" in files
    whole = "\n".join(files.values())
    assert "s3cret-pw" not in whole and "agent_ro" not in whole
    tmdl = files[f"{root}/{root}.SemanticModel/definition/tables/Result.tmdl"]
    assert 'MySQL.Database("db.local:3307", "shop", [Query = "SELECT * FROM t WHERE a = ""x"""])' in tmdl
    assert "partition Result = m" in tmdl
    assert "column 'revenue (bin)' = INT([revenue] / " in tmdl
    # every JSON file parses and references the schema family; the report points at the model
    for path, text in files.items():
        if path.endswith((".json", ".pbip", ".pbir", ".pbism", ".platform")):
            obj = json.loads(text)
            assert obj["$schema"].startswith(powerbi.SCHEMA)
    pbir = json.loads(files[f"{root}/{root}.Report/definition.pbir"])
    assert pbir["datasetReference"]["byPath"]["path"] == f"../{root}.SemanticModel"
    visuals = [json.loads(t) for p, t in files.items() if p.endswith("visual.json")]
    types = sorted(v["visual"]["visualType"] for v in visuals)
    assert types == ["card", "clusteredColumnChart", "clusteredColumnChart", "tableEx", "textbox", "textbox", "textbox"]
    for v in visuals:                          # visuals must stay on the page
        p = v["position"]
        assert 0 <= p["x"] and p["x"] + p["width"] <= powerbi.PAGE_W
        assert 0 <= p["y"] and p["y"] + p["height"] <= powerbi.PAGE_H
    pages = json.loads(files[f"{root}/{root}.Report/definition/pages/pages.json"])
    assert pages["pageOrder"] == ["page1", "page2"]


def test_to_pbip_bytes_inline_embeds_rows_and_extra_tables():
    stats = {"summary": pd.DataFrame({"stat": ["mean", "sd"], "value": [4.5, 1.2]}),
             "empty": pd.DataFrame()}
    b, note = to_pbip_bytes("q", "", "SELECT 1", _df(), [], extra_tables=stats, source="inline", database_url=URL)
    assert note is None
    files = _unzip(b)
    tmdl = files["q/q.SemanticModel/definition/tables/Result.tmdl"]
    assert "partition Result = calculated" in tmdl
    assert '{1, "Hanoi", 10.5, 3, "2025-01-01 00:00:00", TRUE}' in tmdl
    assert '{3, "HCM", BLANK(), 2, "2025-03-01 00:00:00", TRUE}' in tmdl
    assert "MySQL.Database" not in tmdl
    assert "q/q.SemanticModel/definition/tables/summary.tmdl" in files
    assert "q/q.SemanticModel/definition/tables/empty.tmdl" not in files
    model = files["q/q.SemanticModel/definition/model.tmdl"]
    assert 'annotation PBI_QueryOrder = ["Result", "summary"]' in model
    assert "ref table summary" in model


def test_extra_table_name_cannot_escape_tables_folder():
    b, _ = to_pbip_bytes("q", "", "SELECT 1", _df(), [], extra_tables={"../evil/x": pd.DataFrame({"a": [1]})},
                         source="inline", database_url=URL)
    paths = list(_unzip(b))
    assert "q/q.SemanticModel/definition/tables/_evil_x.tmdl" in paths
    assert not any("evil/" in p or "/../" in p for p in paths)


def test_sql_textbox_has_one_paragraph_per_line():
    b, _ = to_pbip_bytes("q", "", "SELECT a\nFROM t\nLIMIT 5", _df(), [], source="inline", database_url=URL)
    files = _unzip(b)
    boxes = [json.loads(t) for p, t in files.items() if "/page2/" in p and p.endswith("visual.json")]
    sql_box = next(v for v in boxes if v["visual"]["visualType"] == "textbox")
    paras = sql_box["visual"]["objects"]["general"][0]["properties"]["paragraphs"]
    assert [p["textRuns"][0]["value"] for p in paras] == ["SQL", "SELECT a", "FROM t", "LIMIT 5"]


def test_to_pbip_bytes_ml_forces_inline_with_note():
    b, note = to_pbip_bytes("q", "", "SELECT 1", _df(), [], source="live", has_ml=True, database_url=URL)
    assert note == powerbi.PBIP_ML_INLINE
    tmdl = _unzip(b)["q/q.SemanticModel/definition/tables/Result.tmdl"]
    assert "partition Result = calculated" in tmdl and "MySQL.Database" not in tmdl


def test_to_pbip_bytes_inline_row_cap():
    df = pd.DataFrame({"n": range(50)})
    b, note = to_pbip_bytes("q", "", "SELECT 1", df, [], source="inline", database_url=URL, inline_max_rows=10)
    assert note == powerbi.PBIP_TRUNCATED.format(n=10)
    tmdl = _unzip(b)["q/q.SemanticModel/definition/tables/Result.tmdl"]
    assert "{10, 9}" in tmdl and "{11, 10}" not in tmdl


def test_to_pbip_bytes_tz_aware_datetimes():
    df = pd.DataFrame({"when": [datetime(2025, 1, 1, 12, tzinfo=timezone.utc)], "n": [1]})
    b, note = to_pbip_bytes("q", "", "SELECT 1", df, [], source="inline", database_url=URL)
    assert b is not None and note is None
    assert '"2025-01-01 12:00:00"' in _unzip(b)["q/q.SemanticModel/definition/tables/Result.tmdl"]


def test_to_pbip_bytes_empty_and_failure():
    assert to_pbip_bytes("q", "", "SELECT 1", pd.DataFrame(), [], database_url=URL) == (None, powerbi.PBIP_NO_DATA)
    assert to_pbip_bytes("q", "", "SELECT 1", None, [], database_url=URL) == (None, powerbi.PBIP_NO_DATA)
    assert to_pbip_bytes("q", "", "SELECT 1", _df(), [], source="live", database_url="not a url") == \
        (None, powerbi.PBIP_FAILED)


def test_skipped_chart_and_box_notes_land_in_readme():
    b, _ = to_pbip_bytes("q", "", "SELECT 1", _df(), [_spec(type="pie"), _spec(type="box", x="city", y="revenue")],
                         source="inline", database_url=URL)
    readme = _unzip(b)["q/README.txt"]
    assert "skipped: pie" in readme and "Box plots are shown as mean / min / max" in readme
    assert "Embedded rows: 4 rows" in readme
