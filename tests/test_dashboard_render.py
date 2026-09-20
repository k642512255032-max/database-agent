"""Hermetic tests for dashboard/render.py (no MySQL/Ollama/network)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import render  # noqa: E402
from dashboard.render import BUNDLE_FILES, _json_safe, bind_widget, bundle_zip, inline_preview, render_bundle  # noqa: E402
from dashboard.spec import DashboardSpec, Widget  # noqa: E402


def _w(kind="bar", x="", y="", width="medium") -> Widget:
    return Widget("w1", "Title", kind, "question", x=x, y=y, width=width)


# --------------------------------------------------------------- _json_safe


def test_json_safe_scalars():
    assert _json_safe(np.float64(2.5)) == 2.5 and type(_json_safe(np.float64(2.5))) is float
    assert _json_safe(np.int64(3)) == 3 and type(_json_safe(np.int64(3))) is int
    assert _json_safe(float("nan")) is None and _json_safe(pd.NaT) is None and _json_safe(pd.NA) is None
    assert _json_safe(pd.Timestamp("2025-01-01")) == "2025-01-01T00:00:00"
    assert _json_safe(Decimal("1.5")) == 1.5
    assert _json_safe(True) is True and _json_safe("s") == "s" and _json_safe(None) is None


# --------------------------------------------------------------- bind_widget


def test_kpi_single_cell():
    out = bind_widget(_w("kpi"), pd.DataFrame({"total": [42.5]}), None, 500)
    assert out["value"] == 42.5 and out["label"] == "total" and out["width"] == 6


def test_kpi_many_rows_counts_rows():
    out = bind_widget(_w("kpi"), pd.DataFrame({"a": range(10)}), None, 500)
    assert out["value"] == 10 and out["label"] == "rows"


def test_bar_groups_duplicates_sorts_desc_and_caps():
    df = pd.DataFrame({"city": [f"c{i % 25}" for i in range(50)], "n": [1] * 50})
    out = bind_widget(_w("bar", x="city", y="n"), df, None, 500)
    assert len(out["labels"]) == 20 and out["values"][0] == 2 and out["series_label"] == "n"


def test_bar_hint_columns_missing_falls_back():
    out = bind_widget(_w("bar", x="nope", y="nope"), pd.DataFrame({"city": ["a", "b"], "n": [1, 2]}), None, 500)
    assert out["labels"] == ["b", "a"] and out["values"] == [2, 1]


def test_pie_caps_eight():
    df = pd.DataFrame({"k": [f"k{i}" for i in range(12)], "v": range(12)})
    assert len(bind_widget(_w("pie"), df, None, 500)["labels"]) == 8


def test_line_sorts_by_date_iso_labels():
    df = pd.DataFrame({"month": pd.to_datetime(["2025-03-01", "2025-01-01", "2025-02-01"]), "revenue": [3.0, 1.0, 2.0]})
    out = bind_widget(_w("line"), df, None, 500)
    assert out["labels"] == ["2025-01-01", "2025-02-01", "2025-03-01"] and out["values"] == [1.0, 2.0, 3.0]


def test_line_with_string_dates_parsed():
    df = pd.DataFrame({"day": ["2025-02-01", "2025-01-01"], "n": [2, 1]})
    assert bind_widget(_w("line"), df, None, 500)["labels"] == ["2025-01-01", "2025-02-01"]


def test_no_numeric_column_counts_categories():
    out = bind_widget(_w("bar"), pd.DataFrame({"status": ["a", "b", "a"]}), None, 500)
    assert out["labels"] == ["a", "b"] and out["values"] == [2, 1] and out["series_label"] == "count"


def test_table_caps_rows_and_serialises():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "d": pd.to_datetime(["2025-01-01"] * 3)})
    out = bind_widget(_w("table"), df, None, 2)
    assert out["columns"] == ["a", "b", "d"] and out["rows"] == [[1, "x", "2025-01-01"], [2, "y", "2025-01-01"]]
    assert out["total_rows"] == 3


def test_empty_and_error_passthrough():
    assert bind_widget(_w(), pd.DataFrame(), None, 500)["empty"] is True
    assert bind_widget(_w(), None, None, 500)["empty"] is True
    out = bind_widget(_w(), pd.DataFrame({"a": [1]}), "boom", 500)
    assert out["error"] == "boom" and "labels" not in out


def test_bind_never_raises(monkeypatch):
    def boom(*a):
        raise ValueError("bad")
    monkeypatch.setattr(render, "_bind_series", boom)
    out = bind_widget(_w(), pd.DataFrame({"a": [1]}), None, 500)
    assert out["error"].startswith("Could not use the result")


# --------------------------------------------------------------- bundle


def _bundle():
    spec = DashboardSpec('T <b>&"', "sub", widgets=[_w("table")])
    bound = [bind_widget(_w("table"), pd.DataFrame({"a": [1], "b": ["</script><img onerror=x>"]}), None, 500)]
    return render_bundle(spec, bound, "2026-09-20 10:00")


def test_render_bundle_files_and_order():
    b = _bundle()
    assert tuple(b) == BUNDLE_FILES
    html = b["index.html"]
    order = [html.index(s) for s in ('href="style.css"', 'src="chart.umd.js"', 'src="data.js"', 'src="app.js"')]
    assert order == sorted(order)
    assert '<meta charset="utf-8">' in html and "Chart.js v4.4.7" in b["chart.umd.js"]


def test_render_bundle_escapes_title_and_data():
    b = _bundle()
    assert "T &lt;b&gt;&amp;&quot;" in b["index.html"] and "<b>" not in b["index.html"]
    assert "</script>" not in b["data.js"] and "<\\/script>" in b["data.js"]
    payload = json.loads(b["data.js"][len("window.DASHBOARD_DATA = "):].rstrip(";\n"))
    assert payload["widgets"][0]["rows"][0][1] == "</script><img onerror=x>"


def test_inline_preview_is_self_contained():
    p = inline_preview(_bundle())
    assert "<link" not in p and "<script src" not in p
    assert "window.DASHBOARD_DATA" in p and "--accent:" in p and "new Chart(" in p


def test_bundle_zip_files_at_root():
    names = zipfile.ZipFile(io.BytesIO(bundle_zip(_bundle()))).namelist()
    assert names == list(BUNDLE_FILES)


def test_missing_chartjs_is_clear(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "CHART_JS", tmp_path / "nope.js")
    with pytest.raises(RuntimeError, match="chart.umd.js is missing"):
        render.chart_js()
