"""Hermetic tests for agent/export.py (no MySQL/Ollama)."""
from __future__ import annotations

import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import export  # noqa: E402
from agent.export import (  # noqa: E402
    export_filenames,
    slugify,
    to_csv_bytes,
)


# --------------------------------------------------------------------- Task 1
def test_slugify_vietnamese():
    assert slugify("Doanh thu theo tháng?") == "doanh-thu-theo-thang"


@pytest.mark.parametrize("text", ["", "   ", "???"])
def test_slugify_default_on_empty(text):
    assert slugify(text) == "result"


def test_slugify_length_cap_no_trailing_dash():
    s = slugify("a " * 100)
    assert len(s) <= 60
    assert not s.endswith("-")


def test_slugify_path_safe_charset():
    s = slugify("Doanh thu theo tháng?!@#$%^&*()")
    assert re.fullmatch(r"[a-z0-9-]+", s)


def test_export_filenames():
    names = export_filenames("Doanh thu theo tháng?")
    assert names["csv"] == "doanh-thu-theo-thang.csv"
    assert names["xlsx"] == "doanh-thu-theo-thang.xlsx"
    assert names["png"].format(i=1) == "doanh-thu-theo-thang-chart-1.png"


def test_to_csv_bytes_bom_and_roundtrip():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    b = to_csv_bytes(df)
    assert b.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(io.BytesIO(b))
    pd.testing.assert_frame_equal(back, df)


def test_to_csv_bytes_empty_header_only():
    df = pd.DataFrame(columns=["a", "b"])
    b = to_csv_bytes(df)
    text_out = b.decode("utf-8-sig")
    assert text_out.strip() == "a,b"


# --------------------------------------------------------------------- Task 2
def test_to_xlsx_bytes_roundtrip_basic_types():
    df = pd.DataFrame({"i": [1, 2, 3], "f": [1.5, 2.5, 3.5], "s": ["a", "b", "c"]})
    b, truncated = export.to_xlsx_bytes(df)
    assert truncated is False
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert list(back.columns) == list(df.columns)
    pd.testing.assert_frame_equal(back, df)


def test_to_xlsx_bytes_tz_aware_datetime_column():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=3, tz="Asia/Ho_Chi_Minh"),
        "v": [1, 2, 3],
    })
    before = df.copy(deep=True)
    b, truncated = export.to_xlsx_bytes(df)
    assert truncated is False
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert back["ts"].dt.tz is None
    pd.testing.assert_frame_equal(df, before)


def test_to_xlsx_bytes_object_column_with_tz_aware_datetimes():
    df = pd.DataFrame({
        "ts": [datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)],
        "v": [1, 2],
    })
    before = df.copy(deep=True)
    b, truncated = export.to_xlsx_bytes(df)
    assert truncated is False
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert len(back) == 2
    pd.testing.assert_frame_equal(df, before)


def test_to_xlsx_bytes_empty_frame_header_only():
    df = pd.DataFrame(columns=["a", "b"])
    b, truncated = export.to_xlsx_bytes(df)
    assert truncated is False
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert list(back.columns) == ["a", "b"]
    assert len(back) == 0


def test_to_xlsx_bytes_row_cap(monkeypatch):
    monkeypatch.setattr(export, "EXCEL_MAX_ROWS", 100)
    df = pd.DataFrame({"a": range(150)})
    b, truncated = export.to_xlsx_bytes(df)
    assert truncated is True
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert len(back) == 100


def test_to_xlsx_bytes_extra_sheets_order():
    df = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"b": [2]})
    df3 = pd.DataFrame({"c": [3]})
    b, truncated = export.to_xlsx_bytes(df, extra_sheets={"anova": df2, "regression": df3})
    assert truncated is False
    sheets = pd.read_excel(io.BytesIO(b), engine="openpyxl", sheet_name=None)
    assert list(sheets.keys()) == ["data", "anova", "regression"]


def test_to_xlsx_bytes_sheet_name_sanitising():
    df = pd.DataFrame({"a": [1]})
    long_key = "x" * 40
    extra = {"t-test: a/b [x]": pd.DataFrame({"b": [1]}), long_key: pd.DataFrame({"c": [1]})}
    b, truncated = export.to_xlsx_bytes(df, extra_sheets=extra)
    sheets = pd.read_excel(io.BytesIO(b), engine="openpyxl", sheet_name=None)
    names = list(sheets.keys())
    assert names[0] == "data"
    for n in names[1:]:
        assert len(n) <= 31
        assert not re.search(r"[" + re.escape("[]:*?/\\") + r"]", n)


def test_to_xlsx_bytes_duplicate_sanitised_names_get_suffix():
    df = pd.DataFrame({"a": [1]})
    extra = {"a/b": pd.DataFrame({"x": [1]}), "a:b": pd.DataFrame({"y": [2]})}
    b, truncated = export.to_xlsx_bytes(df, extra_sheets=extra)
    sheets = pd.read_excel(io.BytesIO(b), engine="openpyxl", sheet_name=None)
    names = list(sheets.keys())
    assert names[0] == "data"
    assert len(names) == 3
    assert len(set(names)) == 3


# --------------------------------------------------------------------- Task 4
altair = pytest.importorskip("altair")


def test_xlsx_strips_illegal_control_characters():
    df = pd.DataFrame({"note": ["ok", "bad\x00\x1fvalue", None], "n": [1, 2, 3]})
    before = df.copy(deep=True)
    b, truncated = export.to_xlsx_bytes(df)
    back = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    assert list(back["note"].fillna("")) == ["ok", "badvalue", ""] and truncated is False
    pd.testing.assert_frame_equal(df, before)       # input untouched


def test_chart_to_png_basic():
    pytest.importorskip("vl_convert")
    chart = altair.Chart(pd.DataFrame({"a": [1, 2], "b": [3, 4]})).mark_bar().encode(x="a:O", y="b:Q")
    png, reason = export.chart_to_png(chart)
    assert reason is None and png is not None
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_chart_to_png_with_configure():
    pytest.importorskip("vl_convert")
    chart = (
        altair.Chart(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        .mark_bar()
        .encode(x="a:O", y="b:Q")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    png, reason = export.chart_to_png(chart)
    assert reason is None and png is not None
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_chart_to_png_returns_none_without_vl_convert(monkeypatch):
    monkeypatch.setitem(sys.modules, "vl_convert", None)
    chart = altair.Chart(pd.DataFrame({"a": [1, 2], "b": [3, 4]})).mark_bar().encode(x="a:O", y="b:Q")
    assert export.chart_to_png(chart) == (None, export.PNG_NOT_INSTALLED)


def test_chart_to_png_returns_none_on_exception(monkeypatch):
    vlc = pytest.importorskip("vl_convert")

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(vlc, "vegalite_to_png", _boom)
    chart = altair.Chart(pd.DataFrame({"a": [1, 2], "b": [3, 4]})).mark_bar().encode(x="a:O", y="b:Q")
    assert export.chart_to_png(chart) == (None, export.PNG_FAILED)


def test_chart_to_png_large_dataframe_short_circuits(monkeypatch):
    vlc = pytest.importorskip("vl_convert")

    def _fail(*a, **k):
        raise AssertionError("vegalite_to_png should not be called for oversized data")

    monkeypatch.setattr(vlc, "vegalite_to_png", _fail)
    big = pd.DataFrame({"a": range(export.MAX_PNG_ROWS + 1), "b": range(export.MAX_PNG_ROWS + 1)})
    chart = altair.Chart(big).mark_bar().encode(x="a:O", y="b:Q")
    assert export.chart_to_png(chart) == (None, export.PNG_TOO_LARGE)
