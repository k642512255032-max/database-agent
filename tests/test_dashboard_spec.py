"""Hermetic tests for dashboard/spec.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.spec import DEFAULT_ACCENT, NO_WIDGETS, DashboardSpec, Widget, spec_from_llm  # noqa: E402


def _w(**kw) -> dict:
    base = {"title": "Revenue", "kind": "bar", "question": "Revenue per city", "x": "city", "y": "revenue",
            "width": "medium", "note": ""}
    base.update(kw)
    return base


def test_spec_from_llm_happy_path_assigns_ids_and_widths():
    spec, reason = spec_from_llm({"title": "Sales", "subtitle": "2025", "accent": "#112233",
                                  "widgets": [_w(kind="kpi", width="small"), _w(), _w(kind="table", width="full")]}, 8)
    assert reason == "" and spec is not None
    assert [w.id for w in spec.widgets] == ["w1", "w2", "w3"]
    assert [w.cols for w in spec.widgets] == [4, 6, 12]
    assert spec.accent == "#112233" and spec.title == "Sales"


def test_spec_from_llm_drops_bad_widgets():
    spec, reason = spec_from_llm({"title": "T", "widgets": [_w(kind="gauge"), _w(question=""), _w(title="ok")]}, 8)
    assert reason == "" and [w.title for w in spec.widgets] == ["ok"]


def test_spec_from_llm_caps_widgets():
    spec, _ = spec_from_llm({"title": "T", "widgets": [_w(title=f"w{i}") for i in range(12)]}, 8)
    assert len(spec.widgets) == 8


def test_spec_from_llm_bad_accent_and_width_fall_back():
    spec, _ = spec_from_llm({"title": "T", "accent": "red", "widgets": [_w(width="huge")]}, 8)
    assert spec.accent == DEFAULT_ACCENT and spec.widgets[0].width == "medium"


def test_spec_from_llm_nothing_usable():
    spec, reason = spec_from_llm({"title": "T", "widgets": [_w(kind="gauge", title="G"), "junk"]}, 8)
    assert spec is None and reason.startswith(NO_WIDGETS) and "G (gauge)" in reason
    assert spec_from_llm({}, 8) == (None, NO_WIDGETS)


def test_spec_round_trip_and_tolerant_from_dict():
    spec, _ = spec_from_llm({"title": "T", "widgets": [_w(), _w(kind="line", title="Trend")]}, 8)
    assert DashboardSpec.from_dict(spec.to_dict()) == spec
    loose = DashboardSpec.from_dict({"widgets": [{"id": "w1", "title": "x", "kind": "kpi", "question": "q"}]})
    assert loose.title == "Dashboard" and loose.widgets == [Widget("w1", "x", "kpi", "q")]
