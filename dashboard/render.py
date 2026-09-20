"""Dashboard renderer: bind result rows to widgets and write the static bundle.

Pure and Streamlit-free. bind_widget() reduces a DataFrame to the small JSON a widget needs
(one number, labels + values, or table rows); render_bundle() writes index.html / style.css /
app.js / data.js from string.Template files and adds the vendored Chart.js. inline_preview()
folds the bundle into one HTML document for st.components.v1.html (an iframe without an
origin, so nothing can be fetched); bundle_zip() packs the files at the zip root for Netlify.
"""
from __future__ import annotations

import html
import io
import json
import logging
import math
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from string import Template

import pandas as pd

from agent.export import _clean_frame

from .spec import DashboardSpec, Widget

log = logging.getLogger(__name__)

BUNDLE_FILES = ("index.html", "style.css", "chart.umd.js", "data.js", "app.js")
MAX_CATEGORIES = {"bar": 20, "pie": 8}
MAX_POINTS = 500
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # app.py
CHART_JS = Path(__file__).parent / "assets" / "chart.umd.js"
CHART_JS_MISSING = ("dashboard/assets/chart.umd.js is missing. Vendor it once with: curl -L -o "
                    "dashboard/assets/chart.umd.js https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.js")


# ====================================================================== binding
def _json_safe(v):
    """numpy / pandas scalars -> plain Python; missing -> None; dates -> ISO strings; Decimal -> float."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if hasattr(v, "item") and not isinstance(v, (pd.Timestamp, bytes)):   # numpy scalar -> python scalar
        try:
            v = v.item()
        except (TypeError, ValueError):
            pass
    if isinstance(v, (int, float)):
        return None if isinstance(v, float) and not math.isfinite(v) else v
    try:
        if pd.isna(v):               # NaT, pd.NA, numpy nan
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    return str(v)


def _label(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else v.isoformat(sep=" ")
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    s = _json_safe(v)
    return "" if s is None else str(s)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            out.append(c)
        elif s.dtype == object and len(s) and all(isinstance(v, (Decimal, int, float)) for v in s.dropna().head(50)):
            out.append(c)
    return out


def _base(w: Widget) -> dict:
    return {"id": w.id, "title": w.title, "kind": w.kind, "width": w.cols, "note": w.note}


def bind_widget(w: Widget, df: pd.DataFrame | None, error: str | None, rows_cap: int) -> dict:
    """Reduce a widget's rows to what the page needs. Never raises: problems become {"error": ...}."""
    out = _base(w)
    if error:
        out["error"] = error
        return out
    if df is None or df.empty:
        out["empty"] = True
        return out
    try:
        df = _clean_frame(df).copy()
        df.columns = [str(c) for c in df.columns]
        numeric = _numeric_columns(df)
        if w.kind == "kpi":
            out.update(_bind_kpi(w, df, numeric))
        elif w.kind == "table":
            out.update(_bind_table(df, rows_cap))
        else:
            out.update(_bind_series(w, df, numeric))
    except Exception as exc:
        log.debug("bind_widget %s failed: %s", w.id, exc, exc_info=True)
        out["error"] = f"Could not use the result for this widget: {exc}"
    return out


def _bind_kpi(w: Widget, df: pd.DataFrame, numeric: list[str]) -> dict:
    if len(df) == 1:
        col = w.y if w.y in numeric else (numeric[0] if numeric else None)
        if col is None and df.shape[1] == 1:
            col = df.columns[0]
        if col is not None:
            return {"value": _json_safe(df[col].iloc[0]), "label": col}
    return {"value": int(len(df)), "label": "rows"}


def _bind_table(df: pd.DataFrame, rows_cap: int) -> dict:
    head = df.iloc[:rows_cap]
    rows = [[_label(v) if isinstance(v, (datetime, date, pd.Timestamp)) else _json_safe(v) for v in row]
            for row in head.itertuples(index=False, name=None)]
    return {"columns": list(head.columns), "rows": rows, "total_rows": int(len(df))}


def _bind_series(w: Widget, df: pd.DataFrame, numeric: list[str]) -> dict:
    non_numeric = [c for c in df.columns if c not in numeric]
    x = w.x if w.x in df.columns else (non_numeric[0] if non_numeric else df.columns[0])
    y = w.y if (w.y in numeric and w.y != x) else next((c for c in numeric if c != x), None)
    if y is None:
        series, label = df[x].value_counts(dropna=False), "count"
    else:
        ys = pd.to_numeric(df[y], errors="coerce")
        if df[x].duplicated().any():
            series = ys.groupby(df[x].values, sort=False).sum()
        else:
            series = pd.Series(ys.values, index=df[x].values)
        label = y
    if w.kind == "line":
        idx = series.index
        if not pd.api.types.is_numeric_dtype(idx):
            parsed = pd.to_datetime(idx, errors="coerce")
            if parsed.notna().all():
                series.index = parsed
        series = series.sort_index().iloc[:MAX_POINTS]
    else:
        series = series.sort_values(ascending=False).iloc[: MAX_CATEGORIES[w.kind]]
    return {"labels": [_label(i) for i in series.index], "values": [_json_safe(v) for v in series.values],
            "series_label": label}


# ====================================================================== templates
INDEX_HTML = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="head">
  <div>
    <h1>$title</h1>
    <p class="sub">$subtitle</p>
  </div>
  <p class="meta">Generated $generated_at</p>
</header>
<main id="grid"></main>
<footer class="foot">Built by Local Data Agent - data snapshot of $generated_at</footer>
<script src="chart.umd.js"></script>
<script src="data.js"></script>
<script src="app.js"></script>
</body>
</html>
""")

STYLE_CSS = Template(""":root {
  --accent: $accent;
  --bg: #F5F7FB;
  --card: #FFFFFF;
  --text: #0F172A;
  --muted: #64748B;
  --border: #E5E9F0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; padding: 28px 24px 12px; }
.head h1 { margin: 0; font-size: 26px; font-weight: 650; }
.head .sub { margin: 4px 0 0; color: var(--muted); }
.head .meta { margin: 0; color: var(--muted); font-size: 13px; }
#grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; padding: 12px 24px 24px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; height: 320px; display: flex; flex-direction: column; min-width: 0; }
.card.kpi { height: 140px; }
.card h2 { margin: 0; font-size: 15px; font-weight: 640; }
.card .note { margin: 2px 0 8px; color: var(--muted); font-size: 12.5px; }
.card .body { flex: 1; min-height: 0; position: relative; display: flex; flex-direction: column; }
.card canvas { width: 100% !important; height: 100% !important; }
.kpi .value { font-size: 36px; font-weight: 700; color: var(--accent); line-height: 1.1; }
.kpi .label { color: var(--muted); font-size: 13px; }
.table-wrap { overflow: auto; flex: 1; min-height: 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { position: sticky; top: 0; background: var(--card); color: var(--muted); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.msg { color: var(--muted); font-size: 13.5px; margin: auto; text-align: center; }
.msg.error { color: #b42318; }
.foot { padding: 0 24px 24px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 800px) { .card { grid-column: span 12 !important; } .head { flex-direction: column; align-items: flex-start; } }
""")

APP_JS = """(function () {
  var data = window.DASHBOARD_DATA || { widgets: [] };
  var palette = PALETTE_JSON;
  var grid = document.getElementById("grid");
  var fmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function num(v) { return typeof v === "number" ? fmt.format(v) : (v === null || v === undefined ? "" : String(v)); }

  data.widgets.forEach(function (w) {
    var card = el("section", "card " + w.kind);
    card.style.gridColumn = "span " + (w.width || 6);
    card.appendChild(el("h2", null, w.title));
    if (w.note) card.appendChild(el("p", "note", w.note));
    var body = el("div", "body");
    card.appendChild(body);
    grid.appendChild(card);

    if (w.error) { body.appendChild(el("p", "msg error", w.error)); return; }
    if (w.empty) { body.appendChild(el("p", "msg", "No data")); return; }

    if (w.kind === "kpi") {
      body.appendChild(el("div", "value", num(w.value)));
      body.appendChild(el("div", "label", w.label));
      return;
    }
    if (w.kind === "table") {
      var wrap = el("div", "table-wrap"), table = el("table"), thead = el("thead"), tr = el("tr");
      w.columns.forEach(function (c) { tr.appendChild(el("th", null, c)); });
      thead.appendChild(tr); table.appendChild(thead);
      var tbody = el("tbody");
      w.rows.forEach(function (row) {
        var r = el("tr");
        row.forEach(function (v) { r.appendChild(el("td", typeof v === "number" ? "num" : null, num(v))); });
        tbody.appendChild(r);
      });
      table.appendChild(tbody); wrap.appendChild(table); body.appendChild(wrap);
      if (w.total_rows > w.rows.length) body.appendChild(el("p", "note", "Showing " + w.rows.length + " of " + w.total_rows + " rows"));
      return;
    }
    var canvas = el("canvas");
    body.appendChild(canvas);
    var type = w.kind === "pie" ? "doughnut" : w.kind;
    var colors = w.kind === "pie" ? w.labels.map(function (_, i) { return palette[i % palette.length]; }) : data.accent;
    new Chart(canvas, {
      type: type,
      data: { labels: w.labels, datasets: [{ label: w.series_label, data: w.values, backgroundColor: colors,
              borderColor: w.kind === "line" ? data.accent : colors, fill: false, tension: 0.25,
              borderWidth: w.kind === "line" ? 2 : 0 }] },
      options: { responsive: true, maintainAspectRatio: false,
                 plugins: { legend: { display: w.kind === "pie", position: "right" } },
                 scales: w.kind === "pie" ? {} : { x: { grid: { display: false } }, y: { beginAtZero: true, grid: { color: "#e5e9f0" } } } }
    });
  });
})();
""".replace("PALETTE_JSON", json.dumps(PALETTE))


def chart_js() -> str:
    if not CHART_JS.exists():
        raise RuntimeError(CHART_JS_MISSING)
    return CHART_JS.read_text(encoding="utf-8")


def render_bundle(spec: DashboardSpec, bound: list[dict], generated_at: str) -> dict[str, str]:
    """{filename: text} for BUNDLE_FILES. Everything user-controlled is HTML-escaped or JSON-encoded."""
    payload = {"title": spec.title, "subtitle": spec.subtitle, "accent": spec.accent,
               "generated_at": generated_at, "widgets": bound}
    data_js = "window.DASHBOARD_DATA = " + json.dumps(payload, ensure_ascii=False).replace("</", "<\\/") + ";\n"
    return {
        "index.html": INDEX_HTML.substitute(title=html.escape(spec.title), subtitle=html.escape(spec.subtitle),
                                            generated_at=html.escape(generated_at)),
        "style.css": STYLE_CSS.substitute(accent=spec.accent),
        "chart.umd.js": chart_js(),
        "data.js": data_js,
        "app.js": APP_JS,
    }


def inline_preview(bundle: dict[str, str]) -> str:
    """One self-contained document for the Streamlit iframe: stylesheet and scripts inlined in order."""
    doc = bundle["index.html"]
    doc = doc.replace('<link rel="stylesheet" href="style.css">', "<style>\n" + bundle["style.css"] + "\n</style>")
    for name in ("chart.umd.js", "data.js", "app.js"):
        doc = doc.replace(f'<script src="{name}"></script>', "<script>\n" + bundle[name] + "\n</script>")
    return doc


def bundle_zip(bundle: dict[str, str]) -> bytes:
    """Files at the zip root - Netlify serves index.html from there."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BUNDLE_FILES:
            zf.writestr(name, bundle[name])
    return buf.getvalue()
