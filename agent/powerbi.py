"""Power BI export: build a Power BI Project (.pbip) from an agent result.

A .pbix is a binary format only Power BI Desktop can write, so we generate the documented
text-based project format instead: a TMDL semantic model (tables, columns, measures, the M
query that hits MySQL) plus a PBIR report (pages and visuals as JSON). The folder is zipped
and the user opens the .pbip in Power BI Desktop.

Two data-source modes:
  * live   - the Result table's partition is an M query running the agent's SQL against MySQL
             (Power BI asks for credentials on first refresh; use the read-only agent_ro user).
  * inline - the rows are embedded as a DAX DATATABLE, so nothing needs to be installed and the
             file is self-contained. ML answers always use inline: predictions only exist in Python.

Pure and Streamlit-free so it can be unit tested in isolation; app.py wires it into a download button.
"""
from __future__ import annotations

import io
import json
import logging
import math
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy.engine import make_url

from agent.config import settings
from agent.export import _clean_frame, slugify

log = logging.getLogger(__name__)

RESULT_TABLE = "Result"
ROW_ID = "row_id"                 # hidden index column: lets scatter/table visuals show every row
ROW_COUNT_MEASURE = "Row count"
PAGE_W, PAGE_H = 1280, 720
HIST_BINS = 30
MAX_TEXT_CHARS = 4000             # answer / SQL text boxes

PBIP_NO_DATA = "Power BI export skipped: no rows"
PBIP_FAILED = "Power BI export failed for this result"
PBIP_TRUNCATED = "Power BI file holds the first {n:,} rows (embedded mode)"
PBIP_ML_INLINE = "ML predictions only exist here, so the rows were embedded instead of a live query"

# Power BI's aggregation function codes used in visual.json projections
AGG_SUM, AGG_AVG, AGG_COUNT, AGG_MIN, AGG_MAX = 0, 1, 2, 3, 4
_AGG_NAME = {AGG_SUM: "Sum", AGG_AVG: "Avg", AGG_COUNT: "Count", AGG_MIN: "Min", AGG_MAX: "Max"}
_AGG_LABEL = {AGG_SUM: "Sum of", AGG_AVG: "Average of", AGG_COUNT: "Count of", AGG_MIN: "Min of", AGG_MAX: "Max of"}
_SPEC_AGG = {"sum": AGG_SUM, "mean": AGG_AVG, "count": AGG_COUNT, "none": AGG_SUM}

SCHEMA = "https://developer.microsoft.com/json-schemas/fabric"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


# ====================================================================== columns
@dataclass
class Column:
    name: str
    kind: str               # int | double | string | dateTime | boolean
    hidden: bool = False

    @property
    def tmdl_type(self) -> str:
        return {"int": "int64", "double": "double", "string": "string",
                "dateTime": "dateTime", "boolean": "boolean"}[self.kind]

    @property
    def dax_type(self) -> str:
        return {"int": "INTEGER", "double": "DOUBLE", "string": "STRING",
                "dateTime": "DATETIME", "boolean": "BOOLEAN"}[self.kind]

    @property
    def m_type(self) -> str:
        return {"int": "Int64.Type", "double": "type number", "string": "type text",
                "dateTime": "type datetime", "boolean": "type logical"}[self.kind]

    @property
    def numeric(self) -> bool:
        return self.kind in ("int", "double")

    @property
    def summarize_by(self) -> str:
        return "sum" if self.numeric and not self.hidden else "none"


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "double"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "dateTime"
    if s.dtype == object:
        sample = [v for v in s.head(200) if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if sample and all(isinstance(v, (Decimal, int, float)) and not isinstance(v, bool) for v in sample):
            return "double"
        if sample and all(isinstance(v, (datetime, date)) for v in sample):
            return "dateTime"
    return "string"


def clean_name(name: object) -> str:
    text = _CTRL.sub("", str(name)).strip()
    return text or "column"


def columns_of(df: pd.DataFrame) -> list[Column]:
    cols, used = [], set()
    for c in df.columns:
        name = clean_name(c)
        base, n = name, 2
        while name.lower() in used or name == ROW_ID:
            name, n = f"{base}_{n}", n + 1
        used.add(name.lower())
        cols.append(Column(name, column_kind(df[c])))
    return cols


# ====================================================================== escaping
def tmdl_name(name: str) -> str:
    return name if _IDENT.match(name) else "'" + name.replace("'", "''") + "'"


def dax_ref(column: str) -> str:
    return "[" + column.replace("]", "]]") + "]"


def dax_str(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


def m_str(text: str) -> str:
    text = text.replace('"', '""').replace("#(", "#(#)(")
    return '"' + text.replace("\r\n", "\n").replace("\n", "#(lf)").replace("\t", "#(tab)") + '"'


def pbi_literal(text: str) -> str:
    """A string literal inside a PBIR expression: single-quoted, apostrophes replaced (no escape ambiguity)."""
    return "'" + text.replace("'", "’").replace("\n", " ") + "'"


def _is_missing(v: object) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def dax_value(v: object, col: Column) -> str:
    if _is_missing(v):
        return "BLANK()"
    if col.kind == "boolean":
        return "TRUE" if bool(v) else "FALSE"
    if col.kind in ("int", "double"):
        try:
            f = float(v)
        except (TypeError, ValueError):          # object column sampled as numeric but holding a stray string
            return "BLANK()"
        if not math.isfinite(f):
            return "BLANK()"
        return str(int(f)) if col.kind == "int" else repr(f)
    if col.kind == "dateTime":
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
        if isinstance(v, datetime):
            return dax_str(v.strftime("%Y-%m-%d %H:%M:%S"))
        if isinstance(v, date):
            return dax_str(v.strftime("%Y-%m-%d"))
        return dax_str(str(v))
    return dax_str(str(v))


# ====================================================================== data source
def mysql_source(database_url: str) -> tuple[str, str]:
    """(server, database) for the MySQL.Database M function. The password is never used."""
    url = make_url(database_url)
    server = url.host or "localhost"
    if url.port:
        server = f"{server}:{url.port}"
    return server, url.database or ""


def m_partition(sql: str, cols: list[Column], server: str, database: str) -> str:
    types = ", ".join("{" + m_str(c.name) + ", " + c.m_type + "}" for c in cols if c.name != ROW_ID)
    return "\n".join([
        "let",
        f"    Source = MySQL.Database({m_str(server)}, {m_str(database)}, [Query = {m_str(sql)}]),",
        f'    Indexed = Table.AddIndexColumn(Source, "{ROW_ID}", 1, 1, Int64.Type),',
        f"    Typed = Table.TransformColumnTypes(Indexed, {{{types}}})",
        "in",
        "    Typed",
    ])


def datatable_expr(df: pd.DataFrame, cols: list[Column]) -> str:
    header = ", ".join(f"{dax_str(c.name)}, {c.dax_type}" for c in cols)
    rows = []
    for values in df.itertuples(index=False, name=None):
        rows.append("{" + ", ".join(dax_value(v, c) for v, c in zip(values, cols)) + "}")
    body = ",\n        ".join(rows) if rows else ""
    return f"DATATABLE(\n    {header},\n    {{\n        {body}\n    }}\n)"


# ====================================================================== histogram bins
def nice_bin_size(lo: float, hi: float, bins: int = HIST_BINS) -> float | None:
    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        return None
    raw = span / bins
    mag = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 5, 10):
        if raw <= step * mag:
            return step * mag
    return 10 * mag


# ====================================================================== TMDL
@dataclass
class Table:
    name: str
    cols: list[Column]
    partition: str          # M expression or DAX DATATABLE
    mode: str               # "m" | "calculated"
    measures: list[tuple[str, str]] = field(default_factory=list)
    calc_columns: list[tuple[str, str, str]] = field(default_factory=list)   # (name, dax, kind)


def _tag() -> str:
    return str(uuid.uuid4())


def table_tmdl(t: Table) -> str:
    calc = t.mode == "calculated"
    out = [f"table {tmdl_name(t.name)}", f"\tlineageTag: {_tag()}", ""]
    for name, dax in t.measures:
        out += [f"\tmeasure {tmdl_name(name)} = {dax}", f"\t\tlineageTag: {_tag()}", ""]
    for c in t.cols:
        out += [f"\tcolumn {tmdl_name(c.name)}", f"\t\tdataType: {c.tmdl_type}"]
        if calc:
            out.append("\t\tisDataTypeInferred: true")
        if c.hidden:
            out.append("\t\tisHidden")
        out += [f"\t\tlineageTag: {_tag()}", f"\t\tsummarizeBy: {c.summarize_by}"]
        if calc:
            out += ["\t\tisNameInferred: true", f"\t\tsourceColumn: {dax_ref(c.name)}"]
        else:
            out.append(f"\t\tsourceColumn: {c.name}")
        out += ["", "\t\tannotation SummarizationSetBy = Automatic", ""]
    for name, dax, kind in t.calc_columns:
        out += [f"\tcolumn {tmdl_name(name)} = {dax}", f"\t\tdataType: {Column(name, kind).tmdl_type}",
                "\t\tisDataTypeInferred: true", f"\t\tlineageTag: {_tag()}", "\t\tsummarizeBy: none", "",
                "\t\tannotation SummarizationSetBy = Automatic", ""]
    out += [f"\tpartition {tmdl_name(t.name)} = {t.mode}", "\t\tmode: import", "\t\tsource ="]
    out += ["\t\t\t\t" + line for line in t.partition.splitlines()]
    out += ["", "\tannotation PBI_ResultType = Table", ""]
    return "\n".join(out)


def model_tmdl(tables: list[Table]) -> str:
    order = json.dumps([t.name for t in tables])
    out = ["model Model", "\tculture: en-US", "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
           "\tsourceQueryCulture: en-US", "\tdataAccessOptions", "\t\tlegacyRedirects",
           "\t\treturnErrorValuesAsNull", "", f"annotation PBI_QueryOrder = {order}", "",
           "annotation __PBI_TimeIntelligenceEnabled = 0", ""]
    out += [f"ref table {tmdl_name(t.name)}" for t in tables]
    out += ["", "ref cultureInfo en-US", ""]
    return "\n".join(out)


DATABASE_TMDL = "database\n\tcompatibilityLevel: 1567\n"
CULTURE_TMDL = ('cultureInfo en-US\n\n\tlinguisticMetadata =\n\t\t\t{\n\t\t\t  "Version": "1.0.0",\n'
                '\t\t\t  "Language": "en-US"\n\t\t\t}\n\t\tcontentType: json\n')


# ====================================================================== PBIR
def _ref(entity: str, column: str) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": column}}


def _measure_ref(entity: str, measure: str) -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": entity}}, "Property": measure}}


def column_projection(entity: str, column: str) -> dict:
    return {"field": _ref(entity, column), "queryRef": f"{entity}.{column}", "nativeQueryRef": column}


def agg_projection(entity: str, column: str, fn: int) -> dict:
    return {"field": {"Aggregation": {"Expression": _ref(entity, column), "Function": fn}},
            "queryRef": f"{_AGG_NAME[fn]}({entity}.{column})",
            "nativeQueryRef": f"{_AGG_LABEL[fn]} {column}"}


def measure_projection(entity: str, measure: str) -> dict:
    return {"field": _measure_ref(entity, measure), "queryRef": f"{entity}.{measure}", "nativeQueryRef": measure}


def _visual_title(text: str) -> dict:
    return {"title": [{"properties": {
        "show": {"expr": {"Literal": {"Value": "true"}}},
        "text": {"expr": {"Literal": {"Value": pbi_literal(text[:200])}}},
    }}]}


def visual_json(name: str, visual_type: str, roles: dict[str, list[dict]], pos: dict,
                title: str | None = None, sort: dict | None = None) -> dict:
    visual: dict = {
        "visualType": visual_type,
        "query": {"queryState": {role: {"projections": projs} for role, projs in roles.items() if projs}},
        "drillFilterOtherVisuals": True,
    }
    if sort:
        visual["query"]["sortDefinition"] = {"sort": [sort], "isDefaultSort": True}
    if title:
        visual["visualContainerObjects"] = _visual_title(title)
    return {"$schema": f"{SCHEMA}/item/report/definition/visualContainer/1.0.0/schema.json",
            "name": name, "position": {"z": 0, **pos}, "visual": visual}


def textbox_json(name: str, paragraphs: list[tuple[str, dict]], pos: dict) -> dict:
    return {"$schema": f"{SCHEMA}/item/report/definition/visualContainer/1.0.0/schema.json",
            "name": name, "position": {"z": 0, **pos},
            "visual": {"visualType": "textbox",
                       "objects": {"general": [{"properties": {"paragraphs": [
                           {"textRuns": [{"value": text, "textStyle": style}]} for text, style in paragraphs]}}]},
                       "drillFilterOtherVisuals": True}}


def _plain_text(markdown: str) -> list[str]:
    lines = []
    for raw in (markdown or "").splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw)
        line = re.sub(r"^\s*[-*+]\s+", "• ", line)
        line = re.sub(r"(\*\*|__|`)", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _vid() -> str:
    return uuid.uuid4().hex[:20]


# ---------------------------------------------------------------- spec -> visual
def chart_visual(spec: dict, cols: dict[str, Column], calc_columns: list[tuple[str, str, str]],
                 df: pd.DataFrame, pos: dict) -> dict | None:
    """Map one validated chart spec (answer agent) to a PBIR visual. Returns None if it cannot be drawn."""
    e = RESULT_TABLE
    t, x, y, color = spec.get("type"), spec.get("x", ""), spec.get("y", ""), spec.get("color") or ""
    agg = _SPEC_AGG.get(spec.get("aggregate", "none"), AGG_SUM)
    title = spec.get("title") or f"{t}: {x}" + (f" vs {y}" if y else "")
    if x not in cols:
        return None
    if y and y != "count" and y not in cols:
        return None
    series = [column_projection(e, color)] if color in cols else []

    if t == "bar":
        y_proj = agg_projection(e, ROW_ID, AGG_COUNT) if y == "count" or agg == AGG_COUNT else agg_projection(e, y, agg)
        many = df[x].nunique() > 8 if x in df.columns else False
        vtype = "clusteredBarChart" if many else "clusteredColumnChart"
        return visual_json(_vid(), vtype, {"Category": [column_projection(e, x)], "Y": [y_proj], "Series": series},
                           pos, title, sort={"field": y_proj["field"], "direction": "Descending"})
    if t == "line":
        y_proj = agg_projection(e, ROW_ID, AGG_COUNT) if y == "count" else agg_projection(e, y, agg)
        return visual_json(_vid(), "lineChart", {"Category": [column_projection(e, x)], "Y": [y_proj], "Series": series},
                           pos, title, sort={"field": _ref(e, x), "direction": "Ascending"})
    if t == "scatter":
        if not y or y == "count":
            return None
        return visual_json(_vid(), "scatterChart", {"Category": [column_projection(e, ROW_ID)],
                                                    "X": [agg_projection(e, x, AGG_SUM)],
                                                    "Y": [agg_projection(e, y, AGG_SUM)], "Series": series}, pos, title)
    if t == "histogram":
        # no native histogram: a DAX bin column + count per bin, like the 30-bin Altair chart
        s = pd.to_numeric(df[x], errors="coerce").dropna() if x in df.columns else pd.Series(dtype=float)
        if s.empty:
            return None
        size = nice_bin_size(float(s.min()), float(s.max()))
        if size is None:
            return None
        bin_name = f"{x} (bin)"
        if not any(n == bin_name for n, _, _ in calc_columns):
            calc_columns.append((bin_name, f"INT({dax_ref(x)} / {size!r}) * {size!r}", "double"))
        return visual_json(_vid(), "clusteredColumnChart",
                           {"Category": [column_projection(e, bin_name)], "Y": [agg_projection(e, ROW_ID, AGG_COUNT)]},
                           pos, title, sort={"field": _ref(e, bin_name), "direction": "Ascending"})
    if t == "box":
        # no core box plot: mean / min / max per group as clustered columns
        if not y or y == "count":
            return None
        return visual_json(_vid(), "clusteredColumnChart",
                           {"Category": [column_projection(e, x)],
                            "Y": [agg_projection(e, y, AGG_AVG), agg_projection(e, y, AGG_MIN), agg_projection(e, y, AGG_MAX)]},
                           pos, f"{title} (mean / min / max)")
    return None


# ====================================================================== assembling
@dataclass
class Layout:
    """Simple grid on a 1280x720 page."""
    margin: int = 20

    def row(self, y: int, height: int, n: int) -> list[dict]:
        width = (PAGE_W - self.margin * (n + 1)) // max(n, 1)
        return [{"x": self.margin + i * (width + self.margin), "y": y, "width": width, "height": height}
                for i in range(n)]


def build_project(name: str, question: str, answer: str, sql: str | None, df: pd.DataFrame,
                  charts: list[dict], extra_tables: dict[str, pd.DataFrame] | None, *, source: str,
                  database_url: str) -> tuple[dict[str, str], list[str]]:
    """Return ({relative path: file text}, notes). `df` already cleaned and capped."""
    notes: list[str] = []
    cols = columns_of(df)
    df = df.copy()
    df.columns = [c.name for c in cols]
    df.insert(0, ROW_ID, range(1, len(df) + 1))
    cols.insert(0, Column(ROW_ID, "int", hidden=True))
    by_name = {c.name: c for c in cols}

    if source == "live":
        server, database = mysql_source(database_url)
        partition, mode = m_partition(sql or "", cols, server, database), "m"
        notes.append(f"Live query: Power BI refreshes from MySQL server {server}, database {database}. "
                     "Enter the read-only user's credentials when asked (MySQL Connector/NET required).")
    else:
        partition, mode = datatable_expr(df, cols), "calculated"
        notes.append(f"Embedded rows: {len(df):,} rows are stored inside the semantic model (no refresh).")

    result = Table(RESULT_TABLE, cols, partition, mode,
                   measures=[(ROW_COUNT_MEASURE, f"COUNTROWS({tmdl_name(RESULT_TABLE)})")])
    tables = [result]
    used = {RESULT_TABLE.lower()}
    for key, frame in (extra_tables or {}).items():
        frame = _clean_frame(frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame))
        if frame.empty:
            continue
        tname = clean_name(key)
        while tname.lower() in used:
            tname += "_"
        used.add(tname.lower())
        ecols = columns_of(frame)
        f2 = frame.copy()
        f2.columns = [c.name for c in ecols]
        tables.append(Table(tname, ecols, datatable_expr(f2, ecols), "calculated"))

    # ---- report
    overview, data_page = "page1", "page2"
    lay = Layout()
    visuals: dict[str, list[dict]] = {overview: [], data_page: []}
    (p,) = lay.row(lay.margin, 50, 1)
    visuals[overview].append(textbox_json(_vid(), [(question[:300], {"fontWeight": "bold", "fontSize": "18pt"})], p))
    answer_pos, card_pos = lay.row(90, 150, 2)
    answer_pos["width"] = int(answer_pos["width"] * 1.6)
    card_pos["x"] = answer_pos["x"] + answer_pos["width"] + lay.margin
    card_pos["width"] = PAGE_W - card_pos["x"] - lay.margin
    lines = _plain_text(answer) or ["(no written answer for this result)"]
    text = "\n".join(lines)[:MAX_TEXT_CHARS]
    visuals[overview].append(textbox_json(_vid(), [(line, {"fontSize": "11pt"}) for line in text.splitlines()], answer_pos))
    visuals[overview].append(visual_json(_vid(), "card", {"Values": [measure_projection(RESULT_TABLE, ROW_COUNT_MEASURE)]},
                                         card_pos, "Rows"))
    drawn = []
    skipped = []
    for spec in charts:
        v = chart_visual(spec, by_name, result.calc_columns, df, {})
        if v is None:
            skipped.append(spec.get("type", "?"))
        else:
            drawn.append(v)
    for v, pos in zip(drawn, lay.row(260, PAGE_H - 260 - lay.margin, len(drawn))):
        v["position"].update(pos)
        visuals[overview].append(v)
    if skipped:
        notes.append("Charts without a Power BI equivalent were skipped: " + ", ".join(skipped) + ".")
    if any(spec.get("type") == "box" for spec in charts):
        notes.append("Box plots are shown as mean / min / max columns (Power BI has no core box plot).")

    (sql_pos,) = lay.row(lay.margin, 110, 1)
    (tbl_pos,) = lay.row(150, PAGE_H - 150 - lay.margin, 1)
    sql_lines = (sql or "")[:MAX_TEXT_CHARS].splitlines() or ["(no SQL for this result)"]
    visuals[data_page].append(textbox_json(_vid(), [("SQL", {"fontWeight": "bold", "fontSize": "12pt"})]
                                           + [(line, {"fontFamily": "Consolas", "fontSize": "10pt"}) for line in sql_lines],
                                           sql_pos))
    visuals[data_page].append(visual_json(_vid(), "tableEx",
                                          {"Values": [column_projection(RESULT_TABLE, c.name) for c in cols]}, tbl_pos))

    # ---- files
    report, model = f"{name}.Report", f"{name}.SemanticModel"
    files: dict[str, str] = {}
    files[f"{name}.pbip"] = _json({
        "$schema": f"{SCHEMA}/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
        "artifacts": [{"report": {"path": report}}], "settings": {"enableAutoRecovery": True}})
    files[f"{report}/.platform"] = _platform("Report", name)
    files[f"{report}/definition.pbir"] = _json({
        "$schema": f"{SCHEMA}/item/report/definition/definitionProperties/1.0.0/schema.json",
        "version": "1.0", "datasetReference": {"byPath": {"path": f"../{model}"}}})
    files[f"{report}/definition/version.json"] = _json({
        "$schema": f"{SCHEMA}/item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"})
    files[f"{report}/definition/report.json"] = _json({
        "$schema": f"{SCHEMA}/item/report/definition/report/1.0.0/schema.json",
        "themeCollection": {"baseTheme": {"name": "CY24SU06", "reportVersionAtImport": "5.55", "type": "SharedResources"}},
        "settings": {"useStylableVisualContainerHeader": True, "useNewFilterPaneExperience": True,
                     "exportDataMode": "AllowSummarizedAndUnderlying"}})
    files[f"{report}/definition/pages/pages.json"] = _json({
        "$schema": f"{SCHEMA}/item/report/definition/pagesMetadata/1.0.0/schema.json",
        "pageOrder": [overview, data_page], "activePageName": overview})
    for page, display in ((overview, "Overview"), (data_page, "Data")):
        files[f"{report}/definition/pages/{page}/page.json"] = _json({
            "$schema": f"{SCHEMA}/item/report/definition/page/1.0.0/schema.json",
            "name": page, "displayName": display, "displayOption": "FitToPage", "height": PAGE_H, "width": PAGE_W})
        for v in visuals[page]:
            files[f"{report}/definition/pages/{page}/visuals/{v['name']}/visual.json"] = _json(v)

    files[f"{model}/.platform"] = _platform("SemanticModel", name)
    files[f"{model}/definition.pbism"] = _json({
        "$schema": f"{SCHEMA}/item/semanticModel/definitionProperties/1.0.0/schema.json", "version": "4.0", "settings": {}})
    files[f"{model}/definition/database.tmdl"] = DATABASE_TMDL
    files[f"{model}/definition/model.tmdl"] = model_tmdl(tables)
    files[f"{model}/definition/cultures/en-US.tmdl"] = CULTURE_TMDL
    for t in tables:
        files[f"{model}/definition/tables/{_file_name(t.name)}.tmdl"] = table_tmdl(t)
    return files, notes


def _file_name(name: str) -> str:
    return re.sub(r'[\/:*?"<>|]', "_", name).strip(". ") or "table"


def _json(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _platform(kind: str, name: str) -> str:
    return _json({"$schema": f"{SCHEMA}/gitIntegration/platformProperties/2.0.0/schema.json",
                  "metadata": {"type": kind, "displayName": name},
                  "config": {"version": "2.0", "logicalId": str(uuid.uuid4())}})


def readme_text(name: str, notes: list[str]) -> str:
    lines = [
        f"Power BI project generated by Local Data Agent: {name}", "",
        "How to open", "-----------",
        "1. Unzip this archive (keep the folder structure).",
        f"2. In Power BI Desktop open  {name}/{name}.pbip  (File > Open, or double-click).",
        "   Older Desktop versions: enable File > Options > Preview features > 'Power BI Project (.pbip)' and",
        "   'Store reports using enhanced metadata format (PBIR)', then restart Desktop.",
        "3. Page 'Overview' has the question, the answer and the charts; page 'Data' has the SQL and the table.",
        "", "Data source", "-----------",
    ]
    lines += [f"- {n}" for n in notes]
    lines += ["", "The database password is never written into these files."]
    return "\n".join(lines) + "\n"


# ====================================================================== public API
def effective_source(source: str, has_ml: bool) -> tuple[str, str | None]:
    """ML outputs exist only in Python, so an ML answer is always embedded."""
    if has_ml and source == "live":
        return "inline", PBIP_ML_INLINE
    return ("inline" if source == "inline" else "live"), None


def to_pbip_bytes(question: str, answer: str, sql: str | None, df: pd.DataFrame, charts: list[dict],
                  extra_tables: dict[str, pd.DataFrame] | None = None, *, source: str = "live",
                  has_ml: bool = False, database_url: str | None = None,
                  inline_max_rows: int | None = None) -> tuple[bytes | None, str | None]:
    """Zip a Power BI project for this result. Returns (zip bytes, note) or (None, user-facing reason).
    A note with bytes is informational (rows truncated, ML forced to embedded rows)."""
    if df is None or df.empty:
        return None, PBIP_NO_DATA
    try:
        source, note = effective_source(source, has_ml)
        cap = inline_max_rows if inline_max_rows is not None else settings.powerbi_inline_max_rows
        frame = _clean_frame(df)
        if source == "inline" and len(frame) > cap:
            frame = frame.iloc[:cap]
            note = PBIP_TRUNCATED.format(n=cap)
        name = slugify(question, default="agent-result", max_len=40)
        files, notes = build_project(name, question, answer, sql, frame, charts, extra_tables,
                                     source=source, database_url=database_url or settings.database_url)
        if note:
            notes.insert(0, note)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{name}/README.txt", readme_text(name, notes))
            for path, text in files.items():
                zf.writestr(f"{name}/{path}", text)
        return buf.getvalue(), note
    except Exception as exc:
        log.debug("to_pbip_bytes failed: %s", exc, exc_info=True)
        return None, PBIP_FAILED
