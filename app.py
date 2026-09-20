"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import html
import json

import altair as alt
import pandas as pd
import streamlit as st

from agent.answer_agent import AnswerAgent
from agent.config import settings
from agent.expert_agent import ExpertAgent, review_markdown
from agent.export import EXCEL_MAX_ROWS, chart_to_png, export_filenames, to_csv_bytes, to_xlsx_bytes
from agent.powerbi import to_pbip_bytes
from agent.llm import OllamaLLM
from agent.orchestrator import AgentResult
from agent.trace import Step
from ui_shared import expert_persona_input, get_agent

st.set_page_config(page_title="Local Data Agent", page_icon="🔎", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
TONE = {"ok": "ok", "warning": "warn", "error": "err", "running": "run"}
INTENT_LABEL = {"data_query": "Data query", "statistics": "Statistics", "machine_learning": "Machine learning"}
MODE_INTENT = {"Data query": "data_query", "Statistics": "statistics", "Machine learning": "machine_learning"}

# how each trace detail is rendered
CODE_KEYS = {"sql", "validated_sql", "schema_given_to_model", "facts_given_to_model",
             "context_given_to_model", "facts", "previous_error", "original_question",
             "standalone_question", "turn_given_to_model", "updated_context", "quality_report_given_to_model",
             "briefing_given_to_model", "thinking"}
TAG_KEYS = {"selected_tables", "columns", "available_models", "explanations",
            "feature_names", "required_columns", "top_features",
            "entities", "filters", "metrics", "grouping", "ambiguities"}
LABELS = {"sql": "Generated SQL", "validated_sql": "Validated SQL (what actually ran)",
          "schema_given_to_model": "Schema sent to the model",
          "facts_given_to_model": "Facts sent to the model",
          "context_given_to_model": "Conversation memory sent to the model",
          "turn_given_to_model": "This turn, as sent to the model",
          "updated_context": "Updated conversation memory",
          "standalone_question": "Standardised request", "ambiguities": "Assumptions made",
          "quality_report_given_to_model": "Material sent to the expert", "persona": "Expert persona",
          "task": "Expert task", "verdict": "Verdict", "quality_score": "Quality score",
          "briefing_given_to_model": "Briefing sent to the expert", "expert_answer": "Expert answer",
          "dropped": "Dropped by validation", "source": "Source", "thinking": "Model's thinking (reasoning model)",
          "previous_error": "Error returned by the database", "plan": "Feature-engineering plan",
          "rows": "Rows", "rows_scored": "Rows scored", "raw_columns": "Raw columns",
          "model_features": "Model features"}

EXAMPLES = {
}

CSS = """
<style>
:root{
  --ink:#0f172a; --ink2:#334155; --muted:#64748b; --faint:#94a3b8;
  --line:#e5e9f0; --surface:#ffffff; --surface2:#f6f8fc;
  --brand:#2f5bea; --brand-soft:#eef2ff; --brand-line:#d5ddfb; --brand-ink:#1e3a8a;
  --ok:#0f9960; --ok-soft:#e9f7f0; --ok-line:#cbe8db; --ok-ink:#0b6b45;
  --warn:#c2740a; --warn-soft:#fdf5e7; --warn-line:#f0dfbc; --warn-ink:#8a5208;
  --err:#dc2626; --err-soft:#fdeded; --err-line:#f5cfcf; --err-ink:#a51b1b;
}
[data-testid="stMainBlockContainer"]{max-width:1060px; padding-top:2.4rem; padding-bottom:7rem;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stAppDeployButton"]{display:none;}
h1,h2,h3,h4,h5{letter-spacing:-.015em;}

/* ---------------------------------------------------------------- header */
.app-head{display:flex; align-items:flex-end; justify-content:space-between; gap:1.5rem;
  border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.8rem;}
.app-title{font-size:1.5rem; font-weight:660; margin:0; letter-spacing:-.025em; color:var(--ink);}
.app-sub{color:var(--muted); font-size:.875rem; margin-top:.35rem; line-height:1.45;}
.head-status{display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end; padding-bottom:.15rem;}

/* ----------------------------------------------------------------- chips */
.chip{display:inline-flex; align-items:center; gap:.42rem; font-size:.75rem; font-weight:550;
  padding:.26rem .62rem; border-radius:999px; border:1px solid var(--line);
  background:var(--surface2); color:var(--ink2); white-space:nowrap; line-height:1.35;}
.chip .dot{width:6px; height:6px; border-radius:50%; background:var(--faint); flex:0 0 auto;}
.chip.ok{background:var(--ok-soft); border-color:var(--ok-line); color:var(--ok-ink);}
.chip.ok .dot{background:var(--ok);}
.chip.err{background:var(--err-soft); border-color:var(--err-line); color:var(--err-ink);}
.chip.err .dot{background:var(--err);}
.chip.warn{background:var(--warn-soft); border-color:var(--warn-line); color:var(--warn-ink);}
.chip.warn .dot{background:var(--warn);}
.chip.brand{background:var(--brand-soft); border-color:var(--brand-line); color:var(--brand-ink);}
.chip.brand .dot{background:var(--brand);}

/* ------------------------------------------------------------ step cards */
.step-head{display:flex; align-items:center; gap:.65rem;}
.step-num{width:23px; height:23px; flex:0 0 auto; border-radius:50%; font-size:.72rem; font-weight:700;
  display:flex; align-items:center; justify-content:center; font-variant-numeric:tabular-nums;
  background:var(--surface2); color:var(--muted); border:1px solid var(--line);}
.step-num.ok{background:var(--ok-soft); color:var(--ok-ink); border-color:var(--ok-line);}
.step-num.warn{background:var(--warn-soft); color:var(--warn-ink); border-color:var(--warn-line);}
.step-num.err{background:var(--err-soft); color:var(--err-ink); border-color:var(--err-line);}
.step-title{font-weight:620; font-size:.95rem; color:var(--ink);}
.step-time{margin-left:auto; font-size:.72rem; color:var(--faint); font-variant-numeric:tabular-nums;}
.step-why{color:var(--muted); font-size:.83rem; line-height:1.5; margin:.5rem 0 .1rem;}
.reasoning{margin:.75rem 0 .2rem; padding:.6rem .8rem; border-left:2px solid var(--brand);
  background:var(--brand-soft); border-radius:0 6px 6px 0; font-size:.84rem;
  color:var(--brand-ink); line-height:1.5;}
.reasoning b{display:block; font-size:.68rem; letter-spacing:.07em; text-transform:uppercase;
  opacity:.75; margin-bottom:.2rem; font-weight:650;}

/* --------------------------------------------------- key/value + tag rows */
.kv-row{display:flex; flex-wrap:wrap; gap:.4rem; margin:.75rem 0 .25rem;}
.kv{display:inline-flex; align-items:baseline; gap:.4rem; font-size:.78rem; padding:.24rem .6rem;
  border:1px solid var(--line); border-radius:6px; background:var(--surface2);}
.kv i{font-style:normal; color:var(--muted); font-size:.72rem; letter-spacing:.01em;}
.kv b{font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums;}
.tag{display:inline-block; font-size:.74rem; padding:.2rem .5rem; border-radius:5px;
  background:var(--brand-soft); color:var(--brand-ink); border:1px solid var(--brand-line);
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.field-label{display:block; font-size:.7rem; letter-spacing:.07em; text-transform:uppercase;
  color:var(--faint); font-weight:650; margin:.85rem 0 .3rem;}

/* ------------------------------------------------------------ answer card */
.answer-meta{display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; margin-bottom:.7rem;}
.rewrite{font-size:.82rem; color:var(--muted); margin-bottom:.6rem;}
.rewrite b{color:var(--ink2); font-weight:600;}

/* -------------------------------------------------------------- examples */
.ex-head{font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--faint);
  font-weight:650; margin:0 0 .55rem 2px;}
.hero{font-size:.9rem; color:var(--muted); line-height:1.6; margin:-.4rem 0 1.5rem;}

/* ----------------------------------------------------- streamlit widgets */
.stButton > button{font-size:.84rem; font-weight:500; text-align:left; line-height:1.4;
  justify-content:flex-start; padding:.55rem .75rem; color:var(--ink2);}
.stButton > button:hover{border-color:var(--brand); color:var(--brand);}
[data-testid="stSidebar"] .stButton > button{justify-content:center; text-align:center;}
[data-testid="stTabs"] button[role="tab"]{font-size:.85rem; font-weight:550;}
[data-testid="stExpander"] summary{font-size:.85rem; font-weight:550;}
[data-testid="stElementToolbar"]{display:none;}
[data-testid="stSidebarContent"] h2{font-size:1.05rem;}
.side-label{font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--faint);
  font-weight:650; margin:1.1rem 0 .45rem;}
.side-note{font-size:.76rem; color:var(--muted); line-height:1.45; margin:.4rem 0 0;
  word-break:break-all;}
.model-row{padding:.5rem 0; border-top:1px solid var(--line);}
.model-row:first-child{border-top:none;}
.model-name{font-size:.82rem; font-weight:600; color:var(--ink);}
.model-meta{font-size:.72rem; color:var(--muted); margin-top:.1rem; line-height:1.45;}
.table-row{font-size:.8rem; padding:.35rem 0; border-top:1px solid var(--line);}
.table-row:first-child{border-top:none;}
.table-row b{font-weight:600; color:var(--ink);}
.table-row span{color:var(--faint); font-size:.72rem;}
.table-cols{color:var(--muted); font-size:.72rem; line-height:1.5; margin-top:.1rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(v) -> str:
    return html.escape(str(v))


def chip(text: str, tone: str = "", dot: bool = True) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip {tone}">{marker}{esc(text)}</span>'


def label_of(key: str) -> str:
    return LABELS.get(key, key.replace("_", " ").capitalize())


def is_scalar(v) -> bool:
    return isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 70 and "\n" not in v)


def tags(values: list, limit: int = 24) -> str:
    shown = [f'<span class="tag">{esc(v)}</span>' for v in values[:limit]]
    if len(values) > limit:
        shown.append(f'<span class="kv"><i>+{len(values) - limit} more</i></span>')
    return f'<div class="kv-row">{"".join(shown)}</div>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">Connection</div>', unsafe_allow_html=True)
    db_url = st.text_input("Database URL", settings.database_url, type="password",
                           label_visibility="collapsed", placeholder="Database URL")
    model = st.text_input("Ollama model", settings.model, label_visibility="collapsed",
                          placeholder="SQL model (Ollama)", help="Writes SQL. e.g. qwen2.5-coder:3b / 7b")
    answer_model = st.text_input("Answer model", settings.answer_model, label_visibility="collapsed",
                                 placeholder="Answer model (empty = same as SQL model)",
                                 help="Explains results and plans charts. A general instruct model "
                                      "(e.g. qwen2.5:7b-instruct) writes better prose than a coder model.")
    charts_on = st.toggle("Draw charts", value=True, help="Let the answer agent plan and draw charts.")
    summary_on = st.toggle("Summarise data queries", value=settings.summarise_data_queries,
                           help="Add a short plain-English summary above the result table of plain data queries "
                                "(one extra answer-model call per question).")
    st.markdown('<div class="side-label">Expert AI</div>', unsafe_allow_html=True)
    expert_on = st.toggle("Expert AI", value=settings.expert_reviews,
                          help="The briefed expert plans the data for the SQL writer and assesses the result before "
                               "the answer is written (two extra model calls per question).")
    expert_model = st.text_input("Expert model", settings.expert_model, label_visibility="collapsed",
                                 placeholder="Expert model (empty = answer model)",
                                 help="A stronger instruct model, e.g. qwen2.5:14b-instruct, gives better judgement.")
    persona = expert_persona_input()
    pbi_source = st.radio("Power BI data source", ["Live MySQL query", "Embedded rows"],
                          index=0 if settings.powerbi_source == "live" else 1, horizontal=True,
                          help="Live: the .pbip runs the generated SQL against MySQL when refreshed (needs MySQL "
                               "Connector/NET on the Power BI machine). Embedded: the rows are stored in the file. "
                               "ML answers are always embedded.")
    st.session_state["pbi_source"] = "live" if pbi_source.startswith("Live") else "inline"
    st.session_state["db_url"] = db_url
    agent = get_agent(db_url, model)          # cached: keeps the introspected schema
    agent.answer_agent = AnswerAgent(OllamaLLM(model=answer_model or model))
    agent.context_builder.llm = agent.answer_agent.llm      # memory summaries are prose: use the answer model
    agent.expert_agent = ExpertAgent(OllamaLLM(model=expert_model or answer_model or model), persona,
                                     briefing=agent.briefing.text if expert_on else "")
    agent.expert = expert_on
    if expert_on:
        with st.expander(f"Database briefing · {agent.briefing.name} · {agent.briefing.source}"):
            st.caption(str(agent.briefing.path) if agent.briefing.path else "auto-generated from the schema")
            st.markdown(agent.briefing.text[:3000] + ("…" if len(agent.briefing.text) > 3000 else ""))
            if st.button("Reload briefing"):
                agent.reload_briefing()
                st.rerun()
    agent.charts = charts_on
    agent.summarise_data = summary_on

    ok_db, msg_db = agent.db.ping()
    ok_llm, msg_llm = agent.llm.health()
    ok_ans, msg_ans = agent.answer_agent.llm.health()
    ok_exp, msg_exp = agent.expert_agent.llm.health() if expert_on else (True, "")
    st.markdown(
        chip("Database" if ok_db else "Database offline", "ok" if ok_db else "err")
        + chip(f"SQL · {model}" if ok_llm else "Ollama offline", "ok" if ok_llm else "err")
        + chip(f"Answer · {agent.answer_agent.llm.model}" if ok_ans else "Answer model missing",
               "ok" if ok_ans else "err")
        + (chip(f"Expert · {agent.expert_agent.llm.model}" if ok_exp else "Expert model missing",
                "ok" if ok_exp else "err") if expert_on else ""),
        unsafe_allow_html=True)
    for ok, msg in ((ok_db, msg_db), (ok_llm, msg_llm), (ok_ans, msg_ans), (ok_exp, msg_exp)):
        if not ok:
            st.markdown(f'<p class="side-note">{esc(msg)}</p>', unsafe_allow_html=True)

    st.markdown('<div class="side-label">Answer mode</div>', unsafe_allow_html=True)
    mode = st.radio("Answer mode", ["Auto", "Data query", "Statistics", "Machine learning"],
                    label_visibility="collapsed",
                    captions=["Let the agent decide", None, None, None])
    force_model = None
    names = agent.registry.names()
    if mode == "Machine learning":
        force_model = st.selectbox("Model", names, label_visibility="collapsed") if names else None
        if not names:
            st.warning("No trained models. Run `python train_models.py`.")

    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
    if ok_db:
        schema = agent.db.schema()
        with st.expander(f"Database schema · {len(schema)} tables"):
            if st.button("Refresh schema", width="stretch"):
                agent.db.schema(refresh=True)
                st.rerun()
            for t in schema.values():
                st.markdown(
                    f'<div class="table-row"><b>{esc(t.name)}</b> <span>{t.row_count} rows</span>'
                    f'<div class="table-cols">{esc(", ".join(t.column_names))}</div></div>',
                    unsafe_allow_html=True)
    with st.expander(f"Trained models · {len(names)}"):
        if not names:
            st.caption("No models yet. Run `python train_models.py`.")
        for c in agent.registry.cards():
            metrics = " · ".join(f"{k} {v}" for k, v in list(c["metrics"].items())[:3])
            st.markdown(
                f'<div class="model-row"><div class="model-name">{esc(c["name"])}</div>'
                f'<div class="model-meta">{esc(c["algorithm"])} / {esc(c["task"])}'
                f'{" · " + esc(metrics) if metrics else ""}<br>{esc(c["description"])}</div></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="side-label">Conversation</div>', unsafe_allow_html=True)
    remember = st.toggle("Remember conversation", value=True,
                         help="After every answer a context-builder agent updates a compact memory (entities, "
                              "filters, preferences, facts found). The next question is standardised against it, "
                              "so follow-ups like 'how old is he?' resolve to concrete IDs.")
    memory = next((r.context for _, r in reversed(st.session_state.get("history", [])) if r.context), None)
    if remember and memory and not memory.is_empty():
        with st.expander(f"Conversation memory · {memory.turns} turn{'s' if memory.turns != 1 else ''}"):
            if memory.summary:
                st.markdown(f'<p class="side-note">{esc(memory.summary)}</p>', unsafe_allow_html=True)
            for title, values in (("Entities", [f"{k}: {v}" for k, v in memory.entities.items()]),
                                  ("Filters", memory.filters), ("Metrics", memory.metrics),
                                  ("Preferences", memory.preferences), ("Tables", memory.tables)):
                if values:
                    st.markdown(f'<span class="field-label">{title}</span>{tags(values)}', unsafe_allow_html=True)
            if memory.findings:
                st.markdown('<span class="field-label">Facts found</span>', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {f}" for f in memory.findings))
    if st.button("Clear chat", width="stretch"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------- rendering
def render_value(key: str, value) -> None:
    if key in CODE_KEYS and isinstance(value, str):
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span>', unsafe_allow_html=True)
        st.code(value, language="sql" if key in {"sql", "validated_sql"} else "text", wrap_lines=True)
    elif key == "plan" and isinstance(value, list):
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(value), hide_index=True, width="stretch")
    elif key in TAG_KEYS and isinstance(value, list):
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span>{tags(value)}',
                    unsafe_allow_html=True)
    elif isinstance(value, dict) and value and all(is_scalar(v) for v in value.values()):
        items = "".join(f'<span class="kv"><i>{esc(k)}</i><b>{esc(v)}</b></span>' for k, v in value.items())
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span><div class="kv-row">{items}</div>',
                    unsafe_allow_html=True)
    elif isinstance(value, (dict, list)):
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span>', unsafe_allow_html=True)
        st.json(json.loads(json.dumps(value, default=str)), expanded=False)
    else:
        st.markdown(f'<span class="field-label">{esc(label_of(key))}</span>'
                    f'<div style="font-size:.86rem;line-height:1.55">{esc(value)}</div>',
                    unsafe_allow_html=True)


def export_row(r: AgentResult, data: pd.DataFrame, where: str) -> None:
    """CSV / Excel / Power BI download buttons for a result table. `where` keeps widget keys unique per placement."""
    if data is None or data.empty:
        return
    names = export_filenames(r.standalone_question or r.question)
    extra_sheets = None
    if r.stats:
        extra_sheets = {k: v if isinstance(v, pd.DataFrame) else pd.DataFrame(v)
                        for k, v in r.stats["tables"].items()}
    cache = r.extras.setdefault("_export", {})    # results persist in session_state: build each file once
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    if "csv" not in cache:
        cache["csv"] = to_csv_bytes(data)
    c1.download_button("Download CSV", cache["csv"], file_name=names["csv"], mime="text/csv",
                       key=f"csv_{where}_{id(r)}")
    export_powerbi(r, data, extra_sheets, names["pbip"], c3, where)
    if "xlsx" not in cache:
        try:
            cache["xlsx"] = to_xlsx_bytes(data, extra_sheets=extra_sheets)
        except Exception as exc:   # a failing export must never break the page
            cache["xlsx"] = (None, str(exc))
    xlsx, truncated = cache["xlsx"]
    if xlsx is None:
        st.caption(f"Excel export unavailable: {truncated}")
        return
    c2.download_button("Download Excel", xlsx, file_name=names["xlsx"],
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"xlsx_{where}_{id(r)}")
    if truncated:
        st.caption(f"Excel sheet limited to {EXCEL_MAX_ROWS:,} rows")


def export_powerbi(r: AgentResult, data: pd.DataFrame, extra_tables: dict | None, fname: str, col, where: str) -> None:
    """Download button for a Power BI project (.pbip in a zip): the result table, the planned charts and the SQL."""
    cache = r.extras["_export"]
    source = st.session_state.get("pbi_source", settings.powerbi_source)
    if cache.get("pbip_source") != source:      # rebuild only when the sidebar choice changes
        try:
            cache["pbip"] = to_pbip_bytes(r.standalone_question or r.question, r.answer, r.sql, data, r.charts,
                                          extra_tables, source=source, has_ml=r.ml is not None,
                                          database_url=st.session_state.get("db_url"))
        except Exception as exc:   # a failing export must never break the page
            cache["pbip"] = (None, str(exc))
        cache["pbip_source"] = source
    pbip, note = cache["pbip"]
    if pbip is None:
        st.caption(f"Power BI export unavailable: {note}")
        return
    col.download_button("Download Power BI", pbip, file_name=fname, mime="application/zip",
                        key=f"pbip_{where}_{id(r)}", help="A Power BI Project (.pbip) with the table, charts and SQL. "
                                                          "Unzip and open the .pbip in Power BI Desktop.")
    if note:
        st.caption(note)


def render_step(s: Step) -> None:
    tone = TONE.get(s.status, "")
    with st.container(border=True):
        st.markdown(
            f'<div class="step-head"><span class="step-num {tone}">{s.index}</span>'
            f'<span class="step-title">{esc(s.name)}</span>'
            f'<span class="step-time">{s.duration_ms:.0f} ms</span></div>'
            f'<div class="step-why">{esc(s.why)}</div>', unsafe_allow_html=True)
        if s.reasoning:
            st.markdown(f'<div class="reasoning"><b>Model reasoning</b>{esc(s.reasoning)}</div>',
                        unsafe_allow_html=True)
        scalars = {k: v for k, v in s.details.items() if k != "error" and is_scalar(v)}
        if scalars:
            items = "".join(f'<span class="kv"><i>{esc(label_of(k))}</i><b>{esc(v)}</b></span>'
                            for k, v in scalars.items())
            st.markdown(f'<div class="kv-row">{items}</div>', unsafe_allow_html=True)
        for k, v in s.details.items():
            if k == "error":
                st.error(v)
            elif k not in scalars:
                render_value(k, v)


# Categorical palette (fixed order, validated for colour-blind safety); a single series uses slot 1.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_BARS = 30
MAX_POINTS = 5000


def _axis_title(col: str, agg: str) -> str:
    if agg == "count":
        return "count"
    return f"{agg} of {col}" if agg not in ("none", "") else col


def render_chart(spec: dict, df: pd.DataFrame, key: str, fname: str) -> None:
    """Draw one validated chart spec (from the answer agent) with Altair."""
    t, x = spec["type"], spec["x"]
    y, color, agg = spec.get("y") or None, spec.get("color") or None, spec.get("aggregate", "none")
    if agg == "count":                 # synthesise the y column: one row = one count
        df = df.assign(count=1)
        y = "count"
    cols = [c for c in dict.fromkeys([x, y, color]) if c]
    d = df[cols].copy()
    if y:
        d[y] = pd.to_numeric(d[y], errors="coerce")
    if t in ("scatter", "histogram"):
        d[x] = pd.to_numeric(d[x], errors="coerce")
    d = d.dropna(subset=[c for c in (x, y) if c])
    if d.empty:
        return
    tooltip = [alt.Tooltip(c, format=",.2f") if (c == y or (t in ("scatter", "histogram") and c == x))
               else alt.Tooltip(c) for c in cols]
    enc_color = (alt.Color(f"{color}:N", scale=alt.Scale(range=PALETTE), legend=alt.Legend(title=color))
                 if color else alt.value(PALETTE[0]))
    keys = [x] + ([color] if color else [])

    if t == "bar":
        if agg != "none":
            d = d.groupby(keys, as_index=False)[y].agg("sum" if agg == "count" else agg)
        d[x] = d[x].astype(str)
        if d[x].nunique() > MAX_BARS:      # keep the largest categories readable
            top = d.groupby(x)[y].sum().nlargest(MAX_BARS).index
            d = d[d[x].isin(top)]
        horizontal = d[x].nunique() > 8 or int(d[x].str.len().max()) > 12
        if horizontal:
            enc = dict(y=alt.Y(f"{x}:N", sort="-x", title=x), x=alt.X(f"{y}:Q", title=_axis_title(y, agg)))
            if color:
                enc["yOffset"] = f"{color}:N"
            height = max(220, 24 * d[x].nunique())
        else:
            enc = dict(x=alt.X(f"{x}:N", sort="-y", title=x), y=alt.Y(f"{y}:Q", title=_axis_title(y, agg)))
            if color:
                enc["xOffset"] = f"{color}:N"
            height = 300
        chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(color=enc_color, tooltip=tooltip, **enc)
        chart = chart.properties(height=height)
    elif t == "line":
        if agg != "none":
            d = d.groupby(keys, as_index=False)[y].agg("sum" if agg == "count" else agg)
        is_time = pd.to_datetime(d[x], errors="coerce").notna().mean() > 0.9
        if is_time:
            d[x] = pd.to_datetime(d[x], errors="coerce")
        d = d.sort_values(x)
        chart = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2).encode(
            x=alt.X(f"{x}:{'T' if is_time else 'O'}", title=x), y=alt.Y(f"{y}:Q", title=_axis_title(y, agg)),
            color=enc_color, tooltip=tooltip).properties(height=300)
    elif t == "scatter":
        if len(d) > MAX_POINTS:
            d = d.sample(MAX_POINTS, random_state=0)
        chart = alt.Chart(d).mark_circle(size=48, opacity=0.7).encode(
            x=alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
            color=enc_color, tooltip=tooltip).properties(height=320).interactive()
    elif t == "histogram":
        chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=30), title=x), y=alt.Y("count():Q", title="count"),
            color=enc_color, tooltip=[alt.Tooltip(f"{x}:Q", bin=True), alt.Tooltip("count():Q")]).properties(height=280)
    elif t == "box":
        d[x] = d[x].astype(str)
        chart = alt.Chart(d).mark_boxplot(extent=1.5, size=24).encode(
            x=alt.X(f"{x}:N", title=x), y=alt.Y(f"{y}:Q", title=y, scale=alt.Scale(zero=False)),
            color=alt.Color(f"{x}:N", scale=alt.Scale(range=PALETTE), legend=None)).properties(height=300)
    else:
        return
    chart = chart.configure_view(strokeWidth=0).configure_axis(gridColor="#e5e9f0", domainColor="#cbd5e1")
    st.markdown(f'<span class="field-label">{esc(spec.get("title") or t)}</span>', unsafe_allow_html=True)
    st.altair_chart(chart, width="stretch")
    png, reason = chart_to_png(chart)
    if png:
        st.download_button("Download PNG", png, file_name=fname, mime="image/png", key=key)
    else:
        st.caption(reason)
    if spec.get("why"):
        st.caption(spec["why"])


def render_charts(r: AgentResult) -> None:
    df = r.frame()
    if df is None or not r.charts:
        return
    names = export_filenames(r.standalone_question or r.question)
    for i, spec in enumerate(r.charts):
        try:
            render_chart(spec, df, key=f"png_{id(r)}_{i}", fname=names["png"].format(i=i + 1))
        except Exception as exc:   # a bad spec must never break the page
            st.caption(f"Could not draw {spec.get('type')} chart: {exc}")


def render_summary(summary: dict) -> None:
    """ML summary: scalars as metrics, distributions as bars, lists as tags."""
    scalars = {k: v for k, v in summary.items() if is_scalar(v)}
    if scalars:
        cols = st.columns(min(len(scalars), 4))
        for i, (k, v) in enumerate(scalars.items()):
            val = f"{v:,.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else v
            cols[i % 4].metric(label_of(k), val)
    for k, v in summary.items():
        if k in scalars:
            continue
        if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
            st.markdown(f'<span class="field-label">{esc(label_of(k))}</span>', unsafe_allow_html=True)
            st.bar_chart(pd.Series(v, name=k).rename(index=str), horizontal=True, height=max(120, 34 * len(v)))
        elif isinstance(v, list) and all(is_scalar(x) for x in v):
            st.markdown(f'<span class="field-label">{esc(label_of(k))}</span>{tags(v)}', unsafe_allow_html=True)
        else:
            render_value(k, v)


def render_result(r: AgentResult) -> None:
    if r.standalone_question and r.standalone_question != r.question:
        st.markdown(f'<div class="rewrite">Understood as: <b>{esc(r.standalone_question)}</b></div>',
                    unsafe_allow_html=True)
    if r.request and r.request.ambiguities:
        st.markdown(f'<div class="rewrite">Assumed: {esc("; ".join(r.request.ambiguities))}</div>',
                    unsafe_allow_html=True)

    meta = chip(INTENT_LABEL.get(r.intent, r.intent), "err" if r.error else "brand")
    if r.model_name:
        meta += chip(r.model_name, dot=False)
    if r.data is not None:
        meta += chip(f"{len(r.data):,} rows", dot=False)
    with st.container(border=True):
        st.markdown(f'<div class="answer-meta">{meta}</div>', unsafe_allow_html=True)
        if r.error:
            st.error(r.answer)
        elif r.intent == "data_query":
            # optional plain-English summary, then 1. the SQL that ran, 2. the result table
            if r.extras.get("summarised"):
                st.markdown(r.answer)
            st.code(r.sql or "-- no SQL was generated", language="sql", wrap_lines=True)
            if r.data is None or r.data.empty:
                st.caption("The query returned no rows.")
            else:
                st.dataframe(r.data, width="stretch", hide_index=True)
                export_row(r, r.frame(), where="card")
            render_charts(r)
        else:
            st.markdown(r.answer)
            render_charts(r)
    if r.expert:
        score = r.expert["quality_score"]
        with st.container(border=True):
            st.markdown(f'<div class="answer-meta">{chip("Expert review", "brand")}'
                        f'{chip(INTENT_LABEL.get(r.expert["task"], r.expert["task"]), dot=False)}'
                        f'{chip(r.expert["model"], dot=False)}'
                        f'{chip(f"quality {score}/5", "ok" if score >= 4 else "warn" if score == 3 else "err")}</div>',
                        unsafe_allow_html=True)
            st.markdown(review_markdown(r.expert))

    n_rows = len(r.data) if r.data is not None else 0
    tabs = st.tabs([f"Step-by-step ({len(r.trace.steps)})", f"Data ({n_rows:,})", "Analysis", "SQL"])
    with tabs[0]:
        for s in r.trace.steps:
            render_step(s)
        st.download_button("Download trace (.txt)", r.trace.as_text(), file_name="agent_trace.txt",
                           key=f"trace_{id(r)}")
    with tabs[1]:
        if r.data is None:
            st.caption("No rows were returned for this request.")
        else:
            data = r.frame()
            st.markdown(f'<div class="kv-row"><span class="kv"><i>rows</i><b>{len(data):,}</b></span>'
                        f'<span class="kv"><i>columns</i><b>{len(data.columns)}</b></span></div>',
                        unsafe_allow_html=True)
            st.dataframe(data, width="stretch", hide_index=True)
            export_row(r, data, where="tab")
    with tabs[2]:
        if r.stats:
            st.markdown(f"##### Statistical model — {r.stats['method']}")
            st.success(r.stats["interpretation"])
            for name, t in r.stats["tables"].items():
                st.markdown(f'<span class="field-label">{esc(name)}</span>', unsafe_allow_html=True)
                st.dataframe(t, width="stretch", hide_index=True)
        elif r.ml:
            st.markdown(f"##### Model — {r.model_name}")
            render_summary(r.ml.summary)
            if {"PC1", "PC2"} <= set(r.ml.output.columns):
                st.markdown('<span class="field-label">Principal components</span>', unsafe_allow_html=True)
                st.scatter_chart(r.ml.output, x="PC1", y="PC2")
            for sec in r.ml.sections:
                st.markdown(f'<span class="field-label">{esc(sec["title"])}</span>', unsafe_allow_html=True)
                if sec["kind"] == "table":
                    st.dataframe(sec["content"], width="stretch", hide_index=True)
                else:
                    st.markdown(sec["content"])
            with st.expander("Feature engineering applied"):
                st.dataframe(r.ml.feature_plan, width="stretch", hide_index=True)
                st.caption("Engineered features the model received (first 20 rows)")
                st.dataframe(r.ml.features.head(20).round(3), width="stretch")
        else:
            st.caption("No statistical or ML analysis for this request.")
    with tabs[3]:
        st.code(r.sql or "-- no SQL was needed for this request", language="sql", wrap_lines=True)


# -------------------------------------------------------------------- main
status_chips = (chip("MySQL" if ok_db else "MySQL offline", "ok" if ok_db else "err")
                + chip("Ollama" if ok_llm else "Ollama offline", "ok" if ok_llm else "err")
                + chip(f"{len(names)} models", dot=False))
st.markdown(
    f'<div class="app-head"><div><h1 class="app-title">Local Data Agent</h1>'
    f'<div class="app-sub">Ask your MySQL database in plain English — SQL, statistics and machine '
    f'learning, entirely on your machine.</div></div>'
    f'<div class="head-status">{status_chips}</div></div>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.history:
    st.markdown('<div class="hero">Every answer shows its full reasoning: the tables it picked, the SQL it '
                'wrote, how the data was transformed and how the result was computed.</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES), gap="medium") if EXAMPLES else []
    for col, (group, questions) in zip(cols, EXAMPLES.items()):
        with col:
            st.markdown(f'<div class="ex-head">{group}</div>', unsafe_allow_html=True)
            for i, ex in enumerate(questions):
                if st.button(ex, key=f"ex_{group}_{i}", width="stretch"):
                    st.session_state.pending = ex
                    st.rerun()

for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        render_result(r)

question = st.chat_input("Ask about your data…") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        intent = MODE_INTENT.get(mode)
        with st.status("Working…", expanded=True) as status:
            def live(s: Step) -> None:
                status.markdown(f'<div class="step-head"><span class="step-num {TONE.get(s.status, "")}">'
                                f'{s.index}</span><span class="step-title">{esc(s.name)}</span>'
                                f'<span class="step-time">{s.duration_ms:.0f} ms</span></div>',
                                unsafe_allow_html=True)

            history = [r for _, r in st.session_state.history] if remember else []
            result = agent.ask(question, on_step=live, force_intent=intent, force_model=force_model,
                               history=history, build_context=remember)
            status.update(label="Done" if not result.error else "Finished with errors",
                          state="complete" if not result.error else "error", expanded=False)
        render_result(result)
    st.session_state.history.append((question, result))
