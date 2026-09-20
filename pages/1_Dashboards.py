"""Dashboards page: describe -> build (LLM + SQL) -> preview -> publish to Netlify.

Second Streamlit page (sidebar nav). Streamlit runs it with the repo root on sys.path, so the
imports below are the same as in app.py.
"""
from __future__ import annotations

import html

import streamlit as st

from agent.answer_agent import AnswerAgent
from agent.config import settings
from agent.export import slugify
from agent.llm import OllamaLLM
from agent.orchestrator import DataAgent
from agent.trace import Step
from dashboard import store
from dashboard.builder import DashboardBuild, DashboardBuilder
from dashboard.netlify import NetlifyClient, PublishError, publish
from dashboard.render import BUNDLE_FILES, bundle_zip, inline_preview
from ui_shared import get_agent

st.set_page_config(page_title="Dashboards", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

TONE = {"ok": "ok", "warning": "warn", "error": "err", "running": "run"}
LANG = {"index.html": "html", "style.css": "css", "app.js": "javascript", "data.js": "javascript"}
PLACEHOLDER = ("Sales overview: total revenue KPI, revenue by month (line), top 10 cities by revenue (bar), "
               "orders by status (pie), latest 20 orders (table)")
PUBLIC_WARNING = ("Publishing makes this dashboard and the data in it publicly readable by anyone with the link. "
                  "It is a snapshot: the site never connects to your database.")

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
[data-testid="stMainBlockContainer"]{max-width:1180px; padding-top:2.4rem;}
[data-testid="stHeader"]{background:transparent;}
.app-head{display:flex; align-items:flex-end; justify-content:space-between; gap:1.5rem;
  border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.4rem;}
.app-title{font-size:1.5rem; font-weight:660; margin:0; letter-spacing:-.025em; color:var(--ink);}
.app-sub{color:var(--muted); font-size:.875rem; margin-top:.35rem; line-height:1.45;}
.head-status{display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end;}
.chip{display:inline-flex; align-items:center; gap:.42rem; font-size:.75rem; font-weight:550;
  padding:.26rem .62rem; border-radius:999px; border:1px solid var(--line);
  background:var(--surface2); color:var(--ink2); white-space:nowrap; line-height:1.35;}
.chip .dot{width:6px; height:6px; border-radius:50%; background:var(--faint); flex:0 0 auto;}
.chip.ok{background:var(--ok-soft); border-color:var(--ok-line); color:var(--ok-ink);} .chip.ok .dot{background:var(--ok);}
.chip.err{background:var(--err-soft); border-color:var(--err-line); color:var(--err-ink);} .chip.err .dot{background:var(--err);}
.chip.warn{background:var(--warn-soft); border-color:var(--warn-line); color:var(--warn-ink);} .chip.warn .dot{background:var(--warn);}
.step-head{display:flex; align-items:center; gap:.65rem;}
.step-num{width:23px; height:23px; flex:0 0 auto; border-radius:50%; font-size:.72rem; font-weight:700;
  display:flex; align-items:center; justify-content:center; background:var(--surface2); color:var(--muted); border:1px solid var(--line);}
.step-num.ok{background:var(--ok-soft); color:var(--ok-ink); border-color:var(--ok-line);}
.step-num.warn{background:var(--warn-soft); color:var(--warn-ink); border-color:var(--warn-line);}
.step-num.err{background:var(--err-soft); color:var(--err-ink); border-color:var(--err-line);}
.step-title{font-weight:620; font-size:.95rem; color:var(--ink);}
.step-time{margin-left:auto; font-size:.72rem; color:var(--faint); font-variant-numeric:tabular-nums;}
.section{font-size:.8rem; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:1.6rem 0 .5rem;}
.side-label{font-size:.72rem; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:.9rem 0 .35rem;}
.side-note{font-size:.78rem; color:var(--muted); line-height:1.45; margin:.2rem 0 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(v) -> str:
    return html.escape(str(v))


def chip(text: str, tone: str = "", dot: bool = True) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip {tone}">{marker}{esc(text)}</span>'


@st.cache_resource(show_spinner=False)
def get_builder_agent(db_url: str, model: str, answer_model: str) -> DataAgent:
    """A DataAgent that shares the chat page's connection and schema but skips charts and summaries."""
    base = get_agent(db_url, model)
    return DataAgent(db=base.db, llm=base.llm, registry=base.registry,
                     answer_agent=AnswerAgent(OllamaLLM(model=answer_model or model)),
                     charts=False, summarise_data=False)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">Connection</div>', unsafe_allow_html=True)
    db_url = st.text_input("Database URL", settings.database_url, type="password",
                           label_visibility="collapsed", placeholder="Database URL")
    model = st.text_input("Ollama model", settings.model, label_visibility="collapsed",
                          placeholder="SQL model (Ollama)", help="Writes the SQL of every widget.")
    answer_model = st.text_input("Answer model", settings.answer_model, label_visibility="collapsed",
                                 placeholder="Answer model (empty = same as SQL model)",
                                 help="Designs the dashboard (which widgets, which questions).")
    builder_agent = get_builder_agent(db_url, model, answer_model)
    builder = DashboardBuilder(builder_agent)

    ok_db, msg_db = builder_agent.db.ping()
    ok_llm, msg_llm = builder_agent.llm.health()
    ok_ans, msg_ans = builder_agent.answer_agent.llm.health()
    token_ok = bool(settings.netlify_token)
    st.markdown(
        chip("Database" if ok_db else "Database offline", "ok" if ok_db else "err")
        + chip(f"SQL · {model}" if ok_llm else "Ollama offline", "ok" if ok_llm else "err")
        + chip(f"Design · {builder_agent.answer_agent.llm.model}" if ok_ans else "Answer model missing",
               "ok" if ok_ans else "err")
        + chip("Netlify token set" if token_ok else "Netlify token missing", "ok" if token_ok else "warn"),
        unsafe_allow_html=True)
    for ok, msg in ((ok_db, msg_db), (ok_llm, msg_llm), (ok_ans, msg_ans)):
        if not ok:
            st.markdown(f'<p class="side-note">{esc(msg)}</p>', unsafe_allow_html=True)
    if not token_ok:
        st.markdown('<p class="side-note">Set NETLIFY_AUTH_TOKEN in .env to enable publishing '
                    '(Netlify → User settings → Applications → Personal access tokens).</p>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Limits</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="side-note">Up to {settings.dashboard_max_widgets} widgets, '
                f'{settings.dashboard_rows_per_widget:,} rows per widget (DASHBOARD_* in .env).</p>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def live_steps(status):
    def live(s: Step) -> None:
        status.markdown(f'<div class="step-head"><span class="step-num {TONE.get(s.status, "")}">{s.index}</span>'
                        f'<span class="step-title">{esc(s.name)}</span>'
                        f'<span class="step-time">{s.duration_ms:.0f} ms</span></div>', unsafe_allow_html=True)
    return live


def run_build(label: str, fn) -> DashboardBuild:
    with st.status(label, expanded=True) as status:
        build = fn(live_steps(status))
        status.update(label="Dashboard ready" if not build.error else "Finished with errors",
                      state="complete" if not build.error else "error", expanded=False)
    return build


def do_publish(build: DashboardBuild, record: dict | None) -> dict | None:
    """Publish a bundle (new site or existing site_id); returns the saved record or None on failure."""
    rec = record or store.new_record(build.description, build.spec, build.sqls())
    try:
        with st.spinner("Publishing to Netlify…"):
            info = publish(NetlifyClient(settings.netlify_token), bundle_zip(build.bundle), rec.get("site_id"), rec["slug"])
    except PublishError as exc:
        st.error(str(exc))
        if rec.get("site_id") and "404" in str(exc):
            st.caption("The site no longer exists on Netlify. Unpublish it here, then publish again as a new site.")
        return None
    rec.update(site_id=info["site_id"], url=info["url"], spec=build.spec.to_dict(), sqls=build.sqls())
    if rec.get("published_at"):
        rec["refreshed_at"] = store.now()
    else:
        rec["published_at"] = store.now()
    store.save(rec)
    st.success(f"Published: {info['url']}")
    st.link_button("Open dashboard", info["url"])
    return rec


# ------------------------------------------------------------------ header
st.markdown(
    '<div class="app-head"><div><h1 class="app-title">Dashboards</h1>'
    '<div class="app-sub">Describe the dashboard you want. The agent designs the widgets, writes and runs the SQL, '
    'generates the HTML / CSS / JavaScript, shows a preview and - once you approve - publishes it.</div></div>'
    f'<div class="head-status">{chip("Netlify" if token_ok else "Netlify token missing", "ok" if token_ok else "warn")}</div></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ 1. describe
st.markdown('<div class="section">1 · Describe</div>', unsafe_allow_html=True)
description = st.text_area("Describe your dashboard", placeholder=PLACEHOLDER, height=110,
                           label_visibility="collapsed")
go = st.button("Design dashboard", type="primary", disabled=not description.strip() or not ok_db)

# ------------------------------------------------------------------ 2. build
if go:
    st.markdown('<div class="section">2 · Build</div>', unsafe_allow_html=True)
    build = run_build("Building the dashboard…", lambda live: builder.build(description.strip(), on_step=live))
    st.session_state.dash_build = build
    st.session_state.dash_record = None

build: DashboardBuild | None = st.session_state.get("dash_build")
if build is not None and build.error and not build.bundle:
    st.error(build.error)

# ------------------------------------------------------------------ 3. preview
if build is not None and build.bundle:
    st.markdown('<div class="section">3 · Preview</div>', unsafe_allow_html=True)
    failed = len(build.results) - build.ok_widgets()
    st.caption(f"{build.spec.title} · {build.ok_widgets()} widget{'s' if build.ok_widgets() != 1 else ''} built"
               + (f", {failed} failed (shown as error cards)" if failed else "") + f" · generated {build.generated_at}")
    st.components.v1.html(inline_preview(build.bundle), height=900, scrolling=True)

    with st.expander("Generated files"):
        for name in BUNDLE_FILES:
            text = build.bundle[name]
            if name == "chart.umd.js":
                st.caption(f"chart.umd.js - Chart.js 4.4.7 (vendored, {len(text):,} chars, MIT)")
                continue
            st.markdown(f"**{name}**")
            st.code(text[:20_000] + ("\n…" if len(text) > 20_000 else ""), language=LANG.get(name, "text"))
    with st.expander("Widgets: questions, SQL and rows"):
        for r in build.results:
            st.markdown(f"**{esc(r.widget.title)}** · {r.widget.kind}" + (" · reused" if r.reused else ""))
            st.caption(r.widget.question)
            if r.sql:
                st.code(r.sql, language="sql")
            if r.error:
                st.error(r.error)
            elif r.data is not None:
                st.caption(f"{len(r.data):,} rows")
    with st.expander("Trace"):
        st.download_button("Download trace (.txt)", build.trace.as_text(), file_name="dashboard_trace.txt",
                           mime="text/plain", key="dash_trace")

    c1, c2 = st.columns([3, 1])
    change = c1.text_input("Refine", placeholder="e.g. make revenue by month a bar chart; add an average order value KPI",
                           label_visibility="collapsed")
    if c2.button("Apply changes", disabled=not change.strip(), width="stretch"):
        st.session_state.dash_build = run_build(
            "Refining the dashboard…", lambda live: builder.build(build.description, on_step=live, previous=build,
                                                                  change=change.strip()))
        st.rerun()
    slug = slugify(build.spec.title, default="dashboard")
    st.download_button("Download zip", bundle_zip(build.bundle), file_name=f"{slug}-dashboard.zip",
                       mime="application/zip", key="dash_zip")

    # -------------------------------------------------------------- 4. publish
    st.markdown('<div class="section">4 · Publish</div>', unsafe_allow_html=True)
    st.warning(PUBLIC_WARNING)
    consent = st.checkbox("I understand this dashboard and its data will be publicly accessible")
    existing = st.session_state.get("dash_record")
    label = "Republish to Netlify" if existing else "Publish to Netlify"
    if st.button(label, type="primary", disabled=not (consent and token_ok)):
        rec = do_publish(build, existing)
        if rec:
            st.session_state.dash_record = rec

# ------------------------------------------------------------------ my dashboards
records = store.list_all()
if records:
    st.markdown('<div class="section">My dashboards</div>', unsafe_allow_html=True)
    for rec in records:
        slug = rec["slug"]
        c1, c2, c3, c4 = st.columns([4, 1.2, 1, 1.2])
        with c1:
            st.markdown(f"**{esc(rec['title'])}**")
            st.caption((f"published {rec.get('published_at')}" if rec.get("published_at") else "not published")
                       + (f" · refreshed {rec['refreshed_at']}" if rec.get("refreshed_at") else "")
                       + (f" · {len(rec.get('sqls') or {})} widgets"))
        if rec.get("url"):
            c2.link_button("Open", rec["url"], width="stretch")
        if c3.button("Refresh", key=f"refresh_{slug}", disabled=not (token_ok and ok_db and rec.get("site_id")),
                     width="stretch", help="Re-run the stored SQL (no LLM) and republish to the same URL."):
            fresh = run_build(f"Refreshing {rec['title']}…", lambda live, r=rec: builder.refresh(r, on_step=live))
            if fresh.bundle:
                do_publish(fresh, rec)
            else:
                st.error(fresh.error or "Refresh failed")
        if st.session_state.get("confirm_unpublish") == slug:
            if c4.button("Confirm unpublish", key=f"unpub_ok_{slug}", type="primary", width="stretch"):
                try:
                    if rec.get("site_id") and token_ok:
                        NetlifyClient(settings.netlify_token).delete_site(rec["site_id"])
                    store.delete(slug)
                    st.session_state.confirm_unpublish = None
                    st.rerun()
                except PublishError as exc:
                    st.error(str(exc))
        elif c4.button("Unpublish", key=f"unpub_{slug}", width="stretch"):
            st.session_state.confirm_unpublish = slug
            st.rerun()
