"""Expert audit page: pick a table or view, profile it with bounded SQL, get the Expert AI's data-quality report.

Third Streamlit page (sidebar nav). Deterministic findings are shown next to the model's report so the
two can be compared; the report can be downloaded as Markdown.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from agent.config import settings
from agent.expert_agent import AuditResult, ExpertAgent, audit_markdown, run_audit
from agent.export import slugify
from agent.llm import OllamaLLM
from agent.trace import Step
from ui_shared import expert_persona_input, get_agent

st.set_page_config(page_title="Expert audit", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

TONE = {"ok": "ok", "warning": "warn", "error": "err", "running": "run"}
SEV_TONE = {"high": "err", "medium": "warn", "low": ""}

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
.chip.brand{background:var(--brand-soft); border-color:var(--brand-line); color:var(--brand-ink);} .chip.brand .dot{background:var(--brand);}
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
.answer-meta{display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.6rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(v) -> str:
    return html.escape(str(v))


def chip(text: str, tone: str = "", dot: bool = True) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip {tone}">{marker}{esc(text)}</span>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">Connection</div>', unsafe_allow_html=True)
    db_url = st.text_input("Database URL", settings.database_url, type="password",
                           label_visibility="collapsed", placeholder="Database URL")
    model = st.text_input("Ollama model", settings.model, label_visibility="collapsed", placeholder="SQL model (Ollama)")
    expert_model = st.text_input("Expert model", settings.expert_model, label_visibility="collapsed",
                                 placeholder="Expert model (empty = answer model)",
                                 help="A stronger instruct model, e.g. qwen2.5:14b-instruct, gives better judgement.")
    agent = get_agent(db_url, model)
    st.markdown('<div class="side-label">Expert AI</div>', unsafe_allow_html=True)
    persona = expert_persona_input()
    expert = ExpertAgent(OllamaLLM(model=expert_model or settings.answer_model or model), persona)

    ok_db, msg_db = agent.db.ping()
    ok_exp, msg_exp = expert.llm.health()
    st.markdown(chip("Database" if ok_db else "Database offline", "ok" if ok_db else "err")
                + chip(f"Expert · {expert.llm.model}" if ok_exp else "Expert model missing", "ok" if ok_exp else "err"),
                unsafe_allow_html=True)
    for ok, msg in ((ok_db, msg_db), (ok_exp, msg_exp)):
        if not ok:
            st.markdown(f'<p class="side-note">{esc(msg)}</p>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Limits</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="side-note">Each audit query is limited to {settings.expert_audit_timeout_ms / 1000:.0f} s; '
                f'sample checks use {settings.expert_audit_sample_rows:,} rows (EXPERT_AUDIT_* in .env).</p>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def live_steps(status):
    def live(s: Step) -> None:
        status.markdown(f'<div class="step-head"><span class="step-num {TONE.get(s.status, "")}">{s.index}</span>'
                        f'<span class="step-title">{esc(s.name)}</span>'
                        f'<span class="step-time">{s.duration_ms:.0f} ms</span></div>', unsafe_allow_html=True)
    return live


# ------------------------------------------------------------------ header
st.markdown(
    '<div class="app-head"><div><h1 class="app-title">Expert audit</h1>'
    '<div class="app-sub">Pick a table or view. The toolkit profiles every column with bounded SQL (missing values, '
    'distinct counts, ranges, negatives, blanks, future dates, orphan foreign keys, duplicates, sample outliers); the '
    'Expert AI turns the findings into a data-quality report with impact, fixes and recommendations.</div></div>'
    f'<div class="head-status">{chip("Expert AI", "brand")}</div></div>', unsafe_allow_html=True)

if not ok_db:
    st.error("Database offline - the audit needs a connection.")
    st.stop()

schema = agent.db.schema()
names = sorted(schema)
table = st.selectbox("Table", names, label_visibility="collapsed")
info = schema[table]
rows_txt = f"{info.row_count:,} rows" if info.row_count is not None else "row count unknown"
st.caption(f"{table} · {rows_txt} · {len(info.columns)} columns"
           + (f" · {info.comment}" if info.comment else "")
           + (f" · PK {', '.join(info.primary_key)}" if info.primary_key else " · no primary key")
           + (f" · {len(info.foreign_keys)} foreign key(s)" if info.foreign_keys else ""))

if st.button("Audit table", type="primary"):
    with st.status(f"Auditing {table}…", expanded=True) as status:
        result = run_audit(agent.db, info, expert, on_step=live_steps(status))
        status.update(label="Audit finished" if not result.error else "Audit failed",
                      state="complete" if not result.error else "error", expanded=False)
    st.session_state.audit_result = result

result: AuditResult | None = st.session_state.get("audit_result")
if result is not None and result.table == table:
    if result.error:
        st.error(result.error)
    counts = {sev: sum(1 for f in result.findings if f.severity == sev) for sev in ("high", "medium", "low")}
    score = result.report["quality_score"] if result.report else None
    st.markdown('<div class="answer-meta">'
                + (chip(f"quality {score}/5", "ok" if score >= 4 else "warn" if score == 3 else "err") if score else "")
                + "".join(chip(f"{n} {sev}", SEV_TONE[sev]) for sev, n in counts.items() if n)
                + chip(f"{result.profile.get('rows', 0):,} rows", dot=False)
                + chip(f"generated {result.generated_at}", dot=False) + "</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown('<div class="section">Findings (computed by tools)</div>', unsafe_allow_html=True)
        if result.findings:
            st.dataframe(pd.DataFrame(result.findings_dicts()), width="stretch", hide_index=True)
        else:
            st.caption("No quality problem detected by the checks.")
    with right:
        st.markdown('<div class="section">Expert report</div>', unsafe_allow_html=True)
        rep = result.report
        if rep is None:
            st.warning("The expert model did not produce a report (see the trace); the findings on the left still apply.")
        else:
            st.markdown(f"**Summary** — {rep['summary']}")
            if rep["issues"]:
                st.markdown("**Issues**")
                st.dataframe(pd.DataFrame(rep["issues"]), width="stretch", hide_index=True)
            for title, key in (("Insights", "insights"), ("Recommendations", "recommendations"),
                               ("Questions for the data owner", "questions_for_owner")):
                if rep.get(key):
                    st.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in rep[key]))

    st.markdown('<div class="section">Column profile</div>', unsafe_allow_html=True)
    prof_rows = [{"column": c, **p} for c, p in result.profile.get("columns", {}).items()]
    if prof_rows:
        st.dataframe(pd.DataFrame(prof_rows), width="stretch", hide_index=True)

    with st.expander(f"Trace ({len(result.trace.steps)} steps)"):
        for s in result.trace.steps:
            st.markdown(f'<div class="step-head"><span class="step-num {TONE.get(s.status, "")}">{s.index}</span>'
                        f'<span class="step-title">{esc(s.name)}</span><span class="step-time">{s.duration_ms:.0f} ms</span></div>',
                        unsafe_allow_html=True)
            st.caption(s.why)
            if s.details.get("note"):
                st.warning(s.details["note"])
        st.download_button("Download trace (.txt)", result.trace.as_text(), file_name="audit_trace.txt",
                           mime="text/plain", key="audit_trace")
    st.download_button("Download report (.md)",
                       audit_markdown(result.report, table, result.findings, result.generated_at),
                       file_name=f"{slugify(table)}-audit.md", mime="text/markdown", key="audit_md")
