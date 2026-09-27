"""Agent settings page: see and customise each agent's prompts, model and knowledge.

Fifth Streamlit page (sidebar nav). Prompts are saved to settings.prompts_file (agent/prompt_store.py), the model
to knowledge/<agent>/model.txt and documents to knowledge/<agent>/ (agent/knowledge.py). The pipeline reads all
three on every call, so a saved change applies from the next question.
"""
from __future__ import annotations

import difflib
import html

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException

from agent.agent_catalog import default_model, specs_for
from agent.config import settings
from agent.knowledge import AGENTS, UPLOAD_TYPES, KnowledgeBase
from agent.llm import OllamaLLM
from agent.prompt_store import store

st.set_page_config(page_title="Agent settings", page_icon="🧩", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root{
  --ink:#0f172a; --ink2:#334155; --muted:#64748b; --faint:#94a3b8;
  --line:#e5e9f0; --surface2:#f6f8fc;
  --brand:#2f5bea; --brand-soft:#eef2ff; --brand-line:#d5ddfb; --brand-ink:#1e3a8a;
  --ok:#0f9960; --ok-soft:#e9f7f0; --ok-line:#cbe8db; --ok-ink:#0b6b45;
  --warn:#c2740a; --warn-soft:#fdf5e7; --warn-line:#f0dfbc; --warn-ink:#8a5208;
}
[data-testid="stMainBlockContainer"]{max-width:1180px; padding-top:2.4rem;}
[data-testid="stHeader"]{background:transparent;}
.app-head{border-bottom:1px solid var(--line); padding-bottom:1.1rem; margin-bottom:1.4rem;}
.app-title{font-size:1.5rem; font-weight:660; margin:0; letter-spacing:-.025em; color:var(--ink);}
.app-sub{color:var(--muted); font-size:.875rem; margin-top:.35rem; line-height:1.45;}
.chip{display:inline-flex; align-items:center; gap:.42rem; font-size:.75rem; font-weight:550;
  padding:.26rem .62rem; border-radius:999px; border:1px solid var(--line);
  background:var(--surface2); color:var(--ink2); white-space:nowrap; line-height:1.35;}
.chip .dot{width:6px; height:6px; border-radius:50%; background:var(--faint); flex:0 0 auto;}
.chip.ok{background:var(--ok-soft); border-color:var(--ok-line); color:var(--ok-ink);} .chip.ok .dot{background:var(--ok);}
.chip.warn{background:var(--warn-soft); border-color:var(--warn-line); color:var(--warn-ink);} .chip.warn .dot{background:var(--warn);}
.chip.brand{background:var(--brand-soft); border-color:var(--brand-line); color:var(--brand-ink);} .chip.brand .dot{background:var(--brand);}
.section{font-size:.8rem; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:1.6rem 0 .5rem;}
.side-label{font-size:.72rem; font-weight:650; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:.9rem 0 .35rem;}
.side-note{font-size:.78rem; color:var(--muted); line-height:1.45; margin:.2rem 0 0;}
.answer-meta{display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.6rem;}
.agent-what{color:var(--ink2); font-size:.9rem; margin:.2rem 0 .1rem;}
.prompt-title{display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; font-weight:620; color:var(--ink);}
.prompt-when{color:var(--muted); font-size:.8rem; margin:.15rem 0 .45rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(v) -> str:
    return html.escape(str(v))


def chip(text: str, tone: str = "", dot: bool = True) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip {tone}">{marker}{esc(text)}</span>'


@st.cache_data(ttl=30, show_spinner=False)
def pulled_models() -> list[str]:
    """Models pulled in Ollama, for the model picker; [] when Ollama is not reachable."""
    try:
        r = requests.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=3)
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


# callbacks run before the rerun, so they may set widget state (Streamlit "widget callbacks")
def _save_prompt(spec) -> None:
    text = st.session_state[f"prompt_{spec.key}"]
    err = store.validate(text, spec.placeholders)
    if not err:
        try:
            store.set(spec.key, text, spec.default)
        except OSError as exc:
            err = f"Could not save: {exc}"
    st.session_state[f"prompt_msg_{spec.key}"] = ("error", err) if err else ("ok", "Saved. Used from the next question.")


def _reset_prompt(spec) -> None:
    store.reset(spec.key)
    st.session_state[f"prompt_{spec.key}"] = spec.default
    st.session_state[f"prompt_msg_{spec.key}"] = ("ok", "Back to the default prompt.")


kb = KnowledgeBase()


def base_model(key: str) -> tuple[str, str]:
    """Model an agent runs on without an override, and where it comes from: the chat page's sidebar boxes this
    session (stored by app.py), else .env."""
    chat = st.session_state.get("chat_models") or {}
    return (chat[key], "chat sidebar") if chat.get(key) else (default_model(key), ".env")


def effective_model(key: str) -> tuple[str, str]:
    """The model the agent's calls really go to (same rule as agent.knowledge.KnowledgeLLM)."""
    override = kb.model_for(key)
    return (override, "Agent settings override") if override else base_model(key)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">How it works</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-note">Saved changes apply from the next question; no restart needed.</p>'
                '<p class="side-note">A model set here wins over the model boxes in the chat page\'s sidebar.</p>'
                '<p class="side-note">Placeholders such as <code>{dialect}</code> or <code>{persona}</code> are '
                'filled in at run time. The JSON output format of each step is enforced separately, so an edit '
                'cannot change the fields an agent returns.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ header
st.markdown('<div class="app-head"><h1 class="app-title">Agent settings</h1>'
            '<div class="app-sub">See and customise what each agent is told, which model runs it and what it '
            'knows. Saved changes apply from the next question.</div></div>', unsafe_allow_html=True)

with st.expander("Which model runs each agent", expanded=True):
    st.dataframe(pd.DataFrame([{"agent": AGENTS[k][0], "model in use": effective_model(k)[0],
                                "set by": effective_model(k)[1], "without override": base_model(k)[0]}
                               for k in AGENTS]), width="stretch", hide_index=True)
    st.caption("An Agent settings override wins over the chat page's model boxes. Each answer's Step-by-step tab "
               "shows the model, prompts and reply of every call. Open the chat page first so its sidebar "
               "models are picked up here; otherwise the .env models are shown.")

agent = st.segmented_control("Agent", list(AGENTS), default="understanding", key="as_agent",
                             format_func=lambda a: AGENTS[a][0], label_visibility="collapsed") or "understanding"
label, what, docs_help = AGENTS[agent]
specs = specs_for(agent)
status = kb.status(agent)
override = kb.model_for(agent)
base, base_source = base_model(agent)
custom = sum(store.is_custom(s.key) for s in specs)

meta = chip(label, "brand")
meta += chip(f"model {override}", "ok") if override else chip(f"model {base} · from {base_source}", dot=False)
meta += (chip(f"{custom} of {len(specs)} prompts customised", "warn") if custom
         else chip(f"{len(specs)} prompt{'s' if len(specs) > 1 else ''} · default", dot=False))
if status["built_at"]:
    meta += chip("knowledge needs re-indexing" if status["stale"] else f"{status['documents']} documents",
                 "warn" if status["stale"] else "ok")
else:
    meta += chip("no knowledge yet", dot=False)
st.markdown(f'<div class="answer-meta">{meta}</div><p class="agent-what">{esc(what)}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 1. model
st.markdown('<div class="section">1 · Model</div>', unsafe_allow_html=True)
options = list(dict.fromkeys([override or base] + pulled_models()))
c1, c2, c3 = st.columns([3, 1, 1])
chosen = c1.selectbox("Model", options, index=0, key=f"as_model_{agent}", accept_new_options=True,
                      label_visibility="collapsed",
                      help=f"Pick a pulled Ollama model or type a name. Without an override: {base} ({base_source})")
if c2.button("Save model", type="primary", width="stretch"):
    name = (chosen or "").strip()
    kb.set_model(agent, name)       # saved even when it equals the .env default: it must win over the chat sidebar
    st.rerun()
if c3.button("Use default", width="stretch", disabled=not override):
    kb.set_model(agent, "")
    st.session_state.pop(f"as_model_{agent}", None)
    st.rerun()
ok, msg = OllamaLLM(model=override or base).health()
st.caption(("✅ " if ok else "⚠️ ") + msg + ("" if override else f" · from {base_source}"))

# ------------------------------------------------------------------ 2. prompts
st.markdown('<div class="section">2 · Prompts</div>', unsafe_allow_html=True)
for spec in specs:
    k = f"prompt_{spec.key}"
    if k not in st.session_state:                          # initialise once; never also pass value=
        st.session_state[k] = store.get(spec.key, spec.default)
    is_custom = store.is_custom(spec.key)
    with st.container(border=True):
        tags = chip("customised", "warn") if is_custom else chip("default", dot=False)
        tags += "".join(chip("{" + p + "}", "brand", dot=False) for p in spec.placeholders)
        st.markdown(f'<div class="prompt-title">{esc(spec.label)} {tags}</div>'
                    f'<p class="prompt-when">{esc(spec.when)}</p>', unsafe_allow_html=True)
        st.text_area(spec.label, key=k, height=280, label_visibility="collapsed")
        if spec.placeholders:
            st.caption("Keep " + ", ".join("{" + p + "}" for p in spec.placeholders)
                       + ": filled in at run time. Write literal braces as {{ and }}.")
        if len(st.session_state[k]) > 2 * len(spec.default):
            st.caption("⚠️ Much longer than the default: every call re-reads it, so answers get slower on CPU.")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.button("Save prompt", key=f"save_{spec.key}", on_click=_save_prompt, args=(spec,),
                  type="primary", width="stretch")
        b2.button("Reset to default", key=f"reset_{spec.key}", on_click=_reset_prompt, args=(spec,),
                  disabled=not is_custom, width="stretch")
        tone, text = st.session_state.pop(f"prompt_msg_{spec.key}", (None, ""))
        if tone == "error":
            st.error(text)
        elif tone:
            st.success(text)
        if is_custom:
            with st.expander("Changes vs default"):
                diff = difflib.unified_diff(spec.default.splitlines(), store.get(spec.key, spec.default).splitlines(),
                                            "default", "yours", lineterm="")
                st.code("\n".join(diff) or "(no difference)", language="diff")

# ------------------------------------------------------------------ 3. knowledge
st.markdown('<div class="section">3 · Knowledge</div>', unsafe_allow_html=True)
st.caption(f"Helpful documents: {docs_help}")
docs = kb.documents(agent)
if docs:
    st.dataframe(pd.DataFrame({"document": docs}), width="stretch", hide_index=True)
files = st.file_uploader("Documents", type=list(UPLOAD_TYPES), accept_multiple_files=True,
                         key=f"as_upload_{agent}", label_visibility="collapsed",
                         help="PDF, Word, Markdown, text, SQL, CSV. A .jsonl file of {\"input\", \"output\"} lines is "
                              "stored as hand-written examples for LoRA training.")
if st.button(f"Add & re-index · {label}", type="primary", disabled=not files):
    with st.status(f"Indexing the {label} agent's documents…", expanded=True) as box:
        failed = False
        for f in files or []:
            try:
                box.write(f"Read **{f.name}**: {kb.add_document(agent, f.name, f.getvalue())}")
            except Exception as exc:
                failed = True
                box.write(f"Skipped **{f.name}**: {exc}")
        index = kb.build(agent)
        box.write(f"Indexed {len(index['documents'])} documents into {len(index['passages'])} passages.")
        box.update(label=f"{label} agent: its next calls use these documents",
                   state="error" if failed and not index["passages"] else "complete", expanded=False)
    st.rerun()
try:
    st.page_link("pages/3_Fine_tune_agents.py", label="Probe passages, remove documents, LoRA training", icon="🎓")
except StreamlitAPIException:        # page opened on its own (not through app.py): no navigation to link into
    st.caption("Probe passages, remove documents and LoRA training: see the Fine-tune agents page.")
