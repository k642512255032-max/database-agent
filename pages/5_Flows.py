"""Flows page: build your own flows (which steps run, in what order, by which agent) and your own agents.

Sixth Streamlit page (sidebar nav). Flows are saved to settings.flows_file (agent/flows.py), custom agents to
settings.custom_agents_file (agent/custom_agents.py) and their documents to knowledge/<agent id>/ (agent/knowledge.py).
The chat page's sidebar picks the active flow; a saved change applies from the next question.
"""
from __future__ import annotations

import html

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException
from streamlit_sortables import sort_items

from agent.config import settings
from agent.custom_agents import CUSTOM_INPUTS, CUSTOM_OUTPUTS, CustomAgent, agent_label, agent_store
from agent.flows import STANDARD_NAME, STEP_KINDS, FlowStep, check_flow, flow_store, step_title
from agent.knowledge import AGENTS, UPLOAD_TYPES, KnowledgeBase

st.set_page_config(page_title="Flows", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")

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
.step-title{display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; font-weight:620; color:var(--ink);}
.step-what{color:var(--muted); font-size:.8rem; margin:.15rem 0 .45rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
NEW_AGENT = "__new__"


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


def agent_of(step: FlowStep) -> str:
    """Who runs the step, for labels: its agent, else the kind's default agent; '' for steps without a model."""
    key = step.agent or STEP_KINDS[step.kind].agent
    return agent_label(key) if key else ""


def title(step: FlowStep) -> str:
    return step_title(step, agent_label)


def bump(flow_name: str) -> None:
    """New key for the drag-and-drop list, so it remounts with the saved order."""
    st.session_state[f"fl_rev_{flow_name}"] = st.session_state.get(f"fl_rev_{flow_name}", 0) + 1


def flash(tone: str, text: str) -> None:
    st.session_state["fl_msg"] = (tone, text)


# ------------------------------------------------------------------ flow callbacks (run before the rerun)
def _new_flow(source: str) -> None:
    flow = flow_store.get(source)
    flow.name = flow_store.unique_name(f"{source} copy" if source != STANDARD_NAME else "My flow")
    flow.builtin = False
    flow_store.save(flow)
    st.session_state["fl_name"] = flow.name
    flash("ok", f"Created '{flow.name}' from '{source}'. Drag, add and remove steps below.")


def _rename_flow(old: str) -> None:
    new = st.session_state.get(f"fl_rename_{old}", "")
    try:
        flow_store.rename(old, new)
    except ValueError as exc:
        flash("error", str(exc))
        return
    st.session_state["fl_name"] = new.strip()
    flash("ok", f"Renamed to '{new.strip()}'.")


def _delete_flow(name: str) -> None:
    if st.session_state.get("fl_confirm_delete") != name:       # first click arms, second click deletes
        st.session_state["fl_confirm_delete"] = name
        flash("warning", f"Click Delete again to delete '{name}'.")
        return
    flow_store.delete(name)
    st.session_state.pop("fl_confirm_delete", None)
    st.session_state["fl_name"] = STANDARD_NAME
    flash("ok", f"Deleted '{name}'.")


def _make_default(name: str) -> None:
    flow_store.set_default(name)
    flash("ok", f"'{name}' is now the default flow for new chat sessions.")


def _set_step(flow_name: str, step_id: str, attr: str, key: str) -> None:
    flow = flow_store.get(flow_name)
    for step in flow.steps:
        if step.id == step_id:
            setattr(step, attr, st.session_state[key] or "")
    flow_store.save(flow)
    bump(flow_name)


def _remove_step(flow_name: str, step_id: str) -> None:
    flow = flow_store.get(flow_name)
    flow.steps = [s for s in flow.steps if s.id != step_id]
    flow_store.save(flow)
    bump(flow_name)


def _add_step(flow_name: str) -> None:
    choice = st.session_state.get(f"fl_add_{flow_name}") or ""
    flow = flow_store.get(flow_name)
    sid = flow_store.next_step_id(flow)
    if choice.startswith("agent:"):
        flow.steps.append(FlowStep(sid, "custom", choice.split(":", 1)[1], "question", "note"))
    elif choice.startswith("kind:"):
        flow.steps.append(FlowStep(sid, choice.split(":", 1)[1]))
    else:
        return
    flow_store.save(flow)
    bump(flow_name)


# ------------------------------------------------------------------ agent callbacks
def _save_agent(agent_id: str) -> None:
    name = st.session_state.get(f"ca_name_{agent_id}", "").strip()
    instructions = st.session_state.get(f"ca_instr_{agent_id}", "")
    model = st.session_state.get(f"ca_model_{agent_id}") or ""
    new_id = agent_id if agent_id != NEW_AGENT else agent_store.new_id(name, set(agent_store.all()))
    try:
        agent_store.save(CustomAgent(new_id, name, instructions, model))
    except ValueError as exc:
        flash("error", str(exc))
        return
    if agent_id == NEW_AGENT:                     # clear the "new agent" form, select the saved agent
        for k in ("ca_name_", "ca_instr_", "ca_model_"):
            st.session_state.pop(k + NEW_AGENT, None)
    st.session_state["ca_sel"] = new_id
    flash("ok", f"Saved '{name}'. Add it to a flow on the Flows tab.")


def _delete_agent(agent_id: str) -> None:
    if st.session_state.get("ca_confirm_delete") != agent_id:
        st.session_state["ca_confirm_delete"] = agent_id
        flash("warning", "Click Delete again to delete this agent.")
        return
    try:
        KnowledgeBase().reset(agent_id)           # its documents go with it: a new agent must not inherit them
    except KeyError:
        pass
    agent_store.delete(agent_id)
    st.session_state.pop("ca_confirm_delete", None)
    st.session_state["ca_sel"] = NEW_AGENT
    flash("ok", "Agent deleted. Steps that used it now show a warning on the Flows tab.")


def flows_using(agent_id: str) -> list[str]:
    return [n for n in flow_store.names() if any(s.agent == agent_id for s in flow_store.get(n).steps)]


# ------------------------------------------------------------------ sidebar + header
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">How it works</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-note">A flow is the list of steps a question goes through. Pick the active flow in '
                'the chat page\'s sidebar; its switches still apply inside any flow.</p>'
                '<p class="side-note">A custom agent can run its own step (it reads one part of the turn and writes '
                'a note, a rewritten question or a section of the answer) or run a built-in step: its instructions '
                'are then added to that step\'s prompt.</p>'
                '<p class="side-note">Problems are shown but never block saving; a flow that cannot run stops with '
                'an explanation.</p>', unsafe_allow_html=True)

st.markdown('<div class="app-head"><h1 class="app-title">Flows</h1>'
            '<div class="app-sub">Choose which steps answer a question, in which order, and which agent runs each '
            'one. Build your own agents for steps the standard flow does not have.</div></div>',
            unsafe_allow_html=True)

msg = st.session_state.pop("fl_msg", None)
if msg:
    {"ok": st.success, "error": st.error, "warning": st.warning}[msg[0]](msg[1])

tab_flows, tab_agents = st.tabs(["Flows", "Custom agents"])

# ------------------------------------------------------------------ flows tab
with tab_flows:
    names = flow_store.names()
    if st.session_state.get("fl_name") not in names:
        st.session_state["fl_name"] = flow_store.default_name()
    default_name = flow_store.default_name()
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    name = c1.selectbox("Flow", names, key="fl_name", label_visibility="collapsed",
                        format_func=lambda n: n + ("  · default" if n == default_name else ""))
    flow = flow_store.get(name)
    c2.button("New copy", on_click=_new_flow, args=(name,), width="stretch",
              help="Make an editable copy of this flow.")
    c3.button("Delete", on_click=_delete_flow, args=(name,), width="stretch", disabled=flow.builtin)
    c4.button("Make default", on_click=_make_default, args=(name,), width="stretch",
              disabled=default_name == name, help="The flow new chat sessions start with.")

    if flow.builtin:
        st.info("Standard is built in and cannot be changed. Click **New copy** to make your own flow from it.")
        for i, step in enumerate(flow.steps, 1):
            kind = STEP_KINDS[step.kind]
            who = agent_of(step)
            st.markdown(f'<div class="step-title">{i} · {esc(kind.title)} '
                        f'{chip(who, "brand") if who else chip("no model", dot=False)}</div>'
                        f'<p class="step-what">{esc(kind.what)}</p>', unsafe_allow_html=True)
    else:
        with st.popover("Rename"):
            st.text_input("New name", value=name, key=f"fl_rename_{name}")
            st.button("Save name", on_click=_rename_flow, args=(name,))

        issues = check_flow(flow, agent_store.exists)
        for issue in (i for i in issues if not i.step_id):
            st.warning(issue.text)
        by_step: dict[str, list[str]] = {}
        for issue in (i for i in issues if i.step_id):
            by_step.setdefault(issue.step_id, []).append(issue.text)
        if issues:
            st.caption("Problems don't block saving; a flow that can't run stops with an explanation.")

        # ---- drag and drop order
        st.markdown('<div class="section">Order · drag to rearrange</div>', unsafe_allow_html=True)
        rev = st.session_state.get(f"fl_rev_{name}", 0)
        labels = [f"{i} · {title(s)}" + (f" · {agent_of(s)}" if agent_of(s) and s.kind != "custom" else "")
                  + ("  ⚠" if s.id in by_step else "") + f"  #{s.id}" for i, s in enumerate(flow.steps, 1)]
        if labels:
            order = sort_items(labels, direction="vertical", key=f"fl_sort_{name}_{rev}")
            ids = [lab.rsplit("#", 1)[1] for lab in order]
            if ids != [s.id for s in flow.steps] and sorted(ids) == sorted(s.id for s in flow.steps):
                flow.steps = [next(s for s in flow.steps if s.id == i) for i in ids]
                flow_store.save(flow)
                bump(name)
                st.rerun()

        # ---- steps: agent, what a custom step reads / writes, remove
        st.markdown('<div class="section">Steps</div>', unsafe_allow_html=True)
        custom = agent_store.all()
        for i, step in enumerate(flow.steps, 1):
            kind = STEP_KINDS[step.kind]
            with st.container(border=True):
                st.markdown(f'<div class="step-title">{i} · {esc(title(step))}</div>'
                            f'<p class="step-what">{esc(kind.what)}</p>', unsafe_allow_html=True)
                for text in by_step.get(step.id, []):
                    st.warning(text)
                cols = st.columns([3, 2, 2, 1]) if step.kind == "custom" else st.columns([3, 4, 1])
                akey = f"fl_{name}_{step.id}_agent"
                if step.kind == "custom":
                    # an empty first choice while the step has no (existing) agent: picking any agent then saves it
                    options = ([""] if step.agent not in custom else []) + list(custom)
                    if st.session_state.get(akey) not in options:
                        st.session_state[akey] = step.agent if step.agent in options else ""
                    cols[0].selectbox("Agent", options, key=akey, on_change=_set_step,
                                      args=(name, step.id, "agent", akey),
                                      format_func=lambda k: agent_label(k) if k else (
                                          "(choose an agent)" if custom else "(create a custom agent first)"))
                    for col, attr, choices in ((cols[1], "input", CUSTOM_INPUTS), (cols[2], "output", CUSTOM_OUTPUTS)):
                        k = f"fl_{name}_{step.id}_{attr}"
                        if st.session_state.get(k) not in choices:
                            current = getattr(step, attr)
                            st.session_state[k] = current if current in choices else next(iter(choices))
                        col.selectbox("Reads" if attr == "input" else "Writes", list(choices), key=k,
                                      format_func=choices.get, on_change=_set_step, args=(name, step.id, attr, k))
                elif kind.agent:
                    options = [""] + list(AGENTS) + list(custom)
                    if st.session_state.get(akey) not in options:
                        st.session_state[akey] = step.agent if step.agent in options else ""
                    cols[0].selectbox("Agent", options, key=akey, on_change=_set_step,
                                      args=(name, step.id, "agent", akey),
                                      format_func=lambda k, d=kind.agent: (
                                          f"Default ({agent_label(d)})" if not k
                                          else agent_label(k) + (" (custom)" if k in custom else "")))
                else:
                    cols[0].caption("Runs without a model.")
                cols[-1].button("Remove", key=f"fl_rm_{name}_{step.id}", on_click=_remove_step,
                                args=(name, step.id), width="stretch")

        # ---- add a step
        st.markdown('<div class="section">Add a step</div>', unsafe_allow_html=True)
        choices = [f"kind:{k}" for k in STEP_KINDS if k != "custom"] + [f"agent:{a}" for a in custom]
        a1, a2 = st.columns([4, 1])
        a1.selectbox("Step", choices, key=f"fl_add_{name}", label_visibility="collapsed",
                     format_func=lambda c: (STEP_KINDS[c[5:]].title if c.startswith("kind:")
                                            else f"{agent_label(c[6:])} (custom agent step)"))
        a2.button("Add", on_click=_add_step, args=(name,), type="primary", width="stretch")
        if not custom:
            st.caption("Custom agent steps appear here once you create an agent on the Custom agents tab.")

# ------------------------------------------------------------------ custom agents tab
with tab_agents:
    custom = agent_store.all()
    options = [NEW_AGENT] + list(custom)
    if st.session_state.get("ca_sel") not in options:
        st.session_state["ca_sel"] = NEW_AGENT
    sel = st.selectbox("Agent", options, key="ca_sel", label_visibility="collapsed",
                       format_func=lambda k: "➕ New agent" if k == NEW_AGENT else custom[k].name)
    agent = custom.get(sel)
    for prefix, value in (("ca_name_", agent.name if agent else ""),
                          ("ca_instr_", agent.instructions if agent else ""),
                          ("ca_model_", agent.model if agent else "")):
        st.session_state.setdefault(prefix + sel, value)          # initialise once; never also pass value=

    st.text_input("Name", key=f"ca_name_{sel}", placeholder="e.g. HR glossary")
    st.text_area("Instructions", key=f"ca_instr_{sel}", height=220,
                 placeholder="Who the agent is and what it does, e.g. \"You know the HR terms of this company. "
                             "Explain which columns and filters the terms in the question mean.\"",
                 help="Its system prompt. On its own step the output format is added automatically; on a built-in "
                      "step these instructions are added at the end of that step's prompt.")
    current_model = st.session_state[f"ca_model_{sel}"]
    models = list(dict.fromkeys([""] + ([current_model] if current_model else []) + pulled_models()))
    st.selectbox("Model", models, key=f"ca_model_{sel}", accept_new_options=True,
                 format_func=lambda m: m or "Default (the chat's answer model)",
                 help="Pick a pulled Ollama model or type a name.")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Save agent", on_click=_save_agent, args=(sel,), type="primary", width="stretch")
    if agent:
        b2.button("Delete", on_click=_delete_agent, args=(sel,), width="stretch")
        used = flows_using(sel)
        if used:
            st.caption("Used by: " + ", ".join(used) + ". Deleting it makes those steps fall back to defaults "
                       "(custom steps are skipped).")

        # ---- documents (same knowledge base as the built-in agents)
        st.markdown('<div class="section">Knowledge</div>', unsafe_allow_html=True)
        kb = KnowledgeBase()
        status = kb.status(sel)
        meta = chip(f"{status['documents']} documents", "ok" if status["documents"] else "")
        meta += chip("needs re-indexing", "warn") if status["stale"] else ""
        st.markdown(f'<div class="answer-meta">{meta}</div>', unsafe_allow_html=True)
        docs = kb.documents(sel)
        if docs:
            st.dataframe(pd.DataFrame({"document": docs}), width="stretch", hide_index=True)
        files = st.file_uploader("Documents", type=list(UPLOAD_TYPES), accept_multiple_files=True,
                                 key=f"ca_upload_{sel}", label_visibility="collapsed",
                                 help="PDF, Word, Markdown, text, SQL, CSV. The most relevant passages are added to "
                                      "this agent's prompt on every call.")
        if st.button(f"Add & re-index · {agent.name}", disabled=not files):
            with st.status(f"Indexing {agent.name}'s documents…", expanded=True) as box:
                failed = False
                for f in files or []:
                    try:
                        box.write(f"Read **{f.name}**: {kb.add_document(sel, f.name, f.getvalue())}")
                    except Exception as exc:
                        failed = True
                        box.write(f"Skipped **{f.name}**: {exc}")
                index = kb.build(sel)
                box.update(label=f"{agent.name}: {len(index['passages'])} passages indexed",
                           state="error" if failed and not index["passages"] else "complete", expanded=False)
            st.rerun()

try:
    st.page_link("app.py", label="Back to the chat", icon="🔎")
except StreamlitAPIException:        # page opened on its own (not through app.py): no navigation to link into
    pass
