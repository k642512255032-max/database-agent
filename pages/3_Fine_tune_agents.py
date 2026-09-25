"""Fine-tune agents page: pick an agent, upload documents, start fine-tuning.

Fourth Streamlit page (sidebar nav). "Start fine-tuning" indexes the agent's documents; from the next question on
the agent gets the relevant passages in its prompt (knowledge learning, agent/knowledge.py). The LoRA section
turns the same documents into a training dataset for real weight training on a GPU (deploy/colab/finetune_agent.py).
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from agent.config import settings
from agent.knowledge import AGENTS, UPLOAD_TYPES, KnowledgeBase
from agent.llm import OllamaLLM

st.set_page_config(page_title="Fine-tune agents", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

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
.agent-docs{color:var(--muted); font-size:.82rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(v) -> str:
    return html.escape(str(v))


def chip(text: str, tone: str = "", dot: bool = True) -> str:
    marker = '<span class="dot"></span>' if dot else ""
    return f'<span class="chip {tone}">{marker}{esc(text)}</span>'


kb = KnowledgeBase()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## Local Data Agent")
    st.markdown('<div class="side-label">Dataset writer</div>', unsafe_allow_html=True)
    gen_model = st.text_input("Dataset model", settings.answer_model or settings.model, label_visibility="collapsed",
                              placeholder="Ollama model that writes training examples",
                              help="Only used by 'Generate training dataset'. A larger model writes better examples.")
    st.markdown('<div class="side-label">How it works</div>', unsafe_allow_html=True)
    st.markdown('<p class="side-note"><b>Knowledge learning</b> runs here in seconds: the documents are indexed and '
                'the most relevant passages are added to the agent\'s prompt on every call. The trace shows them '
                'under <i>knowledge_used</i>.</p><p class="side-note"><b>LoRA training</b> changes the model weights. '
                'It needs a GPU: export the dataset and run it on the Colab host.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ header
st.markdown('<div class="app-head"><h1 class="app-title">Fine-tune agents</h1>'
            '<div class="app-sub">Choose an agent, upload documents that teach it (glossaries, data dictionaries, '
            'business rules, example questions), then start fine-tuning. Each agent learns only from its own '
            'documents.</div></div>', unsafe_allow_html=True)

agent = st.segmented_control("Agent", list(AGENTS), default="understanding", key="ft_agent",
                             format_func=lambda a: AGENTS[a][0], label_visibility="collapsed") or "understanding"
label, what, docs_help = AGENTS[agent]
status = kb.status(agent)

meta = chip(label, "brand")
if status["built_at"]:
    meta += chip("needs re-tuning" if status["stale"] else f"tuned {status['built_at'].replace('T', ' ')}",
                 "warn" if status["stale"] else "ok")
else:
    meta += chip("not tuned yet", dot=False)
meta += chip(f"{status['documents']} documents · {status['passages']} passages", dot=False)
if status["examples"]:
    meta += chip(f"{status['examples']} examples", dot=False)
if status["model"]:
    meta += chip(f"model {status['model']}", "ok")
st.markdown(f'<div class="answer-meta">{meta}</div><p class="agent-what">{esc(what)}</p>'
            f'<p class="agent-docs">Helpful documents: {esc(docs_help)}</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ 1. documents + start
st.markdown('<div class="section">1 · Upload documents</div>', unsafe_allow_html=True)
files = st.file_uploader("Documents", type=list(UPLOAD_TYPES), accept_multiple_files=True,
                         key=f"upload_{agent}", label_visibility="collapsed",
                         help="PDF, Word, Markdown, text, SQL, CSV. A .jsonl file of {\"input\", \"output\"} lines is "
                              "stored as hand-written examples for LoRA training.")

if st.button(f"Start fine-tuning · {label}", type="primary", disabled=not files and not status["documents"]):
    with st.status(f"Fine-tuning the {label} agent…", expanded=True) as box:
        failed = False
        for f in files or []:
            try:
                box.write(f"Read **{f.name}**: {kb.add_document(agent, f.name, f.getvalue())}")
            except Exception as exc:
                failed = True
                box.write(f"Skipped **{f.name}**: {exc}")
        index = kb.build(agent)
        box.write(f"Indexed {len(index['documents'])} documents into {len(index['passages'])} passages.")
        box.update(label=f"{label} agent tuned: its next calls use these documents",
                   state="error" if failed and not index["passages"] else "complete", expanded=False)
    st.rerun()

docs = kb.documents(agent)
if docs:
    rows = pd.DataFrame({"document": docs})
    left, right = st.columns([3, 1])
    left.dataframe(rows, width="stretch", hide_index=True)
    with right:
        drop = st.selectbox("Remove", docs, key=f"drop_{agent}", label_visibility="collapsed")
        if st.button("Remove document", width="stretch"):
            kb.remove_document(agent, drop)
            kb.build(agent)
            st.rerun()
        if st.button("Reset agent", width="stretch", help="Delete this agent's documents, examples, dataset and model."):
            kb.reset(agent)
            st.rerun()

# ------------------------------------------------------------------ 2. try it
st.markdown('<div class="section">2 · Check what the agent will read</div>', unsafe_allow_html=True)
probe = st.text_input("Question", key=f"probe_{agent}", label_visibility="collapsed",
                      placeholder="Type a question to see which passages would be added to the agent's prompt")
if probe:
    hits = kb.search(agent, probe)
    if not status["passages"]:
        st.caption("Nothing learned yet: upload documents and start fine-tuning.")
    elif not hits:
        st.caption("No passage is relevant to this question; the agent runs as usual.")
    for i, p in enumerate(hits, 1):
        with st.container(border=True):
            st.markdown(f'<div class="answer-meta">{chip(f"[{i}] {p.source}", "brand")}{chip(f"score {p.score}", dot=False)}</div>',
                        unsafe_allow_html=True)
            st.markdown(esc(p.text).replace("\n", "<br>"), unsafe_allow_html=True)

# ------------------------------------------------------------------ 3. LoRA
st.markdown('<div class="section">3 · LoRA training (GPU, optional)</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("Real fine-tuning changes the model's weights. **a.** Generate a dataset from the documents (plus any "
                "uploaded `.jsonl` examples). **b.** Train it on the Colab GPU. **c.** Enter the new model below so "
                "this agent uses it.")
    c1, c2 = st.columns([1, 1])
    per = c1.number_input("Examples per passage", 1, 8, 3, key=f"per_{agent}")
    if c2.button("a. Generate training dataset", disabled=not (status["passages"] or status["examples"]),
                 width="stretch"):
        bar = st.progress(0.0, text="Writing examples…")
        n = kb.generate_examples(agent, OllamaLLM(model=gen_model, think=False), per_passage=int(per),
                                 on_progress=lambda i, t: bar.progress(i / t, text=f"Passage {i} of {t}"))
        bar.empty()
        st.success(f"Dataset ready: {n} examples.")
        status = kb.status(agent)
    if status["dataset"]:
        path = kb.dataset_path(agent)
        st.download_button(f"Download dataset ({status['dataset']} examples, .jsonl)", path.read_bytes(),
                           file_name=f"{agent}_train.jsonl", mime="application/jsonl")
        with st.expander("Preview"):
            st.code("\n".join(path.read_text(encoding="utf-8").splitlines()[:5]), language="json", wrap_lines=True)
        if status["dataset"] < 200:
            st.caption(f"{status['dataset']} examples is a small dataset; a few hundred give a steadier result. "
                       "Hand-written .jsonl examples are worth more than generated ones.")
    base = (settings.answer_model or settings.model) if agent in ("answer", "memory", "expert") else settings.model
    st.markdown("**b.** On the Colab host (T4 GPU), upload the dataset and run:")
    st.code(f"!python deploy/colab/finetune_agent.py --data {agent}_train.jsonl --base {base} --name {agent}-ft",
            language="bash")
    st.markdown("**c.** Model for this agent (empty = the default model):")
    m1, m2 = st.columns([3, 1])
    new_model = m1.text_input("Fine-tuned model", status["model"], key=f"model_{agent}", label_visibility="collapsed",
                              placeholder=f"e.g. {agent}-ft")
    if m2.button("Save model", width="stretch"):
        kb.set_model(agent, new_model)
        st.rerun()
    if status["model"]:
        ok, msg = OllamaLLM(model=status["model"]).health()
        st.caption(("✅ " if ok else "⚠️ ") + msg)
