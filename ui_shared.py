"""Objects shared by the Streamlit pages (app.py and pages/*.py).

Defined once so that both pages get the same cached DataAgent - and therefore
the same introspected schema - for a given connection and model.
"""
from __future__ import annotations

import streamlit as st

from agent.config import settings
from agent.db import Database
from agent.llm import OllamaLLM
from agent.orchestrator import DataAgent
from ml.registry import ModelRegistry


@st.cache_resource(show_spinner=False)
def get_agent(db_url: str, model: str) -> DataAgent:
    return DataAgent(db=Database(db_url), llm=OllamaLLM(model=model), registry=ModelRegistry())


def expert_persona_input() -> str:
    """Sidebar 'Domain & goals' box for the Expert AI, shared by all pages through one session_state key."""
    if "expert_persona" not in st.session_state:
        st.session_state.expert_persona = settings.expert_persona
    return st.text_area("Domain & goals", key="expert_persona", height=90, label_visibility="collapsed",
                        placeholder="Domain & goals, e.g. HR analytics for a 300k-employee company; we care about "
                                    "pay equity and retention",
                        help="Who the Expert AI should be and what you care about. Injected into every expert prompt.")
