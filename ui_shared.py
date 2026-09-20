"""Objects shared by the Streamlit pages (app.py and pages/*.py).

Defined once so that both pages get the same cached DataAgent - and therefore
the same introspected schema - for a given connection and model.
"""
from __future__ import annotations

import streamlit as st

from agent.db import Database
from agent.llm import OllamaLLM
from agent.orchestrator import DataAgent
from ml.registry import ModelRegistry


@st.cache_resource(show_spinner=False)
def get_agent(db_url: str, model: str) -> DataAgent:
    return DataAgent(db=Database(db_url), llm=OllamaLLM(model=model), registry=ModelRegistry())
