"""What the "Agent settings" page shows per agent: its system prompts (with their defaults) and its default model.

Keys are the ones the call sites pass to agent.prompt_store.prompt(); tests/test_agent_settings.py checks that the
two stay in sync. Only the page and the tests import this module (it imports the agents, so the agents must not
import it).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .answer_agent import ANSWER_SYSTEM, CHART_SYSTEM, DATA_SUMMARY_SYSTEM
from .config import settings
from .expert_agent import _COMMON, AUDIT_SYSTEM, PLAN_SYSTEM, REVIEW_FORMAT, TASK_FOCUS


@dataclass(frozen=True)
class PromptSpec:
    key: str
    agent: str                            # key in agent.knowledge.AGENTS
    label: str
    when: str                             # when the pipeline sends it
    default: str
    placeholders: tuple[str, ...] = ()    # () = sent as is, never formatted


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    PromptSpec("understanding.system", "understanding", "Request standardiser", "Step 0, every message",
               prompts.REQUEST_SYSTEM),
    PromptSpec("router.system", "router", "Intent router", "Step 1, unless the router shortcut applies",
               prompts.ROUTER_SYSTEM),
    PromptSpec("router.stats", "router", "Statistics method chooser",
               "Step 6a, statistics requests", prompts.STATS_SYSTEM),
    PromptSpec("sql.system", "sql", "SQL writer", "Step 3, every request", prompts.SQL_SYSTEM, ("dialect",)),
    PromptSpec("sql.fix", "sql", "SQL repair", "Step 5, after a failed query", prompts.FIX_SYSTEM, ("dialect",)),
    PromptSpec("answer.full", "answer", "Full answer", "Step 8, statistics and ML answers", ANSWER_SYSTEM),
    PromptSpec("answer.summary", "answer", "Data-query summary", "Step 8, plain data queries", DATA_SUMMARY_SYSTEM),
    PromptSpec("answer.charts", "answer", "Chart planner", "Step 9", CHART_SYSTEM),
    PromptSpec("expert.plan", "expert", "Data plan", "Step 2", PLAN_SYSTEM, ("persona",)),
    PromptSpec("expert.review", "expert", "Assessment: role", "Step 7, start of the prompt", _COMMON, ("persona",)),
    PromptSpec("expert.focus.data_query", "expert", "Assessment: data-query focus", "Step 7, data queries",
               TASK_FOCUS["data_query"]),
    PromptSpec("expert.focus.statistics", "expert", "Assessment: statistics focus", "Step 7, statistics",
               TASK_FOCUS["statistics"]),
    PromptSpec("expert.focus.machine_learning", "expert", "Assessment: ML focus", "Step 7, machine learning",
               TASK_FOCUS["machine_learning"]),
    PromptSpec("expert.review_format", "expert", "Assessment: output format", "Step 7, end of the prompt",
               REVIEW_FORMAT),
    PromptSpec("expert.audit", "expert", "Table audit", "Expert audit page", AUDIT_SYSTEM, ("persona",)),
    PromptSpec("memory.system", "memory", "Context builder", "Step 10, after every answer", prompts.CONTEXT_SYSTEM),
)
SPECS = {s.key: s for s in PROMPT_SPECS}


def specs_for(agent: str) -> list[PromptSpec]:
    return [s for s in PROMPT_SPECS if s.agent == agent]


def default_model(agent: str) -> str:
    """Model an agent runs on without an override (from .env; the chat page's sidebar can change it per session)."""
    if agent == "expert":
        return settings.expert_model or settings.answer_model or settings.model
    if agent in ("answer", "memory"):
        return settings.answer_model or settings.model
    return settings.model
