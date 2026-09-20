"""Context builder agent: keeps a compact, structured memory of the conversation.

After every answered message it rewrites the memory (summary, entities in play, standing
filters, metrics, user preferences, facts already found). The next message's request
standardiser and the SQL generator read this memory instead of a raw dump of earlier turns,
so references like "he", "that department" or "the same period" resolve against concrete
values (IDs, names, dates) and standing instructions ("always top 10") are not forgotten.

It never computes anything: values in the memory are copied from questions, SQL and result
rows, and the LLM call is recorded as a trace step. If the LLM fails the memory is updated
deterministically (tables from the SQL, one finding line) so later turns still have something.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import prompts
from .config import settings
from .llm import OllamaLLM
from .trace import Trace

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .orchestrator import AgentResult

MAX_FINDINGS = 8
MAX_ITEMS = 10          # per list field
MAX_ITEM_CHARS = 160
PREVIEW_ROWS = 5


@dataclass
class ConversationContext:
    summary: str = ""                                   # 1-3 sentences: what the user has been exploring
    entities: dict[str, str] = field(default_factory=dict)   # {"employee": "emp_no 10001 (Georgi Facello)"}
    filters: list[str] = field(default_factory=list)    # conditions the user keeps applying
    metrics: list[str] = field(default_factory=list)    # measures the user cares about
    preferences: list[str] = field(default_factory=list)  # standing instructions ("always show top 10")
    findings: list[str] = field(default_factory=list)   # facts established so far, with concrete values
    tables: list[str] = field(default_factory=list)     # tables used so far (deterministic, from SQL)
    turns: int = 0

    # ------------------------------------------------------------- views
    def is_empty(self) -> bool:
        return self.turns == 0 and not (self.summary or self.entities or self.filters or self.findings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ConversationContext":
        d = d or {}
        ctx = cls()
        ctx.summary = _clip(str(d.get("summary") or ""), 600)
        ents = d.get("entities") or {}
        if isinstance(ents, list):   # tolerate ["employee: ..."] from a sloppy model
            ents = dict(e.split(":", 1) for e in ents if isinstance(e, str) and ":" in e)
        ctx.entities = {_clip(str(k), 40): _clip(str(v), MAX_ITEM_CHARS) for k, v in list(ents.items())[:MAX_ITEMS]
                        if str(v).strip()}
        for name in ("filters", "metrics", "preferences", "tables"):
            setattr(ctx, name, _clean_list(d.get(name)))
        ctx.findings = _clean_list(d.get("findings"), limit=MAX_FINDINGS)
        try:
            ctx.turns = int(d.get("turns") or 0)
        except (TypeError, ValueError):
            ctx.turns = 0
        return ctx

    def as_text(self) -> str:
        """Compact text given to the LLMs (standardiser, SQL generator)."""
        if self.is_empty():
            return ""
        lines = []
        if self.summary:
            lines.append(f"So far: {self.summary}")
        if self.entities:
            lines.append("Entities in play: " + "; ".join(f"{k} = {v}" for k, v in self.entities.items()))
        if self.filters:
            lines.append("Standing filters: " + "; ".join(self.filters))
        if self.metrics:
            lines.append("Metrics of interest: " + "; ".join(self.metrics))
        if self.preferences:
            lines.append("User preferences: " + "; ".join(self.preferences))
        if self.tables:
            lines.append("Tables used: " + ", ".join(self.tables))
        if self.findings:
            lines.append("Facts already found:\n" + "\n".join(f"- {f}" for f in self.findings))
        return "\n".join(lines)


class ContextBuilder:
    def __init__(self, llm: Any | None = None):
        # defaults to the answer model: summarising is prose work, not SQL work
        self.llm = llm or OllamaLLM(model=settings.answer_model or None)

    def update(self, ctx: ConversationContext, res: "AgentResult", trace: Trace) -> ConversationContext:
        turn = turn_text(res)
        with trace.step("Update conversation context",
                        "Fold this turn into the conversation memory (entities, filters, preferences, facts) "
                        "so later questions can refer back to it.") as s:
            new = None
            try:
                out = self.llm.chat_json(prompts.CONTEXT_SYSTEM,
                                         prompts.context_user(json.dumps(ctx.to_dict(), ensure_ascii=False), turn),
                                         prompts.CONTEXT_SCHEMA)
                new = ConversationContext.from_dict(out)
                s.reasoning = out.get("reasoning")
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Context model failed ({exc}); memory updated without the LLM.")
            new = _merge(ctx, new, res)
            s.add(turn_given_to_model=turn, updated_context=json.dumps(new.to_dict(), ensure_ascii=False, indent=1))
        return new


# ---------------------------------------------------------------- helpers
def turn_text(res: "AgentResult") -> str:
    """The latest turn as the context model sees it."""
    lines = [f"User message: {res.question}"]
    if res.standalone_question and res.standalone_question != res.question:
        lines.append(f"Standardised request: {res.standalone_question}")
    req = res.extras.get("request") or {}
    if req.get("filters") or req.get("metrics"):
        lines.append(f"Filters: {'; '.join(req.get('filters') or []) or 'none'}. "
                     f"Metrics: {'; '.join(req.get('metrics') or []) or 'none'}.")
    lines.append(f"Intent: {res.intent}" + (f" (model {res.model_name})" if res.model_name else ""))
    if res.sql:
        lines.append(f"SQL: {res.sql}")
    if res.data is not None:
        frame = res.frame()
        preview = frame.head(PREVIEW_ROWS) if frame is not None else pd.DataFrame()
        lines.append(f"Result: {len(res.data)} rows. First rows:\n{preview.to_string(index=False, max_colwidth=40)}")
    if res.stats:
        lines.append(f"Statistics: {res.stats.get('interpretation', '')[:400]}")
    if res.ml is not None:
        lines.append(f"Model summary: {json.dumps(res.ml.summary, default=str)[:400]}")
    if res.answer:
        lines.append(f"Answer: {res.answer[:500]}")
    if res.error:
        lines.append(f"Error: {res.error}")
    return "\n".join(lines)


def _merge(old: ConversationContext, new: ConversationContext | None, res: "AgentResult") -> ConversationContext:
    """Deterministic part of the update: table bookkeeping, turn count, fallback when the LLM failed."""
    ctx = new or ConversationContext.from_dict(old.to_dict())
    if new is None:   # LLM failed: at least remember the question and what came back
        n = len(res.data) if res.data is not None else 0
        ctx.findings = _clean_list(old.findings + [f"{res.standalone_question or res.question} -> {n} rows"],
                                   limit=MAX_FINDINGS)
    used = re.findall(r"(?:from|join)\s+`?(\w+)", res.sql or "", re.I)
    ctx.tables = _clean_list(old.tables + used)
    ctx.turns = old.turns + 1
    return ctx


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _clean_list(values: Any, limit: int = MAX_ITEMS) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        v = _clip(str(v), MAX_ITEM_CHARS)
        if v and v not in out:
            out.append(v)
    return out[-limit:]   # newest last: keep the tail
