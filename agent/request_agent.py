"""Request standardiser agent: turns whatever the user typed into one explicit, standalone request.

It is the first step of every turn. It gets the conversation memory (see context.py), the
previous query with its first rows, and the new message, and returns a StandardRequest:
a clean standalone question (typos fixed, abbreviations expanded, references such as "he" /
"that department" replaced by concrete IDs and names) plus the structured pieces the SQL
generator benefits from: filters, metrics, grouping, sort order, limit, and the assumptions
it had to make. The router, table linking and SQL generation all work from this request.

If the LLM fails the raw question is used and the memory text is still handed to the SQL step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import pandas as pd
from . import prompts
from .context import ConversationContext, _clean_list, _clip
from .llm import OllamaLLM
from .trace import Trace

TASKS = ("lookup", "list", "count", "aggregate", "rank", "compare", "trend", "analysis", "prediction", "other")
PREVIEW_ROWS = 5


@dataclass
class StandardRequest:
    question: str                       # what the user typed
    standalone_question: str            # explicit, self-contained version
    is_follow_up: bool = False
    task: str = "other"
    entities: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    grouping: list[str] = field(default_factory=list)
    time_range: str = ""
    sort: str = ""
    limit: int = 0
    ambiguities: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def details_text(self) -> str:
        """Short bullet block for the SQL prompt; empty when nothing was extracted."""
        parts = []
        if self.filters:
            parts.append("filters: " + "; ".join(self.filters))
        if self.metrics:
            parts.append("measures: " + "; ".join(self.metrics))
        if self.grouping:
            parts.append("break down by: " + ", ".join(self.grouping))
        if self.time_range:
            parts.append("time range: " + self.time_range)
        if self.sort:
            parts.append("sort: " + self.sort)
        if self.limit:
            parts.append(f"rows wanted: {self.limit}")
        if self.ambiguities:
            parts.append("assumptions made: " + "; ".join(self.ambiguities))
        return "\n".join(f"- {p}" for p in parts)


class RequestStandardizer:
    def __init__(self, llm: Any | None = None):
        self.llm = llm or OllamaLLM()

    def standardize(self, question: str, ctx: ConversationContext, previous: str, trace: Trace) -> StandardRequest:
        """previous = the last turn's question / SQL / first rows (concrete values to copy from)."""
        q = " ".join(question.split())
        req = StandardRequest(question=q, standalone_question=q)
        memory = ctx.as_text()
        with trace.step("Standardise the request",
                        "Rewrite the message into one explicit, standalone request: resolve references to earlier "
                        "turns, fix wording, and pull out filters, measures, grouping, sort and limit.") as s:
            s.add(context_given_to_model=memory or "(empty - first question of the conversation)")
            try:
                out = self.llm.chat_json(prompts.REQUEST_SYSTEM, prompts.request_user(memory, previous, q),
                                         prompts.REQUEST_SCHEMA)
                s.add_thinking(self.llm)
            except Exception as exc:
                s.status = "warning"
                s.add(note=f"Standardiser failed ({exc}); the message is used as typed.")
                return req
            req = _from_llm(q, out)
            s.reasoning = req.reasoning
            extracted = {k: v for k, v in req.to_dict().items() if v and k not in ("question", "reasoning",
                                                                                   "is_follow_up")}
            s.add(decision="Follow-up question" if req.is_follow_up else "Independent question",
                  original_question=q, **extracted)
        return req


# ---------------------------------------------------------------- helpers
def previous_turn_text(question: str, sql: str | None, frame: pd.DataFrame | None, answer: str = "") -> str:
    lines = [f"Question: {question}"]
    if sql:
        lines.append(f"SQL: {sql}")
    if frame is not None:
        lines.append(f"Result: {len(frame)} rows. First rows:\n"
                     f"{frame.head(PREVIEW_ROWS).to_string(index=False, max_colwidth=40)}")
    if answer:
        lines.append(f"Answer: {answer[:300]}")
    return "\n".join(lines)


def _from_llm(question: str, out: dict) -> StandardRequest:
    standalone = " ".join(str(out.get("standalone_question") or "").split())
    task = str(out.get("task") or "other").lower().strip()
    try:
        limit = max(0, int(out.get("limit") or 0))
    except (TypeError, ValueError):
        limit = 0
    return StandardRequest(
        question=question,
        standalone_question=standalone or question,
        is_follow_up=bool(out.get("is_follow_up")) and bool(standalone),
        task=task if task in TASKS else "other",
        entities=_clean_list(out.get("entities")),
        filters=_clean_list(out.get("filters")),
        metrics=_clean_list(out.get("metrics")),
        grouping=_clean_list(out.get("grouping")),
        time_range=_clip(str(out.get("time_range") or ""), 80),
        sort=_clip(str(out.get("sort") or ""), 80),
        limit=limit,
        ambiguities=_clean_list(out.get("ambiguities")),
        reasoning=str(out.get("reasoning") or ""),
    )
