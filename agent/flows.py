"""Custom flows: which steps a question goes through, in what order, and which agent runs each step.

A Flow is an ordered list of FlowSteps. Each step has a kind from STEP_KINDS (the built-in pipeline stages, plus
"custom" for a user-made agent's own step) and optionally an agent that runs it instead of the kind's default agent.
The built-in STANDARD flow is today's pipeline; users copy it on pages/5_Flows.py, then add, remove, reorder steps
and assign agents. DataAgent.ask(flow=...) walks the steps (agent/orchestrator.py).

The editor only warns (check_flow): a flow whose step needs something no earlier step produces can be saved, and
stops at run time with an explanation (FlowError) naming the step to move or add.

Saved to settings.flows_file as {"default": name, "flows": {name: {"steps": [...]}}}. "Standard" is never saved.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .config import settings
from .custom_agents import CUSTOM_INPUTS, CUSTOM_OUTPUTS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepKind:
    key: str
    title: str                  # shown in the editor, e.g. "Write & run SQL"
    what: str                   # one sentence for the editor
    agent: str                  # default agent (knowledge.AGENTS key); "" = deterministic, no model
    needs: tuple[str, ...]      # state that must exist before it runs
    provides: tuple[str, ...]   # state it creates
    switch: str = ""            # chat-sidebar switch that can turn it off
    unique: bool = True         # a second copy in one flow is flagged by check_flow


STEP_KINDS: dict[str, StepKind] = {k.key: k for k in (
    StepKind("standardise", "Standardise the request", "Rewrites the message into one explicit question and pulls "
             "out filters, measures, grouping, sort and limit.", "understanding", (), ("request",), "standardise"),
    StepKind("route", "Understand the request", "Decides whether the question is a plain data query, statistics "
             "or machine learning.", "router", (), ("intent",)),
    StepKind("expert_plan", "Expert data plan", "The expert decides which tables, columns and filters answer the "
             "request and writes the order for the SQL writer.", "expert", (), ("plan",), "expert_plan"),
    StepKind("link_tables", "Select relevant tables", "Picks the tables that go into the SQL writer's prompt "
             "(from the expert plan when there is one). No model.", "", (), ("tables",)),
    StepKind("sql", "Write & run SQL", "Writes the SQL, checks it is read-only, runs it and repairs it when the "
             "database returns an error.", "sql", ("tables",), ("sql", "data")),
    StepKind("analysis", "Statistics / ML analysis", "Runs the statistical test or the trained model when the "
             "question needs one; does nothing for plain data queries.", "router", ("data",), ("analysis",)),
    StepKind("expert_assess", "Expert assessment", "The expert judges the data and writes its own answer, insights "
             "and advice.", "expert", ("data",), ("expert",), "expert_assess"),
    StepKind("answer", "Write the answer", "Turns the computed facts into the plain-English answer.", "answer",
             ("data",), ("answer",)),
    StepKind("charts", "Plan charts", "Chooses charts for the result table.", "answer", ("data",), ("charts",),
             "charts"),
    StepKind("memory", "Update conversation memory", "Remembers this turn for follow-up questions. Always runs "
             "after the answer is shown.", "memory", (), (), "memory"),
    StepKind("custom", "Custom agent step", "A custom agent reads one part of the turn and writes a note, a "
             "rewritten question or a section of the answer.", "", (), (), unique=False),
)}

STATE_LABELS = {"question": "the question", "request": "the standardised request", "intent": "the intent",
                "plan": "the expert plan", "tables": "the selected tables", "sql": "the SQL query",
                "data": "the result table", "analysis": "the analysis", "expert": "the expert assessment",
                "answer": "the answer", "charts": "the charts", "notes": "notes from custom agents"}

# what a custom step's input needs to exist first
_INPUT_NEEDS = {"question": set(), "request": {"request"}, "sql": {"sql"}, "table": {"data"},
                "expert": {"expert"}, "answer": {"answer"}}


@dataclass
class FlowStep:
    id: str                 # "s1", "s2"... unique in its flow and stable across reorders (the editor maps by it)
    kind: str
    agent: str = ""         # "" = the kind's default agent; else a knowledge.AGENTS key or a custom agent id
    input: str = ""         # custom steps: key of CUSTOM_INPUTS
    output: str = ""        # custom steps: key of CUSTOM_OUTPUTS

    def needs(self) -> set[str]:
        if self.kind == "custom":
            return set(_INPUT_NEEDS.get(self.input, set())) | ({"answer"} if self.output == "answer_section" else set())
        kind = STEP_KINDS.get(self.kind)
        return set(kind.needs) if kind else set()

    def provides(self) -> set[str]:
        if self.kind == "custom":
            return {"notes"} if self.output == "note" else set()
        kind = STEP_KINDS.get(self.kind)
        return set(kind.provides) if kind else set()


@dataclass
class Flow:
    name: str
    steps: list[FlowStep] = field(default_factory=list)
    builtin: bool = False

    def to_dict(self) -> dict:
        return {"steps": [asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, name: str, data: dict, builtin: bool = False) -> "Flow":
        steps = []
        for raw in data.get("steps", []) if isinstance(data, dict) else []:
            if not isinstance(raw, dict) or raw.get("kind") not in STEP_KINDS or not raw.get("id"):
                log.warning("flow %s: step %r ignored (unknown kind or no id)", name, raw)
                continue
            steps.append(FlowStep(str(raw["id"]), raw["kind"], str(raw.get("agent") or ""),
                                  str(raw.get("input") or ""), str(raw.get("output") or "")))
        return cls(name, steps, builtin)


STANDARD_NAME = "Standard"
STANDARD = Flow(STANDARD_NAME, [FlowStep(f"s{i}", k) for i, k in enumerate(
    ["standardise", "route", "expert_plan", "link_tables", "sql", "analysis", "expert_assess", "answer", "charts",
     "memory"], 1)], builtin=True)


class FlowError(RuntimeError):
    """A flow that cannot run as ordered; the message says which step to move or add."""


@dataclass(frozen=True)
class Issue:
    step_id: str    # "" for flow-level issues
    text: str


def step_title(step: FlowStep, agent_name: Callable[[str], str] | None = None) -> str:
    if step.kind == "custom":
        name = agent_name(step.agent) if agent_name and step.agent else (step.agent or "Custom agent")
        return f"{name} step"
    kind = STEP_KINDS.get(step.kind)
    return kind.title if kind else step.kind


def producer_of(state: str) -> str:
    """Title of the step kind that produces `state` (for explanations)."""
    if state == "notes":
        return "a custom step that writes a note"
    kind = next((k for k in STEP_KINDS.values() if state in k.provides), None)
    return f"'{kind.title}'" if kind else state


def missing_text(step: FlowStep, missing: set[str], in_flow: bool) -> str:
    """"Step 'Write the answer' needs the result table, produced by 'Write & run SQL'. Move it after that step." """
    what = " and ".join(STATE_LABELS.get(m, m) for m in sorted(missing))
    producers = ", ".join(sorted({producer_of(m) for m in missing}))
    fix = "Move it after that step." if in_flow else "Add that step before it."
    return f"Step '{step_title(step)}' needs {what}, produced by {producers}. {fix}"


def check_flow(flow: Flow, agent_exists: Callable[[str], bool]) -> list[Issue]:
    """Problems the Flows page shows. Warn only: none of them stops a flow from being saved."""
    from .knowledge import AGENTS
    issues: list[Issue] = []
    if not flow.steps:
        return [Issue("", "The flow has no steps.")]
    if "sql" not in {s.kind for s in flow.steps}:
        issues.append(Issue("", "Without 'Write & run SQL' the flow never reads the database."))
    provided = {"question"}
    seen: set[str] = set()
    all_provides = set().union(*(s.provides() for s in flow.steps))
    for i, step in enumerate(flow.steps):
        kind = STEP_KINDS[step.kind]
        if kind.unique and step.kind in seen:
            issues.append(Issue(step.id, f"'{kind.title}' appears twice; it runs twice."))
        seen.add(step.kind)
        if step.kind == "custom":
            if not step.agent:
                issues.append(Issue(step.id, "Choose the custom agent that runs this step."))
            if step.input not in CUSTOM_INPUTS or step.output not in CUSTOM_OUTPUTS:
                issues.append(Issue(step.id, "Choose what this step reads and what it writes."))
        missing = step.needs() - provided
        if missing:
            issues.append(Issue(step.id, missing_text(step, missing, in_flow=missing <= all_provides)))
        if step.kind == "memory" and i != len(flow.steps) - 1:
            issues.append(Issue(step.id, "'Update conversation memory' always runs after the answer is shown, "
                                         "wherever it is in the list."))
        if step.agent and step.agent not in AGENTS and not agent_exists(step.agent):
            issues.append(Issue(step.id, f"Agent '{step.agent}' no longer exists; " +
                                ("this step is skipped." if step.kind == "custom"
                                 else "the default agent runs this step.")))
        provided |= step.provides()
    return issues


class FlowStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.flows_file)
        self._cache: tuple[tuple[int, int], dict] | None = None    # ((mtime_ns, size), data)

    def _data(self) -> dict:
        """{"default": str, "flows": {name: dict}}, re-read only when the file changed."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return {"default": STANDARD_NAME, "flows": {}}
        stamp = (st.st_mtime_ns, st.st_size)          # size too: two quick writes can share a timestamp tick
        if self._cache and self._cache[0] == stamp:
            return self._cache[1]
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("flows", {}), dict):
                raise ValueError('expected {"default": ..., "flows": {...}}')
        except (OSError, ValueError) as exc:           # json.JSONDecodeError is a ValueError
            log.warning("flows in %s ignored: %s", self.path, exc)
            data = {}
        data = {"default": str(data.get("default") or STANDARD_NAME),
                "flows": {k: v for k, v in data.get("flows", {}).items() if k != STANDARD_NAME}}
        self._cache = (stamp, data)
        return data

    def names(self) -> list[str]:
        return [STANDARD_NAME] + sorted(self._data()["flows"])

    def get(self, name: str) -> Flow:
        """A copy of the flow (edits never touch STANDARD or the cache); Standard for unknown names."""
        raw = self._data()["flows"].get(name)
        if raw is None:
            return Flow.from_dict(STANDARD_NAME, STANDARD.to_dict(), builtin=True)
        return Flow.from_dict(name, raw)

    def default_name(self) -> str:
        name = self._data()["default"]
        return name if name in self.names() else STANDARD_NAME

    def save(self, flow: Flow) -> None:
        name = flow.name.strip()
        if not name:
            raise ValueError("Give the flow a name.")
        if name == STANDARD_NAME:
            raise ValueError("Standard is built in; save your changes under another name.")
        data = self._copy()
        data["flows"][name] = flow.to_dict()
        self._write(data)

    def rename(self, old: str, new: str) -> None:
        new = new.strip()
        data = self._copy()
        if old not in data["flows"]:
            raise ValueError(f"No saved flow called '{old}'.")
        if not new or new == STANDARD_NAME:
            raise ValueError("Choose another name.")
        if new != old and new in data["flows"]:
            raise ValueError(f"A flow called '{new}' already exists.")
        data["flows"][new] = data["flows"].pop(old)
        if data["default"] == old:
            data["default"] = new
        self._write(data)

    def delete(self, name: str) -> None:
        data = self._copy()
        if data["flows"].pop(name, None) is not None:
            if data["default"] == name:
                data["default"] = STANDARD_NAME
            self._write(data)

    def set_default(self, name: str) -> None:
        data = self._copy()
        data["default"] = name if name in data["flows"] else STANDARD_NAME
        self._write(data)

    def unique_name(self, base: str) -> str:
        names, n, name = set(self.names()), 1, base
        while name in names:
            n += 1
            name = f"{base} {n}"
        return name

    @staticmethod
    def next_step_id(flow: Flow) -> str:
        """A new id that no earlier step had: the page keys its widgets by step id, so a reused id would show the
        removed step's choices."""
        taken = {s.id for s in flow.steps}
        while True:
            sid = f"s{uuid.uuid4().hex[:8]}"
            if sid not in taken:
                return sid

    def _copy(self) -> dict:
        data = self._data()
        return {"default": data["default"], "flows": dict(data["flows"])}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)                    # atomic: a reader never sees half a file
        self._cache = None


flow_store = FlowStore()
