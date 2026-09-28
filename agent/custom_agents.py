"""User-made agents for custom flows (pages/5_Flows.py, agent/flows.py).

A custom agent is a name, instructions (its system prompt), a model and optional documents (indexed like a built-in
agent's, see agent/knowledge.py). In a flow it either runs its own step - reads one piece of the turn (question, SQL,
result table, ...) and writes a note for later steps, a rewritten question or a section of the answer - or it is
assigned to run a built-in step, whose system prompt then ends with the agent's instructions (InstructedLLM).

Saved to settings.custom_agents_file as {id: {name, instructions, model}}. A missing or broken file means no custom
agents: the pipeline never fails over it.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import settings
from .knowledge import AGENTS

log = logging.getLogger(__name__)


@dataclass
class CustomAgent:
    id: str             # slug, unique, never a key of knowledge.AGENTS
    name: str
    instructions: str   # the agent's system prompt (plain text, never str.format-ed)
    model: str = ""     # "" = the answer model of the session / .env


class AgentStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.custom_agents_file)
        self._cache: tuple[tuple[int, int], dict[str, CustomAgent]] | None = None    # ((mtime_ns, size), agents)

    def all(self) -> dict[str, CustomAgent]:
        """Saved agents, re-read only when the file changed (cheap to call on every step)."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return {}
        stamp = (st.st_mtime_ns, st.st_size)          # size too: two quick writes can share a timestamp tick
        if self._cache and self._cache[0] == stamp:
            return self._cache[1]
        agents: dict[str, CustomAgent] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for key, v in (data.items() if isinstance(data, dict) else []):
                if isinstance(v, dict) and str(v.get("name", "")).strip() and str(v.get("instructions", "")).strip():
                    agents[key] = CustomAgent(key, str(v["name"]), str(v["instructions"]), str(v.get("model") or ""))
        except (OSError, ValueError) as exc:          # ValueError: bad JSON or not UTF-8
            log.warning("custom agents in %s ignored: %s", self.path, exc)
            agents = {}
        self._cache = (stamp, agents)
        return agents

    def get(self, agent_id: str) -> CustomAgent | None:
        return self.all().get(agent_id)

    def exists(self, agent_id: str) -> bool:
        return agent_id in self.all()

    def save(self, agent: CustomAgent) -> None:
        if not agent.name.strip():
            raise ValueError("Give the agent a name.")
        if not agent.instructions.strip():
            raise ValueError("Write the agent's instructions.")
        if agent.id in AGENTS:
            raise ValueError(f"'{agent.id}' is a built-in agent.")
        data = {k: asdict(v) for k, v in self.all().items()}
        data[agent.id] = asdict(agent)
        self._write(data)

    def delete(self, agent_id: str) -> None:
        data = {k: asdict(v) for k, v in self.all().items()}
        if data.pop(agent_id, None) is not None:
            self._write(data)

    @staticmethod
    def new_id(name: str, taken: set[str]) -> str:
        """A slug of the name that is not a built-in agent key, not in `taken` and has no leftover folder in the
        knowledge directory, so a new agent never inherits documents ("agent", "hr_glossary_2", ...)."""
        base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "agent"
        candidate, n = base, 1
        while candidate in AGENTS or candidate in taken or (settings.knowledge_dir / candidate).exists():
            n += 1
            candidate = f"{base}_{n}"
        return candidate

    def _write(self, data: dict[str, dict]) -> None:
        for v in data.values():
            v.pop("id", None)                          # the key is the id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)                    # atomic: a reader never sees half a file
        self._cache = None


agent_store = AgentStore()


def agent_label(key: str) -> str:
    """Display name of a built-in or custom agent; the key itself when it is unknown (e.g. a deleted agent)."""
    if key in AGENTS:
        return AGENTS[key][0]
    agent = agent_store.get(key)
    return agent.name if agent else key


class InstructedLLM:
    """Wraps an LLM: a custom agent assigned to a built-in step keeps the step's system prompt (and output format)
    and adds its own instructions at the end."""

    def __init__(self, inner: Any, name: str, instructions: str):
        self.inner, self.name, self.instructions = inner, name, instructions

    def _sys(self, system: str) -> str:
        return f"{system}\n\nAdditional instructions from the {self.name} agent:\n{self.instructions}"

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        return self.inner.chat_json(self._sys(system), user, schema)

    def chat(self, system: str, user: str, schema: dict | None = None) -> str:
        return self.inner.chat(self._sys(system), user, schema)

    def light(self) -> "InstructedLLM":                # light steps (charts, memory, brief summary) keep the wrapper
        inner = self.inner.light() if hasattr(self.inner, "light") else self.inner
        return InstructedLLM(inner, self.name, self.instructions)

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "?")

    def __getattr__(self, name: str) -> Any:            # last_thinking, last_stats, last_knowledge, health, ...
        return getattr(self.__dict__["inner"], name)


# ------------------------------------------------------------------ custom step contract
CUSTOM_INPUTS = {"question": "The question", "request": "The standardised request", "sql": "The SQL query",
                 "table": "The result table", "expert": "The expert assessment", "answer": "The answer"}
CUSTOM_OUTPUTS = {"note": "A note for later steps", "question": "A rewritten question",
                  "answer_section": "A section added to the answer"}
CUSTOM_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "text": {"type": "string"}},
    "required": ["reasoning", "text"],
}
_OUTPUT_RULES = {
    "note": "In `text`, write a short note (1-4 sentences) that the next agents (SQL writer, answer writer) should "
            "follow. Only facts from your instructions or the material; no SQL.",
    "question": "In `text`, write the question again so the SQL writer understands it better: one complete English "
                "question, same meaning, nothing invented.",
    "answer_section": "In `text`, write a short section (2-6 sentences or bullets) to add at the end of the answer. "
                      "Only facts from the material.",
}


def custom_system(agent: CustomAgent, output: str) -> str:
    """System prompt of a custom agent's own step: its instructions, then what `text` must contain."""
    rule = _OUTPUT_RULES.get(output, "In `text`, write your result.")
    return (f"{agent.instructions.strip()}\n\n{rule}\n"
            "Return JSON only: reasoning (one sentence), text.")
