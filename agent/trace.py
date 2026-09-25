"""Explainability layer: every action the agent takes is recorded as a Step.

The UI renders the Trace so the user can see *what* the agent did, *why*,
*what went in* and *what came out* of every step.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Step:
    index: int
    name: str                      # short title, e.g. "Generate SQL"
    why: str                       # why the agent performs this step
    status: str = "running"        # running | ok | warning | error
    details: dict[str, Any] = field(default_factory=dict)   # inputs / outputs
    reasoning: Optional[str] = None  # the LLM's own explanation, if any
    duration_ms: float = 0.0

    def add(self, **kwargs: Any) -> "Step":
        self.details.update(kwargs)
        return self

    def add_thinking(self, llm: Any, limit: int = 6000) -> "Step":
        """Record a thinking model's reasoning (Ollama message.thinking) so the trace shows it."""
        text = (getattr(llm, "last_thinking", "") or "").strip()
        if text:
            self.details["thinking"] = text[:limit] + ("\n..." if len(text) > limit else "")
        used = getattr(llm, "last_knowledge", None)     # passages from the agent's uploaded documents (knowledge.py)
        if used:
            self.details["knowledge_used"] = used
        # Ollama's per-call stats (nanoseconds) -> tokens and tok/s, so the trace shows where the time goes
        stats = getattr(llm, "last_stats", None) or {}
        if stats.get("eval_count") or stats.get("prompt_eval_count"):
            gen_s = (stats.get("eval_duration") or 0) / 1e9
            self.details["llm"] = {"prompt_tokens": stats.get("prompt_eval_count") or 0,
                                   "prompt_ms": round((stats.get("prompt_eval_duration") or 0) / 1e6),
                                   "gen_tokens": stats.get("eval_count") or 0, "gen_ms": round(gen_s * 1000),
                                   "load_ms": round((stats.get("load_duration") or 0) / 1e6),
                                   "tok_per_s": round((stats.get("eval_count") or 0) / gen_s, 1) if gen_s else 0.0}
        return self

@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)
    on_step: Optional[Callable[[Step], None]] = None   # live callback for the UI
    # steps recorded from another thread (deferred memory update) are kept but not pushed live: the UI container
    # that on_step writes to belongs to the thread that created the trace
    _owner: int = field(default_factory=threading.get_ident, repr=False, compare=False)

    @contextmanager
    def step(self, name: str, why: str):
        s = Step(index=len(self.steps) + 1, name=name, why=why)
        self.steps.append(s)
        t0 = time.perf_counter()
        try:
            yield s
            if s.status == "running":
                s.status = "ok"
        except Exception as exc:  # record the failure, then re-raise
            s.status = "error"
            s.details["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            if self.on_step and threading.get_ident() == self._owner:
                self.on_step(s)

    def timing(self) -> dict[str, float]:
        """Wall time of the whole trace and of the LLM calls in it (from the Ollama stats in each step)."""
        llm = [s.details["llm"] for s in self.steps if isinstance(s.details.get("llm"), dict)]
        return {"total_ms": round(sum(s.duration_ms for s in self.steps), 1),
                "llm_ms": round(sum(x["prompt_ms"] + x["gen_ms"] + x["load_ms"] for x in llm), 1),
                "prompt_tokens": sum(x["prompt_tokens"] for x in llm),
                "gen_tokens": sum(x["gen_tokens"] for x in llm),
                "llm_calls": len(llm)}

    def as_text(self) -> str:
        lines = [f"Question: {self.question}"]
        for s in self.steps:
            lines.append(f"\n[{s.index}] {s.name}  ({s.status}, {s.duration_ms} ms)")
            lines.append(f"    why: {s.why}")
            if s.reasoning:
                lines.append(f"    model reasoning: {s.reasoning}")
            for k, v in s.details.items():
                text = str(v)
                if len(text) > 400:
                    text = text[:400] + " ..."
                lines.append(f"    {k}: {text}")
        return "\n".join(lines)
