"""Explainability layer: every action the agent takes is recorded as a Step.

The UI renders the Trace so the user can see *what* the agent did, *why*,
*what went in* and *what came out* of every step.
"""
from __future__ import annotations

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

@dataclass
class Trace:
    question: str
    steps: list[Step] = field(default_factory=list)
    on_step: Optional[Callable[[Step], None]] = None   # live callback for the UI

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
            if self.on_step:
                self.on_step(s)

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
