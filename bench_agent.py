"""Benchmark the agent end to end: N questions through the real pipeline, per-step timings and token counts.

    python bench_agent.py                          # 3 built-in questions, every agent as configured in .env
    python bench_agent.py --repeat 2 --json before.json
    python bench_agent.py --questions my_questions.txt --no-memory

The agent is built exactly as app.py builds it (same models, same on/off switches from .env), the models are
warmed first, and the questions are chained through the conversation memory like a real chat. Every LLM step
prints the Ollama stats recorded in the trace (prompt tokens, generated tokens, tok/s), so a change can be
judged by where the seconds went, not just by the total. Run it before and after a change and compare.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.answer_agent import AnswerAgent  # noqa: E402
from agent.config import settings  # noqa: E402
from agent.db import Database  # noqa: E402
from agent.expert_agent import ExpertAgent  # noqa: E402
from agent.llm import OllamaLLM  # noqa: E402
from agent.orchestrator import DataAgent  # noqa: E402
from ml.registry import ModelRegistry  # noqa: E402

DEFAULT_QUESTIONS = [
    "average salary per department",
    "how many employees were hired each year since 1995",
    "is there a significant difference in salary between men and women",
]


def build_agent() -> DataAgent:
    """The same wiring as app.py's sidebar, with the .env defaults for every switch."""
    plan_on = settings.expert_reviews if settings.expert_plan is None else settings.expert_plan
    assess_on = settings.expert_reviews if settings.expert_assess is None else settings.expert_assess
    agent = DataAgent(db=Database(settings.database_url), llm=OllamaLLM(), registry=ModelRegistry(),
                      charts=settings.draw_charts, summarise_data=settings.summarise_data_queries,
                      standardise=settings.standardise_requests, expert_plan=plan_on, expert_assess=assess_on)
    agent.answer_agent = AnswerAgent(OllamaLLM(model=settings.answer_model or settings.model))
    agent.context_builder.llm = agent.answer_agent.llm
    agent.expert_agent = ExpertAgent(OllamaLLM(model=settings.expert_model or settings.answer_model or settings.model),
                                     briefing=agent.briefing.text if (plan_on or assess_on) else "")
    return agent


def warm(agent: DataAgent) -> None:
    llms = {l.model: l for l in (agent.llm, agent.answer_agent.llm, agent.expert_agent.llm)}
    for llm in llms.values():
        t0 = time.perf_counter()
        ok, msg = llm.warm()
        print(f"warm {msg} ({time.perf_counter() - t0:.1f} s)" if ok else f"warm FAILED: {msg}")


def step_rows(trace) -> list[dict]:
    rows = []
    for s in trace.steps:
        llm = s.details.get("llm") if isinstance(s.details.get("llm"), dict) else None
        rows.append({"index": s.index, "name": s.name, "status": s.status, "ms": s.duration_ms, "llm": llm})
    return rows


def print_run(question: str, rows: list[dict], timing: dict, wall_ms: float, visible_ms: float, error: str | None):
    print(f"\n=== {question}")
    print(f"{'#':>2}  {'step':46} {'ms':>9} {'prompt':>7} {'gen':>6} {'tok/s':>6}")
    for r in rows:
        l = r["llm"] or {}
        print(f"{r['index']:>2}  {r['name'][:46]:46} {r['ms']:>9.0f} "
              f"{l.get('prompt_tokens', ''):>7} {l.get('gen_tokens', ''):>6} {l.get('tok_per_s', ''):>6}")
    print(f"    answer visible after {visible_ms / 1000:.1f} s; whole turn {wall_ms / 1000:.1f} s; "
          f"LLM {timing['llm_ms'] / 1000:.1f} s in {timing['llm_calls']} calls "
          f"({timing['prompt_tokens']} prompt / {timing['gen_tokens']} generated tokens)"
          + (f"; ERROR: {error}" if error else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--questions", help="file with one question per line (default: 3 built-in questions)")
    ap.add_argument("--repeat", type=int, default=1, help="run the question list N times (default 1)")
    ap.add_argument("--no-memory", action="store_true", help="do not chain questions through the conversation memory")
    ap.add_argument("--json", help="write the per-step results to this file")
    args = ap.parse_args()

    questions = ([q.strip() for q in Path(args.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
                 if args.questions else DEFAULT_QUESTIONS)
    agent = build_agent()
    print(f"models: sql={agent.llm.model} answer={agent.answer_agent.llm.model} expert={agent.expert_agent.llm.model}")
    print(f"agents: standardise={agent.standardise} plan={agent.expert_plan} assess={agent.expert_assess} "
          f"summary={agent.summarise_data} charts={agent.charts} memory={not args.no_memory} | "
          f"think_light_steps={settings.think_light_steps} router_shortcut={settings.router_shortcut} "
          f"defer_memory={settings.defer_memory_update} keep_alive={settings.llm_keep_alive}")
    warm(agent)

    runs = []
    for rep in range(args.repeat):
        history = []
        for q in questions:
            t0 = time.perf_counter()
            res = agent.ask(q, history=history, build_context=not args.no_memory)
            visible_ms = (time.perf_counter() - t0) * 1000
            res.wait_context()                      # include a deferred memory update in the turn's total
            wall_ms = (time.perf_counter() - t0) * 1000
            rows, timing = step_rows(res.trace), res.trace.timing()
            print_run(q, rows, timing, wall_ms, visible_ms, res.error)
            runs.append({"repeat": rep, "question": q, "wall_ms": round(wall_ms), "answer_visible_ms": round(visible_ms),
                         "error": res.error, "timing": timing, "steps": rows})
            history.append(res)

    walls = [r["wall_ms"] for r in runs]
    visible = [r["answer_visible_ms"] for r in runs]
    summary = {"questions": len(runs), "median_wall_ms": round(statistics.median(walls)),
               "median_answer_visible_ms": round(statistics.median(visible)), "total_wall_ms": round(sum(walls)),
               "errors": sum(1 for r in runs if r["error"])}
    print(f"\nmedian question wall = {summary['median_wall_ms'] / 1000:.1f} s "
          f"(answer visible after {summary['median_answer_visible_ms'] / 1000:.1f} s); "
          f"total {summary['total_wall_ms'] / 1000:.1f} s for {len(runs)} questions; errors {summary['errors']}")
    if args.json:
        out = {"config": {"sql_model": agent.llm.model, "answer_model": agent.answer_agent.llm.model,
                          "expert_model": agent.expert_agent.llm.model, "think_light_steps": settings.think_light_steps,
                          "router_shortcut": settings.router_shortcut, "defer_memory_update": settings.defer_memory_update,
                          "keep_alive": settings.llm_keep_alive, "num_ctx": settings.num_ctx},
               "summary": summary, "runs": runs}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"written {args.json}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
