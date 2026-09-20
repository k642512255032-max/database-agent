"""Dashboard builder: description -> spec (LLM) -> rows per widget (DataAgent) -> static bundle.

Every stage is a recorded Step, like the chat pipeline:
  1 Design dashboard          LLM turns the description into widgets with standalone data questions
  2..n Widget i/n             DataAgent.ask(question, force_intent="data_query") -> SQL -> rows
  n+1 Render dashboard files  bind rows to widgets, write index.html / style.css / app.js / data.js

A design failure is fatal for the build; a widget failure only marks that widget. refresh() re-runs
the stored SQL of a published dashboard without any LLM call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from agent.config import settings
from agent.llm import LLMError
from agent.trace import Step, Trace

from . import prompts
from .render import bind_widget, render_bundle
from .spec import DashboardSpec, Widget, spec_from_llm

if TYPE_CHECKING:
    from agent.orchestrator import DataAgent

log = logging.getLogger(__name__)


@dataclass
class WidgetResult:
    widget: Widget
    sql: str | None = None
    data: pd.DataFrame | None = None
    error: str | None = None
    reused: bool = False


@dataclass
class DashboardBuild:
    description: str
    trace: Trace
    spec: DashboardSpec | None = None
    results: list[WidgetResult] = field(default_factory=list)
    bundle: dict[str, str] = field(default_factory=dict)
    generated_at: str = ""
    error: str | None = None

    def sqls(self) -> dict[str, str]:
        return {r.widget.id: r.sql for r in self.results if r.sql}

    def ok_widgets(self) -> int:
        return sum(1 for r in self.results if r.error is None)


class DashboardBuilder:
    def __init__(self, agent: "DataAgent", llm: Any | None = None):
        self.agent = agent                    # a DataAgent built with charts=False, summarise_data=False
        self.llm = llm or agent.answer_agent.llm

    # ------------------------------------------------------------ 1. design
    def design(self, description: str, trace: Trace, previous: DashboardSpec | None = None,
               change: str = "") -> DashboardSpec:
        refining = previous is not None and bool(change.strip())
        with trace.step("Design dashboard" if not refining else "Refine dashboard",
                        "Turn the description into widgets, each with a standalone data question the SQL "
                        "writer can answer." if not refining else
                        "Apply the requested changes to the current widgets; keep the rest identical.") as s:
            tables, _ = self.agent.db.link_tables(description + (" " + change if refining else ""))
            schema = self.agent.db.schema_text(tables, with_samples=False)
            if refining:
                user = prompts.refine_user(description, json.dumps(previous.to_dict(), ensure_ascii=False, indent=1),
                                           change, schema)
            else:
                user = prompts.design_user(description, schema)
            out = self.llm.chat_json(prompts.DESIGN_SYSTEM, user, prompts.DESIGN_SCHEMA)
            s.reasoning = out.get("reasoning")
            spec, reason = spec_from_llm(out, settings.dashboard_max_widgets)
            if spec is None:
                raise LLMError(reason)
            s.add(schema_given_to_model=schema, title=spec.title,
                  widgets=[f"{w.kind}: {w.title} - {w.question}" for w in spec.widgets])
        return spec

    # ------------------------------------------------------------ 2. fetch
    def fetch(self, spec: DashboardSpec, trace: Trace, reuse: dict[str, WidgetResult] | None = None) -> list[WidgetResult]:
        results: list[WidgetResult] = []
        n = len(spec.widgets)
        for i, w in enumerate(spec.widgets, 1):
            with trace.step(f"Widget {i}/{n}: {w.title}", "Answer the widget's data question with SQL.") as s:
                s.add(question=w.question)
                prev = (reuse or {}).get(w.question)
                if prev is not None and prev.error is None:
                    results.append(WidgetResult(w, prev.sql, prev.data, None, reused=True))
                    s.add(decision="Unchanged question - previous result reused", sql=prev.sql,
                          rows=0 if prev.data is None else len(prev.data))
                    continue
                res = self.agent.ask(w.question, force_intent="data_query", build_context=False)
                if res.error or res.data is None:
                    s.status = "warning"
                    s.add(note=f"Widget failed: {res.error or 'no rows returned'}")
                    results.append(WidgetResult(w, res.sql, None, res.error or "no rows returned"))
                    continue
                s.add(sql=res.sql, rows=len(res.data))
                if res.data.empty:
                    s.status = "warning"
                results.append(WidgetResult(w, res.sql, res.data))
        return results

    # ------------------------------------------------------------ 3. build
    def build(self, description: str, on_step: Callable[[Step], None] | None = None,
              previous: DashboardBuild | None = None, change: str = "") -> DashboardBuild:
        trace = Trace(description, on_step=on_step)
        build = DashboardBuild(description, trace)
        try:
            build.spec = self.design(description, trace, previous.spec if previous else None, change)
            reuse = {r.widget.question: r for r in previous.results} if previous else None
            build.results = self.fetch(build.spec, trace, reuse)
            self._render(build)
        except Exception as exc:
            build.error = f"{type(exc).__name__}: {exc}"
        return build

    def refresh(self, record: dict, on_step: Callable[[Step], None] | None = None) -> DashboardBuild:
        """Re-run the stored SQL of every widget (no LLM) and render a fresh bundle."""
        spec = DashboardSpec.from_dict(record.get("spec") or {})
        trace = Trace(record.get("description") or spec.title, on_step=on_step)
        build = DashboardBuild(record.get("description") or "", trace, spec=spec)
        sqls = record.get("sqls") or {}
        n = len(spec.widgets)
        for i, w in enumerate(spec.widgets, 1):
            sql = sqls.get(w.id)
            with trace.step(f"Widget {i}/{n}: {w.title}", "Re-run the widget's stored SQL.") as s:
                if not sql:
                    s.status = "warning"
                    s.add(note="No stored SQL for this widget")
                    build.results.append(WidgetResult(w, None, None, "No stored SQL for this widget"))
                    continue
                try:
                    safe, _ = self.agent.db.validate_sql(sql, max_rows=settings.dashboard_rows_per_widget)
                    df = self.agent.db.run(safe)
                    s.add(sql=safe, rows=len(df))
                    build.results.append(WidgetResult(w, safe, df))
                except Exception as exc:
                    s.status = "warning"
                    s.add(note=f"Query failed: {exc}")
                    build.results.append(WidgetResult(w, sql, None, f"{type(exc).__name__}: {exc}"))
        try:
            self._render(build)
        except Exception as exc:
            build.error = f"{type(exc).__name__}: {exc}"
        return build

    # ------------------------------------------------------------ helpers
    def _render(self, build: DashboardBuild) -> None:
        with build.trace.step("Render dashboard files",
                              "Bind each widget's rows to its chart or table and write the HTML, CSS and JS.") as s:
            build.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            bound = [bind_widget(r.widget, r.data, r.error, settings.dashboard_rows_per_widget) for r in build.results]
            build.bundle = render_bundle(build.spec, bound, build.generated_at)
            s.add(files=[f"{name} ({len(text):,} chars)" for name, text in build.bundle.items()],
                  widgets_ok=build.ok_widgets(), widgets_failed=len(build.results) - build.ok_widgets())
