"""Dashboard spec: what the design agent proposes and what the renderer consumes.

A spec is a title plus 1..N widgets. Each widget has a *kind* (kpi, bar, line, pie, table),
a plain-English *question* that the DataAgent answers with SQL, optional column hints and a
grid width. spec_from_llm() coerces whatever the model returned into a valid spec and drops
anything unusable, mirroring request_agent._from_llm.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

KINDS = ("kpi", "bar", "line", "pie", "table")
WIDTHS = {"small": 4, "medium": 6, "full": 12}          # 12-column CSS grid
DEFAULT_WIDTH = "medium"
DEFAULT_ACCENT = "#2F5BEA"
MAX_TITLE = 80
MAX_QUESTION = 400
NO_WIDGETS = "The model proposed no usable widgets"

_HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass
class Widget:
    id: str                 # "w1".."wN", assigned by spec_from_llm
    title: str
    kind: str               # one of KINDS
    question: str           # standalone data request answered by the DataAgent
    x: str = ""             # preferred category / time column (hint only; the renderer re-checks real columns)
    y: str = ""             # preferred numeric column (hint only)
    width: str = DEFAULT_WIDTH
    note: str = ""          # one line shown under the title

    @property
    def cols(self) -> int:
        return WIDTHS.get(self.width, WIDTHS[DEFAULT_WIDTH])


@dataclass
class DashboardSpec:
    title: str
    subtitle: str = ""
    accent: str = DEFAULT_ACCENT
    widgets: list[Widget] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DashboardSpec":
        d = d or {}
        widgets = [Widget(**{k: w.get(k, "") for k in Widget.__dataclass_fields__}) for w in d.get("widgets") or []
                   if isinstance(w, dict)]
        for w in widgets:
            w.width = w.width or DEFAULT_WIDTH
        return cls(title=str(d.get("title") or "Dashboard"), subtitle=str(d.get("subtitle") or ""),
                   accent=str(d.get("accent") or DEFAULT_ACCENT), widgets=widgets)


def _clip(text: object, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def spec_from_llm(out: dict, max_widgets: int) -> tuple[DashboardSpec | None, str]:
    """(spec, "") or (None, reason). Unknown kinds and empty questions are dropped, titles clipped,
    widths normalised, the accent validated, and the list capped to max_widgets."""
    out = out or {}
    dropped: list[str] = []
    widgets: list[Widget] = []
    for raw in out.get("widgets") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        question = _clip(raw.get("question"), MAX_QUESTION)
        title = _clip(raw.get("title"), MAX_TITLE) or question[:MAX_TITLE] or kind
        if kind not in KINDS or not question:
            dropped.append(f"{title} ({kind or 'no kind'}{'' if question else ', no question'})")
            continue
        width = str(raw.get("width") or "").strip().lower()
        widgets.append(Widget(id="", title=title, kind=kind, question=question,
                              x=_clip(raw.get("x"), MAX_TITLE), y=_clip(raw.get("y"), MAX_TITLE),
                              width=width if width in WIDTHS else DEFAULT_WIDTH,
                              note=_clip(raw.get("note"), 160)))
        if len(widgets) >= max_widgets:
            break
    if not widgets:
        reason = NO_WIDGETS + (": dropped " + "; ".join(dropped) if dropped else "")
        return None, reason
    for i, w in enumerate(widgets, 1):
        w.id = f"w{i}"
    accent = str(out.get("accent") or "").strip()
    spec = DashboardSpec(title=_clip(out.get("title"), MAX_TITLE) or "Dashboard",
                         subtitle=_clip(out.get("subtitle"), 200),
                         accent=accent if _HEX.fullmatch(accent) else DEFAULT_ACCENT,
                         widgets=widgets)
    return spec, ""
