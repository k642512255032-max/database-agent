"""Prompt overrides saved from the "Agent settings" page.

Each agent's system prompts are constants in the code (the defaults). The page saves edited versions to
settings.prompts_file as {key: text}; every LLM call looks its prompt up through prompt(), so a saved edit is
used from the next call on. A missing or broken override falls back to the default: the pipeline never fails
over an edit.
"""
from __future__ import annotations

import json
import logging
import os
import string
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)


def fields(template: str) -> set[str]:
    """Placeholder names in a str.format template ({dialect} -> 'dialect'; {{ }} are literal braces)."""
    return {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}


class PromptStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.prompts_file)
        self._cache: tuple[tuple[int, int], dict[str, str]] | None = None    # ((mtime_ns, size), overrides)

    def overrides(self) -> dict[str, str]:
        """Saved overrides, re-read only when the file changed (cheap to call on every LLM call)."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return {}
        stamp = (st.st_mtime_ns, st.st_size)          # size too: two quick writes can share a timestamp tick
        if self._cache and self._cache[0] == stamp:
            return self._cache[1]
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("prompt overrides in %s ignored: %s", self.path, exc)
            data = {}
        data = {k: v for k, v in data.items() if isinstance(v, str) and v.strip()} if isinstance(data, dict) else {}
        self._cache = (stamp, data)
        return data

    def get(self, key: str, default: str) -> str:
        return self.overrides().get(key, default)

    def is_custom(self, key: str) -> bool:
        return key in self.overrides()

    @staticmethod
    def validate(text: str, placeholders: tuple[str, ...]) -> str:
        """'' when the text can replace a default with these placeholders, else the reason it cannot."""
        if not text.strip():
            return "The prompt is empty."
        if not placeholders:
            return ""                                  # never formatted: any text is fine
        try:
            found = fields(text)
        except ValueError as exc:                     # a single { or } in a formatted prompt
            return f"Unbalanced brace ({exc}). Write literal braces as {{{{ and }}}}."
        missing, unknown = set(placeholders) - found, found - set(placeholders)
        if missing:
            return "Keep the placeholder(s) " + ", ".join("{" + m + "}" for m in sorted(missing)) + "."
        if unknown:
            return ("Unknown placeholder(s) " + ", ".join("{" + u + "}" for u in sorted(unknown))
                    + ". Write literal braces as {{ and }}.")
        return ""

    def set(self, key: str, text: str, default: str) -> None:
        """Save an override; saving the default text (or blank) removes it."""
        data = dict(self.overrides())
        if not text.strip() or text.strip() == default.strip():
            data.pop(key, None)
        else:
            data[key] = text
        self._write(data)

    def reset(self, key: str) -> None:
        data = dict(self.overrides())
        if data.pop(key, None) is not None:
            self._write(data)

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)                    # atomic: a reader never sees half a file
        self._cache = None


store = PromptStore()


def prompt(key: str, default: str, **fmt: str) -> str:
    """The system prompt for `key`: the saved override, else `default`. With fmt the template is formatted.
    If a (hand-edited) override fails to format, the default is used and a warning is logged."""
    text = store.get(key, default)
    if not fmt:
        return text
    try:
        return text.format(**fmt)
    except (KeyError, IndexError, ValueError) as exc:
        log.warning("prompt override %s ignored: %s", key, exc)
        return default.format(**fmt)
