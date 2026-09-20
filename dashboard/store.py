"""Local records of built / published dashboards: one JSON file per dashboard under
settings.dashboards_dir (git-ignored). A record keeps the spec and the SQL of every widget so
"Refresh & republish" can re-run the same queries without the LLM, plus the Netlify site id
and URL so a republish goes to the same address.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from agent.config import settings
from agent.export import slugify

from .spec import DashboardSpec

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _path(slug: str):
    return settings.dashboards_dir / f"{slugify(slug, default='dashboard')}.json"


def new_record(description: str, spec: DashboardSpec, sqls: dict[str, str]) -> dict:
    return {"slug": slugify(spec.title, default="dashboard"), "title": spec.title, "description": description,
            "spec": spec.to_dict(), "sqls": sqls, "site_id": None, "url": None,
            "published_at": None, "refreshed_at": None}


def save(record: dict) -> dict:
    settings.dashboards_dir.mkdir(parents=True, exist_ok=True)
    _path(record["slug"]).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


def load(slug: str) -> dict | None:
    p = _path(slug)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read dashboard record %s: %s", p, exc)
        return None


def list_all() -> list[dict]:
    if not settings.dashboards_dir.exists():
        return []
    records = [r for p in settings.dashboards_dir.glob("*.json") if (r := load(p.stem)) is not None]
    return sorted(records, key=lambda r: r.get("published_at") or "", reverse=True)


def delete(slug: str) -> None:
    p = _path(slug)
    if p.exists():
        p.unlink()
