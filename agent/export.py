"""Export helpers: filenames, CSV/XLSX byte generation and offline PNG chart
rendering.

Pure and Streamlit-free so it can be unit tested in isolation; app.py wires
this into st.download_button widgets.
"""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

EXCEL_MAX_ROWS = 1_048_575        # Excel's 1,048,576 minus the header row
SLUG_MAX = 60
MAX_PNG_ROWS = 20_000
# characters openpyxl refuses to write (XML 1.0 illegal control characters)
ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PNG_NOT_INSTALLED = "PNG export unavailable (install vl-convert-python)"
PNG_TOO_LARGE = f"PNG export skipped: chart data above {MAX_PNG_ROWS:,} rows"
PNG_FAILED = "PNG export failed for this chart"


def slugify(text: str, default: str = "result", max_len: int = SLUG_MAX) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or default


def export_filenames(question: str) -> dict[str, str]:
    slug = slugify(question)
    return {
        "csv": f"{slug}.csv",
        "xlsx": f"{slug}.xlsx",
        "png": f"{slug}-chart-{{i}}.png",
        "pbip": f"{slug}-powerbi.zip",
    }


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _is_tz_aware(v: object) -> bool:
    return isinstance(v, datetime) and v.tzinfo is not None


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame writable by openpyxl: tz-aware datetimes -> naive (Excel can't hold tz) and
    XML-illegal control characters stripped from text. Copies a column only when it must change."""
    out = df
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            fixed = s.dt.tz_localize(None)
        elif s.dtype == object or pd.api.types.is_string_dtype(s):   # pandas 3 uses 'str', not object, for text
            if any(_is_tz_aware(v) for v in s):          # short-circuits on the first hit
                fixed = s.map(lambda v: v.replace(tzinfo=None) if _is_tz_aware(v) else v)
            elif any(isinstance(v, str) and ILLEGAL_XLSX_CHARS.search(v) for v in s):
                fixed = s.map(lambda v: ILLEGAL_XLSX_CHARS.sub("", v) if isinstance(v, str) else v)
            else:
                continue
        else:
            continue
        if out is df:
            out = df.copy()
        out[col] = fixed
    return out


def _sanitise_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]:\*\?/\\]", "", name).strip()
    cleaned = cleaned[:31] or "sheet"
    base = cleaned
    candidate = base
    n = 2
    while candidate in used:
        suffix = f"_{n}"
        candidate = base[: 31 - len(suffix)] + suffix
        n += 1
    used.add(candidate)
    return candidate


def to_xlsx_bytes(
    df: pd.DataFrame,
    sheet_name: str = "data",
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> tuple[bytes, bool]:
    truncated = False

    def _prep(frame: pd.DataFrame) -> pd.DataFrame:
        nonlocal truncated
        frame = _clean_frame(frame)
        if len(frame) > EXCEL_MAX_ROWS:
            frame = frame.iloc[:EXCEL_MAX_ROWS]
            truncated = True
        return frame

    used_names: set[str] = set()
    main_name = _sanitise_sheet_name(sheet_name, used_names)
    main_df = _prep(df)

    sheets: list[tuple[str, pd.DataFrame]] = [(main_name, main_df)]
    for key, frame in (extra_sheets or {}).items():
        sheets.append((_sanitise_sheet_name(key, used_names), _prep(frame)))

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue(), truncated


def chart_to_png(chart, scale: float = 2.0) -> tuple[bytes | None, str | None]:
    """Render an Altair chart to PNG offline. Returns (png bytes, None) or (None, user-facing reason)."""
    try:
        import vl_convert as vlc
    except ImportError:
        return None, PNG_NOT_INSTALLED
    data = getattr(chart, "data", None)
    if isinstance(data, pd.DataFrame) and len(data) > MAX_PNG_ROWS:
        return None, PNG_TOO_LARGE
    try:
        return vlc.vegalite_to_png(chart.to_json(), scale=scale), None
    except Exception as exc:
        log.debug("chart_to_png failed: %s", exc)
        return None, PNG_FAILED
