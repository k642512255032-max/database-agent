"""Automatic, explainable feature engineering.

`AutoFeatureEngineer` inspects raw columns returned by SQL and decides - with
simple, documented rules - how each column becomes model features:

  * ID-like / constant / free-text columns  -> dropped
  * boolean                                 -> 0/1
  * numeric (incl. MySQL DECIMAL)           -> median impute (+ missing flag), log1p if
                                               heavily skewed, standardised
  * dates / datetimes                       -> year, month, day-of-week, days-before-reference
  * low-cardinality categoricals            -> one-hot (top categories + __other__)
  * high-cardinality categoricals           -> frequency encoding

It is fitted ONCE at training time and saved inside the model bundle, so at
inference the exact same transformations are replayed. Every decision is kept
in `self.plan` / `self.log` so the agent can show them to the user.
"""
from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid)", re.I)

class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, max_onehot: int = 15, skew_threshold: float = 1.0,
                 drop_id_like: bool = True, text_unique_ratio: float = 0.5):
        self.max_onehot = max_onehot
        self.skew_threshold = skew_threshold
        self.drop_id_like = drop_id_like
        self.text_unique_ratio = text_unique_ratio

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: Any = None) -> "AutoFeatureEngineer":
        X = X.copy()
        self.input_columns_: list[str] = list(X.columns)
        self.plan: dict[str, dict] = {}        # raw column -> decision
        self.features_: list[dict] = []        # output feature specs (ordered)
        self.log: list[str] = []
        self.baseline_: dict[str, Any] = {}    # "typical" raw value per column (for attributions)
        n = max(len(X), 1)

        for col in X.columns:
            s = X[col]
            kind, s_conv = _infer_kind(s)
            nunique = s_conv.nunique(dropna=True)

            if nunique <= 1:
                self._drop(col, "constant column (no information)")
                continue
            if self.drop_id_like and ID_PATTERN.search(col) and nunique / n > 0.9:
                self._drop(col, "identifier column (unique per row, would cause over-fitting)")
                continue

            if kind == "bool":
                self.baseline_[col] = bool(s_conv.mode().iloc[0])
                self._add(col, "bool", f"{col}", {})
                self.plan[col] = {"type": "boolean", "action": "cast to 0/1"}

            elif kind == "numeric":
                vals = s_conv.astype(float)
                median = float(vals.median())
                has_na = bool(vals.isna().any())
                filled = vals.fillna(median)
                skew = float(filled.skew()) if len(filled) > 2 else 0.0
                use_log = bool(skew > self.skew_threshold and filled.min() >= 0)
                t = np.log1p(filled) if use_log else filled
                mean, std = float(t.mean()), float(t.std() or 1.0)
                self.baseline_[col] = median
                self._add(col, "numeric", col, {"median": median, "log": use_log, "mean": mean, "std": std})
                actions = [f"impute missing with median ({median:.4g})"]
                if use_log:
                    actions.append(f"log1p transform (skew={skew:.2f} > {self.skew_threshold})")
                actions.append("standardise (z-score)")
                if has_na:
                    self._add(col, "missing", f"{col}__is_missing", {})
                    actions.append(f"add missing-indicator feature {col}__is_missing")
                self.plan[col] = {"type": "numeric", "action": "; ".join(actions)}

            elif kind == "datetime":
                dt = s_conv
                ref = dt.max()
                self.baseline_[col] = dt.dropna().sort_values().iloc[len(dt.dropna()) // 2]
                parts = {
                    "year": dt.dt.year, "month": dt.dt.month, "dayofweek": dt.dt.dayofweek,
                    "days_before_ref": (ref - dt).dt.days,
                }
                for part, v in parts.items():
                    v = v.astype(float)
                    med = float(v.median())
                    v = v.fillna(med)
                    self._add(col, "date_part", f"{col}__{part}",
                              {"part": part, "ref": ref, "median": med,
                               "mean": float(v.mean()), "std": float(v.std() or 1.0), "log": False})
                self.plan[col] = {"type": "datetime",
                                  "action": f"expand to year, month, day-of-week, days before {ref.date()}; standardise"}

            else:  # categorical / text
                s_str = s.astype("string")
                self.baseline_[col] = s_str.mode().iloc[0] if s_str.notna().any() else None
                if s_str.str.len().mean() > 40 and nunique / n > self.text_unique_ratio:
                    self._drop(col, "free-text column (too unique to encode)")
                    continue
                if nunique <= self.max_onehot:
                    cats = list(s_str.value_counts().index.astype(str))
                    for c in cats:
                        self._add(col, "onehot", f"{col}={c}", {"category": c})
                    self._add(col, "onehot", f"{col}=__other__", {"category": "__other__", "known": cats})
                    self.plan[col] = {"type": "categorical",
                                      "action": f"one-hot encode {len(cats)} categories (+ __other__ for unseen)"}
                else:
                    freq = (s_str.value_counts(normalize=True)).to_dict()
                    v = s_str.map(freq).astype(float).fillna(0.0)
                    self._add(col, "freq", f"{col}__freq",
                              {"freq": freq, "mean": float(v.mean()), "std": float(v.std() or 1.0)})
                    self.plan[col] = {"type": "categorical (high cardinality)",
                                      "action": f"frequency encoding ({nunique} distinct values); standardise"}

        self.feature_names_out_ = [f["name"] for f in self.features_]
        self.log.append(f"{len(self.input_columns_)} raw columns -> {len(self.feature_names_out_)} model features")
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns_ if c not in X.columns and self.plan.get(c, {}).get("type") != "dropped"]
        if missing:
            raise ValueError(f"Missing required columns for feature engineering: {missing}")
        out: dict[str, np.ndarray] = {}
        cache: dict[str, pd.Series] = {}
        for f in self.features_:
            src, kind, p = f["source"], f["kind"], f["params"]
            if src not in cache:
                cache[src] = _infer_kind(X[src], force=self.plan[src]["type"])[1]
            s = cache[src]
            if kind == "bool":
                v = s.astype(float).fillna(0.0)
            elif kind == "numeric":
                v = s.astype(float).fillna(p["median"])
                if p["log"]:
                    v = np.log1p(v.clip(lower=0))
                v = (v - p["mean"]) / p["std"]
            elif kind == "missing":
                v = s.isna().astype(float)
            elif kind == "date_part":
                part = p["part"]
                if part == "days_before_ref":
                    v = (p["ref"] - s).dt.days
                else:
                    v = getattr(s.dt, part)
                v = (v.astype(float).fillna(p["median"]) - p["mean"]) / p["std"]
            elif kind == "onehot":
                s_str = X[src].astype("string")
                if p["category"] == "__other__":
                    v = (~s_str.isin(p["known"])).astype(float)
                else:
                    v = (s_str == p["category"]).fillna(False).astype(float)
            elif kind == "freq":
                v = X[src].astype("string").map(p["freq"]).astype(float).fillna(0.0)
                v = (v - p["mean"]) / p["std"]
            else:  # pragma: no cover
                raise ValueError(kind)
            out[f["name"]] = np.asarray(v, dtype=float)
        return pd.DataFrame(out, index=X.index, columns=self.feature_names_out_)

    # ------------------------------------------------------- explainability
    def get_feature_names_out(self, input_features=None):  # sklearn API
        return np.array(self.feature_names_out_)

    def source_of(self, feature: str) -> str:
        for f in self.features_:
            if f["name"] == feature:
                return f["source"]
        return feature

    def raw_value(self, feature: str, scaled_value: float) -> float | None:
        """Convert a standardised feature value (e.g. a tree threshold) back to raw units."""
        for f in self.features_:
            if f["name"] == feature and f["kind"] in {"numeric", "date_part", "freq"}:
                v = scaled_value * f["params"]["std"] + f["params"]["mean"]
                if f["params"].get("log"):
                    v = float(np.expm1(v))
                return float(v)
        return None

    def describe_condition(self, feature: str, threshold: float, goes_left: bool) -> str:
        """Human-readable version of a tree split `feature <= threshold`."""
        spec = next((f for f in self.features_ if f["name"] == feature), None)
        if spec is None:
            return f"{feature} {'<=' if goes_left else '>'} {threshold:.3g}"
        if spec["kind"] in {"onehot", "bool", "missing"}:
            base = feature if spec["kind"] != "onehot" else f"{spec['source']} == '{spec['params']['category']}'"
            return f"NOT ({base})" if goes_left else base
        raw = self.raw_value(feature, threshold)
        label = feature.replace("__", " ")
        return f"{label} {'<=' if goes_left else '>'} {raw:.4g}"

    def plan_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"column": c, "detected_type": d["type"], "action": d["action"]} for c, d in self.plan.items()]
        )

    # --------------------------------------------------------------- private
    def _add(self, source: str, kind: str, name: str, params: dict) -> None:
        self.features_.append({"source": source, "kind": kind, "name": name, "params": params})

    def _drop(self, col: str, reason: str) -> None:
        self.plan[col] = {"type": "dropped", "action": f"drop - {reason}"}


# ------------------------------------------------------------------ helpers
def _infer_kind(s: pd.Series, force: str | None = None) -> tuple[str, pd.Series]:
    """Detect a column's semantic type and return a converted series."""
    if force is not None:
        if force == "boolean":
            return "bool", s.map(lambda v: _to_bool(v))
        if force == "numeric":
            return "numeric", pd.to_numeric(s, errors="coerce")
        if force == "datetime":
            return "datetime", _to_datetime(s)
        return "categorical", s

    if pd.api.types.is_bool_dtype(s):
        return "bool", s
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime", _to_datetime(s)
    if pd.api.types.is_numeric_dtype(s):
        uniq = set(pd.unique(s.dropna()))
        if uniq and uniq <= {0, 1} and s.name and re.search(r"^(is_|has_)", str(s.name)):
            return "bool", s.astype(float)
        return "numeric", s.astype(float)

    non_null = s.dropna()
    if len(non_null) == 0:
        return "categorical", s
    # MySQL DECIMAL arrives as Python Decimal objects (object dtype)
    num = pd.to_numeric(non_null, errors="coerce")
    if num.notna().mean() >= 0.95:
        return "numeric", pd.to_numeric(s, errors="coerce")
    sample = non_null.astype(str).head(200)
    if sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}").mean() > 0.9:
        dt = _to_datetime(s)
        if dt.notna().mean() >= 0.9 * s.notna().mean():
            return "datetime", dt
    lowered = sample.str.lower()
    if lowered.isin(["true", "false", "yes", "no"]).all():
        return "bool", s.map(_to_bool)
    return "categorical", s


def _to_datetime(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce", format="mixed") if not pd.api.types.is_datetime64_any_dtype(s) else s
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _to_bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, str):
        return 1.0 if v.strip().lower() in {"true", "yes", "1"} else 0.0
    return float(bool(v))
