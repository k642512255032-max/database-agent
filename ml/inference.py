"""Inference + explanations for already-trained models.

Supported algorithms and how each prediction is explained:
  decision_tree     -> the exact decision path (rules in raw units) + feature importance
  random_forest     -> tree-vote agreement, global importance, local attribution
  knn               -> the k nearest training rows (ids, labels, distances) + local attribution
  kmeans            -> assigned cluster, distance to centroid, cluster profiles
  pca               -> component scores, explained variance, top loadings
  isolation_forest  -> anomaly flag/score + most extreme features of each anomaly
  dbscan            -> nearest core sample within eps (DBSCAN has no native predict)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .registry import ModelBundle

MAX_EXPLAINED_ROWS = 5


@dataclass
class InferenceResult:
    output: pd.DataFrame                      # input keys + predictions
    features: pd.DataFrame                    # engineered features (what the model saw)
    feature_plan: pd.DataFrame                # feature-engineering decisions
    sections: list[dict] = field(default_factory=list)   # {title, kind: text|table, content}
    summary: dict = field(default_factory=dict)          # compact facts for the LLM answer

    def add(self, title: str, content: Any, kind: str | None = None) -> None:
        kind = kind or ("table" if isinstance(content, pd.DataFrame) else "text")
        self.sections.append({"title": title, "kind": kind, "content": content})


def check_columns(bundle: ModelBundle, df: pd.DataFrame) -> list[str]:
    return [c for c in bundle.feature_columns if c not in df.columns]


def run_inference(bundle: ModelBundle, df: pd.DataFrame) -> InferenceResult:
    missing = check_columns(bundle, df)
    if missing:
        raise ValueError(f"Data is missing columns required by model '{bundle.name}': {missing}")
    if df.empty:
        raise ValueError("The query returned no rows to run the model on.")

    fe = bundle.feature_engineer
    X = fe.transform(df[bundle.feature_columns])
    keys = [c for c in [bundle.id_column] if c and c in df.columns]
    res = InferenceResult(output=df[keys].copy() if keys else pd.DataFrame(index=df.index),
                          features=X, feature_plan=fe.plan_table())
    handler = {
        "decision_tree": _supervised, "random_forest": _supervised, "knn": _supervised,
        "kmeans": _kmeans, "pca": _pca, "isolation_forest": _isolation_forest, "dbscan": _dbscan,
    }.get(bundle.algorithm)
    if handler is None:
        raise ValueError(f"Unsupported algorithm: {bundle.algorithm}")
    handler(bundle, df, X, res)
    res.summary.update({"model": bundle.name, "algorithm": bundle.algorithm, "task": bundle.task,
                        "rows_scored": int(len(df)), "training_metrics": bundle.metrics})
    return res


# ================================================================ supervised
def _supervised(b: ModelBundle, df: pd.DataFrame, X: pd.DataFrame, res: InferenceResult) -> None:
    est, fe = b.estimator, b.feature_engineer
    pred = est.predict(X)
    col = f"predicted_{b.target}"
    res.output[col] = pred
    is_clf = b.task == "classification"
    if is_clf and hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)
        res.output["confidence"] = proba.max(axis=1).round(3)
        classes = list(est.classes_)
        if len(classes) == 2:
            res.output[f"prob_{b.target}={classes[1]}"] = proba[:, 1].round(3)
        counts = pd.Series(pred).value_counts().to_dict()
        res.summary["prediction_counts"] = {str(k): int(v) for k, v in counts.items()}
        res.summary["mean_confidence"] = float(proba.max(axis=1).mean().round(3))
    else:
        res.summary["prediction_mean"] = float(np.mean(pred))
        res.summary["prediction_min_max"] = [float(np.min(pred)), float(np.max(pred))]

    # ---- global importance
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({"feature": X.columns, "importance": est.feature_importances_})
        imp["source_column"] = imp["feature"].map(fe.source_of)
        by_src = imp.groupby("source_column")["importance"].sum().sort_values(ascending=False)
        top = by_src.head(10).round(4).reset_index()
        res.add("Global feature importance (which columns the model relies on)", top)
        res.summary["top_features"] = top["source_column"].head(5).tolist()

    n_explain = min(MAX_EXPLAINED_ROWS, len(df))
    label = (lambda i: f"row {i}" if not b.id_column or b.id_column not in df.columns
             else f"{b.id_column}={df.iloc[i][b.id_column]}")

    # ---- decision tree: exact path
    if b.algorithm == "decision_tree":
        tree = est.tree_
        paths = est.decision_path(X.iloc[:n_explain])
        leaves = est.apply(X.iloc[:n_explain])
        lines = []
        for i in range(n_explain):
            nodes = paths.indices[paths.indptr[i]: paths.indptr[i + 1]]
            rules = []
            for node in nodes:
                if node == leaves[i]:
                    continue
                f_idx, thr = tree.feature[node], tree.threshold[node]
                fname = X.columns[f_idx]
                goes_left = X.iloc[i, f_idx] <= thr
                rules.append(fe.describe_condition(fname, thr, goes_left))
            lines.append(f"**{label(i)}** -> predicted `{pred[i]}` because: " + "  AND  ".join(rules))
        res.add("Decision path for each prediction (first rows)", "\n\n".join(lines))

    # ---- random forest: vote agreement
    if b.algorithm == "random_forest" and is_clf:
        votes = np.stack([t.predict(X.iloc[:n_explain].to_numpy()) for t in est.estimators_], axis=1)
        agree = [(votes[i] == np.where(est.classes_ == pred[i])[0][0]).mean() for i in range(n_explain)]
        res.add("Tree vote agreement", pd.DataFrame({
            "row": [label(i) for i in range(n_explain)], "prediction": pred[:n_explain],
            "share_of_trees_agreeing": np.round(agree, 3), "n_trees": len(est.estimators_)}))

    # ---- KNN: neighbours
    if b.algorithm == "knn":
        dist, idx = est.kneighbors(X.iloc[:n_explain])
        train_ids = b.extra.get("train_ids")
        train_y = b.extra.get("train_y")
        rows = []
        for i in range(n_explain):
            for d, j in zip(dist[i], idx[i]):
                rows.append({"row": label(i), "prediction": pred[i],
                             "neighbour": train_ids[j] if train_ids is not None else int(j),
                             "note": "same row seen in training" if d == 0 else "",
                             "neighbour_label": train_y[j] if train_y is not None else None,
                             "distance": round(float(d), 3)})
        res.add(f"Nearest training neighbours (k={est.n_neighbors}) that voted", pd.DataFrame(rows))

    # ---- local attribution (model-agnostic occlusion)
    res.add("Local explanation: how each column pushed the prediction (first rows)",
            _local_attribution(b, df.iloc[:n_explain], pred[:n_explain]))


def _local_attribution(b: ModelBundle, rows: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """Replace one raw column at a time with its typical training value and measure the change.

    Positive effect = this row's actual value pushed the score UP (towards the predicted class
    probability for classifiers, or the predicted number for regressors).
    """
    fe, est = b.feature_engineer, b.estimator
    raw = rows[b.feature_columns]
    base_X = fe.transform(raw)
    is_clf = b.task == "classification" and hasattr(est, "predict_proba")

    def score(X: pd.DataFrame) -> np.ndarray:
        if is_clf:
            p = est.predict_proba(X)
            cls_idx = np.searchsorted(est.classes_, pred)
            return p[np.arange(len(X)), cls_idx]
        return est.predict(X)

    base = score(base_X)
    cols = [c for c in b.feature_columns if fe.plan.get(c, {}).get("type") != "dropped"]
    records = []
    for c in cols:
        mod = raw.copy()
        mod[c] = fe.baseline_.get(c)
        delta = base - score(fe.transform(mod))
        for i, (ix, row) in enumerate(raw.iterrows()):
            records.append({"row": i, "column": c, "value": str(row[c]), "effect": float(delta[i])})
    att = pd.DataFrame(records)
    att["abs"] = att["effect"].abs()
    att = att[att["abs"] > 1e-9]
    if att.empty:
        return pd.DataFrame([{"row": "all", "column": "-", "value": "-", "effect": 0.0,
                              "note": "changing any single column to its typical value does not change the score"}])
    top = att.sort_values(["row", "abs"], ascending=[True, False]).groupby("row").head(4)
    if b.id_column and b.id_column in rows.columns:
        top["row"] = top["row"].map(lambda i: f"{b.id_column}={rows.iloc[i][b.id_column]}")
    top["effect"] = top["effect"].round(4)
    return top.drop(columns="abs").reset_index(drop=True)


# ============================================================== unsupervised
def _profile(fe, centers: np.ndarray, columns: list[str], top: int = 4) -> list[str]:
    lines = []
    for k, center in enumerate(centers):
        order = np.argsort(-np.abs(center))[:top]
        parts = []
        for j in order:
            fname = columns[j]
            raw = fe.raw_value(fname, center[j])
            direction = "high" if center[j] > 0 else "low"
            parts.append(f"{direction} {fname}" + (f" (~{raw:.4g})" if raw is not None else ""))
        lines.append(f"**Cluster {k}**: " + ", ".join(parts))
    return lines


def _kmeans(b: ModelBundle, df, X, res: InferenceResult) -> None:
    est, fe = b.estimator, b.feature_engineer
    labels = est.predict(X)
    dist = est.transform(X)[np.arange(len(X)), labels]
    res.output["cluster"] = labels
    res.output["distance_to_centroid"] = dist.round(3)
    counts = pd.Series(labels).value_counts().sort_index()
    res.summary["cluster_counts"] = {int(k): int(v) for k, v in counts.items()}
    res.add("Cluster sizes in your data", counts.rename_axis("cluster").reset_index(name="rows"))
    res.add("What characterises each cluster (centroid, most distinctive features)",
            "\n\n".join(_profile(fe, est.cluster_centers_, list(X.columns))))
    res.add("Why a row belongs to its cluster",
            "K-Means assigns every row to the centroid with the smallest Euclidean distance in the "
            "standardised feature space; `distance_to_centroid` shows how typical the row is.")


def _pca(b: ModelBundle, df, X, res: InferenceResult) -> None:
    est = b.estimator
    Z = est.transform(X)
    for i in range(Z.shape[1]):
        res.output[f"PC{i + 1}"] = Z[:, i].round(4)
    evr = est.explained_variance_ratio_
    res.summary["explained_variance_ratio"] = [round(float(v), 4) for v in evr]
    res.add("Explained variance per component", pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(evr))],
        "explained_variance": evr.round(4), "cumulative": np.cumsum(evr).round(4)}))
    rows = []
    for i, comp in enumerate(est.components_):
        for j in np.argsort(-np.abs(comp))[:5]:
            rows.append({"component": f"PC{i + 1}", "feature": X.columns[j], "loading": round(float(comp[j]), 4)})
    res.add("Top loadings (which features each component is made of)", pd.DataFrame(rows))


def _isolation_forest(b: ModelBundle, df, X, res: InferenceResult) -> None:
    est = b.estimator
    flag = est.predict(X)
    score = est.decision_function(X)
    res.output["is_anomaly"] = flag == -1
    res.output["anomaly_score"] = (-score).round(4)   # higher = more anomalous
    n_anom = int((flag == -1).sum())
    res.summary["anomalies"] = n_anom
    res.summary["anomaly_rate"] = round(n_anom / len(X), 4)
    idx = np.argsort(score)[: MAX_EXPLAINED_ROWS]
    lines = []
    for i in idx:
        if flag[i] != -1:
            continue
        z = X.iloc[i]
        top = z.abs().sort_values(ascending=False).head(3).index
        ident = f"{b.id_column}={df.iloc[i][b.id_column]}" if b.id_column in df.columns else f"row {i}"
        parts = [f"{f} (z={z[f]:+.2f})" for f in top]
        lines.append(f"**{ident}** score={-score[i]:.3f}: most unusual -> " + ", ".join(parts))
    res.add("Why the top anomalies are unusual",
            "\n\n".join(lines) if lines else "No anomalies detected in these rows.")
    res.add("How the score works",
            "Isolation Forest isolates points with random splits; anomalies need fewer splits. "
            "z-values show how many standard deviations a feature is from the training mean.")


def _dbscan(b: ModelBundle, df, X, res: InferenceResult) -> None:
    core = b.extra["core_points"]
    core_labels = b.extra["core_labels"]
    eps = b.extra["eps"]
    from sklearn.neighbors import NearestNeighbors
    dist, ind = NearestNeighbors(n_neighbors=1).fit(core).kneighbors(X.to_numpy())
    nearest, nd = ind[:, 0], dist[:, 0]
    labels = np.where(nd <= eps, core_labels[nearest], -1)
    res.output["cluster"] = labels
    res.output["distance_to_nearest_core"] = nd.round(3)
    counts = pd.Series(labels).value_counts().sort_index()
    res.summary["cluster_counts"] = {int(k): int(v) for k, v in counts.items()}
    res.add("Cluster sizes (-1 = noise / outlier)", counts.rename_axis("cluster").reset_index(name="rows"))
    res.add("Why a row gets its label",
            f"DBSCAN has no native predict, so each row takes the cluster of its nearest core sample "
            f"from training if that sample is within eps={eps}; otherwise it is labelled noise (-1).")
