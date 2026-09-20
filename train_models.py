"""Train and save the models listed in training_config.yaml (run once, or whenever data changes).

    python train_models.py                       # train everything
    python train_models.py --only churn_knn      # train one model
    python train_models.py --config employees_training_config.yaml   # another config file
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agent.db import Database
from ml.features import AutoFeatureEngineer
from ml.registry import ModelBundle, ModelRegistry

ESTIMATORS = {
    ("decision_tree", "classification"): DecisionTreeClassifier,
    ("decision_tree", "regression"): DecisionTreeRegressor,
    ("random_forest", "classification"): RandomForestClassifier,
    ("random_forest", "regression"): RandomForestRegressor,
    ("knn", "classification"): KNeighborsClassifier,
    ("knn", "regression"): KNeighborsRegressor,
    ("kmeans", "clustering"): KMeans,
    ("pca", "dimensionality_reduction"): PCA,
    ("isolation_forest", "anomaly_detection"): IsolationForest,
    ("dbscan", "clustering"): DBSCAN,
}
RANDOM_STATE = 42


def train_one(spec: dict, db: Database) -> ModelBundle:
    algo, task = spec["algorithm"], spec["task"]
    cls = ESTIMATORS[(algo, task)]
    params = dict(spec.get("params") or {})
    df = db.run(spec["sql"])
    cap = spec.get("max_train_rows")
    if cap and len(df) > cap:
        print(f"   sampling {cap} of {len(df)} rows for training (max_train_rows)")
        df = df.sample(n=int(cap), random_state=RANDOM_STATE).reset_index(drop=True)
    target, id_col = spec.get("target"), spec.get("id_column")
    missing = [c for c in [target, id_col] if c and c not in df.columns]
    if missing:
        raise ValueError(f"columns {missing} are not in the result of: {spec['sql']}. Columns: {list(df.columns)}")
    exclude = set(spec.get("exclude", [])) | {target, id_col}
    feature_columns = [c for c in df.columns if c not in exclude]
    print(f"\n== {spec['name']} ({algo}/{task}) - {len(df)} rows, {len(feature_columns)} raw features")

    fe = AutoFeatureEngineer()
    extra, metrics = {}, {}

    if task in {"classification", "regression"}:
        df = df.dropna(subset=[target])
        y = df[target]
        y = y.astype(int) if task == "classification" and pd.api.types.is_numeric_dtype(y) else y
        if task == "regression":
            y = pd.to_numeric(y, errors="coerce").astype(float)
        strat = y if task == "classification" else None
        tr, te = train_test_split(df.index, test_size=0.25, random_state=RANDOM_STATE, stratify=strat)
        Xtr = fe.fit_transform(df.loc[tr, feature_columns])
        Xte = fe.transform(df.loc[te, feature_columns])
        if "random_state" in cls().get_params():
            params.setdefault("random_state", RANDOM_STATE)
        est = cls(**params).fit(Xtr, y.loc[tr])
        pred = est.predict(Xte)
        if task == "classification":
            metrics = {"test_accuracy": round(accuracy_score(y.loc[te], pred), 4),
                       "test_f1_macro": round(f1_score(y.loc[te], pred, average="macro"), 4)}
            if len(est.classes_) == 2:
                metrics["test_roc_auc"] = round(roc_auc_score(y.loc[te], est.predict_proba(Xte)[:, 1]), 4)
        else:
            metrics = {"test_r2": round(r2_score(y.loc[te], pred), 4),
                       "test_mae": round(mean_absolute_error(y.loc[te], pred), 4)}
        if algo == "knn":
            extra["train_ids"] = (df.loc[tr, id_col].to_numpy() if id_col else np.asarray(tr))
            extra["train_y"] = y.loc[tr].to_numpy()
    else:
        X = fe.fit_transform(df[feature_columns])
        if algo == "dbscan" and params.get("eps", "auto") == "auto":
            k = params.get("min_samples", 5)
            d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
            params["eps"] = round(float(np.percentile(d[:, -1], 90)), 4)
            print(f"   auto eps (90th percentile of {k}-NN distance) = {params['eps']}")
        if "random_state" in cls().get_params():
            params.setdefault("random_state", RANDOM_STATE)
        est = cls(**params)
        if algo == "kmeans":
            est.fit(X)
            metrics = {"silhouette": round(float(silhouette_score(X, est.labels_, sample_size=min(2000, len(X)), random_state=0)), 4),
                       "inertia": round(float(est.inertia_), 2)}
        elif algo == "pca":
            est.fit(X)
            metrics = {"explained_variance_total": round(float(est.explained_variance_ratio_.sum()), 4)}
        elif algo == "isolation_forest":
            est.fit(X)
            metrics = {"train_anomaly_rate": round(float((est.predict(X) == -1).mean()), 4)}
        elif algo == "dbscan":
            est.fit(X)
            labels = est.labels_
            core = est.core_sample_indices_
            extra.update({"core_points": X.to_numpy()[core], "core_labels": labels[core], "eps": params["eps"]})
            n_clusters = len(set(labels) - {-1})
            metrics = {"n_clusters": n_clusters, "noise_ratio": round(float((labels == -1).mean()), 4)}
            mask = labels != -1
            if n_clusters > 1:
                metrics["silhouette_non_noise"] = round(float(silhouette_score(X[mask], labels[mask])), 4)

    print("   feature plan:")
    for col, d in fe.plan.items():
        print(f"     - {col:<24} {d['type']:<14} {d['action']}")
    print(f"   metrics: {metrics}")

    return ModelBundle(
        name=spec["name"], algorithm=algo, task=task, description=spec.get("description", ""),
        base_sql=spec["sql"], feature_columns=feature_columns, estimator=est, feature_engineer=fe,
        target=target, id_column=id_col, metrics=metrics,
        trained_at=datetime.now().isoformat(timespec="seconds"), extra=extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training_config.yaml")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--database-url")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    db, reg = Database(args.database_url), ModelRegistry()
    ok, msg = db.ping()
    print(msg)
    if not ok:
        raise SystemExit(
            "\nCannot connect to the database. Check DATABASE_URL in your .env file "
            "(or pass --database-url). Test with:\n"
            '  python -c "from agent.db import Database; print(Database().ping())"'
        )
    tables = {t.lower() for t in db.schema()}
    failed, trained = [], 0
    for spec in cfg["models"]:
        if args.only and spec["name"] not in args.only:
            continue
        try:
            path = reg.save(train_one(spec, db))
            print(f"   saved -> {path}")
            trained += 1
        except Exception as exc:
            msg = str(exc).split("\n")[0]
            hint = ""
            if "doesn't exist" in msg:
                hint = (f"\n   HINT: this model reads from a table/view that is not in your database. "
                        f"Your tables: {', '.join(sorted(tables))}. Edit 'sql' in {args.config}, "
                        f"or create the view first.")
            print(f"\n!! {spec['name']} FAILED: {msg}{hint}")
            failed.append(spec["name"])

    if not args.only:  # drop old models (e.g. the demo ones) that are not in this config
        keep = {s["name"] for s in cfg["models"]} - set(failed)
        for name in reg.names():
            if name not in keep:
                reg.remove(name)
                print(f"   removed old model '{name}' (not in {args.config})")
    print(f"\nDone: {trained} trained, {len(failed)} failed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
