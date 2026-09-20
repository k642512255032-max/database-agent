"""Generate a training_config.yaml for YOUR database.

Step 1 - see which tables/columns exist:
    python make_training_config.py

Step 2 - generate models for one table (or view) and the column to predict:
    python make_training_config.py --table customers --target churned

It backs up the old training_config.yaml, then writes a new one with:
  decision_tree, random_forest, knn   (predict --target)
  kmeans, pca, isolation_forest, dbscan (on the other numeric/categorical columns)
Then run:  python train_models.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from agent.db import Database

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", help="table or view that has one row per thing to predict")
    ap.add_argument("--target", help="column to predict (optional: unsupervised models only)")
    ap.add_argument("--id", dest="id_column", help="ID column (default: the primary key)")
    ap.add_argument("--out", default=str(ROOT / "training_config.yaml"))
    args = ap.parse_args()

    db = Database()
    ok, msg = db.ping()
    print(msg)
    if not ok:
        raise SystemExit(1)
    schema = db.schema()

    if not args.table:
        print("\nTables and views in your database:\n")
        for t in schema.values():
            print(f"  {t.name}  ({t.row_count} rows)")
            print(f"      columns: {', '.join(t.column_names)}")
        print("\nNext: python make_training_config.py --table <table> --target <column to predict>")
        return

    match = {n.lower(): n for n in schema}
    if args.table.lower() not in match:
        raise SystemExit(f"Table '{args.table}' not found. Available: {', '.join(schema)}")
    table = schema[match[args.table.lower()]]
    cols = table.column_names
    target = None
    if args.target:
        low = {c.lower(): c for c in cols}
        if args.target.lower() not in low:
            raise SystemExit(f"Column '{args.target}' not in {table.name}. Columns: {', '.join(cols)}")
        target = low[args.target.lower()]
    id_col = args.id_column or (table.primary_key[0] if table.primary_key else None)

    df = db.run(f"SELECT * FROM {db.quote(table.name)} LIMIT 5000")
    if len(df) < 30:
        print(f"WARNING: only {len(df)} rows - models need more data to be meaningful.")

    sql = f"SELECT * FROM {table.name}"
    exclude = [c for c in [target] if c]
    # drop free-text-ish columns that would explode feature engineering
    for c in df.columns:
        if c in (target, id_col):
            continue
        if df[c].dtype == object and df[c].astype(str).str.len().mean() > 60:
            exclude.append(c)

    models = []
    if target:
        y = df[target].dropna()
        numeric = pd.to_numeric(y, errors="coerce").notna().mean() > 0.95
        task = "classification" if (not numeric or y.nunique() <= 10) else "regression"
        print(f"Target '{target}' has {y.nunique()} distinct values -> {task}")
        base = {"task": task, "sql": sql, "target": target, "id_column": id_col}
        extra_w = {"class_weight": "balanced"} if task == "classification" else {}
        models += [
            {"name": f"{target}_decision_tree", "algorithm": "decision_tree",
             "description": f"Predicts {target} with explainable decision rules", **base,
             "params": {"max_depth": 4, "min_samples_leaf": 10, **extra_w}},
            {"name": f"{target}_random_forest", "algorithm": "random_forest",
             "description": f"Predicts {target} with a random forest (usually most accurate)", **base,
             "params": {"n_estimators": 200, "max_depth": 8, **extra_w}},
            {"name": f"{target}_knn", "algorithm": "knn",
             "description": f"Predicts {target} from the most similar existing rows", **base,
             "params": {"n_neighbors": 15, "weights": "distance"}},
        ]
    unsup = {"sql": sql, "id_column": id_col, "exclude": exclude}
    models += [
        {"name": f"{table.name}_segments_kmeans", "algorithm": "kmeans", "task": "clustering",
         "description": f"Groups {table.name} rows into segments with similar characteristics", **unsup,
         "params": {"n_clusters": 4, "n_init": 10}},
        {"name": f"{table.name}_pca", "algorithm": "pca", "task": "dimensionality_reduction",
         "description": f"Compresses {table.name} attributes into principal components", **unsup,
         "params": {"n_components": 2}},
        {"name": f"{table.name}_anomaly_isoforest", "algorithm": "isolation_forest", "task": "anomaly_detection",
         "description": f"Detects unusual or suspicious {table.name} rows (outliers)", **unsup,
         "params": {"n_estimators": 300, "contamination": 0.02}},
        {"name": f"{table.name}_density_dbscan", "algorithm": "dbscan", "task": "clustering",
         "description": f"Density-based clusters of {table.name}; flags rows that fit no group as noise", **unsup,
         "params": {"eps": "auto", "min_samples": 10}},
    ]
    for m in models:
        if not m.get("id_column"):
            m.pop("id_column", None)
        if "exclude" in m and not m["exclude"]:
            m.pop("exclude")

    out = Path(args.out)
    if out.exists():
        shutil.copy(out, out.with_suffix(".yaml.bak"))
        print(f"Old config backed up to {out.with_suffix('.yaml.bak').name}")
    out.write_text(yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {len(models)} models to {out.name}")

    # remove old trained models (e.g. the demo ones) so the agent doesn't use them
    models_dir = ROOT / "models"
    old = list(models_dir.glob("*.joblib")) + list(models_dir.glob("manifest.json"))
    for f in old:
        f.unlink()
    if old:
        print(f"Removed {len(old)} old model files from models/")
    print("\nNext: python train_models.py")


if __name__ == "__main__":
    main()
