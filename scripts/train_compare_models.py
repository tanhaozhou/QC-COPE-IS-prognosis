#!/usr/bin/env python
"""
Train and compare multiple classifiers for stroke prognosis prediction.

Example:
  python scripts/train_compare_models.py \
    --data data/train.xlsx --label_col outcome \
    --test_size 0.2 --seed 42 \
    --out_dir outputs/model_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stroke_prognosis.io import read_table, split_xy
from stroke_prognosis.models import get_classifiers
from stroke_prognosis.evaluation import compute_metrics, bootstrap_auc_ci, save_predictions_for_delong
from stroke_prognosis.utils import set_seed, ensure_dir, set_pub_plot_style


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to input table (.xlsx/.csv/.tsv)")
    ap.add_argument("--label_col", required=True, help="Binary outcome column name")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="outputs/model_comparison")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    set_pub_plot_style()

    out_dir = Path(ensure_dir(args.out_dir))
    df = read_table(args.data)

    X, y = split_xy(df, args.label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    results = []
    prob_dict = {}

    for name, clf in get_classifiers(seed=args.seed):
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        prob_dict[name] = prob

        metrics = compute_metrics(y_test.to_numpy(), prob)
        ci_lo, ci_hi = bootstrap_auc_ci(y_test.to_numpy(), prob, seed=args.seed)

        results.append({
            "model": name,
            "auc": metrics.auc,
            "auc_ci_low": ci_lo,
            "auc_ci_high": ci_hi,
            "brier": metrics.brier,
            "n_test": int(len(y_test)),
        })

    res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
    res_path = out_dir / "model_metrics.csv"
    res_df.to_csv(res_path, index=False)

    pred_path = out_dir / "predictions_for_delong.xlsx"
    save_predictions_for_delong(y_test.to_numpy(), prob_dict, str(pred_path))

    meta = {
        "data": str(Path(args.data).resolve()),
        "label_col": args.label_col,
        "test_size": args.test_size,
        "seed": args.seed,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "outputs": {
            "metrics_csv": str(res_path),
            "predictions_xlsx": str(pred_path),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved metrics -> {res_path}")
    print(f"[OK] Saved predictions -> {pred_path}")


if __name__ == "__main__":
    main()
