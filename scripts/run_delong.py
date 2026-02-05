#!/usr/bin/env python
"""
Run DeLong tests from an exported prediction file.

Input file format:
- True_Label column (0/1)
- One probability column per model

Example:
  python scripts/run_delong.py --pred outputs/model_comparison/predictions_for_delong.xlsx \
    --a XGBoost --b LogisticRegression
"""
from __future__ import annotations

import argparse
import pandas as pd

from stroke_prognosis.io import read_table
from stroke_prognosis.delong import delong_roc_test


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Excel/CSV with True_Label + probability columns")
    ap.add_argument("--a", required=True, help="Model column name A")
    ap.add_argument("--b", required=True, help="Model column name B")
    return ap.parse_args()


def main():
    args = parse_args()
    df = read_table(args.pred)
    for col in ["True_Label", args.a, args.b]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Available: {list(df.columns)}")

    y = df["True_Label"].to_numpy()
    auc_a, auc_b, z, p = delong_roc_test(y, df[args.a].to_numpy(), df[args.b].to_numpy())
    print(f"AUC({args.a}) = {auc_a:.4f}")
    print(f"AUC({args.b}) = {auc_b:.4f}")
    print(f"z = {z:.4f}, p = {p:.4g}")


if __name__ == "__main__":
    main()
