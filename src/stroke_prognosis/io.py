"""
Data I/O helpers.

Convention:
- Put raw data under ./data/
- Put generated artifacts under ./outputs/
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV/Excel based on suffix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p.resolve()}")

    suf = p.suffix.lower()
    if suf in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if suf in [".csv"]:
        return pd.read_csv(p)
    if suf in [".tsv", ".txt"]:
        return pd.read_csv(p, sep="\t")
    raise ValueError(f"Unsupported file type: {suf}")


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into X (features) and y (label)."""
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not found in columns: {list(df.columns)}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y
