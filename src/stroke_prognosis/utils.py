"""
Utility helpers for reproducible analysis and publication-quality figures.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (numpy + python)."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike) -> str:
    """Create directory if it doesn't exist and return absolute path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def set_pub_plot_style(font_family: str = "Times New Roman") -> None:
    """Configure matplotlib for vector-friendly PDF export and consistent typography."""
    import matplotlib.pyplot as plt  # local import to keep dependencies light

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [font_family]
    plt.rcParams["axes.unicode_minus"] = False
    # Vector-friendly, editable fonts in PDF/PS
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_fig(path: str, dpi: int = 600, tight: bool = True) -> None:
    """Save current matplotlib figure."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": dpi}
    if tight:
        kwargs["bbox_inches"] = "tight"
    plt.savefig(out, **kwargs)
