"""Shared plotting style so that every figure of the paper looks the same."""

from __future__ import annotations

import matplotlib as mpl

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12

INDEX_COLOR = "tab:blue"
SECOND_COLOR = "tab:orange"
SHELL_COLOR = "red"
DENSITY_CMAP = "inferno"


def use_paper_style() -> None:
    """Apply the figure defaults used throughout the paper."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": TICK_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": TICK_FONTSIZE,
        "axes.grid": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
    })
