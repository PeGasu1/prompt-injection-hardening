"""Report figures (matplotlib, light theme, print-friendly).

Colour roles follow one validated palette: a single blue ramp for magnitudes (heatmaps) and
fixed categorical slots, assigned in order and never cycled, for series. Text always uses the
ink colours, never a series colour; grids and axes stay recessive.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import config

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Segoe UI only exists on Windows

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

FAMILY_LABELS = {"clean": "Clean", "zero_width": "Zero-width", "homoglyph": "Homoglyph",
                 "diacritics": "Diacritics", "leetspeak": "Leetspeak", "char_spacing": "Char spacing",
                 "punct_split": "Punct split"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.labelcolor": INK2,
    })


def _save(fig, name):
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def heatmap(matrix, title: str, name: str, fmt="{:.0%}", vmin=0.0, vmax=1.0, row_labels=None, note=None):
    """Models × families heatmap with the value printed in every cell."""
    _style()
    data = matrix.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * data.shape[1], 0.6 + 0.42 * data.shape[0]))
    ax.imshow(data, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [FAMILY_LABELS.get(c, c) for c in matrix.columns], rotation=30, ha="right")
    ax.set_yticks(range(data.shape[0]), row_labels or list(matrix.index))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)       # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            dark_bg = (v - vmin) / (vmax - vmin) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark_bg else INK)
    ax.set_title(title, pad=18 if note else 10)
    if note:
        ax.text(0.0, 1.01, note, color=MUTED, fontsize=7.5, ha="left", va="bottom", transform=ax.transAxes)
    return _save(fig, name)


def grouped_bars(table, categories: list[str], series: dict, title: str, name: str, ylabel: str,
                 errors: dict | None = None, ylim=(0, 1), percent=True):
    """``table`` indexed by category; ``series`` maps legend label → column name (≤ 8, fixed order)."""
    _style()
    fig, ax = plt.subplots(figsize=(1.2 + 1.05 * len(categories), 3.0))
    k = len(series)
    width = 0.8 / k
    x = np.arange(len(categories))
    for i, (label, col) in enumerate(series.items()):
        vals = table.loc[categories, col].values.astype(float)
        err = None
        if errors and label in errors:
            lo, hi = errors[label]
            err = np.vstack([vals - table.loc[categories, lo].values, table.loc[categories, hi].values - vals])
        ax.bar(x + (i - (k - 1) / 2) * width, vals, width, label=label, color=SERIES[i],
               edgecolor=SURFACE, linewidth=1.0, yerr=err, error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 0})
    ax.set_xticks(x, [FAMILY_LABELS.get(c, c) for c in categories], rotation=0)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=min(k, 4), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    ax.set_title(title, pad=26)
    return _save(fig, name)


def hbar(labels, values, title: str, name: str, xlabel: str, percent=True, log=False, highlight=None, ci=None):
    """Single-series horizontal bars, value labels at the bar end; ``highlight`` indices use slot 2."""
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 0.5 + 0.32 * len(labels)))
    y = np.arange(len(labels))[::-1]
    colors = [SERIES[1] if highlight and i in highlight else SERIES[0] for i in range(len(labels))]
    xerr = None
    if ci is not None:
        xerr = np.vstack([np.asarray(values) - ci[0], ci[1] - np.asarray(values)])
    ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.0, xerr=xerr,
            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 0})
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    if log:
        ax.set_xscale("log")
    for yi, v in zip(y, values):
        ax.text(v * (1.08 if log else 1.0) + (0 if log else 0.01 * max(values)), yi,
                f"{v:.0%}" if percent else (f"{v:.2g}" if v < 10 else f"{v:.0f}"), va="center", fontsize=8, color=INK2)
    if percent and not log:
        ax.set_xlim(0, max(1.0, max(values) * 1.12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title(title, pad=8)
    return _save(fig, name)
