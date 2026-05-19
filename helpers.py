"""
utils/helpers.py
────────────────
Shared constants, matplotlib theme, and small helper functions
used across all pages.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Colour palette ──────────────────────────────────────────
BLUE   = "#38bdf8"
RED    = "#f87171"
GREEN  = "#4ade80"
AMBER  = "#fbbf24"
PURPLE = "#a78bfa"
GRAY   = "#475569"

SEED = 42

# ── Dark matplotlib theme ────────────────────────────────────
def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor":  "#0a0e1a",
        "axes.facecolor":    "#111827",
        "axes.edgecolor":    "#1e2d4a",
        "axes.labelcolor":   "#94a3b8",
        "axes.titlecolor":   "#e2e8f0",
        "axes.grid":         True,
        "grid.color":        "#1e2d4a",
        "grid.linewidth":    0.6,
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "text.color":        "#e2e8f0",
        "legend.facecolor":  "#111827",
        "legend.edgecolor":  "#1e2d4a",
        "legend.labelcolor": "#94a3b8",
        "font.family":       "monospace",
        "axes.titlesize":    12,
        "axes.labelsize":    11,
    })

# ── Reusable plot functions ──────────────────────────────────
def plot_confusion(y_true, y_pred, title, color):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cm = confusion_matrix(y_true, y_pred)
    cmap = sns.light_palette(color, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor="#0a0e1a",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_score_dist(scores, y_true, title):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(scores[y_true == 0], bins=60, alpha=0.65,
            color=BLUE, label="Legitimate", density=True)
    ax.hist(scores[y_true == 1], bins=30, alpha=0.75,
            color=RED,  label="Fraud",      density=True)
    ax.set_title(title)
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig
