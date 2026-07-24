"""IEEE-styled aggregate figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import COLORS, EXPERIMENTS, SHORT_LABELS


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

def save_figure(fig, figure_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def make_figures(rows: List[Dict], summary: List[Dict], output_dir: Path, args) -> None:
    """Create collision-focused benchmark plots; reward plots are intentionally omitted."""
    apply_ieee_style()
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary).set_index("experiment").loc[EXPERIMENTS]
    labels = [SHORT_LABELS[e] for e in EXPERIMENTS]
    colors = [COLORS[e] for e in EXPERIMENTS]
    x = np.arange(len(EXPERIMENTS))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.bar(x, summary_df["selected_event_rmst_steps"], color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(
        float(summary_df["RMST_tau_steps"].iloc[0]),
        color="black", linestyle="--", linewidth=1, label="Restriction tau",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(f"{args.rmst_event.capitalize()} RMST (steps)")
    ax.set_title("MetaDrive Restricted Mean Survival")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_selected_event_rmst")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    width = 0.36
    ax.bar(
        x - width / 2, 100 * summary_df["train_collision_rate"], width,
        label="Train", edgecolor="black",
    )
    ax.bar(
        x + width / 2, 100 * summary_df["test_collision_rate"], width,
        label="Test", edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Episodes with collision (%)")
    ax.set_title("MetaDrive Collision Rate")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collision_rates")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(
        x - width / 2, summary_df["train_collisions_per_1000_steps"], width,
        label="Train", edgecolor="black",
    )
    ax.bar(
        x + width / 2, summary_df["test_collisions_per_1000_steps"], width,
        label="Test", edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Collisions per 1,000 steps")
    ax.set_title("MetaDrive Collision Exposure")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collisions_per_1000_steps")
