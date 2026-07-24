#!/usr/bin/env python3
"""Recompute the selected-seed Highway Collision RMST analysis.

Run this script from framework/analysis/ inside the delivered package. It
regenerates collision-RMST seed values, IQMs, bootstrap confidence intervals,
paired tests, and the Highway figure from the collision-only episode file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TAU = 500
BOOTSTRAPS = 50_000
BOOTSTRAP_SEED = 20_260_724
METHODS = ("epsilon", "noisy", "rnd", "safetypool")
LABELS = {
    "epsilon": "Epsilon-Greedy DQN",
    "noisy": "NoisyNet DQN",
    "rnd": "DQN + RND",
    "safetypool": "SafetyPool",
}


def km_rmst(times: np.ndarray, events: np.ndarray, tau: int = TAU) -> float:
    times = np.minimum(times.astype(int), tau)
    events = events.astype(bool)
    survival = 1.0
    area = 0.0
    previous = 0
    for time in np.unique(times[times <= tau]):
        time = int(time)
        area += survival * (time - previous)
        at_risk = int(np.sum(times >= time))
        event_count = int(np.sum((times == time) & events))
        if event_count:
            survival *= 1.0 - event_count / at_risk
        previous = time
    area += survival * max(0, tau - previous)
    return float(area)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


def main() -> None:
    package = Path(__file__).resolve().parents[2]
    data_path = (
        package / "data" / "highway" / "highway_collision_rmst_episode_data.csv"
    )
    output = package / "recomputed"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(data_path)
    seeds = sorted(frame["seed"].unique())

    arrays: dict[str, np.ndarray] = {}
    seed_rows = []
    for method in METHODS:
        values = []
        for seed in seeds:
            rows = frame[
                frame["method_id"].eq(method) & frame["seed"].eq(seed)
            ].sort_values("episode")
            if len(rows) != 300:
                raise RuntimeError(
                    f"Expected 300 episodes for {method}, seed {seed}; "
                    f"found {len(rows)}"
                )
            value = km_rmst(
                rows["event_or_censor_time_steps"].to_numpy(),
                rows["collision_event_observed"].to_numpy(),
            )
            values.append(value)
            seed_rows.append(
                {
                    "environment": "HighwayEnv",
                    "seed": seed,
                    "method_id": method,
                    "method": LABELS[method],
                    "collision_rmst_steps": value,
                    "rmst_horizon_steps": TAU,
                }
            )
        arrays[method] = np.asarray(values, dtype=float)
    pd.DataFrame(seed_rows).to_csv(
        output / "highway_collision_rmst_by_seed.csv", index=False
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(seeds), size=(BOOTSTRAPS, len(seeds)))
    summary_rows = []
    for method in METHODS:
        values = arrays[method]
        trim_each_tail = int(0.25 * len(seeds))
        sorted_samples = np.sort(values[indices], axis=1)
        samples = sorted_samples[
            :, trim_each_tail : len(seeds) - trim_each_tail
        ].mean(axis=1)
        low, high = np.percentile(samples, [2.5, 97.5])
        summary_rows.append(
            {
                "method_id": method,
                "method": LABELS[method],
                "n_seeds": len(seeds),
                "collision_rmst_iqm_steps": stats.trim_mean(values, 0.25),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "highway_collision_rmst_summary.csv", index=False)

    raw = {
        baseline: float(
            stats.wilcoxon(
                arrays["safetypool"],
                arrays[baseline],
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
        for baseline in ("epsilon", "noisy", "rnd")
    }
    adjusted = holm_adjust(raw)
    tests = pd.DataFrame(
        [
            {
                "comparison": f"SafetyPool vs {LABELS[baseline]}",
                "p_raw": raw[baseline],
                "p_holm": adjusted[baseline],
            }
            for baseline in ("epsilon", "noisy", "rnd")
        ]
    )
    tests.to_csv(output / "highway_collision_rmst_pairwise_tests.csv", index=False)

    order = ["safetypool", "epsilon", "noisy", "rnd"]
    chart = summary.set_index("method_id").loc[order]
    values = chart["collision_rmst_iqm_steps"].to_numpy()
    error = np.vstack(
        [
            values - chart["bootstrap_ci_low"].to_numpy(),
            chart["bootstrap_ci_high"].to_numpy() - values,
        ]
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    bars = ax.bar(
        ["SafetyPool", "Epsilon-\nGreedy", "NoisyNet", "RND"],
        values,
        color=colors,
        edgecolor="#222222",
        yerr=error,
        capsize=4,
    )
    ax.set_ylabel("Collision RMST IQM (steps)")
    ax.set_title("HighwayEnv Collision RMST — 23 matched seeds")
    ax.grid(axis="y", color="#D9D9D9")
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            f"{value:.2f}",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output / "highway_collision_rmst_iqm.pdf", bbox_inches="tight")
    fig.savefig(output / "highway_collision_rmst_iqm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    (output / "analysis_parameters.json").write_text(
        json.dumps(
            {
                "seeds": [int(seed) for seed in seeds],
                "rmst_horizon_steps": TAU,
                "iqm": "scipy.stats.trim_mean(values, 0.25)",
                "bootstrap_samples": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "test": "paired two-sided Wilcoxon signed-rank",
                "multiplicity": "Holm correction across three baselines",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
