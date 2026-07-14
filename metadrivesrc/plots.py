#!/usr/bin/env python3
"""Generate the five paper figures for the Highway Epsilon vs Median 50 study.

Place this file in ``highway/src`` and run ``python3 plot.py``. By default it
reads ``../results/all_episode_results.csv`` and writes JPEG files to
``../results/plots``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("Epsilon Greedy", "Median 50")
COLORS = {"Epsilon Greedy": "#D62728", "Median 50": "#2CA02C"}
ROLLING_WINDOW = 10
CONVERGENCE_FRACTION = 0.90
SURVIVAL_HORIZON = 500
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 20260712
OUTPUT_FILES = (
    "01_mean_frozen_test_reward.jpeg",
    "02_iqm_frozen_test_reward.jpeg",
    "03_highway_rmst_collision_free_survival.jpeg",
    "04_time_to_shared_90pct_target.jpeg",
    "05_frozen_test_reward_boxplot.jpeg",
)


def arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results = script_dir.parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results,
        help="Results directory containing all_episode_results.csv.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def load_results(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = results_dir / "all_episode_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Put plot.py in the Highway src folder "
            "or pass --results-dir /path/to/results."
        )
    data = pd.read_csv(csv_path)
    required = {
        "method", "phase", "episode", "env_reward", "steps",
        "termination_reason", "wall_time_seconds",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    data = data[data["method"].isin(("Epsilon", "Median 50"))].copy()
    data["method"] = data["method"].replace({"Epsilon": "Epsilon Greedy"})
    data = data.rename(
        columns={
            "env_reward": "reward",
            "termination_reason": "term_reason",
            "wall_time_seconds": "wall_seconds",
        }
    )
    train = data[data["phase"].eq("train")].copy()
    test = data[data["phase"].eq("test")].copy()
    absent = [method for method in METHODS if method not in set(test["method"])]
    if absent:
        raise ValueError(f"Frozen-test data missing for: {', '.join(absent)}")
    return train, test


def clean_plot_images(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.jpeg", "*.jpg", "*.png", "*.pdf"):
        for path in output_dir.glob(pattern):
            path.unlink()


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int) -> None:
    fig.tight_layout(pad=0.6)
    fig.savefig(
        output_dir / filename,
        format="jpeg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def iqm(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=float))
    if ordered.size == 0:
        return math.nan
    lower = int(math.floor(0.25 * ordered.size))
    upper = int(math.ceil(0.75 * ordered.size))
    return float(ordered[lower:upper].mean())


def bootstrap_iqm(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    samples = np.sort(values[indices], axis=1)
    lower = int(math.floor(0.25 * len(values)))
    upper = int(math.ceil(0.75 * len(values)))
    return samples[:, lower:upper].mean(axis=1)


def paired_iqm_difference(test: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    paired = test.pivot_table(index="episode", columns="method", values="reward", aggfunc="first")
    paired = paired.dropna(subset=list(METHODS)).sort_index()
    epsilon = paired[METHODS[0]].to_numpy(dtype=float)
    median50 = paired[METHODS[1]].to_numpy(dtype=float)
    indices = rng.integers(0, len(paired), size=(BOOTSTRAP_SAMPLES, len(paired)))
    lower = int(math.floor(0.25 * len(paired)))
    upper = int(math.ceil(0.75 * len(paired)))
    eps_samples = np.sort(epsilon[indices], axis=1)[:, lower:upper].mean(axis=1)
    med_samples = np.sort(median50[indices], axis=1)[:, lower:upper].mean(axis=1)
    return med_samples - eps_samples


def plot_mean_test_reward(test: pd.DataFrame, output_dir: Path, dpi: int) -> Dict[str, float]:
    values = {
        method: float(test.loc[test["method"].eq(method), "reward"].mean())
        for method in METHODS
    }
    x = np.arange(len(METHODS))
    heights = [values[method] for method in METHODS]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(
        x,
        heights,
        width=0.58,
        color=[COLORS[method] for method in METHODS],
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_title("Mean Frozen-Test Reward (Higher Is Better)")
    ax.set_ylabel("Mean episode reward")
    ax.set_xticks(x, METHODS)
    ax.set_ylim(0, max(heights) * 1.18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    save_figure(fig, output_dir, OUTPUT_FILES[0], dpi)
    return values


def plot_iqm(test: pd.DataFrame, output_dir: Path, dpi: int) -> Dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    values: Dict[str, float] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    for method in METHODS:
        rewards = test.loc[test["method"].eq(method), "reward"].to_numpy(dtype=float)
        values[method] = iqm(rewards)
        distribution = bootstrap_iqm(rewards, rng)
        intervals[method] = tuple(np.quantile(distribution, [0.025, 0.975]))

    difference = paired_iqm_difference(test, rng)
    difference_ci = np.quantile(difference, [0.025, 0.975])
    x = np.arange(len(METHODS))
    heights = np.array([values[m] for m in METHODS])
    lower = heights - np.array([intervals[m][0] for m in METHODS])
    upper = np.array([intervals[m][1] for m in METHODS]) - heights

    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(
        x,
        heights,
        width=0.58,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.7,
        yerr=np.vstack([lower, upper]),
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_title("IQM Frozen-Test Reward")
    ax.set_ylabel("Interquartile mean reward")
    ax.set_xticks(x, METHODS)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, height in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{height:.2f}",
                ha="center", va="bottom", fontsize=7)
    save_figure(fig, output_dir, OUTPUT_FILES[1], dpi)

    values["difference_ci_low"] = float(difference_ci[0])
    values["difference_ci_high"] = float(difference_ci[1])
    return values


def kaplan_meier(times: np.ndarray, events: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    times = np.minimum(np.asarray(times, dtype=float), float(horizon))
    events = np.asarray(events, dtype=bool)
    survival = 1.0
    previous = 0.0
    rmst = 0.0
    median_time = math.inf
    curve_t = [0.0]
    curve_s = [1.0]
    for current in np.unique(times):
        rmst += survival * (current - previous)
        at_risk = int(np.sum(times >= current))
        event_count = int(np.sum(events[times == current]))
        if event_count:
            survival *= 1.0 - event_count / at_risk
            if math.isinf(median_time) and survival <= 0.5:
                median_time = float(current)
        curve_t.extend([float(current), float(current)])
        curve_s.extend([curve_s[-1], float(survival)])
        previous = float(current)
    if previous < horizon:
        rmst += survival * (horizon - previous)
        curve_t.append(float(horizon))
        curve_s.append(float(survival))
    return np.asarray(curve_t), np.asarray(curve_s), float(rmst), median_time


def plot_survival(test: pd.DataFrame, output_dir: Path, dpi: int) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for method in METHODS:
        group = test[test["method"].eq(method)].sort_values("episode")
        events = group["term_reason"].eq("collision").to_numpy()
        times = group["steps"].to_numpy(dtype=float)
        _, _, rmst, median_time = kaplan_meier(times, events, SURVIVAL_HORIZON)
        collision_free_rate = 100.0 * (1.0 - events.mean())
        summary[f"{method}_rmst"] = rmst
        summary[f"{method}_collision_free_rate"] = collision_free_rate
        summary[f"{method}_median_time"] = median_time

    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    rmst_values = [summary[f"{method}_rmst"] for method in METHODS]
    difference = rmst_values[1] - rmst_values[0]
    values = [rmst_values[0], rmst_values[1], difference]
    labels = ["Epsilon RMST", "Median 50 RMST", "Difference"]
    colors = [COLORS["Epsilon Greedy"], COLORS["Median 50"], "#777777"]
    x = np.arange(len(values))
    rmst_bars = ax.bar(
        x, values, width=0.58, color=colors, edgecolor="black", linewidth=0.7
    )
    ax.set_title("Highway Restricted Mean Collision-Free Survival")
    ax.set_ylabel(f"RMST (steps; {SURVIVAL_HORIZON}-step horizon)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(values) * 1.28)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, value in zip(rmst_bars, values):
        label_y = value if value > 0 else max(values) * 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    save_figure(fig, output_dir, OUTPUT_FILES[2], dpi)
    return summary


def convergence_metrics(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[float, Dict[str, dict]]:
    test_means = test.groupby("method")["reward"].mean().reindex(METHODS)
    target = CONVERGENCE_FRACTION * float(test_means.max())
    results: Dict[str, dict] = {}
    for method in METHODS:
        group = train[train["method"].eq(method)].sort_values("episode").copy()
        group["rolling_reward"] = group["reward"].rolling(
            ROLLING_WINDOW, min_periods=ROLLING_WINDOW
        ).mean()
        crossed = group[group["rolling_reward"].ge(target)]
        reached = not crossed.empty
        if reached:
            position = int(group.index.get_loc(crossed.index[0]))
            observed = group.iloc[: position + 1]
            episode = int(observed.iloc[-1]["episode"]) + 1
        else:
            observed = group
            episode = None
        results[method] = {
            "reached": reached,
            "episode": episode,
            "seconds": float(observed["wall_seconds"].fillna(0).sum()),
            "interactions": int(observed["steps"].fillna(0).sum()),
        }
    return target, results


def plot_convergence(train: pd.DataFrame, test: pd.DataFrame, output_dir: Path, dpi: int) -> Tuple[float, Dict[str, dict]]:
    target, metrics = convergence_metrics(train, test)
    x = np.arange(len(METHODS))
    heights = [metrics[m]["seconds"] for m in METHODS]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(
        x,
        heights,
        width=0.58,
        color=[COLORS[m] for m in METHODS],
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, method in zip(bars, METHODS):
        record = metrics[method]
        if not record["reached"]:
            bar.set_hatch("///")
            label = f">{record['seconds']:.2f} s\nnot reached"
        else:
            label = f"{record['seconds']:.2f} s\nepisode {record['episode']}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=6.5)
    ax.set_title("Training Time to Shared 90% Reward Target")
    ax.set_ylabel("Cumulative wall-clock time (s)")
    ax.set_xticks(x, METHODS)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)
    save_figure(fig, output_dir, OUTPUT_FILES[3], dpi)
    return target, metrics


def plot_test_boxplot(test: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    arrays = [test.loc[test["method"].eq(method), "reward"].to_numpy(dtype=float) for method in METHODS]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    box = ax.boxplot(
        arrays,
        labels=METHODS,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.45},
    )
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.78)
    ax.set_title("Frozen-Test Reward Distribution")
    ax.set_ylabel("Episode reward")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save_figure(fig, output_dir, OUTPUT_FILES[4], dpi)


def main() -> None:
    args = arguments()
    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "plots"
    configure_ieee_style()
    train, test = load_results(results_dir)
    clean_plot_images(output_dir)

    mean_summary = plot_mean_test_reward(test, output_dir, args.dpi)
    iqm_summary = plot_iqm(test, output_dir, args.dpi)
    survival_summary = plot_survival(test, output_dir, args.dpi)
    target, convergence = plot_convergence(train, test, output_dir, args.dpi)
    plot_test_boxplot(test, output_dir, args.dpi)

    print(f"Created {len(OUTPUT_FILES)} JPEG plots in: {output_dir}")
    for filename in OUTPUT_FILES:
        print(f"  {filename}")
    print("\nHighway paper metrics")
    print(f"  Epsilon mean test reward:   {mean_summary['Epsilon Greedy']:.2f}")
    print(f"  Median 50 mean test reward: {mean_summary['Median 50']:.2f}")
    print(f"  Epsilon IQM:   {iqm_summary['Epsilon Greedy']:.2f}")
    print(f"  Median 50 IQM: {iqm_summary['Median 50']:.2f}")
    print(
        "  Paired-bootstrap IQM difference 95% CI: "
        f"[{iqm_summary['difference_ci_low']:.2f}, {iqm_summary['difference_ci_high']:.2f}]"
    )
    for method in METHODS:
        print(
            f"  {method} collision-free rate: "
            f"{survival_summary[method + '_collision_free_rate']:.2f}%; "
            f"RMST({SURVIVAL_HORIZON})={survival_summary[method + '_rmst']:.2f} steps"
        )
    print(f"  Shared 90% reward target: {target:.2f}")
    for method in METHODS:
        record = convergence[method]
        status = f"episode {record['episode']}" if record["reached"] else "not reached"
        print(
            f"  {method}: {status}; {record['seconds']:.2f} s; "
            f"{record['interactions']:,} interactions"
        )


if __name__ == "__main__":
    main()
