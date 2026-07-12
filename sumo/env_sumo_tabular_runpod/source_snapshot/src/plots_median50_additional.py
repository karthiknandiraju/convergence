#!/usr/bin/env python3
"""Create only the additional IEEE-style Median 50 strength graphs.

Run from:
/home/karthik/Desktop/Karthikeya/sumo_tabular_dqn_original/src

Command:
python3 plots_median50_additional_only.py \
  --results-dir ../results/sumo_tabular_median50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICIES = ["epsilon_greedy", "median_50"]
LABELS = {
    "epsilon_greedy": "Epsilon Greedy",
    "median_50": "Median 50",
}
LINESTYLES = {
    "epsilon_greedy": "--",
    "median_50": "-",
}


def ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(
        out / f"{name}.jpg",
        dpi=600,
        bbox_inches="tight",
        format="jpeg",
    )
    plt.close(fig)


def load_data(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "all_episode_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"phase", "policy", "episode", "reward"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df = df[df["policy"].isin(POLICIES)].copy()
    df["phase"] = df["phase"].astype(str).str.lower()
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
    df = df.dropna(subset=["phase", "policy", "episode", "reward"])

    found = set(df["policy"].unique())
    missing_policies = [p for p in POLICIES if p not in found]
    if missing_policies:
        raise ValueError(f"Missing policies in CSV: {missing_policies}")
    return df


def subset(df: pd.DataFrame, phase: str, policy: str) -> pd.DataFrame:
    return df[(df["phase"] == phase) & (df["policy"] == policy)].sort_values("episode")


def reward_trend(df: pd.DataFrame, phase: str, window: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in POLICIES:
        d = subset(df, phase, policy)
        rewards = d["reward"].to_numpy(float)
        episodes = d["episode"].to_numpy(float)
        smooth = moving_average(rewards, window)
        x = episodes[len(episodes) - len(smooth):]
        ax.plot(x, smooth, linestyle=LINESTYLES[policy], linewidth=1.7, label=LABELS[policy])
    ax.set_xlabel(f"{phase.capitalize()} episode")
    ax.set_ylabel("Moving-average reward")
    ax.set_title(f"{phase.capitalize()} Reward Trend Using a {window}-Episode Window")
    ax.legend(frameon=False)
    save(fig, out, f"median50_{phase}_reward_trend")


def test_boxplot(df: pd.DataFrame, out: Path) -> None:
    groups = [subset(df, "test", p)["reward"].to_numpy(float) for p in POLICIES]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.boxplot(groups, tick_labels=[LABELS[p] for p in POLICIES], showmeans=True, widths=0.55)
    ax.set_xlabel("Exploration strategy")
    ax.set_ylabel("Environment reward")
    ax.set_title("Testing Reward Distribution")
    save(fig, out, "median50_test_reward_distribution")


def average_test_reward(df: pd.DataFrame, out: Path) -> None:
    means = []
    stds = []
    for policy in POLICIES:
        values = subset(df, "test", policy)["reward"].to_numpy(float)
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(POLICIES))
    bars = ax.bar(x, means, yerr=stds, capsize=4, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in POLICIES])
    ax.set_xlabel("Exploration strategy")
    ax.set_ylabel("Average test reward")
    ax.set_title("Average Testing Reward with Standard Deviation")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, out, "median50_average_test_reward")


def episodewise_advantage(df: pd.DataFrame, window: int, out: Path) -> None:
    eps = subset(df, "test", "epsilon_greedy")[["episode", "reward"]].rename(columns={"reward": "epsilon"})
    med = subset(df, "test", "median_50")[["episode", "reward"]].rename(columns={"reward": "median"})
    paired = eps.merge(med, on="episode", how="inner")
    paired["advantage"] = paired["median"] - paired["epsilon"]

    values = paired["advantage"].to_numpy(float)
    episodes = paired["episode"].to_numpy(float)
    smooth = moving_average(values, window)
    x = episodes[len(episodes) - len(smooth):]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.plot(x, smooth, linewidth=1.7)
    ax.set_xlabel("Testing episode")
    ax.set_ylabel("Median 50 reward advantage")
    ax.set_title(f"Episode-Wise Median 50 Advantage Using a {window}-Episode Window")
    save(fig, out, "median50_episodewise_test_advantage")


def training_stability(df: pd.DataFrame, window: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in POLICIES:
        d = subset(df, "train", policy)
        variability = d["reward"].rolling(window=window, min_periods=2).std()
        valid = variability.notna()
        ax.plot(
            d.loc[valid, "episode"],
            variability.loc[valid],
            linestyle=LINESTYLES[policy],
            linewidth=1.7,
            label=LABELS[policy],
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rolling reward standard deviation")
    ax.set_title(f"Training Stability Using a {window}-Episode Window")
    ax.legend(frameon=False)
    save(fig, out, "median50_training_stability")


def reward_ecdf(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in POLICIES:
        rewards = np.sort(subset(df, "test", policy)["reward"].to_numpy(float))
        probability = np.arange(1, len(rewards) + 1) / len(rewards)
        ax.plot(rewards, probability, linestyle=LINESTYLES[policy], linewidth=1.7, label=LABELS[policy])
    ax.set_xlabel("Testing reward")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical Cumulative Distribution of Testing Rewards")
    ax.legend(frameon=False)
    save(fig, out, "median50_test_reward_ecdf")


def high_reward_rate(df: pd.DataFrame, out: Path, threshold: float | None) -> float:
    pooled = df[df["phase"] == "test"]["reward"].to_numpy(float)
    if threshold is None:
        threshold = float(np.percentile(pooled, 75))

    rates = []
    for policy in POLICIES:
        rewards = subset(df, "test", policy)["reward"].to_numpy(float)
        rates.append(float(np.mean(rewards >= threshold) * 100.0))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(POLICIES))
    bars = ax.bar(x, rates, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[p] for p in POLICIES])
    ax.set_xlabel("Exploration strategy")
    ax.set_ylabel("High-reward episodes, percent")
    ax.set_title(f"Testing Episodes at or Above Reward {threshold:.2f}")
    ax.set_ylim(0, max(rates + [1.0]) * 1.18)
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    save(fig, out, "median50_high_reward_episode_rate")
    return threshold


def save_summary(df: pd.DataFrame, threshold: float, out: Path) -> None:
    rows = []
    for phase in ["train", "test"]:
        for policy in POLICIES:
            values = subset(df, phase, policy)["reward"].to_numpy(float)
            rows.append({
                "phase": phase,
                "policy": policy,
                "method": LABELS[policy],
                "episodes": len(values),
                "average_reward": float(np.mean(values)),
                "median_reward": float(np.median(values)),
                "std_reward": float(np.std(values)),
                "min_reward": float(np.min(values)),
                "max_reward": float(np.max(values)),
                "q1_reward": float(np.percentile(values, 25)),
                "q3_reward": float(np.percentile(values, 75)),
                "high_reward_threshold": threshold if phase == "test" else np.nan,
                "high_reward_rate_percent": float(np.mean(values >= threshold) * 100.0) if phase == "test" else np.nan,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "median50_strengths_summary.csv", index=False)

    test = summary[summary["phase"] == "test"].set_index("policy")
    epsilon_mean = float(test.loc["epsilon_greedy", "average_reward"])
    median_mean = float(test.loc["median_50", "average_reward"])
    percentage = ((median_mean - epsilon_mean) / abs(epsilon_mean) * 100.0) if epsilon_mean != 0 else None

    metrics = {
        "epsilon_greedy_average_test_reward": epsilon_mean,
        "median_50_average_test_reward": median_mean,
        "median_50_absolute_improvement": median_mean - epsilon_mean,
        "median_50_percentage_improvement": percentage,
        "high_reward_threshold": threshold,
        "epsilon_greedy_high_reward_rate_percent": float(test.loc["epsilon_greedy", "high_reward_rate_percent"]),
        "median_50_high_reward_rate_percent": float(test.loc["median_50", "high_reward_rate_percent"]),
    }
    (out / "median50_strengths_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results/sumo_tabular_median50")
    parser.add_argument("--train-window", type=int, default=20)
    parser.add_argument("--test-window", type=int, default=20)
    parser.add_argument("--stability-window", type=int, default=20)
    parser.add_argument("--high-reward-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    additional_dir = results_dir / "additionalplots"
    additional_dir.mkdir(parents=True, exist_ok=True)

    ieee_style()
    df = load_data(results_dir)

    # The four graphs already produced by the experiment script are skipped:
    # train_reward_vs_episode, test_reward_vs_episode,
    # test_reward_boxplot, and average_test_reward.
    episodewise_advantage(df, args.test_window, additional_dir)
    training_stability(df, args.stability_window, additional_dir)
    reward_ecdf(df, additional_dir)
    threshold = high_reward_rate(
        df,
        additional_dir,
        args.high_reward_threshold,
    )
    save_summary(df, threshold, additional_dir)

    print("Median 50 additional graph generation completed.")
    print(f"Input: {results_dir / 'all_episode_results.csv'}")
    print(f"Outputs: {additional_dir}")


if __name__ == "__main__":
    main()
