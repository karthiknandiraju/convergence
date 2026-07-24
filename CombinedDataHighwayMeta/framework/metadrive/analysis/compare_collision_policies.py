#!/usr/bin/env python3
"""Compare canonical baselines with arbitrary policy result folders.

The script intentionally uses only collision metrics:
* collision RMST (higher is better),
* collisions per 1,000 environment steps (lower is better), and
* collision rate per episode (lower is better).

It also creates train/test box plots from blockwise collision exposure.  A
block box plot is used instead of binary per-episode collision flags.
It additionally creates train/test box plots from per-episode environment
reward. Rewards are plotted but are not added to the collision metric ranking.
Per-policy training/testing timing summaries and episode wall-time box plots
are generated from the recorded wall and CPU timing columns.

When ``--baseline-root`` is omitted, ``--seed N`` selects baselines from
``canonical_baselines_timed/seed_N`` beside this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-canonical-baselines")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_ORDER = ("epsilon", "noisy", "rnd")
BASELINE_LABELS = {
    "epsilon": "Epsilon Greedy",
    "noisy": "NoisyNet DQN",
    "rnd": "DQN + RND",
}
EMBEDDED_BASELINE_ALIASES = {
    "epsilon",
    "standard_epsilon",
    "epsilon_greedy",
    "noisy",
    "noisy_dqn",
    "noisynet",
    "rnd",
    "dqn_rnd",
    "dqn+rnd",
}
CRITICAL_CONFIG_KEYS = (
    "seed",
    "train_episodes",
    "test_episodes",
    "max_episode_steps",
    "learning_rate",
    "gamma",
    "batch_size",
    "replay_capacity",
    "target_update_steps",
    "hidden_size",
    "discrete_steering_dim",
    "discrete_throttle_dim",
    "map_blocks",
    "traffic_density",
    "accident_prob",
    "success_reward",
    "collision_penalty",
    "out_of_road_penalty",
    "test_seed",
    "rmst_tau",
)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def restricted_mean_survival_time(
    times: Sequence[float], events: Sequence[bool], tau: float
) -> float:
    original_t = np.asarray(times, dtype=float)
    if original_t.size == 0 or tau <= 0:
        return 0.0
    t = np.minimum(original_t, float(tau))
    e = np.asarray(events, dtype=bool) & (original_t <= float(tau))
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(t[e]):
        current = float(current)
        area += survival * max(0.0, current - previous)
        at_risk = int(np.sum(t >= current))
        failures = int(np.sum(e & np.isclose(t, current)))
        if at_risk and failures:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, float(tau) - previous)
    return float(area)


def metric_vector(frame: pd.DataFrame, tau: int) -> Dict[str, float]:
    collisions = as_bool(frame["collision"]).to_numpy(dtype=bool)
    steps = frame["steps"].to_numpy(dtype=float)
    times = frame.get("event_or_censor_time_steps", frame["steps"]).to_numpy(dtype=float)
    count = int(collisions.sum())
    total_steps = float(steps.sum())
    return {
        "episodes": int(len(frame)),
        "collision_count": count,
        "total_steps": int(total_steps),
        "collision_rmst": restricted_mean_survival_time(times, collisions, tau),
        "collisions_per_1000_steps": 1000.0 * count / total_steps if total_steps else 0.0,
        "collision_rate": count / len(frame) if len(frame) else 0.0,
    }


def bootstrap_intervals(
    frame: pd.DataFrame, tau: int, samples: int, rng: np.random.Generator
) -> Dict[str, Tuple[float, float]]:
    if len(frame) == 0 or samples <= 0:
        return {key: (math.nan, math.nan) for key in (
            "collision_rmst", "collisions_per_1000_steps", "collision_rate"
        )}
    values = {key: np.empty(samples, dtype=float) for key in (
        "collision_rmst", "collisions_per_1000_steps", "collision_rate"
    )}
    for index in range(samples):
        selected = rng.integers(0, len(frame), size=len(frame))
        result = metric_vector(frame.iloc[selected], tau)
        for key in values:
            values[key][index] = result[key]
    return {
        key: tuple(np.quantile(array, [0.025, 0.975]).astype(float))
        for key, array in values.items()
    }


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_csv(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "all_episode_results.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing all_episode_results.csv in {path}")
    return candidate


def unique_series_id(existing: set, desired: str, source_name: str) -> str:
    identifier = desired
    if identifier not in existing:
        existing.add(identifier)
        return identifier
    identifier = f"{source_name}:{desired}"
    counter = 2
    while identifier in existing:
        identifier = f"{source_name}:{desired}:{counter}"
        counter += 1
    existing.add(identifier)
    return identifier


def normalize_frame(
    csv_path: Path,
    source_kind: str,
    source_name: str,
    existing_ids: set,
    exclude_embedded_baselines: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {"phase", "episode", "steps", "collision", "env_reward"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} lacks required columns: {sorted(missing)}")
    # Timing was added after some canonical baselines had already been trained.
    # Preserve compatibility: unavailable timing stays NaN and is excluded only
    # from timing summaries/plots, not from collision or reward comparisons.
    for optional_timing_column in ("wall_time_seconds", "cpu_time_seconds"):
        if optional_timing_column not in frame.columns:
            frame[optional_timing_column] = np.nan
    if "experiment" not in frame.columns:
        frame["experiment"] = source_name
    if "method" not in frame.columns:
        frame["method"] = frame["experiment"]
    if "scenario_seed" not in frame.columns:
        frame["scenario_seed"] = frame["episode"]
    parts = []
    for experiment, group in frame.groupby("experiment", sort=False):
        if exclude_embedded_baselines and str(experiment).strip().lower() in EMBEDDED_BASELINE_ALIASES:
            continue
        label = str(group["method"].iloc[0])
        desired = str(experiment)
        if source_kind == "baseline" and source_name in BASELINE_LABELS:
            desired = source_name
            label = BASELINE_LABELS[source_name]
        series_id = unique_series_id(existing_ids, desired, source_name)
        if series_id != desired:
            label = f"{label} [{source_name}]"
        copy = group.copy()
        copy["series_id"] = series_id
        copy["display_label"] = label
        copy["source_kind"] = source_kind
        copy["source_name"] = source_name
        copy["source_csv"] = str(csv_path.resolve())
        parts.append(copy)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def natural_path_key(path: Path):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path).lower())
    )


def resolve_policy_paths(args, baseline_root: Path) -> List[Path]:
    """Collect explicit and automatically discovered policy result sources."""
    candidates = [path.resolve() for path in args.policy_dir]
    roots: List[Path] = []
    if args.policy_root is not None:
        roots.append(args.policy_root.resolve())
    if args.auto_discover_policies:
        inferred = (
            baseline_root.parent.parent
            / "policy_results"
            / baseline_root.name
        ).resolve()
        if inferred.exists():
            roots.append(inferred)
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Policy root not found: {root}")
        if root.is_file():
            candidates.append(root)
            continue
        direct_csv = root / "all_episode_results.csv"
        if direct_csv.is_file():
            candidates.append(root)
            continue
        candidates.extend(path.parent for path in root.rglob("all_episode_results.csv"))

    unique: Dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        unique[str(resolved)] = resolved
    return sorted(unique.values(), key=natural_path_key)


def collect_sources(args) -> Tuple[pd.DataFrame, List[Tuple[str, Dict]]]:
    frames = []
    configs: List[Tuple[str, Dict]] = []
    existing_ids: set = set()
    baseline_root = args.baseline_root.resolve()
    for method in BASELINE_ORDER:
        method_dir = baseline_root / method
        if not method_dir.exists():
            if args.require_all_baselines:
                raise FileNotFoundError(f"Missing canonical baseline: {method_dir}")
            warnings.warn(f"Skipping missing baseline: {method_dir}")
            continue
        frames.append(
            normalize_frame(
                discover_csv(method_dir), "baseline", method, existing_ids
            )
        )
        configs.append((method, read_json(method_dir / "config.json")))
    policy_paths = resolve_policy_paths(args, baseline_root)
    args.resolved_policy_paths = policy_paths
    for policy_path in policy_paths:
        source_name = policy_path.stem if policy_path.is_file() else policy_path.name
        policy_frame = normalize_frame(
            discover_csv(policy_path), "policy", source_name, existing_ids,
            exclude_embedded_baselines=not args.include_embedded_baselines,
        )
        if policy_frame.empty:
            warnings.warn(f"No non-baseline policy experiments found in {policy_path}")
            continue
        for series_id in policy_frame["series_id"].drop_duplicates():
            selected = policy_frame["series_id"].eq(series_id)
            original_method = str(policy_frame.loc[selected, "method"].iloc[0])
            prefix = "Median Policy" if "median" in original_method.lower() else "Policy"
            policy_frame.loc[selected, "display_label"] = f"{prefix} ({source_name})"
            policy_frame.loc[selected, "policy_folder"] = source_name
        frames.append(policy_frame)
        config_path = policy_path.parent / "config.json" if policy_path.is_file() else policy_path / "config.json"
        configs.append((source_name, read_json(config_path)))
    if not frames:
        raise ValueError("No baseline or policy data were found")
    return pd.concat(frames, ignore_index=True), configs


def validate_configs(configs: List[Tuple[str, Dict]], allow_mismatch: bool) -> List[str]:
    nonempty = [(name, cfg) for name, cfg in configs if cfg]
    if not nonempty:
        return ["No config.json files were available for validation"]
    reference_name, reference = nonempty[0]
    messages = []
    for name, config in nonempty[1:]:
        differences = []
        for key in CRITICAL_CONFIG_KEYS:
            if key in reference and key in config and reference[key] != config[key]:
                differences.append(f"{key}: {reference[key]!r} != {config[key]!r}")
        if differences:
            message = f"Config mismatch {reference_name} vs {name}: " + "; ".join(differences)
            if not allow_mismatch:
                raise ValueError(message)
            messages.append(message)
    return messages


def validate_test_scenarios(frame: pd.DataFrame, allow_mismatch: bool) -> List[str]:
    test = frame[frame["phase"].eq("test")]
    series = list(test["series_id"].drop_duplicates())
    if not series:
        raise ValueError("No test rows were found")
    reference = set(test.loc[test["series_id"].eq(series[0]), "scenario_seed"])
    messages = []
    for identifier in series[1:]:
        current = set(test.loc[test["series_id"].eq(identifier), "scenario_seed"])
        if current != reference:
            message = (
                f"Test scenario mismatch for {identifier}: "
                f"missing={len(reference-current)}, extra={len(current-reference)}"
            )
            if not allow_mismatch:
                raise ValueError(message)
            messages.append(message)
    return messages


def build_metrics(
    frame: pd.DataFrame, tau: int, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for (series_id, phase), group in frame.groupby(["series_id", "phase"], sort=False):
        values = metric_vector(group, tau)
        intervals = bootstrap_intervals(group, tau, bootstrap_samples, rng)
        record = {
            "series_id": series_id,
            "method": group["display_label"].iloc[0],
            "source_kind": group["source_kind"].iloc[0],
            "source_name": group["source_name"].iloc[0],
            "phase": phase,
            "rmst_tau": tau,
            **values,
        }
        for metric, (lower, upper) in intervals.items():
            record[f"{metric}_ci_low"] = lower
            record[f"{metric}_ci_high"] = upper
        records.append(record)
    return pd.DataFrame(records)


def block_collision_values(frame: pd.DataFrame, block_size: int) -> pd.DataFrame:
    records = []
    for (series_id, phase), group in frame.groupby(["series_id", "phase"], sort=False):
        ordered = group.sort_values("episode").reset_index(drop=True).copy()
        ordered["block"] = np.arange(len(ordered)) // block_size + 1
        ordered["collision_bool"] = as_bool(ordered["collision"])
        for block, block_frame in ordered.groupby("block"):
            collisions = int(block_frame["collision_bool"].sum())
            steps = int(block_frame["steps"].sum())
            records.append(
                {
                    "series_id": series_id,
                    "method": group["display_label"].iloc[0],
                    "phase": phase,
                    "block": int(block),
                    "episode_start": int(block_frame["episode"].min()),
                    "episode_end": int(block_frame["episode"].max()),
                    "episodes_in_block": int(len(block_frame)),
                    "collision_count": collisions,
                    "total_steps": steps,
                    "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
                    "collision_rate": collisions / len(block_frame),
                }
            )
    return pd.DataFrame(records)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def comparison_colors(count: int) -> List[str]:
    baseline_colors = ["#D62728", "#1F77B4", "#9467BD"]
    policy_colors = ["#2CA02C", "#17BECF", "#BCBD22", "#E377C2", "#8C564B", "#7F7F7F"]
    colors = baseline_colors[: min(count, len(baseline_colors))]
    while len(colors) < count:
        colors.append(policy_colors[(len(colors) - len(baseline_colors)) % len(policy_colors)])
    return colors


def compatible_boxplot(ax, groups, labels):
    options = {"showmeans": True, "patch_artist": True}
    try:
        return ax.boxplot(groups, tick_labels=labels, **options)
    except TypeError:  # Matplotlib before tick_labels was introduced
        return ax.boxplot(groups, labels=labels, **options)


def ordered_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["phase"].eq("test")].copy()
    baseline_rank = {name: index for index, name in enumerate(BASELINE_ORDER)}
    test["order"] = [
        baseline_rank.get(identifier, len(BASELINE_ORDER) + index)
        for index, identifier in enumerate(test["series_id"])
    ]
    return test.sort_values("order").reset_index(drop=True)


def save_bar_plot(
    test: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    values = test[metric].to_numpy(dtype=float)
    lower = test[f"{metric}_ci_low"].to_numpy(dtype=float)
    upper = test[f"{metric}_ci_high"].to_numpy(dtype=float)
    errors = np.vstack((np.maximum(0, values - lower), np.maximum(0, upper - values)))
    colors = comparison_colors(len(test))
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(test)), 4.0))
    bars = ax.bar(
        np.arange(len(test)), values, yerr=errors, capsize=3,
        color=colors[: len(test)], edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(test)), test["method"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        text = f"{100*value:.1f}%" if metric == "collision_rate" else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom", fontsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_box_plot(blocks: pd.DataFrame, phase: str, output_path: Path) -> None:
    phase_data = blocks[blocks["phase"].eq(phase)]
    identifiers = list(phase_data["series_id"].drop_duplicates())
    if not identifiers:
        warnings.warn(f"No {phase} rows were available for {output_path.name}")
        return
    groups = [
        phase_data.loc[phase_data["series_id"].eq(identifier), "collisions_per_1000_steps"].to_numpy()
        for identifier in identifiers
    ]
    labels = [
        str(phase_data.loc[phase_data["series_id"].eq(identifier), "method"].iloc[0])
        for identifier in identifiers
    ]
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(groups)), 4.0))
    box = compatible_boxplot(ax, groups, labels)
    colors = comparison_colors(len(groups))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Collisions per 1,000 steps per episode block")
    ax.set_title(f"{phase.title()} Collision Exposure Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_box_plot(blocks: pd.DataFrame, output_path: Path) -> None:
    identifiers = list(blocks["series_id"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.35 * len(identifiers)), 4.2), sharey=True)
    for ax, phase in zip(axes, ("train", "test")):
        phase_data = blocks[blocks["phase"].eq(phase)]
        phase_identifiers = [
            identifier for identifier in identifiers
            if phase_data["series_id"].eq(identifier).any()
        ]
        groups = [
            phase_data.loc[phase_data["series_id"].eq(identifier), "collisions_per_1000_steps"].to_numpy()
            for identifier in phase_identifiers
        ]
        labels = [
            str(blocks.loc[blocks["series_id"].eq(identifier), "method"].iloc[0])
            for identifier in phase_identifiers
        ]
        if not groups:
            ax.set_axis_off()
            ax.set_title(f"{phase.title()} (no data)")
            continue
        box = compatible_boxplot(ax, groups, labels)
        colors = comparison_colors(len(groups))
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(phase.title())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Collisions per 1,000 steps per episode block")
    fig.suptitle("Training and Testing Collision Exposure")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def reward_box_data(frame: pd.DataFrame, phase: str):
    """Return ordered per-method environment-reward arrays and labels."""
    phase_data = frame[frame["phase"].eq(phase)]
    groups, labels = [], []
    for identifier in phase_data["series_id"].drop_duplicates():
        group = phase_data[phase_data["series_id"].eq(identifier)]
        rewards = pd.to_numeric(group["env_reward"], errors="coerce").dropna()
        rewards = rewards[np.isfinite(rewards.to_numpy(dtype=float))]
        if rewards.empty:
            warnings.warn(f"No finite {phase} rewards for {identifier}")
            continue
        groups.append(rewards.to_numpy(dtype=float))
        labels.append(str(group["display_label"].iloc[0]))
    return groups, labels


def color_boxes(box, count: int) -> None:
    colors = comparison_colors(count)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)


def save_reward_box_plot(frame: pd.DataFrame, phase: str, output_path: Path) -> None:
    groups, labels = reward_box_data(frame, phase)
    if not groups:
        warnings.warn(f"No {phase} rewards were available for {output_path.name}")
        return
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(groups)), 4.0))
    box = compatible_boxplot(ax, groups, labels)
    color_boxes(box, len(groups))
    ax.set_ylabel("Environment reward per episode")
    title_phase = "Frozen-Test" if phase == "test" else "Training"
    ax.set_title(f"{title_phase} Environment Reward Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_reward_box_plot(frame: pd.DataFrame, output_path: Path) -> None:
    identifiers = list(frame["series_id"].drop_duplicates())
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 1.35 * len(identifiers)), 4.2), sharey=True
    )
    for ax, phase in zip(axes, ("train", "test")):
        groups, labels = reward_box_data(frame, phase)
        if not groups:
            ax.set_axis_off()
            ax.set_title(f"{phase.title()} (no data)")
            continue
        box = compatible_boxplot(ax, groups, labels)
        color_boxes(box, len(groups))
        ax.set_title("Frozen Test" if phase == "test" else "Train")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Environment reward per episode")
    fig.suptitle("Training and Frozen-Test Environment Reward Distributions")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def build_timing_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize recorded per-episode wall and CPU time by method and phase."""
    columns = (
        "series_id",
        "method",
        "source_kind",
        "source_name",
        "phase",
        "episodes",
        "summed_episode_wall_time_seconds",
        "summed_episode_wall_time_minutes",
        "average_wall_time_seconds_per_episode",
        "median_wall_time_seconds_per_episode",
        "std_wall_time_seconds_per_episode",
        "min_wall_time_seconds_per_episode",
        "p25_wall_time_seconds_per_episode",
        "p75_wall_time_seconds_per_episode",
        "p95_wall_time_seconds_per_episode",
        "max_wall_time_seconds_per_episode",
        "summed_episode_cpu_time_seconds",
        "average_cpu_time_seconds_per_episode",
        "total_environment_steps",
        "average_environment_steps_per_episode",
        "average_wall_time_ms_per_environment_step",
    )
    records = []
    for (series_id, phase), group in frame.groupby(["series_id", "phase"], sort=False):
        wall = finite_numeric(group["wall_time_seconds"])
        cpu = finite_numeric(group["cpu_time_seconds"])
        steps = finite_numeric(group["steps"])
        if wall.size == 0:
            continue
        total_wall = float(wall.sum())
        total_steps = float(steps.sum())
        records.append(
            {
                "series_id": series_id,
                "method": str(group["display_label"].iloc[0]),
                "source_kind": str(group["source_kind"].iloc[0]),
                "source_name": str(group["source_name"].iloc[0]),
                "phase": phase,
                "episodes": int(wall.size),
                "summed_episode_wall_time_seconds": total_wall,
                "summed_episode_wall_time_minutes": total_wall / 60.0,
                "average_wall_time_seconds_per_episode": float(wall.mean()),
                "median_wall_time_seconds_per_episode": float(np.median(wall)),
                "std_wall_time_seconds_per_episode": float(np.std(wall, ddof=0)),
                "min_wall_time_seconds_per_episode": float(wall.min()),
                "p25_wall_time_seconds_per_episode": float(np.quantile(wall, 0.25)),
                "p75_wall_time_seconds_per_episode": float(np.quantile(wall, 0.75)),
                "p95_wall_time_seconds_per_episode": float(np.quantile(wall, 0.95)),
                "max_wall_time_seconds_per_episode": float(wall.max()),
                "summed_episode_cpu_time_seconds": float(cpu.sum()) if cpu.size else math.nan,
                "average_cpu_time_seconds_per_episode": float(cpu.mean()) if cpu.size else math.nan,
                "total_environment_steps": int(total_steps),
                "average_environment_steps_per_episode": (
                    total_steps / wall.size if wall.size else math.nan
                ),
                "average_wall_time_ms_per_environment_step": (
                    1000.0 * total_wall / total_steps if total_steps else math.nan
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def episode_time_box_data(frame: pd.DataFrame, phase: str):
    phase_data = frame[frame["phase"].eq(phase)]
    groups, labels = [], []
    for identifier in phase_data["series_id"].drop_duplicates():
        group = phase_data[phase_data["series_id"].eq(identifier)]
        values = finite_numeric(group["wall_time_seconds"])
        if values.size == 0:
            continue
        groups.append(values)
        labels.append(str(group["display_label"].iloc[0]))
    return groups, labels


def save_episode_time_box_plot(
    frame: pd.DataFrame, phase: str, output_path: Path
) -> None:
    groups, labels = episode_time_box_data(frame, phase)
    if not groups:
        warnings.warn(f"No {phase} timing rows were available for {output_path.name}")
        return
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(groups)), 4.0))
    box = compatible_boxplot(ax, groups, labels)
    color_boxes(box, len(groups))
    ax.set_ylabel("Wall time per episode (seconds)")
    title_phase = "Frozen-Test" if phase == "test" else "Training"
    ax.set_title(f"{title_phase} Episode-Time Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_episode_time_box_plot(frame: pd.DataFrame, output_path: Path) -> None:
    identifiers = list(frame["series_id"].drop_duplicates())
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 1.35 * len(identifiers)), 4.2), sharey=False
    )
    for ax, phase in zip(axes, ("train", "test")):
        groups, labels = episode_time_box_data(frame, phase)
        if not groups:
            ax.set_axis_off()
            ax.set_title(f"{phase.title()} (no data)")
            continue
        box = compatible_boxplot(ax, groups, labels)
        color_boxes(box, len(groups))
        ax.set_title("Frozen Test" if phase == "test" else "Train")
        ax.set_ylabel("Wall time per episode (seconds)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Training and Frozen-Test Episode-Time Distributions")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_average_episode_time_plot(timing: pd.DataFrame, output_path: Path) -> None:
    identifiers = list(timing["series_id"].drop_duplicates())
    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 1.35 * len(identifiers)), 4.2), sharey=False
    )
    for ax, phase in zip(axes, ("train", "test")):
        phase_data = timing[timing["phase"].eq(phase)].copy()
        if phase_data.empty:
            ax.set_axis_off()
            ax.set_title(f"{phase.title()} (no data)")
            continue
        values = phase_data["average_wall_time_seconds_per_episode"].to_numpy(dtype=float)
        labels = phase_data["method"].astype(str).tolist()
        colors = comparison_colors(len(values))
        bars = ax.bar(
            np.arange(len(values)), values, color=colors[: len(values)],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(np.arange(len(values)), labels, rotation=25, ha="right")
        ax.set_title("Frozen Test" if phase == "test" else "Train")
        ax.set_ylabel("Average wall time per episode (seconds)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.suptitle("Average Training and Frozen-Test Episode Time")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare canonical baselines and policies using collision metrics"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed used to derive canonical_baselines_timed/seed_<seed> when "
            "--baseline-root is omitted."
        ),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit seed directory containing epsilon/, noisy/, and rnd/. "
            "Otherwise use canonical_baselines_timed/seed_<seed>."
        ),
    )
    parser.add_argument(
        "--policy-dir", type=Path, nargs="*", default=[],
        help="Policy result directories or all_episode_results.csv files",
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=None,
        help=(
            "Optional directory recursively containing policy outputs. By default, "
            "policy_results/<baseline seed folder> is inferred automatically."
        ),
    )
    parser.add_argument(
        "--auto-discover-policies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically include every saved policy for the baseline seed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rmst-tau", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-scenario-mismatch", action="store_true")
    parser.add_argument(
        "--include-embedded-baselines", action="store_true",
        help="Also include baseline experiments found inside policy folders",
    )
    parser.add_argument("--require-all-baselines", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.baseline_root is None:
        if args.seed is None:
            parser.error("provide --seed or an explicit --baseline-root")
        project = Path(__file__).resolve().parent
        args.baseline_root = (
            project / "canonical_baselines_timed" / f"seed_{args.seed}"
        ).resolve()
    if args.rmst_tau <= 0 or args.block_size <= 0 or args.bootstrap_samples < 0:
        parser.error("tau/block size must be positive and bootstrap samples non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Timed baseline source: {args.baseline_root.resolve()}")
    frame, configs = collect_sources(args)
    messages = validate_configs(configs, args.allow_config_mismatch)
    messages.extend(validate_test_scenarios(frame, args.allow_scenario_mismatch))
    metrics = build_metrics(
        frame, args.rmst_tau, args.bootstrap_samples, args.bootstrap_seed
    )
    blocks = block_collision_values(frame, args.block_size)
    timing = build_timing_statistics(frame)
    timed_series = set(timing["series_id"].astype(str))
    timing_missing = (
        frame.loc[~frame["series_id"].astype(str).isin(timed_series), ["series_id", "display_label"]]
        .drop_duplicates()
    )
    if not timing_missing.empty:
        missing_labels = timing_missing["display_label"].astype(str).tolist()
        timing_message = (
            "Timing statistics exclude sources whose existing episode CSVs do not "
            f"contain wall_time_seconds: {', '.join(missing_labels)}"
        )
        warnings.warn(timing_message)
        messages.append(timing_message)
    metrics.to_csv(args.output_dir / "collision_metrics.csv", index=False)
    blocks.to_csv(args.output_dir / "collision_block_values.csv", index=False)
    timing.to_csv(args.output_dir / "timing_statistics.csv", index=False)
    frame.to_csv(args.output_dir / "combined_episode_results.csv", index=False)
    policy_index_columns = (
        "policy_folder",
        "display_label",
        "source_name",
        "source_csv",
        "experiment",
        "method",
    )
    if "policy_folder" in frame.columns:
        policy_index = (
            frame.loc[frame["source_kind"].eq("policy"), list(policy_index_columns)]
            .drop_duplicates()
        )
    else:
        policy_index = pd.DataFrame(columns=policy_index_columns)
    policy_index.to_csv(args.output_dir / "policy_source_index.csv", index=False)
    style()
    test = ordered_test_metrics(metrics)
    save_bar_plot(
        test, "collision_rmst", "Frozen-Test Collision RMST (Higher Is Better)",
        "Collision-free RMST (steps)", args.output_dir / "test_collision_rmst.png",
    )
    save_bar_plot(
        test, "collisions_per_1000_steps",
        "Frozen-Test Collisions per 1,000 Steps (Lower Is Better)",
        "Collisions per 1,000 steps",
        args.output_dir / "test_collisions_per_1000_steps.png",
    )
    save_bar_plot(
        test, "collision_rate", "Frozen-Test Collision Rate (Lower Is Better)",
        "Collision rate", args.output_dir / "test_collision_rate.png",
    )
    save_box_plot(blocks, "train", args.output_dir / "train_collision_boxplot.png")
    save_box_plot(blocks, "test", args.output_dir / "test_collision_boxplot.png")
    save_combined_box_plot(blocks, args.output_dir / "train_test_collision_boxplots.png")
    save_reward_box_plot(frame, "train", args.output_dir / "train_reward_boxplot.png")
    save_reward_box_plot(frame, "test", args.output_dir / "test_reward_boxplot.png")
    save_combined_reward_box_plot(
        frame, args.output_dir / "train_test_reward_boxplots.png"
    )
    save_episode_time_box_plot(
        frame, "train", args.output_dir / "train_episode_time_boxplot.png"
    )
    save_episode_time_box_plot(
        frame, "test", args.output_dir / "test_episode_time_boxplot.png"
    )
    save_combined_episode_time_box_plot(
        frame, args.output_dir / "train_test_episode_time_boxplots.png"
    )
    save_average_episode_time_plot(
        timing, args.output_dir / "average_episode_wall_time.png"
    )
    report = {
        "baseline_root": str(args.baseline_root.resolve()),
        "policy_sources": [
            str(path) for path in getattr(args, "resolved_policy_paths", [])
        ],
        "automatic_policy_discovery": args.auto_discover_policies,
        "rmst_tau": args.rmst_tau,
        "block_size": args.block_size,
        "bootstrap_samples": args.bootstrap_samples,
        "timing_statistics": "timing_statistics.csv",
        "validation_messages": messages,
        "series": test[["series_id", "method", "source_kind", "source_name"]].to_dict("records"),
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Collision comparison saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
