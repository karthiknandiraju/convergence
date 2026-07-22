#!/usr/bin/env python3
"""Create a compact comparison of canonical baselines and safety policies.

The default report intentionally uses only fields shared by every result file:
environment reward, collision, out-of-road, combined safety failure, goal,
maximum-step completion, observed episode steps, and recorded train/test time.
It creates exactly eight requested graphs and concise metric/timing CSVs instead of the legacy
collection of RMST, bootstrap, timing, and duplicate train/test plots.

Policy test plots show only frozen Pure-DQN results. New policy runs store that
evaluation directly as the canonical ``test`` phase. For compatibility with
older policy folders, a sibling ``pure_dqn_episode_results.csv`` or embedded
``test_pure_dqn`` phase is preferred and remapped to ``test``; any corresponding
DQN-plus-safety test rows are excluded. Training rows still appear once.

When ``--baseline-root`` is omitted, ``--seed N`` selects baselines from
``canonical_baselines_timed/seed_N`` beside this script.

Example::

    python -u compare_safety_policies.py --seed 5 \
        --output-dir comparisons/seed_5/all_policies_compact
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
PURE_DQN_SOURCE_PHASE = "test_pure_dqn"
PURE_DQN_VARIANT = "pure_dqn"
STANDARD_VARIANT = "standard"
SAFETY_BOOLEAN_COLUMNS = (
    "out_of_road",
    "goal_reached",
    "max_steps_reached",
)
BOOTSTRAP_METRICS = (
    "collision_rmst",
    "safety_failure_rmst",
    "collision_rmst_common_support",
    "safety_failure_rmst_common_support",
    "collisions_per_1000_steps",
    "out_of_road_per_1000_steps",
    "safety_failures_per_1000_steps",
    "collision_rate",
    "out_of_road_rate",
    "safety_failure_rate",
    "goal_rate",
    "max_steps_rate",
    "other_termination_rate",
    "mean_episode_steps",
    "median_episode_steps",
    "at_risk_at_tau_rate",
)
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


def metric_vector(
    frame: pd.DataFrame, tau: int, common_support_tau: int | None = None
) -> Dict[str, float]:
    collisions = as_bool(frame["collision"]).to_numpy(dtype=bool)
    out_of_road = as_bool(frame["out_of_road"]).to_numpy(dtype=bool)
    goals = as_bool(frame["goal_reached"]).to_numpy(dtype=bool)
    max_steps = as_bool(frame["max_steps_reached"]).to_numpy(dtype=bool)
    safety_failures = collisions | out_of_road
    steps = frame["steps"].to_numpy(dtype=float)
    times = frame.get("event_or_censor_time_steps", frame["steps"]).to_numpy(dtype=float)
    collision_count = int(collisions.sum())
    out_of_road_count = int(out_of_road.sum())
    safety_failure_count = int(safety_failures.sum())
    goal_count = int(goals.sum())
    max_steps_count = int(max_steps.sum())
    terminal_collision = collisions
    terminal_out_of_road = (~collisions) & out_of_road
    terminal_goal = (~safety_failures) & goals
    terminal_max_steps = (~safety_failures) & (~goals) & max_steps
    terminal_classified = (
        terminal_collision | terminal_out_of_road | terminal_goal | terminal_max_steps
    )
    other_count = int((~terminal_classified).sum())
    total_steps = float(steps.sum())
    episodes = int(len(frame))
    at_risk_at_tau_count = int(np.sum(times >= float(tau)))
    support_tau = int(common_support_tau if common_support_tau is not None else tau)
    return {
        "episodes": episodes,
        "collision_count": collision_count,
        "out_of_road_count": out_of_road_count,
        "safety_failure_count": safety_failure_count,
        "goal_count": goal_count,
        "max_steps_count": max_steps_count,
        "other_termination_count": other_count,
        "terminal_collision_rate": float(np.mean(terminal_collision)) if episodes else 0.0,
        "terminal_out_of_road_rate": (
            float(np.mean(terminal_out_of_road)) if episodes else 0.0
        ),
        "terminal_goal_rate": float(np.mean(terminal_goal)) if episodes else 0.0,
        "terminal_max_steps_rate": (
            float(np.mean(terminal_max_steps)) if episodes else 0.0
        ),
        "total_steps": int(total_steps),
        "collision_rmst": restricted_mean_survival_time(times, collisions, tau),
        "safety_failure_rmst": restricted_mean_survival_time(
            times, safety_failures, tau
        ),
        "common_support_tau": support_tau,
        "collision_rmst_common_support": restricted_mean_survival_time(
            times, collisions, support_tau
        ),
        "safety_failure_rmst_common_support": restricted_mean_survival_time(
            times, safety_failures, support_tau
        ),
        "collisions_per_1000_steps": (
            1000.0 * collision_count / total_steps if total_steps else 0.0
        ),
        "out_of_road_per_1000_steps": (
            1000.0 * out_of_road_count / total_steps if total_steps else 0.0
        ),
        "safety_failures_per_1000_steps": (
            1000.0 * safety_failure_count / total_steps if total_steps else 0.0
        ),
        "collision_rate": collision_count / episodes if episodes else 0.0,
        "out_of_road_rate": out_of_road_count / episodes if episodes else 0.0,
        "safety_failure_rate": safety_failure_count / episodes if episodes else 0.0,
        "goal_rate": goal_count / episodes if episodes else 0.0,
        "max_steps_rate": max_steps_count / episodes if episodes else 0.0,
        "other_termination_rate": other_count / episodes if episodes else 0.0,
        "mean_episode_steps": float(np.mean(steps)) if episodes else 0.0,
        "median_episode_steps": float(np.median(steps)) if episodes else 0.0,
        "min_episode_steps": float(np.min(steps)) if episodes else 0.0,
        "max_episode_steps_observed": float(np.max(steps)) if episodes else 0.0,
        "at_risk_at_tau_count": at_risk_at_tau_count,
        "at_risk_at_tau_rate": at_risk_at_tau_count / episodes if episodes else 0.0,
        "collision_rmst_followup_warning": bool(
            episodes and at_risk_at_tau_count == 0
        ),
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    tau: int,
    common_support_tau: int,
    samples: int,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    if len(frame) == 0 or samples <= 0:
        return {key: (math.nan, math.nan) for key in BOOTSTRAP_METRICS}
    values = {key: np.empty(samples, dtype=float) for key in BOOTSTRAP_METRICS}
    for index in range(samples):
        selected = rng.integers(0, len(frame), size=len(frame))
        result = metric_vector(frame.iloc[selected], tau, common_support_tau)
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

    # New policy outputs write Pure DQN directly as ``test``. For legacy
    # outputs, prefer the explicit Pure-DQN audit rows and discard the old
    # DQN-plus-safety test rows so the report always compares one test policy.
    if source_kind == "policy":
        legacy_pure_path = csv_path.parent / "pure_dqn_episode_results.csv"
        phases = set(frame["phase"].astype(str))
        if (
            csv_path.name != legacy_pure_path.name
            and legacy_pure_path.is_file()
        ):
            pure_frame = pd.read_csv(legacy_pure_path)
            pure_missing = required.difference(pure_frame.columns)
            if pure_missing:
                raise ValueError(
                    f"{legacy_pure_path} lacks required columns: "
                    f"{sorted(pure_missing)}"
                )
            pure_frame["phase"] = "test"
            frame = pd.concat(
                [frame[frame["phase"].eq("train")], pure_frame],
                ignore_index=True,
                sort=False,
            )
        elif PURE_DQN_SOURCE_PHASE in phases:
            pure_frame = frame[
                frame["phase"].eq(PURE_DQN_SOURCE_PHASE)
            ].copy()
            pure_frame["phase"] = "test"
            frame = pd.concat(
                [frame[frame["phase"].eq("train")], pure_frame],
                ignore_index=True,
                sort=False,
            )
    # Older result files may not contain every safety-outcome flag.  Missing
    # flags are retained as False so legacy collision-only inputs still work.
    for safety_column in SAFETY_BOOLEAN_COLUMNS:
        if safety_column not in frame.columns:
            frame[safety_column] = False
    if "event_or_censor_time_steps" not in frame.columns:
        frame["event_or_censor_time_steps"] = frame["steps"]
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
    # Preserve the phase written by the source runner. Pure-DQN rows are
    # remapped to the common comparison phase below, but remain auditable.
    frame["source_phase"] = frame["phase"].astype(str)
    parts = []
    for experiment, group in frame.groupby("experiment", sort=False):
        if exclude_embedded_baselines and str(experiment).strip().lower() in EMBEDDED_BASELINE_ALIASES:
            continue
        label = str(group["method"].iloc[0])
        desired = str(experiment)
        if source_kind == "baseline" and source_name in BASELINE_LABELS:
            desired = source_name
            label = BASELINE_LABELS[source_name]

        variants = [(group.copy(), desired, label, STANDARD_VARIANT)]

        for variant_group, variant_id, variant_label, variant in variants:
            series_id = unique_series_id(
                existing_ids, variant_id, source_name
            )
            if series_id != variant_id:
                variant_label = f"{variant_label} [{source_name}]"
            copy = variant_group.copy()
            copy["series_id"] = series_id
            copy["display_label"] = variant_label
            copy["evaluation_variant"] = variant
            if source_kind == "policy":
                test_rows = copy["phase"].eq("test")
                copy.loc[test_rows, "display_label"] = (
                    f"{variant_label} Pure DQN"
                )
                copy.loc[test_rows, "evaluation_variant"] = PURE_DQN_VARIANT
                copy.loc[test_rows, "method"] = f"{label} Pure DQN"
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
            variant = str(
                policy_frame.loc[selected, "evaluation_variant"].iloc[0]
            )
            standard_selected = selected & policy_frame[
                "evaluation_variant"
            ].eq(STANDARD_VARIANT)
            if standard_selected.any():
                prefix = (
                    "Median Policy"
                    if "median" in original_method.lower()
                    else "Policy"
                )
                policy_frame.loc[standard_selected, "display_label"] = (
                    f"{prefix} ({source_name})"
                )
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
    common_support_by_phase = {}
    for phase, phase_frame in frame.groupby("phase", sort=False):
        maxima = []
        for _, group in phase_frame.groupby("series_id", sort=False):
            times = pd.to_numeric(
                group.get("event_or_censor_time_steps", group["steps"]), errors="coerce"
            ).dropna()
            if not times.empty:
                maxima.append(float(times.max()))
        common_support_by_phase[phase] = int(min(float(tau), min(maxima))) if maxima else int(tau)
    for (series_id, phase), group in frame.groupby(["series_id", "phase"], sort=False):
        common_support_tau = common_support_by_phase[phase]
        values = metric_vector(group, tau, common_support_tau)
        intervals = bootstrap_intervals(
            group, tau, common_support_tau, bootstrap_samples, rng
        )
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


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    discordant = int(improved + worsened)
    if discordant == 0:
        return 1.0
    tail = min(int(improved), int(worsened))
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * probability))


def paired_mean_ci(
    values: np.ndarray, samples: int, rng: np.random.Generator
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or samples <= 0:
        return math.nan, math.nan
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, values.size, size=values.size)
        estimates[index] = float(np.mean(values[selected]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def paired_test_comparisons(
    frame: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    """Pair every frozen-test method with Epsilon Greedy by scenario seed."""
    test = frame[frame["phase"].eq("test")].copy()
    if "epsilon" not in set(test["series_id"].astype(str)):
        return pd.DataFrame()
    reference = test[test["series_id"].eq("epsilon")].copy()
    rng = np.random.default_rng(seed + 1)
    records = []
    for series_id, candidate in test.groupby("series_id", sort=False):
        if series_id == "epsilon":
            continue
        paired = reference.merge(
            candidate,
            on="scenario_seed",
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        if paired.empty:
            continue
        outcomes = {}
        for side in ("reference", "candidate"):
            collision = as_bool(paired[f"collision_{side}"]).to_numpy(dtype=bool)
            out_of_road = as_bool(paired[f"out_of_road_{side}"]).to_numpy(dtype=bool)
            outcomes[f"collision_{side}"] = collision
            outcomes[f"out_of_road_{side}"] = out_of_road
            outcomes[f"safety_{side}"] = collision | out_of_road
            outcomes[f"goal_{side}"] = as_bool(
                paired[f"goal_reached_{side}"]
            ).to_numpy(dtype=bool)

        record = {
            "reference_series_id": "epsilon",
            "reference_method": str(reference["display_label"].iloc[0]),
            "candidate_series_id": series_id,
            "candidate_method": str(candidate["display_label"].iloc[0]),
            "paired_episodes": int(len(paired)),
        }
        definitions = (
            ("collision", "collision"),
            ("out_of_road", "out_of_road"),
            ("safety_failure", "safety"),
        )
        for output_name, key in definitions:
            ref = outcomes[f"{key}_reference"]
            cand = outcomes[f"{key}_candidate"]
            reduction = ref.astype(float) - cand.astype(float)
            improved = int(np.sum(ref & ~cand))
            worsened = int(np.sum(~ref & cand))
            ci_low, ci_high = paired_mean_ci(reduction, bootstrap_samples, rng)
            record.update(
                {
                    f"reference_{output_name}_rate": float(np.mean(ref)),
                    f"candidate_{output_name}_rate": float(np.mean(cand)),
                    f"{output_name}_rate_reduction": float(np.mean(reduction)),
                    f"{output_name}_rate_reduction_ci_low": ci_low,
                    f"{output_name}_rate_reduction_ci_high": ci_high,
                    f"{output_name}_improved_scenarios": improved,
                    f"{output_name}_worsened_scenarios": worsened,
                    f"{output_name}_mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
                }
            )

        goal_gain = (
            outcomes["goal_candidate"].astype(float)
            - outcomes["goal_reference"].astype(float)
        )
        step_gain = (
            paired["steps_candidate"].to_numpy(dtype=float)
            - paired["steps_reference"].to_numpy(dtype=float)
        )
        goal_low, goal_high = paired_mean_ci(goal_gain, bootstrap_samples, rng)
        step_low, step_high = paired_mean_ci(step_gain, bootstrap_samples, rng)
        record.update(
            {
                "goal_rate_gain": float(np.mean(goal_gain)),
                "goal_rate_gain_ci_low": goal_low,
                "goal_rate_gain_ci_high": goal_high,
                "mean_episode_steps_gain": float(np.mean(step_gain)),
                "mean_episode_steps_gain_ci_low": step_low,
                "mean_episode_steps_gain_ci_high": step_high,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def block_collision_values(frame: pd.DataFrame, block_size: int) -> pd.DataFrame:
    records = []
    for (series_id, phase), group in frame.groupby(["series_id", "phase"], sort=False):
        ordered = group.sort_values("episode").reset_index(drop=True).copy()
        ordered["block"] = np.arange(len(ordered)) // block_size + 1
        ordered["collision_bool"] = as_bool(ordered["collision"])
        ordered["out_of_road_bool"] = as_bool(ordered["out_of_road"])
        ordered["safety_failure_bool"] = (
            ordered["collision_bool"] | ordered["out_of_road_bool"]
        )
        for block, block_frame in ordered.groupby("block"):
            collisions = int(block_frame["collision_bool"].sum())
            out_of_road = int(block_frame["out_of_road_bool"].sum())
            safety_failures = int(block_frame["safety_failure_bool"].sum())
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
                    "out_of_road_count": out_of_road,
                    "safety_failure_count": safety_failures,
                    "total_steps": steps,
                    "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
                    "collision_rate": collisions / len(block_frame),
                    "out_of_road_per_1000_steps": (
                        1000.0 * out_of_road / steps if steps else 0.0
                    ),
                    "out_of_road_rate": out_of_road / len(block_frame),
                    "safety_failures_per_1000_steps": (
                        1000.0 * safety_failures / steps if steps else 0.0
                    ),
                    "safety_failure_rate": safety_failures / len(block_frame),
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
    test["original_order"] = np.arange(len(test))
    test["order_group"] = 2
    test["order_within_group"] = test["original_order"]
    for index, row in test.iterrows():
        identifier = str(row["series_id"])
        variant = str(row.get("evaluation_variant", STANDARD_VARIANT))
        if identifier in baseline_rank:
            test.at[index, "order_group"] = 0
            test.at[index, "order_within_group"] = baseline_rank[identifier]
        elif variant == PURE_DQN_VARIANT:
            test.at[index, "order_group"] = 1
            test.at[index, "order_within_group"] = 0
    return test.sort_values(
        ["order_group", "order_within_group", "original_order"]
    ).reset_index(drop=True)


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
        text = f"{100*value:.1f}%" if metric.endswith("_rate") else f"{value:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom", fontsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_rmst_comparison_plot(
    test: pd.DataFrame, tau: int, output_path: Path
) -> None:
    """Show why collision-only RMST must be read with follow-up information."""
    labels = test["method"].astype(str).tolist()
    positions = np.arange(len(test), dtype=float)
    width = 0.2
    series = (
        ("Collision RMST", "collision_rmst", "#1F77B4"),
        ("Collision/out-of-road RMST", "safety_failure_rmst", "#D62728"),
        (
            "Common-support safety RMST",
            "safety_failure_rmst_common_support",
            "#2CA02C",
        ),
        ("Median observed steps", "median_episode_steps", "#7F7F7F"),
    )
    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(test)), 4.4))
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, (label, metric, color) in zip(offsets, series):
        ax.bar(
            positions + offset,
            test[metric].to_numpy(dtype=float),
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
    ax.axhline(float(tau), color="black", linestyle="--", linewidth=0.9, label=f"tau={tau}")
    support_tau = int(test["common_support_tau"].iloc[0])
    if support_tau != tau:
        ax.axhline(
            float(support_tau), color="#2CA02C", linestyle=":", linewidth=1.0,
            label=f"common support={support_tau}",
        )
    ax.set_xticks(positions, labels, rotation=22, ha="right")
    ax.set_ylabel("Steps")
    ax.set_title("Frozen-Test RMST and Observed Follow-up Diagnostic")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_outcome_rate_plot(test: pd.DataFrame, output_path: Path) -> None:
    """Plot mutually prioritized terminal outcomes for frozen testing."""
    labels = test["method"].astype(str).tolist()
    positions = np.arange(len(test), dtype=float)
    width = 0.2
    series = (
        ("Collision", "terminal_collision_rate", "#D62728"),
        ("Out of road", "terminal_out_of_road_rate", "#FF7F0E"),
        ("Goal", "terminal_goal_rate", "#2CA02C"),
        ("Max steps", "terminal_max_steps_rate", "#1F77B4"),
    )
    fig, ax = plt.subplots(figsize=(max(7.0, 1.05 * len(test)), 4.4))
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    for offset, (label, metric, color) in zip(offsets, series):
        ax.bar(
            positions + offset,
            100.0 * test[metric].to_numpy(dtype=float),
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
    ax.set_xticks(positions, labels, rotation=22, ha="right")
    ax.set_ylabel("Episodes (%)")
    ax.set_title("Frozen-Test Terminal Outcome Rates")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_box_plot(
    blocks: pd.DataFrame,
    phase: str,
    output_path: Path,
    metric: str = "collisions_per_1000_steps",
    ylabel: str = "Collisions per 1,000 steps per episode block",
    title_subject: str = "Collision Exposure",
) -> None:
    phase_data = blocks[blocks["phase"].eq(phase)]
    identifiers = list(phase_data["series_id"].drop_duplicates())
    if not identifiers:
        warnings.warn(f"No {phase} rows were available for {output_path.name}")
        return
    groups = [
        phase_data.loc[phase_data["series_id"].eq(identifier), metric].to_numpy()
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
    ax.set_ylabel(ylabel)
    ax.set_title(f"{phase.title()} {title_subject} Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_box_plot(
    blocks: pd.DataFrame,
    output_path: Path,
    metric: str = "collisions_per_1000_steps",
    ylabel: str = "Collisions per 1,000 steps per episode block",
    title_subject: str = "Collision Exposure",
) -> None:
    identifiers = list(blocks["series_id"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.35 * len(identifiers)), 4.2), sharey=True)
    for ax, phase in zip(axes, ("train", "test")):
        phase_data = blocks[blocks["phase"].eq(phase)]
        phase_identifiers = [
            identifier for identifier in identifiers
            if phase_data["series_id"].eq(identifier).any()
        ]
        groups = [
            phase_data.loc[phase_data["series_id"].eq(identifier), metric].to_numpy()
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
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"Training and Testing {title_subject}")
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


def build_common_test_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize only reward and safety fields shared by every result source."""
    records = []
    test_frame = frame[frame["phase"].eq("test")]
    for series_id, group in test_frame.groupby("series_id", sort=False):
        collisions = as_bool(group["collision"]).to_numpy(dtype=bool)
        out_of_road = as_bool(group["out_of_road"]).to_numpy(dtype=bool)
        goals = as_bool(group["goal_reached"]).to_numpy(dtype=bool)
        max_steps = as_bool(group["max_steps_reached"]).to_numpy(dtype=bool)
        safety_failures = collisions | out_of_road
        steps = pd.to_numeric(group["steps"], errors="coerce").to_numpy(dtype=float)
        rewards = pd.to_numeric(group["env_reward"], errors="coerce").to_numpy(dtype=float)
        episodes = int(len(group))
        total_steps = float(np.nansum(steps))
        records.append(
            {
                "series_id": series_id,
                "method": str(group["display_label"].iloc[0]),
                "source_kind": str(group["source_kind"].iloc[0]),
                "source_name": str(group["source_name"].iloc[0]),
                "evaluation_variant": str(
                    group["evaluation_variant"].iloc[0]
                ),
                "phase": "test",
                "episodes": episodes,
                "mean_reward": float(np.nanmean(rewards)),
                "median_reward": float(np.nanmedian(rewards)),
                "mean_episode_steps": float(np.nanmean(steps)),
                "median_episode_steps": float(np.nanmedian(steps)),
                "collision_count": int(collisions.sum()),
                "collision_rate": float(collisions.mean()),
                "collisions_per_1000_steps": (
                    1000.0 * float(collisions.sum()) / total_steps
                    if total_steps else 0.0
                ),
                "out_of_road_count": int(out_of_road.sum()),
                "out_of_road_rate": float(out_of_road.mean()),
                "safety_failure_count": int(safety_failures.sum()),
                "safety_failure_rate": float(safety_failures.mean()),
                "goal_count": int(goals.sum()),
                "goal_rate": float(goals.mean()),
                "max_steps_count": int(max_steps.sum()),
                "max_steps_rate": float(max_steps.mean()),
            }
        )
    if not records:
        raise ValueError("No frozen-test rows were available for comparison.")
    return pd.DataFrame(records)


def save_simple_bar_plot(
    test: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save one fast bar plot without bootstrap confidence intervals."""
    values = test[metric].to_numpy(dtype=float)
    colors = comparison_colors(len(test))
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(test)), 4.0))
    bars = ax.bar(
        np.arange(len(test)), values, color=colors[: len(test)],
        edgecolor="black", linewidth=0.6,
    )
    ax.set_xticks(np.arange(len(test)), test["method"], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        label = f"{100.0 * value:.1f}%" if metric.endswith("_rate") else f"{value:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_test_step_box_plot(
    frame: pd.DataFrame, test: pd.DataFrame, output_path: Path
) -> None:
    """Compare frozen-test episode-step distributions across methods."""
    groups = []
    labels = []
    test_frame = frame[frame["phase"].eq("test")]
    for row in test.itertuples(index=False):
        values = pd.to_numeric(
            test_frame.loc[test_frame["series_id"].eq(row.series_id), "steps"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(str(row.method))
    if not groups:
        warnings.warn("No frozen-test episode-step values were available.")
        return
    fig, ax = plt.subplots(figsize=(max(6.2, 0.8 * len(groups)), 4.0))
    box = compatible_boxplot(ax, groups, labels)
    color_boxes(box, len(groups))
    ax.set_ylabel("Observed steps per test episode")
    ax.set_title("Frozen-Test Episode-Step Distribution")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def remove_legacy_outputs(output_dir: Path) -> None:
    """Remove only files generated by earlier, larger versions of this report."""
    plot_stems = (
        "test_collision_rmst",
        "test_collisions_per_1000_steps",
        "test_collision_rate",
        "test_safety_failure_rmst",
        "test_common_support_safety_rmst",
        "test_safety_failures_per_1000_steps",
        "test_out_of_road_rate",
        "test_goal_rate",
        "test_mean_episode_steps",
        "test_rmst_followup_rate",
        "test_rmst_comparison",
        "test_terminal_outcome_rates",
        "train_collision_boxplot",
        "test_collision_boxplot",
        "train_test_collision_boxplots",
        "train_safety_failure_boxplot",
        "test_safety_failure_boxplot",
        "train_test_safety_failure_boxplots",
        "train_reward_boxplot",
        "test_reward_boxplot",
        "train_test_reward_boxplots",
        "train_episode_time_boxplot",
        "test_episode_time_boxplot",
        "train_test_episode_time_boxplots",
        "average_episode_wall_time",
        "train_collision_exposure_boxplot",
        "test_episode_steps_boxplot",
    )
    for stem in plot_stems:
        for suffix in (".png", ".pdf"):
            path = output_dir / f"{stem}{suffix}"
            if path.is_file():
                path.unlink()
    for filename in (
        "collision_metrics.csv",
        "safety_metrics.csv",
        "paired_test_safety_comparisons.csv",
        "timing_statistics.csv",
        "test_safety_summary.csv",
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create eight compact reward, collision, goal, max-step, "
            "episode-step, and train/test timing comparison graphs"
        )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. When omitted, results are saved to "
            "comparisons/seed_<seed>/all_policies_compact beside this script. "
            "If a supplied path contains a different seed_<N> component, it is "
            "automatically corrected to match --seed."
        ),
    )
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--allow-scenario-mismatch", action="store_true")
    parser.add_argument(
        "--include-embedded-baselines", action="store_true",
        help="Also include baseline experiments found inside policy folders",
    )
    parser.add_argument("--require-all-baselines", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    if args.baseline_root is None:
        if args.seed is None:
            parser.error("provide --seed or an explicit --baseline-root")
        args.baseline_root = (
            project / "canonical_baselines_timed" / f"seed_{args.seed}"
        ).resolve()

    # Infer the seed from an explicit baseline root when --seed was omitted.
    if args.seed is None:
        match = re.fullmatch(r"seed_(\d+)", args.baseline_root.name)
        if match:
            args.seed = int(match.group(1))

    # Derive a canonical seed-specific output folder by default.  If the user
    # accidentally supplies comparisons/seed_13 while running --seed 3, replace
    # only that seed_<N> path component and preserve the rest of the path.
    args.output_dir_was_corrected = False
    args.original_output_dir = None
    if args.output_dir is None:
        if args.seed is None:
            parser.error(
                "cannot derive --output-dir because no seed was provided or "
                "inferred from --baseline-root"
            )
        args.output_dir = (
            project / "comparisons" / f"seed_{args.seed}" / "all_policies_compact"
        )
    elif args.seed is not None:
        supplied = args.output_dir.expanduser()
        parts = list(supplied.parts)
        expected = f"seed_{args.seed}"
        for index, part in enumerate(parts):
            if re.fullmatch(r"seed_\d+", part) and part != expected:
                args.original_output_dir = str(args.output_dir)
                parts[index] = expected
                supplied = Path(*parts)
                args.output_dir_was_corrected = True
                break
        args.output_dir = supplied

    args.output_dir = args.output_dir.expanduser()
    if not args.output_dir.is_absolute():
        args.output_dir = (project / args.output_dir).resolve()
    else:
        args.output_dir = args.output_dir.resolve()

    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Timed baseline source: {args.baseline_root.resolve()}")
    if getattr(args, "output_dir_was_corrected", False):
        print(
            "Corrected output directory to match --seed: "
            f"{args.original_output_dir} -> {args.output_dir}"
        )
    print(f"Output directory: {args.output_dir}")
    remove_legacy_outputs(args.output_dir)
    frame, configs = collect_sources(args)
    messages = validate_configs(configs, args.allow_config_mismatch)
    messages.extend(validate_test_scenarios(frame, args.allow_scenario_mismatch))
    common_metrics = build_common_test_metrics(frame)
    blocks = block_collision_values(frame, args.block_size)
    timing = build_timing_statistics(frame)
    common_metrics.to_csv(args.output_dir / "test_common_metrics.csv", index=False)
    blocks.to_csv(args.output_dir / "collision_block_values.csv", index=False)
    timing.to_csv(args.output_dir / "timing_statistics.csv", index=False)
    frame.to_csv(args.output_dir / "combined_episode_results.csv", index=False)
    policy_index_columns = (
        "policy_folder",
        "display_label",
        "evaluation_variant",
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
    test = ordered_test_metrics(common_metrics)
    save_reward_box_plot(frame, "test", args.output_dir / "test_reward_boxplot.png")
    save_box_plot(
        blocks,
        "train",
        args.output_dir / "train_collision_exposure_boxplot.png",
    )
    save_simple_bar_plot(
        test,
        "max_steps_rate",
        "Frozen-Test Maximum-Step Completion Rate",
        "Episodes reaching maximum steps",
        args.output_dir / "test_max_steps_rate.png",
    )
    save_simple_bar_plot(
        test,
        "goal_rate",
        "Frozen-Test Goal Rate",
        "Goal-reaching episode rate",
        args.output_dir / "test_goal_rate.png",
    )
    save_simple_bar_plot(
        test, "collisions_per_1000_steps",
        "Frozen-Test Collisions per 1,000 Steps",
        "Collisions per 1,000 steps",
        args.output_dir / "test_collisions_per_1000_steps.png",
    )
    save_simple_bar_plot(
        test, "collision_rate", "Frozen-Test Collision Rate",
        "Collision rate", args.output_dir / "test_collision_rate.png",
    )
    save_test_step_box_plot(
        frame, test, args.output_dir / "test_episode_steps_boxplot.png"
    )
    if timing.empty:
        warnings.warn(
            "No wall_time_seconds values were available; the train/test timing "
            "graph is omitted."
        )
    else:
        save_average_episode_time_plot(
            timing, args.output_dir / "average_episode_wall_time.png"
        )
    report = {
        "baseline_root": str(args.baseline_root.resolve()),
        "policy_sources": [
            str(path) for path in getattr(args, "resolved_policy_paths", [])
        ],
        "automatic_policy_discovery": args.auto_discover_policies,
        "block_size": args.block_size,
        "test_common_metrics": "test_common_metrics.csv",
        "timing_statistics": "timing_statistics.csv",
        "graphs": [
            "test_reward_boxplot.png",
            "train_collision_exposure_boxplot.png",
            "test_max_steps_rate.png",
            "test_goal_rate.png",
            "test_collisions_per_1000_steps.png",
            "test_collision_rate.png",
            "test_episode_steps_boxplot.png",
            "average_episode_wall_time.png",
        ],
        "metric_definitions": {
            "collision_rate": "Episodes with a collision divided by test episodes.",
            "collisions_per_1000_steps": (
                "Collision count divided by observed test steps, multiplied by 1,000."
            ),
            "goal_rate": "Episodes reaching the goal divided by test episodes.",
            "max_steps_rate": (
                "Episodes reaching the configured horizon divided by test episodes."
            ),
            "environment_reward": "Raw environment reward per frozen-test episode.",
        },
        "validation_messages": messages,
        "series": test[
            [
                "series_id",
                "method",
                "evaluation_variant",
                "source_kind",
                "source_name",
            ]
        ].to_dict("records"),
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Eight-graph compact comparison saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
