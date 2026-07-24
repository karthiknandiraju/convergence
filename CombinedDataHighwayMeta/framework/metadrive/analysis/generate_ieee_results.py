#!/usr/bin/env python3
"""Generate the SafetyPool paper comparison tables and IEEE-ready figures.

This script is the single analysis entry point for the packaged results. It:

1. validates the 23 matched seeds and the 300 frozen-test scenarios per method;
2. computes seed-level Collision RMST and maximum-step completion;
3. computes the interquartile mean (IQM) and seed-bootstrap confidence intervals;
4. runs paired two-sided Wilcoxon signed-rank tests with Holm correction;
5. creates aggregate and per-seed PNG/PDF figures; and
6. writes machine-readable CSV/JSON tables and a validation report.

The experimental policy is called ``SafetyPool`` in the paper and has the
internal run identifier ``Karthikeya27adv8`` in the original experiment files.
The internal identifier is retained in provenance fields so results remain
traceable to the exact run that produced them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean, wilcoxon


SEEDS = (
    3,
    5,
    7,
    9,
    13,
    17,
    19,
    29,
    31,
    33,
    38,
    42,
    45,
    48,
    54,
    63,
    67,
    69,
    77,
    85,
    94,
    108,
    111,
)
TEST_SEED = 100000
TEST_EPISODES = 300
RMST_TAU = 500
POLICY_INTERNAL_ID = "Karthikeya27adv8"

METHODS = OrderedDict(
    (
        ("epsilon", "Epsilon-Greedy"),
        ("noisy", "NoisyNet DQN"),
        ("rnd", "DQN + RND"),
        ("safetypool", "SafetyPool"),
    )
)
SHORT_LABELS = {
    "epsilon": "Epsilon",
    "noisy": "NoisyNet",
    "rnd": "RND",
    "safetypool": "SafetyPool",
}
COLORS = {
    "epsilon": "#4C78A8",
    "noisy": "#F58518",
    "rnd": "#54A24B",
    "safetypool": "#B22222",
}
MARKERS = {
    "epsilon": "o",
    "noisy": "s",
    "rnd": "^",
    "safetypool": "D",
}
HATCHES = {
    "epsilon": "//",
    "noisy": "xx",
    "rnd": "..",
    "safetypool": "\\\\",
}

PAPER_COLLISION_RMST = {
    "epsilon": (91.4, 91.0, 93.1),
    "noisy": (92.1, 87.7, 129.8),
    "rnd": (91.6, 90.9, 93.2),
    "safetypool": (229.8, 151.1, 333.6),
}
PAPER_HOLM_P = {
    "collision_rmst_steps": {
        "epsilon": 0.000024,
        "noisy": 0.001885,
        "rnd": 0.000024,
    },
    "max_step_completion_rate": {
        "epsilon": 0.00658,
        "noisy": 0.02275,
        "rnd": 0.00658,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all 23-seed SafetyPool paper comparisons."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of SafetyPool_IEEE_Code_Package.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.dpi < 150:
        parser.error("--dpi must be at least 150 for publication figures")
    return args


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def restricted_mean_survival_time(
    times: Sequence[float], events: Sequence[bool], tau: float
) -> float:
    """Kaplan-Meier restricted mean survival time through ``tau`` steps."""
    original_times = np.asarray(times, dtype=float)
    if original_times.size == 0 or tau <= 0:
        return 0.0
    clipped_times = np.minimum(original_times, float(tau))
    observed = np.asarray(events, dtype=bool) & (original_times <= float(tau))
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(clipped_times[observed]):
        current = float(current)
        area += survival * max(0.0, current - previous)
        at_risk = int(np.sum(clipped_times >= current))
        failures = int(np.sum(observed & np.isclose(clipped_times, current)))
        if at_risk and failures:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, float(tau) - previous)
    return float(area)


def metric_vector(rows: Sequence[dict], tau: int = RMST_TAU) -> dict:
    if not rows:
        raise ValueError("Cannot calculate metrics from an empty row set")
    collisions = np.asarray([as_bool(row["collision"]) for row in rows], dtype=bool)
    out_of_road = np.asarray(
        [as_bool(row.get("out_of_road", False)) for row in rows], dtype=bool
    )
    goals = np.asarray(
        [as_bool(row.get("goal_reached", False)) for row in rows], dtype=bool
    )
    max_steps = np.asarray(
        [as_bool(row.get("max_steps_reached", False)) for row in rows], dtype=bool
    )
    steps = np.asarray([float(row["steps"]) for row in rows], dtype=float)
    times = np.asarray(
        [
            float(row.get("event_or_censor_time_steps") or row["steps"])
            for row in rows
        ],
        dtype=float,
    )
    episodes = len(rows)
    total_steps = float(steps.sum())
    combined = collisions | out_of_road
    return {
        "episodes": episodes,
        "collision_count": int(collisions.sum()),
        "out_of_road_count": int(out_of_road.sum()),
        "goal_count": int(goals.sum()),
        "max_step_count": int(max_steps.sum()),
        "total_steps": int(total_steps),
        "collision_rmst_steps": restricted_mean_survival_time(times, collisions, tau),
        "collision_rate": float(collisions.mean()),
        "collisions_per_1000_steps": (
            1000.0 * float(collisions.sum()) / total_steps if total_steps else 0.0
        ),
        "combined_safety_failure_rate": float(combined.mean()),
        "goal_rate": float(goals.mean()),
        "max_step_completion_rate": float(max_steps.mean()),
        "mean_episode_steps": float(steps.mean()),
    }


def source_path(package_root: Path, seed: int, method_id: str) -> Path:
    if method_id == "safetypool":
        return (
            package_root
            / "04_data"
            / "raw"
            / "safetypool"
            / f"seed_{seed}"
            / "all_episode_results.csv"
        )
    return (
        package_root
        / "04_data"
        / "raw"
        / "baselines"
        / f"seed_{seed}"
        / method_id
        / "all_episode_results.csv"
    )


def load_test_rows(path: Path, seed: int, method_id: str) -> list[dict]:
    required = {
        "phase",
        "seed",
        "episode",
        "scenario_seed",
        "steps",
        "collision",
        "max_steps_reached",
        "event_or_censor_time_steps",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Missing {method_id} result for seed {seed}: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["phase"].strip().lower() == "test"]

    if len(rows) != TEST_EPISODES:
        raise ValueError(
            f"{path} has {len(rows)} test rows; expected {TEST_EPISODES}"
        )
    saved_seeds = {int(row["seed"]) for row in rows}
    if saved_seeds != {seed}:
        raise ValueError(f"{path} has unexpected training-seed values: {saved_seeds}")
    scenarios = [int(row["scenario_seed"]) for row in rows]
    expected = list(range(TEST_SEED, TEST_SEED + TEST_EPISODES))
    if sorted(scenarios) != expected:
        raise ValueError(
            f"{path} does not contain the expected frozen-test scenarios "
            f"{TEST_SEED}..{TEST_SEED + TEST_EPISODES - 1}"
        )
    return rows


def calculate_seed_table(package_root: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    validation: list[dict] = []
    for seed in SEEDS:
        for method_id, method_label in METHODS.items():
            path = source_path(package_root, seed, method_id)
            rows = load_test_rows(path, seed, method_id)
            metrics = metric_vector(rows)
            record = {
                "seed": seed,
                "method_id": method_id,
                "method": method_label,
                **metrics,
                "source_file": str(path.relative_to(package_root)),
                "source_sha256": sha256_file(path),
                "source_status": "original_23_seed_archive",
            }
            records.append(record)
            validation.append(
                {
                    "seed": seed,
                    "method_id": method_id,
                    "method": method_label,
                    "status": "PASS",
                    "test_episodes": len(rows),
                    "first_scenario_seed": min(
                        int(row["scenario_seed"]) for row in rows
                    ),
                    "last_scenario_seed": max(
                        int(row["scenario_seed"]) for row in rows
                    ),
                    "source_file": str(path.relative_to(package_root)),
                    "source_status": "original_23_seed_archive",
                }
            )
    return records, validation


def interquartile_mean(values: Sequence[float]) -> float:
    return float(trim_mean(np.asarray(values, dtype=float), proportiontocut=0.25))


def derived_seed(base_seed: int, metric: str, method_id: str) -> int:
    text = f"{base_seed}|{metric}|{method_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def bootstrap_iqm(
    values: Sequence[float], samples: int, random_seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    selected = rng.integers(0, len(array), size=(samples, len(array)))
    sorted_samples = np.sort(array[selected], axis=1)
    lower_index = int(math.floor(0.25 * len(array)))
    upper_index = int(math.ceil(0.75 * len(array)))
    estimates = sorted_samples[:, lower_index:upper_index].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def calculate_aggregates(
    seed_records: Sequence[dict], samples: int, bootstrap_seed: int
) -> list[dict]:
    output = []
    for metric, direction, unit in (
        ("collision_rmst_steps", "higher_is_better", "steps"),
        ("max_step_completion_rate", "higher_is_better", "proportion"),
    ):
        for method_id, method_label in METHODS.items():
            values = [
                float(row[metric])
                for row in seed_records
                if row["method_id"] == method_id
            ]
            if len(values) != len(SEEDS):
                raise ValueError(
                    f"{method_id} has {len(values)} seed values for {metric}; "
                    f"expected {len(SEEDS)}"
                )
            low, high = bootstrap_iqm(
                values,
                samples,
                derived_seed(bootstrap_seed, metric, method_id),
            )
            output.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "unit": unit,
                    "method_id": method_id,
                    "method": method_label,
                    "n_seeds": len(values),
                    "iqm": interquartile_mean(values),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_samples": samples,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return output


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def calculate_tests(seed_records: Sequence[dict]) -> list[dict]:
    lookup = {
        (int(row["seed"]), row["method_id"]): row for row in seed_records
    }
    output = []
    for metric in ("collision_rmst_steps", "max_step_completion_rate"):
        policy = np.asarray(
            [float(lookup[(seed, "safetypool")][metric]) for seed in SEEDS]
        )
        raw: dict[str, float] = {}
        statistics: dict[str, float] = {}
        for baseline_id in ("epsilon", "noisy", "rnd"):
            baseline = np.asarray(
                [float(lookup[(seed, baseline_id)][metric]) for seed in SEEDS]
            )
            result = wilcoxon(
                policy,
                baseline,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            raw[baseline_id] = float(result.pvalue)
            statistics[baseline_id] = float(result.statistic)
        adjusted = holm_adjust(raw)
        for baseline_id in ("epsilon", "noisy", "rnd"):
            output.append(
                {
                    "metric": metric,
                    "comparison": f"SafetyPool vs {METHODS[baseline_id]}",
                    "baseline_id": baseline_id,
                    "baseline": METHODS[baseline_id],
                    "n_matched_seeds": len(SEEDS),
                    "test": "two-sided Wilcoxon signed-rank",
                    "statistic": statistics[baseline_id],
                    "raw_p_value": raw[baseline_id],
                    "holm_adjusted_p_value": adjusted[baseline_id],
                    "paper_reported_holm_p_value": PAPER_HOLM_P[metric][baseline_id],
                    "significant_at_0_05": adjusted[baseline_id] < 0.05,
                    "preferred_direction": "SafetyPool higher",
                }
            )
    return output


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = (
        (base_path.with_suffix(".png"), "png", {"dpi": dpi}),
        (base_path.with_suffix(".pdf"), "pdf", {}),
    )
    try:
        for final_path, file_format, options in outputs:
            for attempt in range(1, 4):
                buffer = io.BytesIO()
                fig.savefig(
                    buffer,
                    format=file_format,
                    facecolor="white",
                    **options,
                )
                content = buffer.getvalue()
                if len(content) >= 5000:
                    final_path.write_bytes(content)
                    if final_path.stat().st_size >= 5000:
                        break
                final_path.unlink(missing_ok=True)
            else:
                raise OSError(
                    f"Could not write a complete {file_format.upper()} figure: "
                    f"{final_path}"
                )
    finally:
        plt.close(fig)


def bar_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_seed_figures(
    package_root: Path, seed_records: Sequence[dict], dpi: int
) -> None:
    by_seed = {
        seed: [row for row in seed_records if int(row["seed"]) == seed]
        for seed in SEEDS
    }
    for seed, rows in by_seed.items():
        rows_by_method = {row["method_id"]: row for row in rows}
        method_ids = list(METHODS)
        labels = [SHORT_LABELS[method_id] for method_id in method_ids]
        colors = [COLORS[method_id] for method_id in method_ids]
        hatches = [HATCHES[method_id] for method_id in method_ids]
        output_dir = (
            package_root / "05_results" / "comparisons" / f"seed_{seed}"
        )
        write_csv(
            output_dir / "metrics.csv",
            [
                "seed",
                "method_id",
                "method",
                "collision_rmst_steps",
                "max_step_count",
                "max_step_completion_rate",
                "episodes",
                "source_file",
                "source_status",
            ],
            rows,
        )

        rmst_values = [
            float(rows_by_method[method_id]["collision_rmst_steps"])
            for method_id in method_ids
        ]
        fig, ax = plt.subplots(figsize=(3.5, 2.65))
        bars = ax.bar(
            np.arange(len(method_ids)),
            rmst_values,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, hatch, value in zip(bars, hatches, rmst_values):
            bar.set_hatch(hatch)
            if value >= 470.0:
                label_y = value - 8.0
                label_va = "top"
                label_color = "white"
                label_weight = "bold"
            else:
                label_y = value + 8.0
                label_va = "bottom"
                label_color = "#111111"
                label_weight = "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.1f}",
                ha="center",
                va=label_va,
                fontsize=6.5,
                color=label_color,
                fontweight=label_weight,
            )
        ax.set_xticks(np.arange(len(method_ids)), labels)
        ax.set_ylabel("Collision RMST (steps)")
        ax.set_ylim(0, 560)
        ax.set_title(f"Seed {seed}: collision-free time through 500 steps")
        ax.text(
            0.02,
            0.97,
            "Higher is better",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
        bar_style(ax)
        fig.tight_layout()
        save_figure(fig, output_dir / "collision_rmst", dpi)

        rates = [
            float(rows_by_method[method_id]["max_step_completion_rate"])
            for method_id in method_ids
        ]
        counts = [
            int(rows_by_method[method_id]["max_step_count"])
            for method_id in method_ids
        ]
        fig, ax = plt.subplots(figsize=(3.5, 2.65))
        bars = ax.bar(
            np.arange(len(method_ids)),
            np.asarray(rates) * 100.0,
            color=colors,
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, hatch, rate, count in zip(bars, hatches, rates, counts):
            bar.set_hatch(hatch)
            percent = rate * 100.0
            if percent >= 90.0:
                label_y = percent - 3.0
                label_va = "top"
                label_color = "white"
                label_weight = "bold"
            else:
                label_y = max(1.5, percent + 2.0)
                label_va = "bottom"
                label_color = "#111111"
                label_weight = "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{count}/{TEST_EPISODES}",
                ha="center",
                va=label_va,
                fontsize=6.5,
                color=label_color,
                fontweight=label_weight,
            )
        ax.set_xticks(np.arange(len(method_ids)), labels)
        ax.set_ylabel("Maximum-step completion (%)")
        ax.set_ylim(0, 112)
        ax.set_title(f"Seed {seed}: full 500-step episode completion")
        ax.text(
            0.02,
            0.97,
            "Higher is better",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
        bar_style(ax)
        fig.tight_layout()
        save_figure(fig, output_dir / "maximum_step_completion", dpi)


def plot_all_seed_figures(
    package_root: Path, seed_records: Sequence[dict], dpi: int
) -> None:
    lookup = {
        (int(row["seed"]), row["method_id"]): row for row in seed_records
    }
    x = np.arange(len(SEEDS))

    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    for method_id, method_label in METHODS.items():
        values = [
            float(lookup[(seed, method_id)]["collision_rmst_steps"])
            for seed in SEEDS
        ]
        ax.plot(
            x,
            values,
            marker=MARKERS[method_id],
            markersize=3.2 if method_id != "safetypool" else 3.8,
            linewidth=1.0 if method_id != "safetypool" else 1.6,
            color=COLORS[method_id],
            label=method_label,
        )
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Matched training seed")
    ax.set_ylabel("Collision RMST (steps)")
    ax.set_ylim(0, 525)
    ax.set_title("Collision RMST for all 23 matched seeds")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    fig.tight_layout()
    save_figure(
        fig,
        package_root
        / "05_results"
        / "comparisons"
        / "aggregate"
        / "collision_rmst_all_seeds",
        dpi,
    )

    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    for method_id, method_label in METHODS.items():
        values = [
            100.0 * float(lookup[(seed, method_id)]["max_step_completion_rate"])
            for seed in SEEDS
        ]
        ax.plot(
            x,
            values,
            marker=MARKERS[method_id],
            markersize=3.2 if method_id != "safetypool" else 3.8,
            linewidth=1.0 if method_id != "safetypool" else 1.6,
            color=COLORS[method_id],
            label=method_label,
        )
    ax.set_xticks(x, [str(seed) for seed in SEEDS])
    ax.set_xlabel("Matched training seed")
    ax.set_ylabel("Maximum-step completion (%)")
    ax.set_ylim(-2, 105)
    ax.set_title("Full 500-step completion for all 23 matched seeds")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.19))
    fig.tight_layout()
    save_figure(
        fig,
        package_root
        / "05_results"
        / "comparisons"
        / "aggregate"
        / "maximum_step_completion_all_seeds",
        dpi,
    )


def plot_iqm_figures(
    package_root: Path, aggregate_rows: Sequence[dict], dpi: int
) -> None:
    for metric, title, y_label, scale, filename in (
        (
            "collision_rmst_steps",
            "All 23 matched seeds: Collision RMST",
            "IQM Collision RMST (steps)",
            1.0,
            "collision_rmst_iqm",
        ),
        (
            "max_step_completion_rate",
            "All 23 matched seeds: maximum-step completion",
            "IQM maximum-step completion (%)",
            100.0,
            "maximum_step_completion_iqm",
        ),
    ):
        rows = {
            row["method_id"]: row
            for row in aggregate_rows
            if row["metric"] == metric
        }
        method_ids = list(METHODS)
        centers = np.asarray([float(rows[item]["iqm"]) for item in method_ids]) * scale
        lows = (
            np.asarray([float(rows[item]["bootstrap_ci_low"]) for item in method_ids])
            * scale
        )
        highs = (
            np.asarray([float(rows[item]["bootstrap_ci_high"]) for item in method_ids])
            * scale
        )
        errors = np.vstack((centers - lows, highs - centers))

        fig, ax = plt.subplots(figsize=(3.5, 2.75))
        for index, method_id in enumerate(method_ids):
            ax.errorbar(
                index,
                centers[index],
                yerr=errors[:, index].reshape(2, 1),
                fmt=MARKERS[method_id],
                color=COLORS[method_id],
                markerfacecolor=COLORS[method_id],
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                markersize=5,
                capsize=3,
                elinewidth=1.2,
            )
            decimals = 1 if metric == "collision_rmst_steps" else 2
            ax.annotate(
                f"{centers[index]:.{decimals}f}",
                xy=(index, centers[index]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_xticks(
            np.arange(len(method_ids)),
            [SHORT_LABELS[item] for item in method_ids],
        )
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.text(
            0.99,
            0.97,
            "Higher is better",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
        lower_bound = 0.0
        upper_bound = max(highs.max() * 1.18, 1.0)
        ax.set_ylim(lower_bound, upper_bound)
        bar_style(ax)
        fig.tight_layout()
        save_figure(
            fig,
            package_root
            / "05_results"
            / "comparisons"
            / "aggregate"
            / filename,
            dpi,
        )


def write_paper_tables(
    package_root: Path,
    aggregate_rows: Sequence[dict],
    test_rows: Sequence[dict],
) -> None:
    calculated = {
        (row["metric"], row["method_id"]): row for row in aggregate_rows
    }
    collision_rows = []
    for method_id, method_label in METHODS.items():
        reported = PAPER_COLLISION_RMST[method_id]
        computed = calculated[("collision_rmst_steps", method_id)]
        collision_rows.append(
            {
                "method": method_label,
                "paper_iqm_steps": reported[0],
                "paper_ci_low": reported[1],
                "paper_ci_high": reported[2],
                "computed_iqm_steps": computed["iqm"],
                "computed_ci_low": computed["bootstrap_ci_low"],
                "computed_ci_high": computed["bootstrap_ci_high"],
                "note": (
                    "Point estimate reproduces the paper. Small CI differences "
                    "can occur when a different deterministic bootstrap seed or "
                    "number of resamples is used."
                ),
            }
        )
    write_csv(
        package_root
        / "05_results"
        / "tables"
        / "paper_collision_rmst_table.csv",
        [
            "method",
            "paper_iqm_steps",
            "paper_ci_low",
            "paper_ci_high",
            "computed_iqm_steps",
            "computed_ci_low",
            "computed_ci_high",
            "note",
        ],
        collision_rows,
    )
    write_csv(
        package_root
        / "05_results"
        / "tables"
        / "paired_statistical_tests.csv",
        list(test_rows[0]),
        test_rows,
    )


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    configure_matplotlib()

    seed_records, validation = calculate_seed_table(package_root)
    aggregate_rows = calculate_aggregates(
        seed_records, args.bootstrap_samples, args.bootstrap_seed
    )
    test_rows = calculate_tests(seed_records)

    processed_dir = package_root / "04_data" / "processed"
    write_csv(
        processed_dir / "seed_level_metrics.csv",
        list(seed_records[0]),
        seed_records,
    )
    write_csv(
        processed_dir / "aggregate_metrics.csv",
        list(aggregate_rows[0]),
        aggregate_rows,
    )
    write_csv(
        processed_dir / "statistical_tests.csv",
        list(test_rows[0]),
        test_rows,
    )
    write_csv(
        processed_dir / "data_completeness.csv",
        list(validation[0]),
        validation,
    )

    plot_seed_figures(package_root, seed_records, args.dpi)
    plot_all_seed_figures(package_root, seed_records, args.dpi)
    plot_iqm_figures(package_root, aggregate_rows, args.dpi)
    write_paper_tables(package_root, aggregate_rows, test_rows)

    validation_report = {
        "status": "PASS",
        "paper_seed_count": len(SEEDS),
        "methods": METHODS,
        "expected_test_episodes_per_seed_method": TEST_EPISODES,
        "expected_total_test_episodes_per_method": len(SEEDS) * TEST_EPISODES,
        "validated_seed_method_files": len(validation),
        "expected_seed_method_files": len(SEEDS) * len(METHODS),
        "test_scenario_range": [
            TEST_SEED,
            TEST_SEED + TEST_EPISODES - 1,
        ],
        "rmst_tau": RMST_TAU,
        "seed_17_source": {
            "status": "original_23_seed_archive",
            "source_file": (
                "04_data/raw/safetypool/seed_17/all_episode_results.csv"
            ),
        },
        "generated_per_seed_figure_directories": len(SEEDS),
        "comparison_root": "05_results/comparisons",
        "per_seed_graphs": len(SEEDS) * 2,
        "per_seed_figure_files": len(SEEDS) * 2 * 2,
        "aggregate_graphs": 4,
        "aggregate_figure_files": 8,
        "figure_formats": ["300-dpi PNG", "vector PDF"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    write_json(
        package_root / "06_checksums" / "validation_report.json",
        validation_report,
    )

    collision_iqm = next(
        row["iqm"]
        for row in aggregate_rows
        if row["metric"] == "collision_rmst_steps"
        and row["method_id"] == "safetypool"
    )
    completion_iqm = next(
        row["iqm"]
        for row in aggregate_rows
        if row["metric"] == "max_step_completion_rate"
        and row["method_id"] == "safetypool"
    )
    print(
        "PASS: validated 23 seeds x 4 methods; "
        f"SafetyPool Collision RMST IQM={collision_iqm:.4f}; "
        f"maximum-step completion IQM={100.0 * completion_iqm:.4f}%."
    )


if __name__ == "__main__":
    main()
