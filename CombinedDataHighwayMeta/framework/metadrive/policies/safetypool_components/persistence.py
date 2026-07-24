"""Canonical CSV/JSON/model outputs, manifests, and baseline indexes."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .constants import (
    BASELINE_SHARED_CONFIG_KEYS,
    CANONICAL_EPISODE_COLUMNS,
    CRITICAL_CONFIG_KEYS,
    OBSERVATION_SOURCE,
    POLICY_NAME,
    TEST_POLICY,
)
from .environment import metadrive_version
from .memory import SimilarStateActionPools
from .metrics import (
    collision_summary,
    make_summary,
    policy_safety_stop_summary,
)
from .plotting import make_figures
from .utils import (
    canonical_json_sha256,
    json_safe,
    sha256_file,
)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def critical_config(args) -> Dict:
    values = vars(args)
    return {key: values[key] for key in CRITICAL_CONFIG_KEYS}

def baseline_shared_config(args) -> Dict:
    """Return only settings shared with the canonical trainer."""
    values = vars(args)
    return {key: values[key] for key in BASELINE_SHARED_CONFIG_KEYS}

def canonical_episode_rows(
    rows: Sequence[Dict],
    phases: Sequence[str],
    phase_override: Optional[str] = None,
) -> List[Dict]:
    """Project detailed policy rows onto the exact baseline CSV contract."""
    selected: List[Dict] = []
    allowed = set(phases)
    for row in rows:
        if row["phase"] not in allowed:
            continue
        missing = [key for key in CANONICAL_EPISODE_COLUMNS if key not in row]
        if missing:
            raise KeyError(
                "Episode row is missing canonical fields: "
                + ", ".join(missing)
            )
        projected = {key: row[key] for key in CANONICAL_EPISODE_COLUMNS}
        if phase_override is not None:
            projected["phase"] = phase_override
        selected.append(projected)
    return selected

def save_framework_compatibility_outputs(
    rows: List[Dict],
    runtimes: List[Dict],
    args,
    output_dir: Path,
) -> None:
    """Write the files expected by baseline comparison/timing workflows."""
    canonical_runtimes = [
        row for row in runtimes if row["phase"] in {"train", "test"}
    ]
    runtime_path = output_dir / "runtime_statistics.csv"
    write_csv(runtime_path, canonical_runtimes)

    metric_rows = []
    for phase in ("train", "test"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        metric_rows.append(
            {
                "method": POLICY_NAME,
                "method_label": POLICY_NAME,
                "phase": phase,
                **collision_summary(phase_rows, args.rmst_tau),
            }
        )
    metrics_path = output_dir / "collision_metrics.csv"
    write_csv(metrics_path, metric_rows)

    intervention_rows = []
    for phase in ("train", "test"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        intervention_rows.append(
            {
                "method": POLICY_NAME,
                "method_label": POLICY_NAME,
                "phase": phase,
                **policy_safety_stop_summary(phase_rows, args.rmst_tau),
            }
        )
    write_csv(
        output_dir / "policy_safety_stop_metrics.csv",
        intervention_rows,
    )

    # Keep the existing model location and also expose the baseline-style name.
    nested_model = output_dir / "models" / "Karthikeya27adv8_model.pt"
    root_model = output_dir / "model.pt"
    if nested_model.is_file():
        root_model.write_bytes(nested_model.read_bytes())

def write_completion_manifest(
    rows: List[Dict],
    runtimes: List[Dict],
    args,
    output_dir: Path,
    action_pools: SimilarStateActionPools,
) -> Dict:
    results_path = output_dir / "all_episode_results.csv"
    metrics_path = output_dir / "collision_metrics.csv"
    runtime_path = output_dir / "runtime_statistics.csv"
    root_model = output_dir / "model.pt"
    config_path = output_dir / "config.json"
    pool_stats_path = output_dir / "state_pool_statistics.csv"
    candidate_stats_path = output_dir / "state_candidate_statistics.csv"
    retired_stats_path = output_dir / "state_retired_pool_statistics.csv"
    hazard_stats_path = output_dir / "state_hazard_memory_statistics.csv"
    calibration_path = output_dir / "state_matching_calibration.csv"
    matching_rejection_path = (
        output_dir / "state_matching_rejection_statistics.csv"
    )
    capacity_growth_path = output_dir / "state_pool_capacity_growth.csv"
    safety_memory_path = (
        output_dir / "models" / "Karthikeya27adv8_safety_memory.pkl"
    )
    detailed_results_path = output_dir / "all_episode_results_detailed.csv"
    policy_safety_stop_metrics_path = (
        output_dir / "policy_safety_stop_metrics.csv"
    )
    canonical_runtimes = [
        row for row in runtimes if row["phase"] in {"train", "test"}
    ]
    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": metadrive_version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "seed": args.seed,
        "method": POLICY_NAME,
        "method_label": POLICY_NAME,
        "observation_source": OBSERVATION_SOURCE,
        "uses_engine_object_safety_scan": True,
        "test_uses_engine_object_safety_scan": False,
        "test_policy": TEST_POLICY,
        "first_pass_action_policy": (
            "all_nonblocked_actions_once_then_one_unknown_only_pass"
        ),
        "scheduled_unknown_passes": 1,
        "unknown_pass_excludes_blocked_actions": True,
        "unknown_pass_retirement_waits_for_final_safety_horizon": True,
        "missing_safety_behavior": (
            "skip_pool_lookup_and_safe_evidence_use_pure_dqn"
        ),
        "missing_safety_values_are_never_imputed_as_safe": True,
        "post_first_pass_selection": (
            "80_percent_eligible_dqn_max_20_percent_unknown_random_"
            "with_greedy_fallback_when_no_unknown"
        ),
        "post_first_pass_blocked_actions_eligible": False,
        "all_blocked_fallback_overrides_exclusion": True,
        "all_actions_blocked_behavior": "execute_least_risk_blocked_action",
        "all_actions_blocked_metric_treatment": (
            "executed_action_reward_and_safety_outcome_recorded_normally"
        ),
        "canonical_collision_metrics_include_policy_safety_stop": False,
        "policy_safety_stop_audit_counts_stop_as_failure": True,
        "hazard_blocked_evidence_inheritance_scope": "mutable_training_only",
        "capacity_fallback_scope": (
            "candidate_hard_limit_permanent_full_or_eviction"
        ),
        "normal_general_cosine_relaxation": (
            args.general_cosine_relaxation
        ),
        "normal_general_rms_relaxation": args.general_rms_relaxation,
        "normal_general_relaxation_requires_both_gates": True,
        "normal_general_relaxation_safety_gates_remain_strict": True,
        "capacity_fallback_requires_both_relaxed_general_gates": True,
        "capacity_fallback_general_cosine_relaxation": (
            args.general_cosine_relaxation
        ),
        "capacity_fallback_general_rms_relaxation": (
            args.capacity_fallback_general_variation
        ),
        "capacity_fallback_safety_cosine_rms_gates_remain_strict": True,
        "capacity_fallback_all_directional_safety_nonworsening": True,
        "capacity_fallback_any_directional_safety_improves_10_percent": True,
        "evicted_candidate_history_capacity": (
            action_pools.evicted_candidate_history_capacity
        ),
        "evicted_candidate_history_rows_retained": len(
            action_pools.evicted_candidate_rows
        ),
        "evicted_candidate_history_rows_dropped": (
            action_pools.evicted_candidate_history_dropped
        ),
        "near_promotion_candidates_are_never_evicted": True,
        "frozen_final_training_uses_blocked_filter": False,
        "final_training_policy": "pure_dqn_argmax",
        "final_training_uses_pool_lookup": False,
        "final_training_uses_engine_object_safety_scan": False,
        "final_training_uses_safety_masks": False,
        "test_uses_safety_masks": False,
        "model_selection": "final_training_episode",
        "critical_config": critical_config(args),
        "critical_config_sha256": canonical_json_sha256(
            critical_config(args)
        ),
        "baseline_shared_config": baseline_shared_config(args),
        "baseline_shared_config_sha256": canonical_json_sha256(
            baseline_shared_config(args)
        ),
        "model_sha256": sha256_file(root_model),
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_statistics_sha256": sha256_file(runtime_path),
        "detailed_results_sha256": sha256_file(detailed_results_path),
        "policy_safety_stop_metrics_sha256": sha256_file(
            policy_safety_stop_metrics_path
        ),
        "config_sha256": sha256_file(config_path),
        "state_pool_statistics_sha256": sha256_file(pool_stats_path),
        "state_candidate_statistics_sha256": sha256_file(candidate_stats_path),
        "state_retired_pool_statistics_sha256": sha256_file(
            retired_stats_path
        ),
        "state_hazard_memory_statistics_sha256": sha256_file(
            hazard_stats_path
        ),
        "state_matching_calibration_sha256": sha256_file(calibration_path),
        "state_matching_rejection_statistics_sha256": sha256_file(
            matching_rejection_path
        ),
        "state_pool_capacity_growth_sha256": sha256_file(
            capacity_growth_path
        ),
        "safety_memory_sha256": sha256_file(safety_memory_path),
        "phase_runtime": canonical_runtimes,
        "created_at_unix": time.time(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest

def update_baseline_index(
    args,
    output_dir: Path,
    manifest: Dict,
) -> None:
    """Add or replace this policy in the seed-level comparison index."""
    seed_root = output_dir.parent
    index_path = seed_root / "baseline_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot safely update baseline index: {index_path}"
            ) from exc
        if int(index.get("seed", args.seed)) != int(args.seed):
            raise ValueError(
                f"Baseline index seed does not match --seed: {index_path}"
            )
        if index.get("environment", "MetaDrive") != "MetaDrive":
            raise ValueError(
                f"Baseline index environment is not MetaDrive: {index_path}"
            )
        indexed_config = index.get("critical_config", {})
        if not isinstance(indexed_config, dict):
            raise ValueError(
                f"Baseline index critical_config is invalid: {index_path}"
            )
        missing_shared = [
            key
            for key in BASELINE_SHARED_CONFIG_KEYS
            if key not in indexed_config
        ]
        if missing_shared:
            raise ValueError(
                "Baseline index is missing required shared settings: "
                + ", ".join(missing_shared)
            )
        indexed_hash = index.get("critical_config_sha256")
        computed_indexed_hash = canonical_json_sha256(indexed_config)
        if indexed_hash != computed_indexed_hash:
            raise ValueError(
                "Baseline index critical_config hash is missing or invalid: "
                f"{index_path}"
            )
        current_shared = baseline_shared_config(args)
        mismatches = {
            key: {
                "indexed": indexed_config[key],
                "current": current_shared[key],
            }
            for key in BASELINE_SHARED_CONFIG_KEYS
            if key in indexed_config
            and indexed_config[key] != current_shared[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}={values['indexed']!r}!={values['current']!r}"
                for key, values in sorted(mismatches.items())
            )
            raise ValueError(
                "Advanced run does not match the baseline index shared "
                f"configuration: {details}"
            )
    else:
        shared_config = baseline_shared_config(args)
        index = {
            "seed": int(args.seed),
            "environment": "MetaDrive",
            "methods": [],
            "critical_config": shared_config,
            "critical_config_sha256": canonical_json_sha256(
                shared_config
            ),
            "critical_config_scope": "baseline_shared_settings",
        }

    existing_methods = index.get("methods", [])
    if not isinstance(existing_methods, list):
        raise ValueError(
            f"Baseline index methods must be a list: {index_path}"
        )
    methods = [
        item
        for item in existing_methods
        if not (
            isinstance(item, dict)
            and item.get("method") == POLICY_NAME
        )
    ]
    methods.append(json_safe(manifest))
    index["methods"] = methods

    observation_sources = {
        str(item.get("method")): item.get("observation_source", "unknown")
        for item in methods
        if isinstance(item, dict)
    }
    scan_usage = {
        str(item.get("method")): bool(
            item.get("uses_engine_object_safety_scan", False)
        )
        for item in methods
        if isinstance(item, dict)
    }
    index["observation_source"] = "method_specific_see_manifests"
    index["uses_engine_object_safety_scan"] = any(scan_usage.values())
    index["method_observation_sources"] = observation_sources
    index["method_engine_object_safety_scan"] = scan_usage
    index["method_critical_config_sha256"] = {
        str(item.get("method")): item.get("critical_config_sha256")
        for item in methods
        if isinstance(item, dict)
    }
    index["method_baseline_shared_config_sha256"] = {
        str(item.get("method")): item.get(
            "baseline_shared_config_sha256"
        )
        for item in methods
        if isinstance(item, dict)
    }

    temporary_path = index_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(json_safe(index), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(index_path)

def save_outputs(rows: List[Dict], runtimes: List[Dict], args, output_dir: Path) -> None:
    summary = make_summary(rows, args)
    canonical_rows = canonical_episode_rows(rows, ("train", "test"))
    write_csv(output_dir / "all_episode_results.csv", canonical_rows)
    pd.DataFrame(rows).to_csv(
        output_dir / "all_episode_results_detailed.csv", index=False
    )
    pd.DataFrame(runtimes).to_csv(
        output_dir / "all_experiments_runtime_logs.csv", index=False
    )
    (output_dir / "config.json").write_text(
        json.dumps(json_safe(vars(args)), indent=2), encoding="utf-8"
    )
    save_framework_compatibility_outputs(rows, runtimes, args, output_dir)
    try:
        make_figures(rows, summary, output_dir, args)
    except Exception as exc:
        plt.close("all")
        print(
            "WARNING: benchmark plots were not created; all required result "
            f"files were already saved: {exc}",
            file=sys.stderr,
            flush=True,
        )
