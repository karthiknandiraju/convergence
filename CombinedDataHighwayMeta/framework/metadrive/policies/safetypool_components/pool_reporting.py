"""Safety-memory CSV diagnostics and pool-specific figures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .constants import SAFETY_VECTOR_NAMES
from .memory import SimilarStateActionPools
from .plotting import apply_ieee_style, save_figure


def save_pool_statistics(
    action_pools: SimilarStateActionPools,
    args,
    output_dir: Path,
) -> None:
    action_pools.validate_invariants()
    pool_rows = action_pools.pool_statistics()
    candidate_rows = action_pools.candidate_statistics()
    retired_rows = action_pools.retired_pool_statistics()
    hazard_rows = action_pools.hazard_statistics()
    matching_rows = action_pools.matching_rejection_statistics()
    global_row = action_pools.global_statistics()

    def save_rows(
        path: Path, rows: Sequence[Dict], columns: List[str]
    ) -> None:
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    evidence_columns = [
        "total_action_attempts", "total_safe_outcomes",
        "total_collision_outcomes", "total_out_of_road_outcomes",
        "total_warnings",
        "action_attempt_counts", "safe_outcome_counts",
        "collision_outcome_counts", "out_of_road_outcome_counts",
        "warning_counts",
    ]
    pool_columns = [
        "pool_id", "status", "promotion_evidence_visits",
        "absorbed_candidate_visits", "absorbed_candidate_action_evidence",
        "active_pool_mask_visits", "matched_state_count",
        "exploration_pass",
        "actions_tried", "remaining_actions", "coverage_percent",
        "unknown_actions", "safe_actions", "blocked_actions",
        "unknown_mask", "safe_mask", "blocked_mask",
        *evidence_columns,
        "first_episode_created", "last_episode_visited",
        "mean_general_cosine_similarity", "mean_general_rms_distance",
        "mean_safety_cosine_similarity", "mean_safety_rms_distance",
        "centroid_updates", "centroid_frozen_by_stability",
        "centroid_frozen_by_cap",
    ]
    candidate_columns = [
        "candidate_id", "status", "visit_count", "first_episode", "last_episode",
        "visits_remaining_for_promotion", "unique_actions_executed",
        "unknown_mask", "safe_mask", "blocked_mask",
        *evidence_columns,
    ]
    retired_columns = [
        "retired_pool_id", "original_pool_id", "status", "retirement_reason",
        "episode_created", "episode_retired", "permanent_pool_visits",
        "actions_explored", "hits_after_retirement",
        "unknown_actions", "safe_actions", "blocked_actions",
        "unknown_mask", "safe_mask", "blocked_mask",
        *evidence_columns,
        "last_hit_episode", "last_hit_step",
        "mean_general_similarity", "mean_general_rms_distance",
        "mean_safety_similarity", "mean_safety_rms_distance",
        "retirement_trigger_action", "retirement_trigger_reward",
        "retirement_trigger_collision",
        "retirement_trigger_out_of_road",
        "retirement_trigger_done",
        "retirement_trigger_type",
    ]
    evicted_columns = [
        "eviction_episode", "visit_count", "first_episode",
        "last_episode", "age_episodes", "unique_actions_executed",
        "near_promotion", "recent_candidate",
        "unknown_mask", "safe_mask", "blocked_mask",
        "total_action_attempts", "total_safe_outcomes",
        "total_collision_outcomes", "total_out_of_road_outcomes",
        "total_warnings",
        "blocked_evidence_archived",
    ]
    hazard_columns = [
        "hazard_id", "status", "archived_episode", "last_hit_episode",
        "hit_count", "archive_cycles", "source_candidate_visits",
        "unknown_actions", "safe_actions", "blocked_actions",
        "unknown_mask", "safe_mask", "blocked_mask",
        *evidence_columns,
    ]
    matching_columns = [
        "scope", "status", "queries", "empty_queries", "records_compared",
        "strict_matches", "general_relaxation_attempts",
        "general_relaxation_matches", "fallback_attempts", "fallback_matches",
        "no_match_queries", "general_cosine_failed_records",
        "general_rms_failed_records", "safety_cosine_failed_records",
        "safety_rms_failed_records", "safety_direction_failed_records",
        "safety_worsened_records",
        "safety_no_10_percent_improvement_records",
        "general_cosine_sole_block_queries",
        "general_rms_sole_block_queries", "safety_cosine_sole_block_queries",
        "safety_rms_sole_block_queries", "total_matches", "match_rate",
        "fallback_acceptance_rate", "mean_records_per_nonempty_query",
        "close_enough_fallback_enabled", "fallback_scope",
        "general_threshold_variation", "general_cosine_relaxation",
        "general_rms_relaxation", "safety_directional_improvement",
        "general_cosine_threshold",
        "general_rms_threshold", "safety_cosine_threshold",
        "safety_rms_threshold", "general_gates_relaxable",
        "pressure_fallback_eligible",
        "safety_gates_relaxable",
    ]
    save_rows(
        output_dir / "state_pool_statistics.csv", pool_rows, pool_columns
    )
    save_rows(
        output_dir / "state_candidate_statistics.csv",
        candidate_rows,
        candidate_columns,
    )
    save_rows(
        output_dir / "state_retired_pool_statistics.csv",
        retired_rows,
        retired_columns,
    )
    save_rows(
        output_dir / "state_hazard_memory_statistics.csv",
        hazard_rows,
        hazard_columns,
    )
    save_rows(
        output_dir / "state_matching_rejection_statistics.csv",
        matching_rows,
        matching_columns,
    )
    save_rows(
        output_dir / "state_evicted_candidate_history.csv",
        action_pools.evicted_candidate_rows,
        evicted_columns,
    )
    pd.DataFrame([global_row]).to_csv(
        output_dir / "state_pool_global_summary.csv", index=False
    )
    pd.DataFrame(
        action_pools.creation_events,
        columns=["episode", "cumulative_pools_created"],
    ).to_csv(output_dir / "state_pool_creation_timeline.csv", index=False)
    capacity_growth_columns = [
        "episode", "capacity_before", "capacity_after",
        "candidate_soft_before", "candidate_soft_after",
        "candidate_hard_before", "candidate_hard_after",
        "new_pool_capacity_pressure", "promoted_since_last_review",
        "waiting_events_since_last_review",
        "waiting_candidates_promoted_after_growth",
    ]
    pd.DataFrame(
        action_pools.capacity_growth_rows,
        columns=capacity_growth_columns,
    ).to_csv(
        output_dir / "state_pool_capacity_growth.csv", index=False
    )

    calibration_row = {
        "pool_representation": "general observation AND safety vector",
        "safety_vector_names": "|".join(SAFETY_VECTOR_NAMES),
        "calibration_target_states": action_pools.calibration_state_count,
        "calibration_states_collected": action_pools.total_calibration_states,
        "calibration_complete": action_pools.thresholds_frozen,
        "general_similarity_threshold": action_pools.similarity_threshold,
        "general_rms_threshold": action_pools.distance_threshold,
        "candidate_general_rms_threshold": (
            action_pools.candidate_distance_threshold
        ),
        "safety_similarity_threshold": (
            action_pools.safety_similarity_threshold
        ),
        "safety_rms_threshold": action_pools.safety_distance_threshold,
        "candidate_safety_rms_threshold": (
            action_pools.candidate_safety_distance_threshold
        ),
        "close_enough_fallback_enabled": (
            action_pools.close_enough_fallback
        ),
        "normal_general_cosine_relaxation": (
            action_pools.general_cosine_relaxation
        ),
        "normal_general_rms_relaxation": (
            action_pools.general_rms_relaxation
        ),
        "normal_general_relaxation_requires_both_gates": True,
        "normal_general_relaxation_keeps_safety_gates_strict": True,
        "capacity_fallback_scope": (
            "candidate_hard_limit_permanent_full_or_eviction"
        ),
        "capacity_fallback_general_variation": (
            action_pools.capacity_fallback_general_variation
        ),
        "capacity_fallback_general_cosine_relaxation": (
            action_pools.general_cosine_relaxation
        ),
        "capacity_fallback_general_rms_relaxation": (
            action_pools.capacity_fallback_general_variation
        ),
        "capacity_fallback_safety_improvement": (
            action_pools.capacity_fallback_safety_improvement
        ),
        "capacity_fallback_directional_features": (
            "lane_clearance:+|threat_distance:+|"
            "lane_offset:-|heading_error:-"
        ),
        "capacity_fallback_speed_directional": False,
        "close_enough_safety_gates_relaxed": False,
        **{
            f"general_{k}": v
            for k, v in action_pools.normalizer.statistics().items()
        },
        **{
            f"safety_{k}": v
            for k, v in action_pools.safety_normalizer.statistics().items()
        },
    }
    pd.DataFrame([calibration_row]).to_csv(
        output_dir / "state_matching_calibration.csv", index=False
    )

    try:
        make_pool_figures(action_pools, pool_rows, output_dir)
    except Exception as exc:
        plt.close("all")
        print(
            "WARNING: pool plots were not created; all required pool CSV "
            f"outputs were already saved: {exc}",
            file=sys.stderr,
            flush=True,
        )

def make_pool_figures(
    action_pools: SimilarStateActionPools,
    pool_rows: Sequence[Dict],
    output_dir: Path,
) -> None:
    """Create optional pool plots after required pool data has been saved."""
    if not pool_rows:
        return
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_ieee_style()
    pool_ids = [int(row["pool_id"]) for row in pool_rows]
    visits = [int(row["active_pool_mask_visits"]) for row in pool_rows]
    actions = [int(row["actions_tried"]) for row in pool_rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(pool_ids, visits)
    ax.set_xlabel("Active pool ID")
    ax.set_ylabel("Mask-controlled visits")
    ax.set_title("Active State Pool Occupancy")
    save_figure(fig, figure_dir, "state_pool_occupancy")

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(pool_ids, actions)
    ax.axhline(
        action_pools.action_count,
        linestyle="--",
        linewidth=1.0,
        label=f"Complete coverage ({action_pools.action_count} actions)",
    )
    ax.set_xlabel("Active pool ID")
    ax.set_ylabel("Actions attempted")
    ax.set_ylim(0, action_pools.action_count + 0.5)
    ax.set_title("Action Coverage per Active Pool")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "state_pool_action_coverage")
