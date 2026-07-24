"""Seed/episode summaries for collision RMST and safety-stop reporting."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from .constants import EXPERIMENTS, SHORT_LABELS
from .utils import restricted_mean_survival_time


def make_summary(rows: List[Dict], args) -> List[Dict]:
    summary: List[Dict] = []
    for experiment in EXPERIMENTS:
        train = [
            row for row in rows
            if row["experiment"] == experiment and row["phase"] == "train"
        ]
        test = [
            row for row in rows
            if row["experiment"] == experiment and row["phase"] == "test"
        ]
        test_times = [float(row["event_or_censor_time_steps"]) for row in test]
        test_events = [bool(row["rmst_event_observed"]) for row in test]
        rmst = restricted_mean_survival_time(test_times, test_events, args.rmst_tau)
        total_train_time = sum(
            float(row["wall_time_seconds"]) for row in train
        )
        summary.append(
            {
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "learning_rate": args.learning_rate,
                "selected_event_rmst_steps": rmst,
                "RMST_tau_steps": args.rmst_tau,
                "RMST_event_definition": args.rmst_event,
                "total_training_wall_time_seconds": total_train_time,
                "train_collision_rate": float(np.mean([bool(r["collision"]) for r in train])),
                "test_collision_rate": float(np.mean([bool(r["collision"]) for r in test])),
                "train_policy_safety_stop_rate": float(np.mean([
                    bool(r.get("policy_safety_stop", False)) for r in train
                ])),
                "test_policy_safety_stop_rate": float(np.mean([
                    bool(r.get("policy_safety_stop", False)) for r in test
                ])),
                "train_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in train])),
                "test_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in test])),
                "test_off_road_rate": float(np.mean([bool(r["out_of_road"]) for r in test])),
                "train_combined_safety_failure_rate": float(np.mean([
                    bool(r["collision"])
                    or bool(r["out_of_road"])
                    for r in train
                ])),
                "test_combined_safety_failure_rate": float(np.mean([
                    bool(r["collision"])
                    or bool(r["out_of_road"])
                    for r in test
                ])),
                "train_policy_adjusted_safety_failure_rate": float(np.mean([
                    bool(r["collision"])
                    or bool(r["out_of_road"])
                    or bool(r.get("policy_safety_stop", False))
                    for r in train
                ])),
                "test_policy_adjusted_safety_failure_rate": float(np.mean([
                    bool(r["collision"])
                    or bool(r["out_of_road"])
                    or bool(r.get("policy_safety_stop", False))
                    for r in test
                ])),
                "train_collisions_per_1000_steps": (
                    1000.0 * sum(bool(r["collision"]) for r in train)
                    / max(1, sum(int(r["steps"]) for r in train))
                ),
                "test_collisions_per_1000_steps": (
                    1000.0 * sum(bool(r["collision"]) for r in test)
                    / max(1, sum(int(r["steps"]) for r in test))
                ),
                "train_mean_minimum_lane_boundary_clearance": float(
                    np.nanmean([
                        r["minimum_lane_boundary_clearance"] for r in train
                    ])
                ),
                "test_mean_minimum_lane_boundary_clearance": math.nan,
                "train_mean_minimum_collision_hazard_center_distance": float(
                    np.nanmean([
                        r["minimum_nearest_collision_hazard_center_distance"] for r in train
                    ])
                ),
                "test_mean_minimum_collision_hazard_center_distance": math.nan,
                "network_frozen_during_testing": True,
            }
        )
    return summary

def bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)

def collision_summary(rows: Sequence[Dict], tau: int) -> Dict[str, float]:
    collisions = sum(bool_value(row["collision"]) for row in rows)
    out_of_road = sum(bool_value(row["out_of_road"]) for row in rows)
    goals = sum(bool_value(row["goal_reached"]) for row in rows)
    combined = sum(
        bool_value(row["collision"])
        or bool_value(row["out_of_road"])
        for row in rows
    )
    steps = sum(int(row["steps"]) for row in rows)
    episodes = len(rows)
    return {
        "episodes": episodes,
        "collision_count": collisions,
        "out_of_road_count": out_of_road,
        "total_steps": steps,
        "collision_rmst_event_definition": "collision",
        "collision_rmst": restricted_mean_survival_time(
            [int(row["event_or_censor_time_steps"]) for row in rows],
            [bool_value(row["collision"]) for row in rows],
            tau,
        ),
        "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "out_of_road_rate": out_of_road / episodes if episodes else 0.0,
        "goal_count": goals,
        "goal_rate": goals / episodes if episodes else 0.0,
        "combined_safety_failure_count": combined,
        "combined_safety_failure_rate": (
            combined / episodes if episodes else 0.0
        ),
        "combined_safety_failures_per_1000_steps": (
            1000.0 * combined / steps if steps else 0.0
        ),
        "combined_safety_rmst_event_definition": "collision_or_out_of_road",
        "combined_safety_rmst": restricted_mean_survival_time(
            [int(row["event_or_censor_time_steps"]) for row in rows],
            [
                bool_value(row["collision"])
                or bool_value(row["out_of_road"])
                for row in rows
            ],
            tau,
        ),
    }

def policy_safety_stop_summary(
    rows: Sequence[Dict], tau: int
) -> Dict[str, object]:
    """Audit policy stops without changing canonical collision semantics."""
    episodes = len(rows)
    stops = sum(
        bool_value(row.get("policy_safety_stop", False)) for row in rows
    )
    environment_steps = sum(int(row["steps"]) for row in rows)
    # Each stop happens at a real policy decision point even though no action
    # is sent to the environment. Counting that point prevents a zero-step
    # stopped episode from incorrectly reporting zero failures per exposure.
    policy_decisions = environment_steps + stops
    adjusted_events = [
        bool_value(row["collision"])
        or bool_value(row["out_of_road"])
        or bool_value(row.get("policy_safety_stop", False))
        for row in rows
    ]
    adjusted_failures = sum(adjusted_events)
    return {
        "episodes": int(episodes),
        "policy_safety_stop_count": int(stops),
        "policy_safety_stop_rate": stops / episodes if episodes else 0.0,
        "environment_steps": int(environment_steps),
        "policy_decision_opportunities": int(policy_decisions),
        "policy_adjusted_safety_failure_count": int(adjusted_failures),
        "policy_adjusted_safety_failure_rate": (
            adjusted_failures / episodes if episodes else 0.0
        ),
        "policy_adjusted_safety_failures_per_1000_decisions": (
            1000.0 * adjusted_failures / policy_decisions
            if policy_decisions
            else 0.0
        ),
        "policy_adjusted_safety_rmst_event_definition": (
            "collision_or_out_of_road_or_policy_safety_stop"
        ),
        "policy_adjusted_safety_rmst": restricted_mean_survival_time(
            [int(row["event_or_censor_time_steps"]) for row in rows],
            adjusted_events,
            tau,
        ),
        "metric_treatment": (
            "canonical_collision_metrics_unchanged;policy_stop_is_separate_"
            "policy_failure"
        ),
    }
