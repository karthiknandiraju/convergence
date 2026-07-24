"""Shared immutable names and experiment contracts for SafetyPool."""

from __future__ import annotations

import math

import numpy as np


EXPERIMENTS = ["Karthikeya27adv8"]

SHORT_LABELS = {
    "Karthikeya27adv8": "Karthikeya27adv8",
}

COLORS = {
    "Karthikeya27adv8": "#2ca02c",
}

POLICY_NAME = "Karthikeya27adv8"

OBSERVATION_SOURCE = (
    "flattened_metadrive_observation_for_dqn_plus_engine_safety_features_"
    "during_first_pool_training_phase_only"
)

TEST_POLICY = "final_trained_frozen_dqn_argmax"

DIRECTIONAL_SAFETY_RELATIVE_FLOORS = np.asarray(
    (0.10, 1.0, 0.10, math.radians(1.0)), dtype=np.float32
)

CANONICAL_EPISODE_COLUMNS = (
    "phase",
    "experiment",
    "method",
    "seed",
    "episode",
    "scenario_seed",
    "initial_observation_sha256",
    "env_reward",
    "training_reward",
    "steps",
    "termination_reason",
    "collision",
    "crash_vehicle",
    "crash_object",
    "out_of_road",
    "goal_reached",
    "max_steps_reached",
    "rmst_event_definition",
    "rmst_event_observed",
    "event_or_censor_time_steps",
    "wall_time_seconds",
    "cpu_time_seconds",
    "average_loss",
    "average_rnd_loss",
    "average_rnd_bonus",
    "replay_buffer_size",
    "learn_steps",
    "epsilon",
    "rnd_beta",
    "noisy_sigma_init",
    "network_frozen",
    "updates_during_test",
    "action_source_counts",
)

CRITICAL_CONFIG_KEYS = (
    "seed",
    "train_episodes",
    "test_episodes",
    "max_episode_steps",
    "max_state_pools",
    "maximum_pool_capacity",
    "max_state_candidates",
    "candidate_hard_limit",
    "candidate_batch_evict_count",
    "candidate_promotion_visits",
    "hazard_memory_fraction",
    "hazard_memory_capacity",
    "safe_confirmation_visits",
    "safety_horizon_steps",
    "minimum_progress_reward",
    "warning_block_threshold",
    "auto_calibrate_thresholds",
    "calibration_state_count",
    "calibration_max_pairs",
    "safety_similarity_threshold",
    "safety_distance_threshold",
    "candidate_safety_distance_threshold",
    "safety_nearest_object_cap",
    "safety_lane_boundary_cap",
    "safety_speed_cap",
    "safety_speed_fallback_unit",
    "capacity_review_interval",
    "candidate_recent_protection_episodes",
    "state_similarity_threshold",
    "state_distance_threshold",
    "candidate_similarity_threshold",
    "candidate_distance_threshold",
    "close_enough_fallback",
    "general_cosine_relaxation",
    "general_rms_relaxation",
    "capacity_fallback_general_variation",
    "capacity_fallback_safety_improvement",
    "candidate_centroid_shift_threshold",
    "candidate_stable_updates",
    "max_candidate_centroid_updates",
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
    "rmst_event",
    "progress_every",
    "deterministic",
    "pool_training_fraction",
    "centroid_shift_threshold",
    "centroid_stable_updates",
    "max_centroid_updates",
    "centroid_stability_distance_threshold",
    "pool_storage_dtype",
    "hybrid_design_version",
)

BASELINE_SHARED_CONFIG_KEYS = (
    "seed",
    "deterministic",
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
    "progress_every",
)

SAFETY_VECTOR_NAMES = (
    "lane_boundary_clearance",
    "nearest_collision_hazard_center_distance",
    "absolute_lane_offset",
    "absolute_heading_error",
    "ego_speed_km_h",
)
