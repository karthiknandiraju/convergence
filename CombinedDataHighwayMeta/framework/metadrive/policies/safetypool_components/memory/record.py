"""Typed state record shared by every SafetyPool lifecycle component."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass
class UnifiedStateRecord:
    """One centroid record whose lifecycle is controlled by a status tag."""

    record_id: int
    status: str
    state: np.ndarray
    safety: np.ndarray
    state_norm: float
    safety_norm: float
    unknown_mask: int
    safe_mask: int = 0
    blocked_mask: int = 0
    action_history_mask: int = 0
    availability_mask: int = 0
    exploration_pass: int = 0
    retirement_pending: bool = False
    action_attempt_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    safe_outcome_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    collision_outcome_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    out_of_road_outcome_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    warning_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    candidate_visits: int = 0
    candidate_first_episode: int = -1
    candidate_last_episode: int = -1
    candidate_centroid_updates: int = 0
    candidate_stable_updates: int = 0
    candidate_last_shift: float = 0.0
    candidate_centroid_frozen: bool = False
    last_permanent_match_generation: int = -1
    permanent_generation: int = -1

    promotion_evidence_visits: int = 0
    absorbed_candidate_visits: int = 0
    absorbed_candidate_actions: int = 0
    active_mask_visits: int = 0
    match_count: int = 0
    first_episode_created: int = -1
    last_episode_visited: int = -1
    similarity_sum: float = 0.0
    distance_sum: float = 0.0
    safety_similarity_sum: float = 0.0
    safety_distance_sum: float = 0.0
    centroid_updates: int = 0
    centroid_stable_updates: int = 0
    centroid_last_shift: float = 0.0
    centroid_frozen_by_stability: bool = False
    centroid_frozen_by_cap: bool = False
    recent_distance_window: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5)
    )

    retirement_reason: str = ""
    episode_retired: int = -1
    retired_permanent_visits: int = 0
    retired_actions_explored: int = 0
    retired_hit_count: int = 0
    retired_last_hit_episode: int = -1
    retired_last_hit_step: int = -1
    retired_similarity_sum: float = 0.0
    retired_distance_sum: float = 0.0
    retired_safety_similarity_sum: float = 0.0
    retired_safety_distance_sum: float = 0.0
    retirement_trigger_action: int = -1
    retirement_trigger_collision: bool = False
    retirement_trigger_out_of_road: bool = False
    retirement_trigger_done: bool = False
    retirement_trigger_step: int = -1
    retirement_trigger_type: str = ""

    hazard_archived_episode: int = -1
    hazard_last_hit_episode: int = -1
    hazard_hit_count: int = 0
    hazard_archive_cycles: int = 0
