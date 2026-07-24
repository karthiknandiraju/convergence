"""Pool record storage, status indexes, invariants, and evidence merging."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..safety import (
    RunningObservationNormalizer,
    SimilarityThresholdCalibrator,
)
from .record import UnifiedStateRecord


class PoolStorageMixin:
    """Pool record storage, status indexes, invariants, and evidence merging."""

    def __init__(
        self,
        max_pools: int,
        maximum_pool_capacity: int,
        max_candidates: int,
        candidate_hard_limit: int,
        candidate_batch_evict_count: int,
        candidate_promotion_visits: int,
        hazard_memory_capacity: int,
        safe_confirmation_visits: int,
        safety_horizon_steps: int,
        minimum_progress_reward: float,
        warning_block_threshold: int,
        candidate_recent_protection_episodes: int,
        capacity_review_interval: int,
        action_count: int,
        observation_size: int,
        safety_size: int,
        similarity_threshold: float,
        distance_threshold: float,
        candidate_similarity_threshold: float,
        candidate_distance_threshold: float,
        safety_similarity_threshold: float,
        safety_distance_threshold: float,
        candidate_safety_distance_threshold: float,
        close_enough_fallback: bool,
        general_cosine_relaxation: float,
        general_rms_relaxation: float,
        capacity_fallback_general_variation: float,
        capacity_fallback_safety_improvement: float,
        auto_calibrate_thresholds: bool,
        calibration_state_count: int,
        calibration_max_pairs: int,
        seed: int,
        candidate_centroid_shift_threshold: float,
        candidate_stable_updates: int,
        max_candidate_centroid_updates: int,
        centroid_shift_threshold: float,
        centroid_stable_updates: int,
        max_centroid_updates: int,
        centroid_stability_distance_threshold: float,
        pool_storage_dtype: str,
    ):
        if max_pools <= 0 or maximum_pool_capacity < max_pools:
            raise ValueError("Invalid permanent-pool capacities.")
        if max_candidates <= 0 or candidate_hard_limit <= max_candidates:
            raise ValueError("Invalid candidate capacities.")
        if hazard_memory_capacity <= 0:
            raise ValueError("Hazard-memory capacity must be positive.")
        if safe_confirmation_visits <= 0:
            raise ValueError("Safe-confirmation visits must be positive.")
        if safety_horizon_steps <= 0:
            raise ValueError("Safety horizon must be positive.")
        if (
            not math.isfinite(float(minimum_progress_reward))
            or minimum_progress_reward < 0.0
        ):
            raise ValueError(
                "Minimum progress reward must be finite and non-negative."
            )
        if warning_block_threshold <= 0:
            raise ValueError("Warning block threshold must be positive.")
        if not 0.0 <= capacity_fallback_general_variation < 1.0:
            raise ValueError(
                "Capacity-fallback general variation must be in [0, 1)."
            )
        if not 0.0 <= general_cosine_relaxation < 1.0:
            raise ValueError("General cosine relaxation must be in [0, 1).")
        if general_rms_relaxation < 0.0:
            raise ValueError("General RMS relaxation must be non-negative.")
        if capacity_fallback_safety_improvement < 0.0:
            raise ValueError(
                "Capacity-fallback safety improvement must be non-negative."
            )
        if candidate_hard_limit - candidate_batch_evict_count != max_candidates:
            raise ValueError(
                "candidate_hard_limit - candidate_batch_evict_count "
                "must equal max_candidates."
            )

        self.initial_max_pools = int(max_pools)
        self.max_pools = int(max_pools)
        self.maximum_pool_capacity = int(maximum_pool_capacity)
        self.max_candidates = int(max_candidates)
        self.candidate_hard_limit = int(candidate_hard_limit)
        self.candidate_batch_evict_count = int(candidate_batch_evict_count)
        self.candidate_promotion_visits = int(candidate_promotion_visits)
        self.hazard_memory_capacity = int(hazard_memory_capacity)
        self.safe_confirmation_visits = int(safe_confirmation_visits)
        self.safety_horizon_steps = int(safety_horizon_steps)
        self.minimum_progress_reward = float(minimum_progress_reward)
        self.warning_block_threshold = int(warning_block_threshold)
        self.candidate_recent_protection_episodes = int(
            candidate_recent_protection_episodes
        )
        self.capacity_review_interval = int(capacity_review_interval)
        self.action_count = int(action_count)
        self.observation_size = int(observation_size)
        self.safety_size = int(safety_size)
        self.full_mask = (1 << self.action_count) - 1
        # Stateless keyed draws keep epsilon-greedy selection reproducible
        # without maintaining or serializing another random-number generator.
        self.selection_seed = int(seed)
        self.storage_dtype = (
            np.float16 if pool_storage_dtype == "float16" else np.float32
        )

        self.configured_similarity_threshold = float(similarity_threshold)
        self.configured_distance_threshold = float(distance_threshold)
        self.configured_safety_similarity_threshold = float(
            safety_similarity_threshold
        )
        self.configured_safety_distance_threshold = float(
            safety_distance_threshold
        )
        self.similarity_threshold = self.configured_similarity_threshold
        self.distance_threshold = self.configured_distance_threshold
        self.candidate_similarity_threshold = float(
            candidate_similarity_threshold
        )
        self.configured_candidate_distance_threshold = float(
            candidate_distance_threshold
        )
        self.candidate_distance_threshold = (
            self.configured_candidate_distance_threshold
        )
        self.safety_similarity_threshold = (
            self.configured_safety_similarity_threshold
        )
        self.safety_distance_threshold = (
            self.configured_safety_distance_threshold
        )
        self.configured_candidate_safety_distance_threshold = float(
            candidate_safety_distance_threshold
        )
        self.candidate_safety_distance_threshold = (
            self.configured_candidate_safety_distance_threshold
        )
        self.close_enough_fallback = bool(close_enough_fallback)
        self.general_cosine_relaxation = float(
            general_cosine_relaxation
        )
        self.general_rms_relaxation = float(general_rms_relaxation)
        self.capacity_fallback_general_variation = float(
            capacity_fallback_general_variation
        )
        self.capacity_fallback_safety_improvement = float(
            capacity_fallback_safety_improvement
        )

        self.auto_calibrate_thresholds = bool(auto_calibrate_thresholds)
        self.calibration_state_count = int(calibration_state_count)
        self.normalization_state_count = (
            max(1, self.calibration_state_count // 2)
            if self.auto_calibrate_thresholds else 0
        )
        self.thresholds_frozen = not self.auto_calibrate_thresholds
        self.normalizer = RunningObservationNormalizer(observation_size)
        self.safety_normalizer = RunningObservationNormalizer(safety_size)
        self.calibrator = SimilarityThresholdCalibrator(
            calibration_max_pairs,
            similarity_threshold,
            distance_threshold,
            candidate_distance_threshold,
            seed + 91_337,
        )
        self.safety_calibrator = SimilarityThresholdCalibrator(
            calibration_max_pairs,
            safety_similarity_threshold,
            safety_distance_threshold,
            candidate_safety_distance_threshold,
            seed + 191_337,
        )

        self.candidate_centroid_shift_threshold = float(
            candidate_centroid_shift_threshold
        )
        self.candidate_stable_updates_required = int(candidate_stable_updates)
        self.max_candidate_centroid_updates = int(
            max_candidate_centroid_updates
        )
        self.centroid_shift_threshold = float(centroid_shift_threshold)
        self.centroid_stable_updates_required = int(centroid_stable_updates)
        self.max_centroid_updates = int(max_centroid_updates)
        self.centroid_stability_distance_threshold = float(
            centroid_stability_distance_threshold
        )

        # One canonical collection plus O(1) status membership indexes.
        self.records: Dict[int, UnifiedStateRecord] = {}
        self.status_ids: Dict[str, Dict[int, None]] = {
            self.CANDIDATE: {},
            self.ACTIVE: {},
            self.RETIRED: {},
            self.HAZARD: {},
        }
        # Stable record IDs are mapped onto recyclable dense slots. Exact
        # matching is vectorized over these preallocated centroid matrices.
        self.slot_capacity = int(
            self.maximum_pool_capacity
            + self.candidate_hard_limit
            + self.hazard_memory_capacity
        )
        self.state_matrix = np.zeros(
            (self.slot_capacity, self.observation_size),
            dtype=self.storage_dtype,
        )
        self.safety_matrix = np.zeros(
            (self.slot_capacity, self.safety_size),
            dtype=self.storage_dtype,
        )
        self.state_norms = np.zeros(self.slot_capacity, dtype=np.float32)
        self.safety_norms = np.zeros(self.slot_capacity, dtype=np.float32)
        self.slot_record_ids = np.full(
            self.slot_capacity, -1, dtype=np.int64
        )
        self.record_slots: Dict[int, int] = {}
        self.free_slots: List[int] = list(
            range(self.slot_capacity - 1, -1, -1)
        )
        self.status_slots: Dict[str, List[int]] = {
            self.CANDIDATE: [],
            self.ACTIVE: [],
            self.RETIRED: [],
            self.HAZARD: [],
        }
        self.slot_status_positions = np.full(
            self.slot_capacity, -1, dtype=np.int64
        )

        # Only promotion-ready capacity waiters enter this O(1) queue. The
        # generation log lets a waiter recheck only permanent centroids that
        # changed after its last exact full scan.
        self.ready_waiting_ids: Dict[int, None] = {}
        self.permanent_generation = 0
        self.permanent_change_log: List[int] = []
        self.next_record_id = 0
        self.pending_action_outcome: Optional[Dict] = None
        self.recent_safety_actions: Deque[Dict] = deque()
        self.policy_frozen = False
        self.policy_frozen_at_episode: Optional[int] = None
        self.policy_freeze_events = 0
        self.calibration_incomplete_at_freeze = False
        # Keep only a bounded recent audit trail. Aggregate eviction counters
        # below remain all-time totals, while this deque prevents long runs
        # from retaining one Python dictionary per historical eviction.
        self.evicted_candidate_history_capacity = int(
            self.maximum_pool_capacity
        )
        self.evicted_candidate_rows: Deque[Dict] = deque(
            maxlen=self.evicted_candidate_history_capacity
        )
        self.evicted_candidate_history_dropped = 0
        self.creation_events: List[Tuple[int, int]] = []
        self.capacity_growth_rows: List[Dict] = []

        self.total_states_seen = 0
        self.total_calibration_states = 0
        self.total_pool_matches = 0
        self.total_pool_creations = 0
        self.total_retired_pool_hits = 0
        # Retained for backward-compatible statistics. The final phase no
        # longer performs frozen safety matching, so this remains zero.
        self.total_final_frozen_safety_states = 0
        self.total_final_pure_dqn_states = 0
        self.candidates_created = 0
        self.candidates_promoted = 0
        self.candidates_merged_into_active_pool = 0
        self.candidate_action_history_transfers = 0
        self.candidate_action_bits_removed_by_transfer = 0
        self.candidates_evicted = 0
        self.candidates_blocked_by_capacity = 0
        self.candidate_capacity_wait_events = 0
        self.candidates_promoted_after_capacity_growth = 0
        self.absolute_capacity_argmax_states = 0
        self.candidates_suppressed_by_retired_pool = 0
        self.candidate_history_exhaustion_events = 0
        self.candidate_retired_suppression_events = 0
        self.pre_eviction_candidates_checked = 0
        self.pre_eviction_candidates_merged = 0
        self.near_promotion_candidates_protected = 0
        self.recent_candidates_protected = 0
        self.active_pools_retired_by_mask_exhaustion = 0
        self.direct_mask_exhaustion_retirements = 0
        self.candidate_history_exhaustion_retirements = 0
        self.retired_direct_evidence_updates = 0
        self.retired_candidate_evidence_merges = 0
        self.retired_mask_filtered_actions = 0
        self.post_first_pass_greedy_selections = 0
        self.post_first_pass_random_selections = 0
        self.post_first_pass_all_blocked_rejections = 0
        self.blocked_actions_skipped_from_scheduled_passes = 0
        self.hard_blocked_empty_selection_rejections = 0
        self.least_risk_fallbacks = 0
        self.initial_passes_completed = 0
        self.unknown_passes_completed = 0
        self.missing_safety_pure_dqn_states = 0
        self.active_mask_filtered_actions = 0
        self.candidate_mask_filtered_actions = 0
        self.hazard_mask_filtered_actions = 0
        self.safe_action_outcomes = 0
        self.no_progress_safe_rejections = 0
        self.blocked_action_outcomes = 0
        self.blocked_candidates_protected = 0
        self.hazard_records_archived = 0
        self.hazard_records_merged = 0
        self.hazard_records_evicted = 0
        self.hazard_queries = 0
        self.hazard_matches = 0
        self.hazard_actions_transferred = 0
        self.hazard_actions_filtered = 0
        self.test_hazard_queries = 0
        self.test_hazard_matches = 0
        self.test_hazard_actions_filtered = 0
        self.hazard_blocked_bits_preserved = 0
        self.hazard_blocked_bits_dropped = 0
        self.delayed_safe_confirmations = 0
        self.precursor_warnings_recorded = 0
        self.precursor_warning_blocks = 0
        self.attempt_accounting_repair_events = 0
        self.attempt_accounting_repaired_attempts = 0
        self.pending_eviction_protections = 0
        self.last_review_promoted = 0
        self.last_review_capacity_wait_events = 0
        self.candidate_full_fallback_queries = 0
        self.candidate_full_fallback_matches = 0
        self.permanent_full_fallback_queries = 0
        self.permanent_full_fallback_matches = 0
        self.eviction_fallback_queries = 0
        self.eviction_fallback_matches = 0
        self.capacity_fallback_active_matches = 0
        self.capacity_fallback_retired_matches = 0
        diagnostic_fields = (
            "queries",
            "empty_queries",
            "records_compared",
            "strict_matches",
            "general_relaxation_attempts",
            "general_relaxation_matches",
            "fallback_attempts",
            "fallback_matches",
            "no_match_queries",
            "general_cosine_failed_records",
            "general_rms_failed_records",
            "safety_cosine_failed_records",
            "safety_rms_failed_records",
            "safety_direction_failed_records",
            "safety_worsened_records",
            "safety_no_10_percent_improvement_records",
            "general_cosine_sole_block_queries",
            "general_rms_sole_block_queries",
            "safety_cosine_sole_block_queries",
            "safety_rms_sole_block_queries",
        )
        self.match_rejection_counters = {
            scope: {
                status: {field: 0 for field in diagnostic_fields}
                for status in (
                    self.CANDIDATE,
                    self.ACTIVE,
                    self.RETIRED,
                    self.HAZARD,
                )
            }
            for scope in ("train", "test")
        }

    def _ids(self, status: str) -> Iterable[int]:
        return self.status_ids[status].keys()

    def _require_policy_mutable(self, operation: str) -> None:
        if self.policy_frozen:
            raise RuntimeError(
                f"Pool policy is frozen; cannot perform {operation}."
            )

    def freeze_policy(self, episode: int) -> None:
        """Freeze pool evidence/lifecycle state at the pure-DQN boundary."""
        if self.policy_frozen:
            return
        self.calibration_incomplete_at_freeze = bool(
            self.auto_calibrate_thresholds
            and self.total_calibration_states < self.calibration_state_count
        )
        if self.pending_action_outcome is not None or self.recent_safety_actions:
            raise RuntimeError(
                "Cannot freeze pool policy with unresolved delayed safety evidence."
            )
        # Calibration must not resume or derive new gates in the final phase.
        self.normalizer.freeze()
        self.safety_normalizer.freeze()
        self.calibrator.freeze()
        self.safety_calibrator.freeze()
        self.thresholds_frozen = True
        self.policy_frozen = True
        self.policy_frozen_at_episode = int(episode)
        self.policy_freeze_events += 1

    def _attach_status_slot(self, status: str, slot: int) -> None:
        position = len(self.status_slots[status])
        self.status_slots[status].append(int(slot))
        self.slot_status_positions[int(slot)] = position

    def _detach_status_slot(self, status: str, slot: int) -> None:
        slot = int(slot)
        position = int(self.slot_status_positions[slot])
        slots = self.status_slots[status]
        if position < 0 or position >= len(slots) or slots[position] != slot:
            raise RuntimeError("Dense status-slot index is inconsistent.")
        last_slot = slots[-1]
        slots[position] = last_slot
        self.slot_status_positions[last_slot] = position
        slots.pop()
        self.slot_status_positions[slot] = -1

    def _allocate_record_slot(self, record: UnifiedStateRecord) -> int:
        if not self.free_slots:
            raise RuntimeError("Preallocated unified centroid storage is full.")
        slot = int(self.free_slots.pop())
        self.record_slots[record.record_id] = slot
        self.slot_record_ids[slot] = record.record_id
        self._attach_status_slot(record.status, slot)
        self._sync_record_vectors(record)
        return slot

    def _release_record_slot(self, record: UnifiedStateRecord) -> None:
        slot = int(self.record_slots.pop(record.record_id))
        self._detach_status_slot(record.status, slot)
        self.slot_record_ids[slot] = -1
        self.state_norms[slot] = 0.0
        self.safety_norms[slot] = 0.0
        self.free_slots.append(slot)

    def _sync_record_vectors(self, record: UnifiedStateRecord) -> None:
        slot = self.record_slots.get(record.record_id)
        if slot is None:
            return
        self.state_matrix[slot] = np.asarray(
            record.state, dtype=self.storage_dtype
        )
        self.safety_matrix[slot] = np.asarray(
            record.safety, dtype=self.storage_dtype
        )
        self.state_norms[slot] = float(record.state_norm)
        self.safety_norms[slot] = float(record.safety_norm)

    def _record_permanent_change(self, record: UnifiedStateRecord) -> None:
        if record.status not in {self.ACTIVE, self.RETIRED}:
            return
        self.permanent_generation += 1
        record.permanent_generation = self.permanent_generation
        self.permanent_change_log.append(record.record_id)

    def _changed_permanent_ids_since(self, generation: int) -> List[int]:
        if generation < 0:
            return list(self._ids(self.ACTIVE)) + list(
                self._ids(self.RETIRED)
            )
        changed = self.permanent_change_log[int(generation):]
        # Dict insertion order provides deterministic O(delta) de-duplication.
        return list(dict.fromkeys(changed))

    def _set_status(self, record_id: int, status: str) -> None:
        record = self.records[int(record_id)]
        if status not in self.status_ids:
            raise ValueError(f"Unknown record status: {status}")
        if record.status == status:
            return
        old_status = record.status
        slot = int(self.record_slots[record.record_id])
        self._detach_status_slot(old_status, slot)
        del self.status_ids[record.status][record.record_id]
        self.status_ids[status][record.record_id] = None
        record.status = status
        self._attach_status_slot(status, slot)
        if old_status == self.CANDIDATE:
            self.ready_waiting_ids.pop(record.record_id, None)
        if status in {self.ACTIVE, self.RETIRED}:
            self._record_permanent_change(record)

    def _pending_record_ids(self) -> set[int]:
        pending: set[int] = set()
        if self.pending_action_outcome is not None:
            pending.update(self.pending_action_outcome.get("record_ids", ()))
        for item in self.recent_safety_actions:
            pending.update(item.get("record_ids", ()))
        return pending

    def _record_is_pending(self, record_id: int) -> bool:
        return int(record_id) in self._pending_record_ids()

    def _remap_pending_record(self, old_id: int, new_id: int) -> None:
        """Retarget bounded delayed evidence after an O(1) lifecycle merge."""
        old_id, new_id = int(old_id), int(new_id)
        items: List[Dict] = list(self.recent_safety_actions)
        if self.pending_action_outcome is not None:
            items.append(self.pending_action_outcome)
        for item in items:
            record_ids = list(item.get("record_ids", ()))
            if old_id not in record_ids:
                continue
            record_ids = [new_id if value == old_id else value for value in record_ids]
            item["record_ids"] = tuple(dict.fromkeys(record_ids))

    def _delete_candidate(self, record_id: int) -> None:
        record = self.records[int(record_id)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Only candidate records can be recycled.")
        if self._record_is_pending(record.record_id):
            raise RuntimeError("A candidate with unresolved safety evidence cannot be evicted.")
        self.ready_waiting_ids.pop(record.record_id, None)
        self._release_record_slot(record)
        del self.status_ids[self.CANDIDATE][record.record_id]
        del self.records[record.record_id]

    def _delete_hazard(self, record_id: int) -> None:
        record = self.records[int(record_id)]
        if record.status != self.HAZARD:
            raise RuntimeError("Only hazard records can be recycled.")
        if self._record_is_pending(record.record_id):
            raise RuntimeError("A hazard with unresolved safety evidence cannot be evicted.")
        self._release_record_slot(record)
        del self.status_ids[self.HAZARD][record.record_id]
        del self.records[record.record_id]

    def _empty_action_counts(self) -> np.ndarray:
        return np.zeros(self.action_count, dtype=np.int64)

    def _rebuild_outcome_masks(self, record: UnifiedStateRecord) -> None:
        """Derive mutually exclusive masks from conservative outcome counts."""
        unknown_mask = 0
        safe_mask = 0
        blocked_mask = 0
        for action in range(self.action_count):
            bit = 1 << action
            unsafe_count = int(record.collision_outcome_counts[action]) + int(
                record.out_of_road_outcome_counts[action]
            )
            warning_blocked = (
                int(record.warning_counts[action])
                >= self.warning_block_threshold
            )
            if unsafe_count > 0 or warning_blocked:
                blocked_mask |= bit
            elif (
                int(record.safe_outcome_counts[action])
                >= self.safe_confirmation_visits
            ):
                safe_mask |= bit
            else:
                unknown_mask |= bit
        record.unknown_mask = int(unknown_mask)
        record.safe_mask = int(safe_mask)
        record.blocked_mask = int(blocked_mask)

    @staticmethod
    def _required_action_attempts(record: UnifiedStateRecord) -> np.ndarray:
        """Return the minimum attempts implied by retained outcome evidence."""
        return record.safe_outcome_counts + np.maximum(
            record.collision_outcome_counts,
            record.out_of_road_outcome_counts,
        )

    def _reconcile_attempt_accounting(
        self, record: UnifiedStateRecord
    ) -> int:
        """Repair rare lifecycle-remap deficits without changing outcomes.

        SAFE observations and direct failures remain the authoritative
        evidence.  A merge, hazard inheritance, or delayed confirmation may
        move that evidence between records, so the bookkeeping counter must
        be at least the number of retained outcomes for every action.
        """
        required = self._required_action_attempts(record)
        deficit = np.maximum(
            required - record.action_attempt_counts,
            0,
        )
        repaired = int(np.sum(deficit))
        if repaired:
            np.maximum(
                record.action_attempt_counts,
                required,
                out=record.action_attempt_counts,
            )
            self.attempt_accounting_repair_events += 1
            self.attempt_accounting_repaired_attempts += repaired
        return repaired

    def _merge_outcome_evidence(
        self, target: UnifiedStateRecord, source: UnifiedStateRecord
    ) -> None:
        target.action_attempt_counts += source.action_attempt_counts
        target.safe_outcome_counts += source.safe_outcome_counts
        target.collision_outcome_counts += source.collision_outcome_counts
        target.out_of_road_outcome_counts += source.out_of_road_outcome_counts
        target.warning_counts += source.warning_counts
        target.action_history_mask |= source.action_history_mask
        self._reconcile_attempt_accounting(target)
        self._rebuild_outcome_masks(target)

    def _merge_hazard_danger_evidence(
        self, target: UnifiedStateRecord, source: UnifiedStateRecord
    ) -> None:
        """Merge already mirrored danger evidence without double counting it."""
        target.collision_outcome_counts = np.maximum(
            target.collision_outcome_counts,
            source.collision_outcome_counts,
        )
        target.out_of_road_outcome_counts = np.maximum(
            target.out_of_road_outcome_counts,
            source.out_of_road_outcome_counts,
        )
        target.warning_counts = np.maximum(
            target.warning_counts,
            source.warning_counts,
        )
        direct_attempts = np.maximum(
            target.collision_outcome_counts,
            target.out_of_road_outcome_counts,
        )
        target.action_attempt_counts = np.maximum(
            target.action_attempt_counts, direct_attempts
        )
        # Hazard memory is danger-only. Confirmed-safe evidence remains local.
        target.safe_outcome_counts.fill(0)
        self._reconcile_attempt_accounting(target)
        self._rebuild_outcome_masks(target)

    def _new_candidate(self, state, safety, episode: int) -> int:
        state = np.asarray(state, dtype=self.storage_dtype).copy()
        safety = np.asarray(safety, dtype=self.storage_dtype).copy()
        record_id = self.next_record_id
        self.next_record_id += 1
        record = UnifiedStateRecord(
            record_id=record_id,
            status=self.CANDIDATE,
            state=state,
            safety=safety,
            state_norm=float(np.linalg.norm(state)),
            safety_norm=float(np.linalg.norm(safety)),
            unknown_mask=self.full_mask,
            action_attempt_counts=self._empty_action_counts(),
            safe_outcome_counts=self._empty_action_counts(),
            collision_outcome_counts=self._empty_action_counts(),
            out_of_road_outcome_counts=self._empty_action_counts(),
            warning_counts=self._empty_action_counts(),
            candidate_visits=1,
            candidate_first_episode=int(episode),
            candidate_last_episode=int(episode),
        )
        self.records[record_id] = record
        self.status_ids[self.CANDIDATE][record_id] = None
        self._allocate_record_slot(record)
        self.candidates_created += 1
        return record_id

    def _new_hazard(self, state, safety, episode: int) -> int:
        state = np.asarray(state, dtype=self.storage_dtype).copy()
        safety = np.asarray(safety, dtype=self.storage_dtype).copy()
        record_id = self.next_record_id
        self.next_record_id += 1
        record = UnifiedStateRecord(
            record_id=record_id,
            status=self.HAZARD,
            state=state,
            safety=safety,
            state_norm=float(np.linalg.norm(state)),
            safety_norm=float(np.linalg.norm(safety)),
            unknown_mask=self.full_mask,
            action_attempt_counts=self._empty_action_counts(),
            safe_outcome_counts=self._empty_action_counts(),
            collision_outcome_counts=self._empty_action_counts(),
            out_of_road_outcome_counts=self._empty_action_counts(),
            warning_counts=self._empty_action_counts(),
            hazard_archived_episode=int(episode),
            hazard_archive_cycles=1,
        )
        self.records[record_id] = record
        self.status_ids[self.HAZARD][record_id] = None
        self._allocate_record_slot(record)
        self.hazard_records_archived += 1
        return record_id

    def validate_invariants(self) -> None:
        if len(self.records) != len(self.record_slots):
            raise RuntimeError("Record-to-slot cardinality invariant failed.")
        occupied_slots = set(self.record_slots.values())
        if occupied_slots & set(self.free_slots):
            raise RuntimeError("A centroid slot is both occupied and free.")
        if len(occupied_slots) + len(self.free_slots) != self.slot_capacity:
            raise RuntimeError("Centroid slot accounting invariant failed.")
        for status in (
            self.CANDIDATE,
            self.ACTIVE,
            self.RETIRED,
            self.HAZARD,
        ):
            if len(self.status_ids[status]) != len(self.status_slots[status]):
                raise RuntimeError("Status membership cardinality failed.")
            for position, slot in enumerate(self.status_slots[status]):
                record_id = int(self.slot_record_ids[slot])
                record = self.records.get(record_id)
                if (
                    record is None
                    or record.status != status
                    or self.record_slots.get(record_id) != slot
                    or int(self.slot_status_positions[slot]) != position
                ):
                    raise RuntimeError("Dense status membership is inconsistent.")
        for record in self.records.values():
            if record.exploration_pass not in {0, 1}:
                raise RuntimeError("Only initial and UNKNOWN passes are scheduled.")
            if record.retirement_pending and (
                record.status != self.ACTIVE
                or record.exploration_pass != 1
                or record.availability_mask != 0
            ):
                raise RuntimeError("Pending retirement state is inconsistent.")
            for counts in (
                record.action_attempt_counts,
                record.safe_outcome_counts,
                record.collision_outcome_counts,
                record.out_of_road_outcome_counts,
                record.warning_counts,
            ):
                if counts.shape != (self.action_count,):
                    raise RuntimeError("Per-action evidence shape invariant failed.")
                if np.any(counts < 0):
                    raise RuntimeError("Per-action evidence cannot be negative.")
            direct_outcomes = self._required_action_attempts(record)
            if np.any(record.action_attempt_counts < direct_outcomes):
                action = int(
                    np.flatnonzero(
                        record.action_attempt_counts < direct_outcomes
                    )[0]
                )
                raise RuntimeError(
                    "Per-action attempt accounting failed: "
                    f"record={record.record_id}, status={record.status}, "
                    f"action={action}, "
                    f"attempts={int(record.action_attempt_counts[action])}, "
                    f"safe={int(record.safe_outcome_counts[action])}, "
                    "collision="
                    f"{int(record.collision_outcome_counts[action])}, "
                    "out_of_road="
                    f"{int(record.out_of_road_outcome_counts[action])}."
                )
            if record.unknown_mask & record.safe_mask:
                raise RuntimeError("Unknown and safe masks overlap.")
            if record.unknown_mask & record.blocked_mask:
                raise RuntimeError("Unknown and blocked masks overlap.")
            if record.safe_mask & record.blocked_mask:
                raise RuntimeError("Safe and blocked masks overlap.")
            if (
                record.unknown_mask
                | record.safe_mask
                | record.blocked_mask
            ) != self.full_mask:
                raise RuntimeError("Outcome masks do not cover all actions.")
            expected_masks = (
                record.unknown_mask,
                record.safe_mask,
                record.blocked_mask,
            )
            self._rebuild_outcome_masks(record)
            if expected_masks != (
                record.unknown_mask,
                record.safe_mask,
                record.blocked_mask,
            ):
                raise RuntimeError("Outcome masks disagree with evidence counts.")
        for record_id in self.ready_waiting_ids:
            record = self.records.get(record_id)
            if (
                record is None
                or record.status != self.CANDIDATE
                or record.candidate_visits < self.candidate_promotion_visits
            ):
                raise RuntimeError("Ready-waiter queue invariant failed.")
        if self.total_permanent_records() > self.max_pools:
            raise RuntimeError("Permanent capacity invariant failed.")
        if len(self.status_ids[self.CANDIDATE]) > self.candidate_hard_limit:
            raise RuntimeError("Candidate hard-limit invariant failed.")
        if len(self.status_ids[self.HAZARD]) > self.hazard_memory_capacity:
            raise RuntimeError("Hazard-memory capacity invariant failed.")
