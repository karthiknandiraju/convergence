"""Centroid updates, matched-state acceptance, and candidate reception."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .record import UnifiedStateRecord


class PoolStateProcessingMixin:
    """Centroid updates, matched-state acceptance, and candidate reception."""

    def _update_active_centroid(
        self, record: UnifiedStateRecord, state, safety
    ) -> None:
        if (
            record.centroid_frozen_by_stability
            or record.centroid_frozen_by_cap
        ):
            return
        old = np.asarray(record.state, dtype=np.float32).copy()
        old_safety = np.asarray(record.safety, dtype=np.float32).copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)
        prior = max(
            1,
            record.promotion_evidence_visits
            + record.absorbed_candidate_visits
            + record.active_mask_visits - 1,
        )
        updated = old + (
            np.asarray(state, dtype=np.float32) - old
        ) / float(prior + 1)
        updated_safety = old_safety + (
            np.asarray(safety, dtype=np.float32) - old_safety
        ) / float(prior + 1)
        record.state = updated.astype(self.storage_dtype)
        record.safety = updated_safety.astype(self.storage_dtype)
        record.state_norm = float(np.linalg.norm(record.state))
        record.safety_norm = float(np.linalg.norm(record.safety))
        shift = max(
            float(np.linalg.norm(updated - old) / old_norm),
            self._rms(updated, old),
            self._rms(updated_safety, old_safety),
        )
        record.centroid_updates += 1
        record.centroid_last_shift = shift
        recent_distance = (
            float(np.mean(record.recent_distance_window))
            if record.recent_distance_window
            else float("inf")
        )
        if (
            shift < self.centroid_shift_threshold
            and recent_distance <= self.centroid_stability_distance_threshold
        ):
            record.centroid_stable_updates += 1
        else:
            record.centroid_stable_updates = 0
        if (
            record.centroid_stable_updates
            >= self.centroid_stable_updates_required
        ):
            record.centroid_frozen_by_stability = True
        elif record.centroid_updates >= self.max_centroid_updates:
            record.centroid_frozen_by_cap = True
        self._sync_record_vectors(record)
        self._record_permanent_change(record)

    def _update_candidate_centroid(
        self, record: UnifiedStateRecord, state, safety, new_visits: int
    ) -> None:
        if record.candidate_centroid_frozen:
            return
        old = np.asarray(record.state, dtype=np.float32).copy()
        old_safety = np.asarray(record.safety, dtype=np.float32).copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)
        updated = old + (
            np.asarray(state, dtype=np.float32) - old
        ) / float(new_visits)
        updated_safety = old_safety + (
            np.asarray(safety, dtype=np.float32) - old_safety
        ) / float(new_visits)
        record.state = updated.astype(self.storage_dtype)
        record.safety = updated_safety.astype(self.storage_dtype)
        record.state_norm = float(np.linalg.norm(record.state))
        record.safety_norm = float(np.linalg.norm(record.safety))
        shift = max(
            float(np.linalg.norm(updated - old) / old_norm),
            self._rms(updated, old),
            self._rms(updated_safety, old_safety),
        )
        record.candidate_centroid_updates += 1
        record.candidate_last_shift = shift
        if shift < self.candidate_centroid_shift_threshold:
            record.candidate_stable_updates += 1
        else:
            record.candidate_stable_updates = 0
        if (
            record.candidate_stable_updates
            >= self.candidate_stable_updates_required
            or record.candidate_centroid_updates
            >= self.max_candidate_centroid_updates
        ):
            record.candidate_centroid_frozen = True
        self._sync_record_vectors(record)

    def _accept_active_state_match(
        self,
        record: UnifiedStateRecord,
        match: Tuple[Optional[int], float, float, float, float],
        state: np.ndarray,
        safety: np.ndarray,
        episode: int,
    ) -> int:
        """Apply one accepted ACTIVE hit with O(1) lifecycle bookkeeping."""
        record.active_mask_visits += 1
        record.last_episode_visited = int(episode)
        record.match_count += 1
        self.total_pool_matches += 1
        record.similarity_sum += float(match[1])
        record.distance_sum += float(match[2])
        record.safety_similarity_sum += float(match[3])
        record.safety_distance_sum += float(match[4])
        record.recent_distance_window.append(float(match[2]))
        self._update_active_centroid(record, state, safety)
        return record.record_id

    def process_state(
        self,
        state,
        safety,
        episode,
        active_match=None,
        active_match_precomputed: bool = False,
        retired_match=None,
        retired_match_precomputed: bool = False,
    ):
        self._require_policy_mutable("state-pool lifecycle processing")
        self.total_states_seen += 1
        active = (
            active_match
            if active_match_precomputed
            else self.find_active_match(state, safety)
        )
        active_id = active[0]
        if active_id is not None:
            record = self.records[int(active_id)]
            return (
                self._accept_active_state_match(
                    record, active, state, safety, episode
                ),
                None,
                "permanent_matched",
            )

        # Retired lookup is deliberately before candidate lookup. The caller
        # normally handles the retired hit to apply its DQN action mask.
        retired = (
            retired_match
            if retired_match_precomputed
            else self.find_retired_match(state, safety)
        )
        if retired[0] is not None:
            return None, None, "retired_match_requires_filtered_dqn"

        candidate_match = self.find_candidate_match(state, safety)
        if candidate_match[0] is not None:
            candidate = self.records[int(candidate_match[0])]
            new_visits = candidate.candidate_visits + 1
            self._update_candidate_centroid(
                candidate, state, safety, new_visits
            )
            candidate.candidate_visits = new_visits
            candidate.candidate_last_episode = int(episode)
            if new_visits >= self.candidate_promotion_visits:
                active_candidate_match = self.find_active_match(
                    candidate.state, candidate.safety
                )
                if active_candidate_match[0] is not None:
                    active_id = int(active_candidate_match[0])
                    retired_id = self.absorb_candidate_into_active(
                        candidate.record_id, active_id, episode
                    )
                    if retired_id is not None:
                        return (
                            None,
                            retired_id,
                            "candidate_history_exhausted_retired_pool",
                        )
                    return (
                        active_id,
                        None,
                        "candidate_merged_into_active_pool",
                    )

                retired_candidate_match = self.find_retired_match(
                    candidate.state, candidate.safety
                )
                if retired_candidate_match[0] is not None:
                    retired_id = self.absorb_candidate_into_retired(
                        candidate.record_id,
                        int(retired_candidate_match[0]),
                    )
                    return (
                        None,
                        retired_id,
                        "candidate_suppressed_by_retired_pool",
                    )

                if self.permanent_capacity_available():
                    promoted_id = self._promote_candidate(
                        candidate.record_id, episode
                    )
                    if promoted_id is None:
                        return (
                            None,
                            None,
                            "candidate_history_exhausted_pool",
                        )
                    return (
                        promoted_id,
                        None,
                        "candidate_promoted_before_action",
                    )

                if not self.absolute_permanent_capacity_reached():
                    self.candidate_capacity_wait_events += 1
                    candidate.last_permanent_match_generation = (
                        self.permanent_generation
                    )
                    self.ready_waiting_ids[candidate.record_id] = None
                    return (
                        None,
                        candidate.record_id,
                        "candidate_waiting_for_pool_capacity_argmax",
                    )

                self.candidates_blocked_by_capacity += 1
                if self._record_is_pending(candidate.record_id):
                    self.pending_eviction_protections += 1
                    return (
                        None,
                        candidate.record_id,
                        "candidate_pending_absolute_capacity",
                    )
                hazard_id = self._archive_or_delete_candidate(
                    candidate.record_id, episode
                )
                if hazard_id is not None:
                    return (
                        None,
                        hazard_id,
                        "candidate_absolute_capacity_hazard",
                    )
                return (
                    None, None, "candidate_absolute_capacity_argmax"
                )
            return None, candidate.record_id, "candidate_matched_argmax"

        permanent_full = self.absolute_permanent_capacity_reached()
        candidate_full = (
            len(self.status_ids[self.CANDIDATE])
            >= self.candidate_hard_limit
        )
        if permanent_full or candidate_full:
            fallback_context = (
                "permanent_full" if permanent_full else "candidate_full"
            )
            fallback_status, fallback_match = self.find_capacity_fallback(
                state,
                safety,
                context=fallback_context,
            )
            if fallback_status == self.ACTIVE:
                active_record = self.records[int(fallback_match[0])]
                return (
                    self._accept_active_state_match(
                        active_record,
                        fallback_match,
                        state,
                        safety,
                        episode,
                    ),
                    None,
                    f"{fallback_context}_fallback_active_pool",
                )
            if fallback_status == self.RETIRED:
                return (
                    None,
                    int(fallback_match[0]),
                    f"{fallback_context}_fallback_retired_pool",
                )
            if permanent_full:
                self.absolute_capacity_argmax_states += 1
                return None, None, "absolute_pool_capacity_argmax"
            self._batch_evict_candidates(episode)
            if (
                len(self.status_ids[self.CANDIDATE])
                >= self.candidate_hard_limit
            ):
                # The remaining candidates are pending or one visit from
                # promotion. Preserve them and avoid exceeding preallocated
                # candidate storage.
                return None, None, "candidate_hard_limit_protected_argmax"
        created_id = self._new_candidate(state, safety, episode)
        if len(self.status_ids[self.CANDIDATE]) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)
            if created_id not in self.records:
                return (
                    None, None, "candidate_created_then_evicted_argmax"
                )
        return None, created_id, "candidate_created_argmax"
