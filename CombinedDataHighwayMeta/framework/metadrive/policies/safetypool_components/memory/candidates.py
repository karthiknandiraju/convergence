"""Candidate absorption, hazard archiving, and bounded eviction."""

from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple

import numpy as np


class PoolCandidateManagementMixin:
    """Candidate absorption, hazard archiving, and bounded eviction."""

    def absorb_candidate_into_active(
        self,
        candidate_index: int,
        pool_index: int,
        episode: int,
        pre_eviction: bool = False,
    ) -> Optional[int]:
        candidate = self.records[int(candidate_index)]
        active = self.records[int(pool_index)]
        if candidate.status != self.CANDIDATE or active.status != self.ACTIVE:
            raise RuntimeError("Candidate-to-active merge received bad tags.")

        visits = int(candidate.candidate_visits)
        history = int(candidate.action_history_mask)
        active_weight = max(
            1,
            active.promotion_evidence_visits
            + active.absorbed_candidate_visits
            + active.active_mask_visits,
        )
        if not (
            active.centroid_frozen_by_stability
            or active.centroid_frozen_by_cap
        ):
            old_state = np.asarray(active.state, dtype=np.float32).copy()
            old_safety = np.asarray(active.safety, dtype=np.float32).copy()
            combined_weight = active_weight + visits
            merged_state = (
                old_state * active_weight
                + np.asarray(candidate.state, dtype=np.float32) * visits
            ) / float(combined_weight)
            merged_safety = (
                old_safety * active_weight
                + np.asarray(candidate.safety, dtype=np.float32) * visits
            ) / float(combined_weight)
            relative_shift = float(
                np.linalg.norm(merged_state - old_state)
                / max(float(np.linalg.norm(old_state)), 1e-8)
            )
            shift = max(
                relative_shift,
                self._rms(merged_state, old_state),
                self._rms(merged_safety, old_safety),
            )
            active.state = merged_state.astype(self.storage_dtype)
            active.safety = merged_safety.astype(self.storage_dtype)
            active.state_norm = float(np.linalg.norm(active.state))
            active.safety_norm = float(np.linalg.norm(active.safety))
            active.centroid_updates += 1
            active.centroid_last_shift = shift
            active.recent_distance_window.append(
                self._rms(candidate.state, old_state)
            )
            recent_distance = float(
                np.mean(active.recent_distance_window)
            )
            if (
                shift < self.centroid_shift_threshold
                and recent_distance
                <= self.centroid_stability_distance_threshold
            ):
                active.centroid_stable_updates += 1
            else:
                active.centroid_stable_updates = 0
            if (
                active.centroid_stable_updates
                >= self.centroid_stable_updates_required
            ):
                active.centroid_frozen_by_stability = True
            elif active.centroid_updates >= self.max_centroid_updates:
                active.centroid_frozen_by_cap = True
            self._sync_record_vectors(active)
            self._record_permanent_change(active)

        before = int(active.availability_mask)
        active.availability_mask = before & ~history
        removed_bits = (before & history).bit_count()
        self._merge_outcome_evidence(active, candidate)
        active.absorbed_candidate_visits += visits
        active.absorbed_candidate_actions += history.bit_count()
        self.candidates_merged_into_active_pool += 1
        self.candidate_action_history_transfers += 1
        self.candidate_action_bits_removed_by_transfer += removed_bits
        if pre_eviction:
            self.pre_eviction_candidates_merged += 1
        self._remap_pending_record(candidate.record_id, active.record_id)
        self._delete_candidate(candidate.record_id)

        if active.availability_mask == 0:
            self.candidate_history_exhaustion_events += 1
            self._complete_active_scheduled_pass(
                active.record_id,
                episode,
                {
                    "action": -1,
                    "collision": False,
                    "out_of_road": False,
                    "done": False,
                    "step": -1,
                    "retirement_trigger_type": "candidate_history_evidence",
                },
            )
            return (
                active.record_id
                if active.status == self.RETIRED
                else None
            )
        return None

    def absorb_candidate_into_retired(
        self,
        candidate_index: int,
        retired_index: int,
        pre_eviction: bool = False,
    ) -> int:
        """Merge fixed-size action evidence without reopening retirement."""
        candidate = self.records[int(candidate_index)]
        retired = self.records[int(retired_index)]
        if (
            candidate.status != self.CANDIDATE
            or retired.status != self.RETIRED
        ):
            raise RuntimeError("Candidate-to-retired merge received bad tags.")
        self.candidates_suppressed_by_retired_pool += 1
        self.candidate_retired_suppression_events += 1
        self._merge_outcome_evidence(retired, candidate)
        self.retired_candidate_evidence_merges += 1
        if pre_eviction:
            self.pre_eviction_candidates_merged += 1
        self._remap_pending_record(candidate.record_id, retired.record_id)
        self._delete_candidate(candidate.record_id)
        return retired.record_id

    def _promote_waiting_candidates_after_growth(
        self, episode: int
    ) -> int:
        promoted_to_new_pool = 0
        for candidate_id in list(self.ready_waiting_ids):
            candidate = self.records.get(candidate_id)
            if (
                candidate is None
                or candidate.status != self.CANDIDATE
                or candidate.candidate_visits
                < self.candidate_promotion_visits
            ):
                self.ready_waiting_ids.pop(candidate_id, None)
                continue
            changed_ids = self._changed_permanent_ids_since(
                candidate.last_permanent_match_generation
            )
            active_match = self._best_status_match(
                candidate.state,
                candidate.safety,
                self.ACTIVE,
                self.similarity_threshold,
                self.distance_threshold,
                self.safety_similarity_threshold,
                self.safety_distance_threshold,
                record_ids=changed_ids,
            )
            if active_match[0] is not None:
                self.absorb_candidate_into_active(
                    candidate.record_id, int(active_match[0]), episode
                )
                continue
            retired_match = self._best_status_match(
                candidate.state,
                candidate.safety,
                self.RETIRED,
                self.similarity_threshold,
                self.distance_threshold,
                self.safety_similarity_threshold,
                self.safety_distance_threshold,
                record_ids=changed_ids,
            )
            if retired_match[0] is not None:
                self.absorb_candidate_into_retired(
                    candidate.record_id, int(retired_match[0])
                )
                continue
            candidate.last_permanent_match_generation = (
                self.permanent_generation
            )
            if not self.permanent_capacity_available():
                break
            if self._promote_candidate(
                candidate.record_id, episode, after_growth=True
            ) is not None:
                promoted_to_new_pool += 1
        return promoted_to_new_pool

    def _record_evicted_candidate(
        self, record_id: int, episode: int
    ) -> None:
        record = self.records[int(record_id)]
        visits = int(record.candidate_visits)
        if (
            len(self.evicted_candidate_rows)
            >= self.evicted_candidate_history_capacity
        ):
            self.evicted_candidate_history_dropped += 1
        self.evicted_candidate_rows.append(
            {
                "eviction_episode": int(episode),
                "visit_count": visits,
                "first_episode": int(record.candidate_first_episode),
                "last_episode": int(record.candidate_last_episode),
                "age_episodes": int(
                    episode - record.candidate_first_episode
                ),
                "unique_actions_executed": int(
                    record.action_history_mask.bit_count()
                ),
                "near_promotion": bool(
                    visits >= self.candidate_promotion_visits - 1
                ),
                "recent_candidate": bool(
                    episode - record.candidate_first_episode
                    < self.candidate_recent_protection_episodes
                ),
                "unknown_mask": int(record.unknown_mask),
                "safe_mask": int(record.safe_mask),
                "blocked_mask": int(record.blocked_mask),
                "total_action_attempts": int(
                    np.sum(record.action_attempt_counts)
                ),
                "total_safe_outcomes": int(
                    np.sum(record.safe_outcome_counts)
                ),
                "total_collision_outcomes": int(
                    np.sum(record.collision_outcome_counts)
                ),
                "total_out_of_road_outcomes": int(
                    np.sum(record.out_of_road_outcome_counts)
                ),
                "total_warnings": int(np.sum(record.warning_counts)),
                "blocked_evidence_archived": bool(record.blocked_mask),
            }
        )

    def _evict_weakest_hazard(self) -> None:
        eligible = [
            record_id for record_id in self._ids(self.HAZARD)
            if not self._record_is_pending(record_id)
        ]
        if not eligible:
            raise RuntimeError(
                "Hazard capacity is full and every hazard has unresolved evidence."
            )
        if not self.status_ids[self.HAZARD]:
            return
        record_id = min(
            eligible,
            key=lambda hazard_id: (
                int(
                    np.sum(
                        self.records[hazard_id].collision_outcome_counts
                        + self.records[hazard_id].out_of_road_outcome_counts
                    )
                ),
                self.records[hazard_id].blocked_mask.bit_count(),
                self.records[hazard_id].hazard_last_hit_episode,
                self.records[hazard_id].hazard_archived_episode,
                int(np.sum(self.records[hazard_id].warning_counts)),
                hazard_id,
            ),
        )
        record = self.records[int(record_id)]
        self.hazard_blocked_bits_dropped += record.blocked_mask.bit_count()
        self._delete_hazard(record.record_id)
        self.hazard_records_evicted += 1

    def _archive_or_delete_candidate(
        self, record_id: int, episode: int
    ) -> Optional[int]:
        """Keep blocked candidate evidence in bounded lookup-only memory."""
        candidate = self.records[int(record_id)]
        if candidate.status != self.CANDIDATE:
            raise RuntimeError("Only a candidate can enter hazard memory.")
        if candidate.blocked_mask == 0:
            if self._record_is_pending(candidate.record_id):
                raise RuntimeError(
                    "Pending candidate reached an eviction path despite protection."
                )
            self._delete_candidate(candidate.record_id)
            return None

        hazard_match = self.find_hazard_match(
            candidate.state, candidate.safety
        )
        if hazard_match[0] is not None:
            hazard = self.records[int(hazard_match[0])]
            self._merge_hazard_danger_evidence(hazard, candidate)
            hazard.hazard_archived_episode = int(episode)
            hazard.hazard_archive_cycles += 1
            self.hazard_blocked_bits_preserved += (
                candidate.blocked_mask.bit_count()
            )
            self.hazard_records_merged += 1
            self._remap_pending_record(candidate.record_id, hazard.record_id)
            self._delete_candidate(candidate.record_id)
            return hazard.record_id

        if len(self.status_ids[self.HAZARD]) >= self.hazard_memory_capacity:
            self._evict_weakest_hazard()
        self.hazard_blocked_bits_preserved += candidate.blocked_mask.bit_count()
        candidate.action_attempt_counts = np.maximum(
            candidate.collision_outcome_counts,
            candidate.out_of_road_outcome_counts,
        ).astype(np.int64, copy=True)
        candidate.safe_outcome_counts.fill(0)
        self._reconcile_attempt_accounting(candidate)
        self._rebuild_outcome_masks(candidate)
        candidate.hazard_archived_episode = int(episode)
        candidate.hazard_last_hit_episode = -1
        candidate.hazard_hit_count = 0
        candidate.hazard_archive_cycles += 1
        self._set_status(candidate.record_id, self.HAZARD)
        self.hazard_records_archived += 1
        return candidate.record_id

    def _weakest_candidates(
        self,
        eligible: Sequence[int],
        protected: Sequence[int],
        removal_needed: int,
        limit: int,
    ) -> List[int]:
        def key(record_id: int) -> Tuple[int, int, int, int, int, int]:
            record = self.records[record_id]
            visits = int(record.candidate_visits)
            # Near-promotion records are excluded by the caller. Among the
            # remaining records, ordinary one-visit candidates are removed
            # before ordinary two-visit candidates; age then breaks ties.
            # Blocked evidence retains its existing protection.
            near_promotion = int(
                visits >= self.candidate_promotion_visits - 1
            )
            return (
                near_promotion,
                int(record.blocked_mask != 0),
                visits,
                int(record.candidate_last_episode),
                int(record.candidate_first_episode),
                int(record_id),
            )
        if len(eligible) >= removal_needed:
            return heapq.nsmallest(
                min(limit, len(eligible)), eligible, key=key
            )
        selected = heapq.nsmallest(len(eligible), eligible, key=key)
        selected.extend(
            heapq.nsmallest(
                min(limit - len(selected), len(protected)),
                protected,
                key=key,
            )
        )
        return selected

    def _batch_evict_candidates(self, episode: int) -> None:
        candidate_count = len(self.status_ids[self.CANDIDATE])
        if candidate_count < self.candidate_hard_limit:
            return
        removal_needed = max(0, candidate_count - self.max_candidates)
        if removal_needed == 0:
            return

        eligible, protected = [], []
        for record_id in self._ids(self.CANDIDATE):
            record = self.records[record_id]
            near = (
                record.candidate_visits
                >= self.candidate_promotion_visits - 1
            )
            recent = (
                episode - record.candidate_first_episode
                < self.candidate_recent_protection_episodes
            )
            blocked = bool(record.blocked_mask)
            pending = self._record_is_pending(record_id)
            if pending:
                self.pending_eviction_protections += 1
                continue
            # A candidate one visit from promotion is never an eviction
            # target. If these candidates fill the hard limit, process_state
            # uses fresh DQN without creating another candidate.
            if near:
                self.near_promotion_candidates_protected += 1
                continue
            if recent or blocked:
                protected.append(record_id)
                self.recent_candidates_protected += int(recent)
                self.blocked_candidates_protected += int(blocked)
            else:
                eligible.append(record_id)

        shortlist = self._weakest_candidates(
            eligible,
            protected,
            removal_needed,
            max(removal_needed, self.candidate_batch_evict_count),
        )
        for candidate_id in shortlist:
            candidate = self.records.get(candidate_id)
            if candidate is None or candidate.status != self.CANDIDATE:
                continue
            self.pre_eviction_candidates_checked += 1
            active_match = self.find_active_match(
                candidate.state, candidate.safety
            )
            if active_match[0] is not None:
                self.absorb_candidate_into_active(
                    candidate.record_id,
                    int(active_match[0]),
                    episode,
                    pre_eviction=True,
                )
                continue
            retired_match = self.find_retired_match(
                candidate.state, candidate.safety
            )
            if retired_match[0] is not None:
                self.absorb_candidate_into_retired(
                    candidate.record_id,
                    int(retired_match[0]),
                    pre_eviction=True,
                )
                continue
            fallback_status, fallback_match = self.find_capacity_fallback(
                candidate.state,
                candidate.safety,
                context="eviction",
            )
            if fallback_status == self.ACTIVE:
                self.absorb_candidate_into_active(
                    candidate.record_id,
                    int(fallback_match[0]),
                    episode,
                    pre_eviction=True,
                )
            elif fallback_status == self.RETIRED:
                self.absorb_candidate_into_retired(
                    candidate.record_id,
                    int(fallback_match[0]),
                    pre_eviction=True,
                )

        removal_needed = max(
            0, len(self.status_ids[self.CANDIDATE]) - self.max_candidates
        )
        if removal_needed == 0:
            return
        eligible, protected = [], []
        for record_id in self._ids(self.CANDIDATE):
            record = self.records[record_id]
            near = (
                record.candidate_visits
                >= self.candidate_promotion_visits - 1
            )
            recent = (
                episode - record.candidate_first_episode
                < self.candidate_recent_protection_episodes
            )
            blocked = bool(record.blocked_mask)
            if self._record_is_pending(record_id):
                self.pending_eviction_protections += 1
                continue
            if near:
                continue
            (protected if recent or blocked else eligible).append(record_id)
        to_remove = self._weakest_candidates(
            eligible, protected, removal_needed, removal_needed
        )
        for record_id in to_remove:
            if record_id not in self.records:
                continue
            self._record_evicted_candidate(record_id, episode)
            self._archive_or_delete_candidate(record_id, episode)
        self.candidates_evicted += len(to_remove)
