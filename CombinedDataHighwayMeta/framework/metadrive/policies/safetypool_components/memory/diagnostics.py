"""Pool, evidence, matching, and lifecycle diagnostic summaries."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .record import UnifiedStateRecord


class PoolDiagnosticsMixin:
    """Pool, evidence, matching, and lifecycle diagnostic summaries."""

    @staticmethod
    def _counts_text(counts: np.ndarray) -> str:
        return "|".join(str(int(value)) for value in counts)

    def _evidence_statistics(self, record: UnifiedStateRecord) -> Dict:
        return {
            "total_action_attempts": int(np.sum(record.action_attempt_counts)),
            "total_safe_outcomes": int(np.sum(record.safe_outcome_counts)),
            "total_collision_outcomes": int(
                np.sum(record.collision_outcome_counts)
            ),
            "total_out_of_road_outcomes": int(
                np.sum(record.out_of_road_outcome_counts)
            ),
            "total_warnings": int(np.sum(record.warning_counts)),
            "action_attempt_counts": self._counts_text(
                record.action_attempt_counts
            ),
            "safe_outcome_counts": self._counts_text(
                record.safe_outcome_counts
            ),
            "collision_outcome_counts": self._counts_text(
                record.collision_outcome_counts
            ),
            "out_of_road_outcome_counts": self._counts_text(
                record.out_of_road_outcome_counts
            ),
            "warning_counts": self._counts_text(record.warning_counts),
        }

    def pool_statistics(self):
        rows = []
        for record_id in self._ids(self.ACTIVE):
            record = self.records[record_id]
            remaining = record.availability_mask.bit_count()
            matches = record.match_count
            rows.append(
                {
                    "pool_id": int(record.record_id),
                    "status": record.status,
                    "promotion_evidence_visits": int(
                        record.promotion_evidence_visits
                    ),
                    "absorbed_candidate_visits": int(
                        record.absorbed_candidate_visits
                    ),
                    "absorbed_candidate_action_evidence": int(
                        record.absorbed_candidate_actions
                    ),
                    "active_pool_mask_visits": int(
                        record.active_mask_visits
                    ),
                    "matched_state_count": int(matches),
                    "exploration_pass": (
                        "INITIAL"
                        if record.exploration_pass == 0
                        else "UNKNOWN"
                    ),
                    "actions_tried": int(
                        record.action_history_mask.bit_count()
                    ),
                    "remaining_actions": int(remaining),
                    "coverage_percent": float(
                        100.0
                        * record.action_history_mask.bit_count()
                        / self.action_count
                    ),
                    "unknown_actions": int(record.unknown_mask.bit_count()),
                    "safe_actions": int(record.safe_mask.bit_count()),
                    "blocked_actions": int(record.blocked_mask.bit_count()),
                    "unknown_mask": int(record.unknown_mask),
                    "safe_mask": int(record.safe_mask),
                    "blocked_mask": int(record.blocked_mask),
                    **self._evidence_statistics(record),
                    "first_episode_created": int(
                        record.first_episode_created
                    ),
                    "last_episode_visited": int(
                        record.last_episode_visited
                    ),
                    "mean_general_cosine_similarity": (
                        record.similarity_sum / matches if matches else 0.0
                    ),
                    "mean_general_rms_distance": (
                        record.distance_sum / matches if matches else 0.0
                    ),
                    "mean_safety_cosine_similarity": (
                        record.safety_similarity_sum / matches
                        if matches else 0.0
                    ),
                    "mean_safety_rms_distance": (
                        record.safety_distance_sum / matches
                        if matches else 0.0
                    ),
                    "centroid_updates": int(record.centroid_updates),
                    "centroid_frozen_by_stability": bool(
                        record.centroid_frozen_by_stability
                    ),
                    "centroid_frozen_by_cap": bool(
                        record.centroid_frozen_by_cap
                    ),
                }
            )
        return rows

    def candidate_statistics(self):
        rows = []
        for record_id in self._ids(self.CANDIDATE):
            record = self.records[record_id]
            rows.append(
                {
                    "candidate_id": int(record.record_id),
                    "status": record.status,
                    "visit_count": int(record.candidate_visits),
                    "first_episode": int(record.candidate_first_episode),
                    "last_episode": int(record.candidate_last_episode),
                    "visits_remaining_for_promotion": max(
                        0,
                        self.candidate_promotion_visits
                        - record.candidate_visits,
                    ),
                    "unique_actions_executed": int(
                        record.action_history_mask.bit_count()
                    ),
                    "unknown_mask": int(record.unknown_mask),
                    "safe_mask": int(record.safe_mask),
                    "blocked_mask": int(record.blocked_mask),
                    **self._evidence_statistics(record),
                }
            )
        return rows

    def retired_pool_statistics(self):
        rows = []
        for record_id in self._ids(self.RETIRED):
            record = self.records[record_id]
            hits = record.retired_hit_count
            rows.append(
                {
                    "retired_pool_id": int(record.record_id),
                    "original_pool_id": int(record.record_id),
                    "status": record.status,
                    "retirement_reason": record.retirement_reason,
                    "episode_created": int(record.first_episode_created),
                    "episode_retired": int(record.episode_retired),
                    "permanent_pool_visits": int(
                        record.retired_permanent_visits
                    ),
                    "actions_explored": int(record.retired_actions_explored),
                    "unknown_actions": int(record.unknown_mask.bit_count()),
                    "safe_actions": int(record.safe_mask.bit_count()),
                    "blocked_actions": int(record.blocked_mask.bit_count()),
                    "unknown_mask": int(record.unknown_mask),
                    "safe_mask": int(record.safe_mask),
                    "blocked_mask": int(record.blocked_mask),
                    **self._evidence_statistics(record),
                    "hits_after_retirement": int(hits),
                    "last_hit_episode": int(record.retired_last_hit_episode),
                    "last_hit_step": int(record.retired_last_hit_step),
                    "mean_general_similarity": (
                        record.retired_similarity_sum / hits if hits else 0.0
                    ),
                    "mean_general_rms_distance": (
                        record.retired_distance_sum / hits if hits else 0.0
                    ),
                    "mean_safety_similarity": (
                        record.retired_safety_similarity_sum / hits
                        if hits else 0.0
                    ),
                    "mean_safety_rms_distance": (
                        record.retired_safety_distance_sum / hits
                        if hits else 0.0
                    ),
                    "retirement_trigger_action": int(
                        record.retirement_trigger_action
                    ),
                    "retirement_trigger_reward": math.nan,
                    "retirement_trigger_collision": bool(
                        record.retirement_trigger_collision
                    ),
                    "retirement_trigger_out_of_road": bool(
                        record.retirement_trigger_out_of_road
                    ),
                    "retirement_trigger_done": bool(
                        record.retirement_trigger_done
                    ),
                    "retirement_trigger_type": (
                        record.retirement_trigger_type
                    ),
                }
            )
        return rows

    def hazard_statistics(self):
        rows = []
        for record_id in self._ids(self.HAZARD):
            record = self.records[record_id]
            rows.append(
                {
                    "hazard_id": int(record.record_id),
                    "status": record.status,
                    "archived_episode": int(record.hazard_archived_episode),
                    "last_hit_episode": int(record.hazard_last_hit_episode),
                    "hit_count": int(record.hazard_hit_count),
                    "archive_cycles": int(record.hazard_archive_cycles),
                    "source_candidate_visits": int(record.candidate_visits),
                    "unknown_actions": int(record.unknown_mask.bit_count()),
                    "safe_actions": int(record.safe_mask.bit_count()),
                    "blocked_actions": int(record.blocked_mask.bit_count()),
                    "unknown_mask": int(record.unknown_mask),
                    "safe_mask": int(record.safe_mask),
                    "blocked_mask": int(record.blocked_mask),
                    **self._evidence_statistics(record),
                }
            )
        return rows

    def matching_rejection_statistics(self) -> List[Dict]:
        """Return constant-size audit rows for strict and fallback matching."""
        rows: List[Dict] = []
        for scope in ("train", "test"):
            for status in (
                self.CANDIDATE,
                self.ACTIVE,
                self.RETIRED,
                self.HAZARD,
            ):
                counters = self.match_rejection_counters[scope][status]
                if status == self.CANDIDATE:
                    general_cosine_threshold = (
                        self.candidate_similarity_threshold
                    )
                    general_rms_threshold = (
                        self.candidate_distance_threshold
                    )
                    safety_rms_threshold = (
                        self.candidate_safety_distance_threshold
                    )
                else:
                    general_cosine_threshold = self.similarity_threshold
                    general_rms_threshold = self.distance_threshold
                    safety_rms_threshold = self.safety_distance_threshold
                queries = int(counters["queries"])
                strict_matches = int(counters["strict_matches"])
                general_relaxation_matches = int(
                    counters["general_relaxation_matches"]
                )
                fallback_matches = int(counters["fallback_matches"])
                records_compared = int(counters["records_compared"])
                nonempty_queries = queries - int(counters["empty_queries"])
                rows.append(
                    {
                        "scope": scope,
                        "status": status,
                        **{key: int(value) for key, value in counters.items()},
                        "total_matches": (
                            strict_matches
                            + general_relaxation_matches
                            + fallback_matches
                        ),
                        "match_rate": float(
                            strict_matches
                            + general_relaxation_matches
                            + fallback_matches
                        ) / max(1, queries),
                        "fallback_acceptance_rate": float(
                            fallback_matches
                        ) / max(1, int(counters["fallback_attempts"])),
                        "mean_records_per_nonempty_query": float(
                            records_compared
                        ) / max(1, nonempty_queries),
                        "close_enough_fallback_enabled": bool(
                            self.close_enough_fallback
                        ),
                        "fallback_scope": (
                            "candidate_hard_limit_permanent_full_or_eviction"
                        ),
                        "general_threshold_variation": (
                            self.capacity_fallback_general_variation
                        ),
                        "general_cosine_relaxation": (
                            self.general_cosine_relaxation
                        ),
                        "general_rms_relaxation": (
                            self.general_rms_relaxation
                        ),
                        "safety_directional_improvement": (
                            self.capacity_fallback_safety_improvement
                        ),
                        "general_cosine_threshold": (
                            general_cosine_threshold
                        ),
                        "general_rms_threshold": general_rms_threshold,
                        "safety_cosine_threshold": (
                            self.safety_similarity_threshold
                        ),
                        "safety_rms_threshold": safety_rms_threshold,
                        "general_gates_relaxable": True,
                        "pressure_fallback_eligible": status in {
                            self.ACTIVE, self.RETIRED
                        },
                        "safety_gates_relaxable": False,
                    }
                )
        return rows

    def matching_diagnostic_totals(self) -> Dict[str, int]:
        """Aggregate the fixed train/test and status diagnostic table."""
        fields = next(
            iter(self.match_rejection_counters["train"].values())
        ).keys()
        return {
            field: int(
                sum(
                    counters[field]
                    for scope_rows in self.match_rejection_counters.values()
                    for counters in scope_rows.values()
                )
            )
            for field in fields
        }

    def global_statistics(self):
        remaining_candidates = len(self.status_ids[self.CANDIDATE])
        matching_totals = self.matching_diagnostic_totals()
        accounted = (
            self.candidates_promoted
            + self.candidates_merged_into_active_pool
            + self.candidates_evicted
            + self.candidates_blocked_by_capacity
            + self.candidates_suppressed_by_retired_pool
            + remaining_candidates
        )
        pressure = self.candidate_capacity_wait_events / max(
            1,
            self.candidates_promoted + self.candidate_capacity_wait_events,
        )
        return {
            "storage_model": "unified_status_tagged_records",
            "matching_backend": (
                "preallocated_vectorized_strict_plus_general_relaxation_linear"
            ),
            "general_cosine_relaxation": self.general_cosine_relaxation,
            "general_rms_relaxation": self.general_rms_relaxation,
            "close_enough_fallback_enabled": self.close_enough_fallback,
            "capacity_fallback_scope": (
                "candidate_hard_limit_permanent_full_or_eviction"
            ),
            "capacity_fallback_general_variation": (
                self.capacity_fallback_general_variation
            ),
            "capacity_fallback_general_cosine_relaxation": (
                self.general_cosine_relaxation
            ),
            "capacity_fallback_general_rms_relaxation": (
                self.capacity_fallback_general_variation
            ),
            "capacity_fallback_safety_improvement": (
                self.capacity_fallback_safety_improvement
            ),
            "capacity_fallback_directional_features": (
                "lane_clearance:+|threat_distance:+|"
                "lane_offset:-|heading_error:-"
            ),
            "capacity_fallback_speed_directional": False,
            "close_enough_safety_gates_relaxed": False,
            "candidate_full_fallback_queries": (
                self.candidate_full_fallback_queries
            ),
            "candidate_full_fallback_matches": (
                self.candidate_full_fallback_matches
            ),
            "permanent_full_fallback_queries": (
                self.permanent_full_fallback_queries
            ),
            "permanent_full_fallback_matches": (
                self.permanent_full_fallback_matches
            ),
            "eviction_fallback_queries": self.eviction_fallback_queries,
            "eviction_fallback_matches": self.eviction_fallback_matches,
            "capacity_fallback_active_matches": (
                self.capacity_fallback_active_matches
            ),
            "capacity_fallback_retired_matches": (
                self.capacity_fallback_retired_matches
            ),
            "matching_queries": matching_totals["queries"],
            "matching_records_compared": matching_totals[
                "records_compared"
            ],
            "strict_matching_hits": matching_totals["strict_matches"],
            "general_relaxation_attempts": matching_totals[
                "general_relaxation_attempts"
            ],
            "general_relaxation_hits": matching_totals[
                "general_relaxation_matches"
            ],
            "close_enough_fallback_attempts": (
                self.candidate_full_fallback_queries
                + self.permanent_full_fallback_queries
                + self.eviction_fallback_queries
            ),
            # Context counters count the one selected ACTIVE/RETIRED match.
            # Per-status diagnostics can record two eligible matches for one
            # query and therefore are not an accepted-hit counter.
            "close_enough_fallback_hits": (
                self.candidate_full_fallback_matches
                + self.permanent_full_fallback_matches
                + self.eviction_fallback_matches
            ),
            "general_cosine_rejected_records": matching_totals[
                "general_cosine_failed_records"
            ],
            "general_rms_rejected_records": matching_totals[
                "general_rms_failed_records"
            ],
            "safety_cosine_rejected_records": matching_totals[
                "safety_cosine_failed_records"
            ],
            "safety_rms_rejected_records": matching_totals[
                "safety_rms_failed_records"
            ],
            "safety_direction_rejected_records": matching_totals[
                "safety_direction_failed_records"
            ],
            "safety_worsened_records": matching_totals[
                "safety_worsened_records"
            ],
            "safety_without_10_percent_improvement_records": matching_totals[
                "safety_no_10_percent_improvement_records"
            ],
            "pool_policy_frozen": bool(self.policy_frozen),
            "pool_policy_frozen_at_episode": self.policy_frozen_at_episode,
            "pool_policy_freeze_events": self.policy_freeze_events,
            "calibration_incomplete_at_freeze": (
                self.calibration_incomplete_at_freeze
            ),
            "final_phase_pool_mutations_permitted": False,
            "final_phase_read_only_safety_matching": False,
            "final_phase_uses_safety_extraction": False,
            "final_phase_uses_safety_masks": False,
            "final_phase_pure_dqn_argmax": True,
            "final_phase_frozen_safety_states": (
                self.total_final_frozen_safety_states
            ),
            "final_phase_pure_dqn_states": (
                self.total_final_pure_dqn_states
            ),
            "missing_safety_pure_dqn_states": (
                self.missing_safety_pure_dqn_states
            ),
            "missing_safety_behavior": (
                "skip_pool_lookup_and_safe_evidence_use_pure_dqn"
            ),
            "preallocated_centroid_slots": self.slot_capacity,
            "occupied_centroid_slots": len(self.record_slots),
            "candidate_status_tag": self.CANDIDATE,
            "active_status_tag": self.ACTIVE,
            "retired_status_tag": self.RETIRED,
            "hazard_status_tag": self.HAZARD,
            "hazard_memory_capacity": self.hazard_memory_capacity,
            "hazard_records_at_end": len(self.status_ids[self.HAZARD]),
            "safe_confirmation_visits": self.safe_confirmation_visits,
            "safety_horizon_steps": self.safety_horizon_steps,
            "warning_block_threshold": self.warning_block_threshold,
            "candidate_promotion_visits": self.candidate_promotion_visits,
            "calibration_target_states": self.calibration_state_count,
            "calibration_states_collected": self.total_calibration_states,
            "initial_pool_capacity": self.initial_max_pools,
            "current_pool_capacity": self.max_pools,
            "maximum_pool_capacity": self.maximum_pool_capacity,
            "capacity_growth_events": len(self.capacity_growth_rows),
            "new_pool_capacity_pressure": float(pressure),
            "active_permanent_pools": len(self.status_ids[self.ACTIVE]),
            "retired_permanent_pools": len(self.status_ids[self.RETIRED]),
            "combined_permanent_records": self.total_permanent_records(),
            "combined_capacity_invariant_holds": (
                self.total_permanent_records() <= self.max_pools
            ),
            "candidates_created": self.candidates_created,
            "candidates_promoted_to_new_pool": self.candidates_promoted,
            "candidates_merged_into_active_pool": (
                self.candidates_merged_into_active_pool
            ),
            "candidate_action_history_transfers": (
                self.candidate_action_history_transfers
            ),
            "candidate_action_bits_removed_by_transfer": (
                self.candidate_action_bits_removed_by_transfer
            ),
            "candidates_evicted": self.candidates_evicted,
            "evicted_candidate_history_capacity": (
                self.evicted_candidate_history_capacity
            ),
            "evicted_candidate_history_rows_retained": len(
                self.evicted_candidate_rows
            ),
            "evicted_candidate_history_rows_dropped": (
                self.evicted_candidate_history_dropped
            ),
            "candidates_blocked_by_permanent_capacity": (
                self.candidates_blocked_by_capacity
            ),
            "candidate_capacity_wait_events": (
                self.candidate_capacity_wait_events
            ),
            "promotion_ready_candidates_waiting": len(
                self.ready_waiting_ids
            ),
            "candidates_promoted_after_capacity_growth": (
                self.candidates_promoted_after_capacity_growth
            ),
            "absolute_capacity_argmax_states": (
                self.absolute_capacity_argmax_states
            ),
            "candidates_suppressed_by_retired_pool": (
                self.candidates_suppressed_by_retired_pool
            ),
            "candidate_retired_suppression_events": (
                self.candidate_retired_suppression_events
            ),
            "candidate_history_exhaustion_events": (
                self.candidate_history_exhaustion_events
            ),
            "candidates_remaining_at_end": remaining_candidates,
            "candidate_to_active_match_ratio": (
                float(self.total_pool_matches)
                / max(1, self.candidates_created)
            ),
            "candidate_eviction_fraction": (
                float(self.candidates_evicted)
                / max(1, self.candidates_created)
            ),
            "candidate_accounting_total": int(accounted),
            "candidate_accounting_matches_created": (
                accounted == self.candidates_created
            ),
            "near_promotion_protection_occurrences": (
                self.near_promotion_candidates_protected
            ),
            "recent_candidate_protection_occurrences": (
                self.recent_candidates_protected
            ),
            "blocked_candidate_protection_occurrences": (
                self.blocked_candidates_protected
            ),
            "hazard_records_archived": self.hazard_records_archived,
            "hazard_records_merged": self.hazard_records_merged,
            "hazard_records_evicted": self.hazard_records_evicted,
            "hazard_pool_queries": self.hazard_queries,
            "states_matched_to_hazard_memory": self.hazard_matches,
            "hazard_match_rate": (
                float(self.hazard_matches) / max(1, self.hazard_queries)
            ),
            "hazard_blocked_actions_transferred": (
                self.hazard_actions_transferred
            ),
            "hazard_actions_filtered": self.hazard_actions_filtered,
            "test_hazard_queries": self.test_hazard_queries,
            "test_hazard_matches": self.test_hazard_matches,
            "test_hazard_match_rate": (
                float(self.test_hazard_matches)
                / max(1, self.test_hazard_queries)
            ),
            "test_hazard_actions_filtered": (
                self.test_hazard_actions_filtered
            ),
            "hazard_blocked_bits_preserved": (
                self.hazard_blocked_bits_preserved
            ),
            "hazard_blocked_bits_dropped": self.hazard_blocked_bits_dropped,
            "states_matched_to_active_pools": self.total_pool_matches,
            "states_matched_to_retired_pools": self.total_retired_pool_hits,
            "retired_mask_filtered_actions": (
                self.retired_mask_filtered_actions
            ),
            "post_first_pass_greedy_probability": (
                self.POST_FIRST_PASS_GREEDY_PROBABILITY
            ),
            "post_first_pass_greedy_selections": (
                self.post_first_pass_greedy_selections
            ),
            "post_first_pass_random_selections": (
                self.post_first_pass_random_selections
            ),
            "post_first_pass_all_blocked_rejections": (
                self.post_first_pass_all_blocked_rejections
            ),
            "post_first_pass_blocked_actions_eligible": False,
            "blocked_actions_skipped_from_scheduled_passes": (
                self.blocked_actions_skipped_from_scheduled_passes
            ),
            "hard_blocked_empty_selection_rejections": (
                self.hard_blocked_empty_selection_rejections
            ),
            "least_risk_fallbacks": self.least_risk_fallbacks,
            "initial_passes_completed": self.initial_passes_completed,
            "unknown_passes_completed": self.unknown_passes_completed,
            "scheduled_unknown_pass_count": 1,
            "retired_evidence_writable_after_scheduled_passes": True,
            "availability_mask_refilled_for_unknown_pass": True,
            "retired_direct_evidence_updates": (
                self.retired_direct_evidence_updates
            ),
            "retired_candidate_evidence_merges": (
                self.retired_candidate_evidence_merges
            ),
            "active_mask_filtered_actions": (
                self.active_mask_filtered_actions
            ),
            "candidate_mask_filtered_actions": (
                self.candidate_mask_filtered_actions
            ),
            "hazard_mask_filtered_actions": (
                self.hazard_mask_filtered_actions
            ),
            "safe_action_outcomes": self.safe_action_outcomes,
            "minimum_progress_reward": self.minimum_progress_reward,
            "no_progress_safe_rejections": (
                self.no_progress_safe_rejections
            ),
            "blocked_action_outcomes": self.blocked_action_outcomes,
            "delayed_safe_confirmations": self.delayed_safe_confirmations,
            "precursor_warnings_recorded": self.precursor_warnings_recorded,
            "precursor_warning_blocks": self.precursor_warning_blocks,
            "attempt_accounting_repair_events": (
                self.attempt_accounting_repair_events
            ),
            "attempt_accounting_repaired_attempts": (
                self.attempt_accounting_repaired_attempts
            ),
            "pending_eviction_protections": self.pending_eviction_protections,
            "direct_mask_exhaustion_retirements": (
                self.direct_mask_exhaustion_retirements
            ),
            "candidate_history_exhaustion_retirements": (
                self.candidate_history_exhaustion_retirements
            ),
            "retired_due_to_complete_action_coverage": (
                self.active_pools_retired_by_mask_exhaustion
            ),
            "general_similarity_threshold": self.similarity_threshold,
            "general_rms_threshold": self.distance_threshold,
            "safety_similarity_threshold": self.safety_similarity_threshold,
            "safety_rms_threshold": self.safety_distance_threshold,
            **{
                f"general_{key}": value
                for key, value in self.normalizer.statistics().items()
            },
            **{
                f"safety_{key}": value
                for key, value in self.safety_normalizer.statistics().items()
            },
        }
