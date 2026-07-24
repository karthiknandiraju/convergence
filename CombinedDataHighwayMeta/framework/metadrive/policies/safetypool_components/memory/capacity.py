"""Permanent capacity review, promotion, and retirement transitions."""

from __future__ import annotations

import math
from typing import Optional


class PoolCapacityMixin:
    """Permanent capacity review, promotion, and retirement transitions."""

    def total_permanent_records(self) -> int:
        return len(self.status_ids[self.ACTIVE]) + len(
            self.status_ids[self.RETIRED]
        )

    def permanent_capacity_available(self) -> bool:
        return self.total_permanent_records() < self.max_pools

    def absolute_permanent_capacity_reached(self) -> bool:
        return self.total_permanent_records() >= self.maximum_pool_capacity

    def review_capacity(self, episode: int) -> None:
        self._require_policy_mutable("capacity review")
        if (
            episode <= 0
            or episode % self.capacity_review_interval != 0
        ):
            return
        self.validate_invariants()
        if self.max_pools >= self.maximum_pool_capacity:
            return
        promoted_delta = self.candidates_promoted - self.last_review_promoted
        waiting_delta = (
            self.candidate_capacity_wait_events
            - self.last_review_capacity_wait_events
        )
        pressure = waiting_delta / max(1, promoted_delta + waiting_delta)
        has_ready_waiter = bool(self.ready_waiting_ids)
        if (
            self.total_permanent_records() >= self.max_pools
            and has_ready_waiter
        ):
            pressure = 1.0

        old_capacity = self.max_pools
        if pressure > 0.15:
            self.max_pools = min(
                self.maximum_pool_capacity,
                int(math.ceil(self.max_pools * 1.50)),
            )
        elif pressure >= 0.05:
            self.max_pools = min(
                self.maximum_pool_capacity,
                int(math.ceil(self.max_pools * 1.20)),
            )

        if self.max_pools != old_capacity:
            promoted_after_growth = self._promote_waiting_candidates_after_growth(
                episode
            )
            self.capacity_growth_rows.append(
                {
                    "episode": int(episode),
                    "capacity_before": int(old_capacity),
                    "capacity_after": int(self.max_pools),
                    "candidate_soft_before": int(self.max_candidates),
                    "candidate_soft_after": int(self.max_candidates),
                    "candidate_hard_before": int(self.candidate_hard_limit),
                    "candidate_hard_after": int(self.candidate_hard_limit),
                    "new_pool_capacity_pressure": float(pressure),
                    "promoted_since_last_review": int(promoted_delta),
                    "waiting_events_since_last_review": int(waiting_delta),
                    "waiting_candidates_promoted_after_growth": int(
                        promoted_after_growth
                    ),
                }
            )
        self.last_review_promoted = self.candidates_promoted
        self.last_review_capacity_wait_events = (
            self.candidate_capacity_wait_events
        )
        self.validate_invariants()

    def _promote_candidate(
        self, record_id: int, episode: int, after_growth: bool = False
    ) -> Optional[int]:
        record = self.records[int(record_id)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Only a candidate can be promoted.")
        if not self.permanent_capacity_available():
            return None

        record.promotion_evidence_visits = int(record.candidate_visits)
        record.availability_mask = (
            self.full_mask & ~record.action_history_mask
        )
        record.first_episode_created = int(episode)
        record.last_episode_visited = int(episode)
        self._set_status(record.record_id, self.ACTIVE)
        self.candidates_promoted += 1
        self.total_pool_creations += 1
        self.creation_events.append(
            (int(episode), int(self.total_pool_creations))
        )
        if after_growth:
            self.candidates_promoted_after_capacity_growth += 1

        if record.availability_mask == 0:
            self.candidate_history_exhaustion_events += 1
            self._complete_active_scheduled_pass(
                record.record_id,
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
                record.record_id
                if record.status == self.ACTIVE
                else None
            )
        return record.record_id

    def _complete_active_scheduled_pass(
        self,
        record_id: int,
        episode: int,
        trigger=None,
        externally_blocked_mask: int = 0,
    ) -> int:
        """Advance INITIAL -> UNKNOWN, then retire after UNKNOWN coverage."""
        record = self.records[int(record_id)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Only an active pool can complete a pass.")
        trigger = trigger or {}
        if record.exploration_pass == 0:
            self.initial_passes_completed += 1
            record.exploration_pass = 1
            blocked_mask = (
                int(record.blocked_mask)
                | int(externally_blocked_mask)
            ) & self.full_mask
            record.availability_mask = (
                int(record.unknown_mask) & ~blocked_mask & self.full_mask
            )
            self.blocked_actions_skipped_from_scheduled_passes += (
                int(record.unknown_mask) & blocked_mask
            ).bit_count()
            if record.availability_mask != 0:
                return record.record_id
            # No UNKNOWN action remains after the initial pass.
        elif record.exploration_pass == 1:
            self.unknown_passes_completed += 1
        else:
            raise RuntimeError("Active pool has an invalid exploration pass.")

        record.availability_mask = 0
        record.retirement_pending = False
        return self._retire_active_pool(
            record.record_id,
            episode,
            (
                "UNKNOWN_PASS_EXHAUSTED"
                if record.exploration_pass == 1
                else "MASK_EXHAUSTED"
            ),
            trigger,
        )

    def _retire_active_pool(
        self, record_id: int, episode: int, reason: str, trigger=None
    ) -> int:
        record = self.records[int(record_id)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Only an active pool can be retired.")
        trigger = trigger or {}
        record.retirement_reason = str(reason)
        record.episode_retired = int(episode)
        record.retired_permanent_visits = int(record.active_mask_visits)
        record.retired_actions_explored = int(
            record.action_history_mask.bit_count()
        )
        record.retirement_trigger_action = int(trigger.get("action", -1))
        record.retirement_trigger_collision = bool(
            trigger.get("collision", False)
        )
        record.retirement_trigger_out_of_road = bool(
            trigger.get("out_of_road", False)
        )
        record.retirement_trigger_done = bool(trigger.get("done", False))
        record.retirement_trigger_step = int(trigger.get("step", -1))
        record.retirement_trigger_type = str(
            trigger.get("retirement_trigger_type", "direct_final_action")
        )
        record.retirement_pending = False
        self._set_status(record.record_id, self.RETIRED)
        if reason in {"MASK_EXHAUSTED", "UNKNOWN_PASS_EXHAUSTED"}:
            self.direct_mask_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        elif reason == "CANDIDATE_HISTORY_EXHAUSTED":
            self.candidate_history_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        return record.record_id

    def _record_attempt(self, record_id: int, action: int) -> None:
        record = self.records.get(int(record_id))
        if record is None:
            return
        record.action_attempt_counts[int(action)] += 1
