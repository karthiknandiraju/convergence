"""Immediate, delayed, warning, and interrupted outcome processing."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


class PoolOutcomeMixin:
    """Immediate, delayed, warning, and interrupted outcome processing."""

    def _record_direct_failure(
        self, record_id: int, action: int, collision: bool, out_of_road: bool
    ) -> None:
        record = self.records.get(int(record_id))
        if record is None:
            return
        action = int(action)
        record.collision_outcome_counts[action] += int(collision)
        record.out_of_road_outcome_counts[action] += int(out_of_road)
        self._reconcile_attempt_accounting(record)
        self._rebuild_outcome_masks(record)

    def _record_delayed_safe(self, record_id: int, action: int) -> None:
        record = self.records.get(int(record_id))
        if record is None:
            return
        # Hazard memory is deliberately danger-only.  A lifecycle remap must
        # never turn lookup-only hazard evidence into a SAFE classification.
        if record.status == self.HAZARD:
            return
        action = int(action)
        before = bool(record.safe_mask & (1 << action))
        record.safe_outcome_counts[action] += 1
        self.safe_action_outcomes += 1
        self._reconcile_attempt_accounting(record)
        self._rebuild_outcome_masks(record)
        if not before and record.safe_mask & (1 << action):
            self.delayed_safe_confirmations += 1

    def _record_warning(self, record_id: int, action: int) -> None:
        record = self.records.get(int(record_id))
        if record is None:
            return
        action = int(action)
        before = bool(record.blocked_mask & (1 << action))
        record.warning_counts[action] += 1
        self._rebuild_outcome_masks(record)
        if not before and record.blocked_mask & (1 << action):
            self.precursor_warning_blocks += 1

    def _upsert_hazard_evidence(
        self,
        state: np.ndarray,
        safety: np.ndarray,
        episode: int,
        action: int,
        collision: bool = False,
        out_of_road: bool = False,
        warning: bool = False,
    ) -> int:
        """Insert or update one bounded hazard record in O(H(D+S))."""
        match = self.find_hazard_match(state, safety)
        if match[0] is None:
            if len(self.status_ids[self.HAZARD]) >= self.hazard_memory_capacity:
                self._evict_weakest_hazard()
            hazard_id = self._new_hazard(state, safety, episode)
        else:
            hazard_id = int(match[0])
            hazard = self.records[hazard_id]
            hazard.hazard_archived_episode = int(episode)
            hazard.hazard_archive_cycles += 1
            self.hazard_records_merged += 1
        hazard = self.records[hazard_id]
        action = int(action)
        before = int(hazard.blocked_mask)
        hazard.action_attempt_counts[action] += int(collision or out_of_road)
        hazard.collision_outcome_counts[action] += int(collision)
        hazard.out_of_road_outcome_counts[action] += int(out_of_road)
        hazard.warning_counts[action] += int(warning)
        self._reconcile_attempt_accounting(hazard)
        self._rebuild_outcome_masks(hazard)
        if (
            warning
            and not (before & (1 << action))
            and hazard.blocked_mask & (1 << action)
        ):
            self.precursor_warning_blocks += 1
        self.hazard_blocked_bits_preserved += (
            hazard.blocked_mask & ~before
        ).bit_count()
        return hazard_id

    def mark_pending_action_outcome(
        self,
        record_ids: Sequence[int],
        state: np.ndarray,
        safety: Optional[np.ndarray],
        episode: int,
        step: int,
        action: int,
        retire_record_id: Optional[int] = None,
    ) -> None:
        if self.pending_action_outcome is not None:
            raise RuntimeError("A prior pool action outcome is still pending.")
        self.pending_action_outcome = {
            "record_ids": tuple(dict.fromkeys(int(v) for v in record_ids)),
            "state": np.asarray(state, dtype=self.storage_dtype).copy(),
            "safety": (
                None
                if safety is None
                else np.asarray(safety, dtype=self.storage_dtype).copy()
            ),
            "safety_valid": safety is not None,
            "episode": int(episode),
            "step": int(step),
            "action": int(action),
            "retire_record_id": (
                None if retire_record_id is None else int(retire_record_id)
            ),
        }

    def _resolve_interrupted_retirements(
        self, items: Iterable[Dict], episode: int, reason: str
    ) -> None:
        """Restore an unconfirmed final UNKNOWN action in bounded O(H) time."""
        for item in items:
            if not item.get("defer_second_pass_retirement", False):
                continue
            record_id = item.get("retire_record_id")
            record = self.records.get(record_id)
            if (
                record is None
                or record.status != self.ACTIVE
                or not record.retirement_pending
            ):
                continue
            record.retirement_pending = False
            action = int(item["action"])
            bit = 1 << action
            if record.unknown_mask & bit:
                record.availability_mask |= bit
                continue
            self._complete_active_scheduled_pass(
                record.record_id,
                int(episode),
                {
                    **item,
                    "retirement_trigger_type": str(reason),
                },
            )

    def _complete_matured_second_pass(self, matured: Dict) -> None:
        if not matured.get("defer_second_pass_retirement", False):
            return
        record_id = matured.get("retire_record_id")
        record = self.records.get(record_id)
        if (
            record is None
            or record.status != self.ACTIVE
            or not record.retirement_pending
        ):
            return
        record.retirement_pending = False
        self._complete_active_scheduled_pass(
            record.record_id,
            int(matured["episode"]),
            {
                **matured,
                "collision": False,
                "out_of_road": False,
                "done": False,
                "retirement_trigger_type": "unknown_pass_final_outcome_matured",
            },
        )

    def finalize_pending_action_outcome(self, reward, parsed, done) -> None:
        self._require_policy_mutable("delayed action-outcome update")
        if self.pending_action_outcome is None:
            return
        pending = self.pending_action_outcome
        self.pending_action_outcome = None
        pending["progress_reward"] = float(reward)
        collision = bool(parsed.get("collision", False))
        out_of_road = bool(parsed.get("out_of_road", False))
        unsafe = collision or out_of_road
        action = int(pending["action"])
        record_ids = tuple(
            record_id
            for record_id in pending["record_ids"]
            if record_id in self.records
        )
        for record_id in record_ids:
            status = self.records[record_id].status
            if status == self.RETIRED:
                # Retirement closes scheduled pass availability only.
                # Outcome evidence remains writable for later matches.
                self.retired_direct_evidence_updates += 1
            if status != self.HAZARD:
                self._record_attempt(record_id, action)
        retire_record_id = pending["retire_record_id"]
        retire_record = self.records.get(retire_record_id)
        defer_second_pass_retirement = bool(
            not unsafe
            and retire_record_id is not None
            and retire_record is not None
            and retire_record.status == self.ACTIVE
            and retire_record.exploration_pass == 1
        )
        pending["defer_second_pass_retirement"] = (
            defer_second_pass_retirement
        )
        if defer_second_pass_retirement:
            retire_record.retirement_pending = True
        if unsafe:
            # Count the observed unsafe transition once, regardless of how
            # many matching local/hazard records receive its evidence.
            self.blocked_action_outcomes += 1
            for record_id in record_ids:
                if self.records[record_id].status != self.HAZARD:
                    self._record_direct_failure(
                        record_id, action, collision, out_of_road
                    )
            if pending["safety_valid"]:
                self._upsert_hazard_evidence(
                    pending["state"], pending["safety"], pending["episode"],
                    action, collision=collision, out_of_road=out_of_road,
                )
            # The current action is directly blocked. Only the preceding
            # bounded window receives suspicion warnings.
            for previous in self.recent_safety_actions:
                self.precursor_warnings_recorded += 1
                previous_action = int(previous["action"])
                for record_id in previous["record_ids"]:
                    if (
                        record_id in self.records
                        and self.records[record_id].status != self.HAZARD
                    ):
                        self._record_warning(record_id, previous_action)
                if previous.get("safety_valid", True):
                    self._upsert_hazard_evidence(
                        previous["state"], previous["safety"],
                        pending["episode"], previous_action, warning=True,
                    )
            self._resolve_interrupted_retirements(
                self.recent_safety_actions,
                int(pending["episode"]),
                "unknown_pass_outcome_interrupted_by_failure",
            )
            self.recent_safety_actions.clear()
        else:
            # Hazard records are updated through _upsert_hazard_evidence on a
            # failure. On a clean transition their attempt must be counted
            # here so a later delayed-safe outcome preserves the invariant
            # safe_count <= attempt_count.
            for record_id in record_ids:
                if self.records[record_id].status == self.HAZARD:
                    self._record_attempt(record_id, action)
            self.recent_safety_actions.append(pending)
            if len(self.recent_safety_actions) >= self.safety_horizon_steps:
                matured = self.recent_safety_actions.popleft()
                matured_record_ids = [
                    record_id
                    for record_id in matured["record_ids"]
                    if record_id in self.records
                ]
                progress_reward = float(matured["progress_reward"])
                progress_qualified = (
                    math.isfinite(progress_reward)
                    and progress_reward >= self.minimum_progress_reward
                )
                if progress_qualified:
                    for record_id in matured_record_ids:
                        self._record_delayed_safe(
                            record_id, int(matured["action"])
                        )
                else:
                    # A clean but non-progressing action is neither SAFE nor
                    # BLOCKED. It remains UNKNOWN and can collect more evidence.
                    self.no_progress_safe_rejections += len(
                        matured_record_ids
                    )
                self._complete_matured_second_pass(matured)

        record = self.records.get(retire_record_id)
        if (
            retire_record_id is not None
            and record is not None
            and record.status == self.ACTIVE
            and not defer_second_pass_retirement
        ):
            self._complete_active_scheduled_pass(
                record.record_id,
                pending["episode"],
                {
                    **pending,
                    "collision": bool(parsed.get("collision", False)),
                    "out_of_road": bool(parsed.get("out_of_road", False)),
                    "done": bool(done),
                    "retirement_trigger_type": "direct_final_action",
                },
            )

        if done and not unsafe:
            # A final UNKNOWN-pass action that did not receive its full future
            # horizon is restored for one later verified attempt.
            self._resolve_interrupted_retirements(
                self.recent_safety_actions,
                int(pending["episode"]),
                "unknown_pass_outcome_censored_at_episode_end",
            )
            self.recent_safety_actions.clear()

    def end_episode_safety_window(self, episode: Optional[int] = None) -> None:
        if self.pending_action_outcome is not None:
            raise RuntimeError("Episode ended with an unresolved immediate outcome.")
        self._resolve_interrupted_retirements(
            self.recent_safety_actions,
            int(episode) if episode is not None else -1,
            "unknown_pass_outcome_censored_at_episode_end",
        )
        self.recent_safety_actions.clear()

    def finalize_pending_retirement(self, reward, parsed, done) -> None:
        """Compatibility alias for older training-loop integrations."""
        self.finalize_pending_action_outcome(reward, parsed, done)
