"""Candidate, active, retired, and hazard action-selection behavior."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .record import UnifiedStateRecord


class PoolActionSelectionMixin:
    """Candidate, active, retired, and hazard action-selection behavior."""

    def record_candidate_action(self, candidate_index, action) -> None:
        record = self.records[int(candidate_index)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Candidate action targeted a non-candidate.")
        record.action_history_mask |= 1 << int(action)

    def record_retired_hit(
        self,
        retired_index,
        episode,
        step,
        similarity,
        distance,
        safety_similarity,
        safety_distance,
    ) -> None:
        self._require_policy_mutable("retired-pool hit update")
        record = self.records[int(retired_index)]
        if record.status != self.RETIRED:
            raise RuntimeError("Retired hit targeted a non-retired record.")
        record.retired_hit_count += 1
        record.retired_last_hit_episode = int(episode)
        record.retired_last_hit_step = int(step)
        record.retired_similarity_sum += float(similarity)
        record.retired_distance_sum += float(distance)
        record.retired_safety_similarity_sum += float(safety_similarity)
        record.retired_safety_distance_sum += float(safety_distance)
        self.total_retired_pool_hits += 1

    def record_retired_state_hit(
        self, retired_index, state, safety, episode, step
    ) -> None:
        record = self.records[int(retired_index)]
        state = np.asarray(state, dtype=np.float32)
        safety = np.asarray(safety, dtype=np.float32)
        self.record_retired_hit(
            retired_index,
            episode,
            step,
            self._cosine(
                state,
                float(np.linalg.norm(state)),
                record.state,
                record.state_norm,
            ),
            self._rms(state, record.state),
            self._cosine(
                safety,
                float(np.linalg.norm(safety)),
                record.safety,
                record.safety_norm,
            ),
            self._rms(safety, record.safety),
        )

    @staticmethod
    def _best_action_from_mask(
        q_values: np.ndarray, mask: int, action_count: int, key: str
    ) -> Optional[int]:
        maximum = float("-inf")
        candidates: List[int] = []
        for action in range(action_count):
            if not (mask & (1 << action)):
                continue
            value = float(q_values[action])
            if value > maximum:
                maximum = value
                candidates = [action]
            elif value == maximum:
                candidates.append(action)
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % len(candidates)
        return int(candidates[offset])

    def _uniform_action_from_mask(self, mask: int, digest: bytes) -> int:
        """Select one set bit uniformly; A=9 makes this constant-time."""
        count = int(mask).bit_count()
        if count <= 0:
            raise RuntimeError("Random action selection received an empty mask.")
        target = int.from_bytes(digest[8:16], "big") % count
        for action in range(self.action_count):
            if not (mask & (1 << action)):
                continue
            if target == 0:
                return int(action)
            target -= 1
        raise RuntimeError("Random action selection could not resolve a set bit.")

    @staticmethod
    def _combined_blocked_mask(
        local: Optional[UnifiedStateRecord],
        hazard: Optional[UnifiedStateRecord],
    ) -> int:
        blocked_mask = 0
        if local is not None:
            blocked_mask |= int(local.blocked_mask)
        if hazard is not None:
            blocked_mask |= int(hazard.blocked_mask)
        return int(blocked_mask)

    def _inherit_hazard_blocked_evidence(
        self,
        local: Optional[UnifiedStateRecord],
        hazard: Optional[UnifiedStateRecord],
    ) -> int:
        """Permanently copy matched hazard danger evidence into a local pool.

        Copying the evidence counts, rather than only OR-ing a mask bit, makes
        the inherited BLOCKED result survive every later outcome-mask rebuild.
        The loop is over the fixed nine-action space and is therefore O(1).
        """
        if local is None or hazard is None:
            return 0
        hazard_bits = int(hazard.blocked_mask) & self.full_mask
        if hazard_bits == 0:
            return 0
        newly_inherited = hazard_bits & ~int(local.blocked_mask)
        for action in range(self.action_count):
            if not (hazard_bits & (1 << action)):
                continue
            local.collision_outcome_counts[action] = max(
                int(local.collision_outcome_counts[action]),
                int(hazard.collision_outcome_counts[action]),
            )
            local.out_of_road_outcome_counts[action] = max(
                int(local.out_of_road_outcome_counts[action]),
                int(hazard.out_of_road_outcome_counts[action]),
            )
            local.warning_counts[action] = max(
                int(local.warning_counts[action]),
                int(hazard.warning_counts[action]),
            )
        self._reconcile_attempt_accounting(local)
        self._rebuild_outcome_masks(local)
        inherited = newly_inherited & int(local.blocked_mask)
        local.availability_mask &= ~int(local.blocked_mask)
        self.hazard_actions_transferred += inherited.bit_count()
        return int(inherited)

    def _post_first_pass_action(
        self,
        local: UnifiedStateRecord,
        hazard: Optional[UnifiedStateRecord],
        q_values: np.ndarray,
        base_mask: int,
        key: str,
        allow_random: bool = True,
        count: bool = True,
    ) -> Tuple[int, str]:
        """Use 80% eligible greedy and 20% UNKNOWN-only exploration."""
        blocked_mask = self._combined_blocked_mask(local, hazard)
        eligible_status_mask = int(local.safe_mask) | int(local.unknown_mask)
        eligible_mask = (
            int(base_mask)
            & eligible_status_mask
            & ~blocked_mask
            & self.full_mask
        )
        if eligible_mask == 0:
            if count:
                self.post_first_pass_all_blocked_rejections += 1
            selected = self._least_risk_action(
                [record for record in (local, hazard) if record is not None],
                q_values,
                int(base_mask) & self.full_mask,
                key,
                count=count,
            )
            if selected is None:
                raise RuntimeError(
                    "All-BLOCKED fallback received an empty base mask."
                )
            return int(selected), "all_blocked_least_risk"

        digest = hashlib.sha256(
            f"{self.selection_seed}|post_first_pass|{key}".encode()
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        explore = (
            allow_random
            and draw >= self.POST_FIRST_PASS_GREEDY_PROBABILITY
        )
        random_mask = (
            eligible_mask & int(local.unknown_mask) & self.full_mask
        )
        if explore and random_mask:
            selected = self._uniform_action_from_mask(random_mask, digest)
            if count:
                self.post_first_pass_random_selections += 1
            return int(selected), "epsilon_random_unknown"

        selected = self._best_action_from_mask(
            q_values, eligible_mask, self.action_count, key
        )
        if selected is None:
            raise RuntimeError("Post-first-pass greedy selection produced no action.")
        if count:
            self.post_first_pass_greedy_selections += 1
        prefix = "epsilon_greedy" if allow_random else "greedy"
        category = (
            f"{prefix}_confirmed_safe"
            if local.safe_mask & (1 << selected)
            else f"{prefix}_unknown"
        )
        return int(selected), category

    def _least_risk_action(
        self,
        records: Sequence[UnifiedStateRecord],
        q_values: np.ndarray,
        mask: int,
        key: str,
        count: bool = True,
    ) -> Optional[int]:
        """Minimize recorded collision plus out-of-road failures, then DQN Q."""
        best_score: Optional[Tuple[int, float]] = None
        candidates: List[int] = []
        for action in range(self.action_count):
            if not (int(mask) & (1 << action)):
                continue
            # Local and matched HAZARD records can contain the same event.
            # Maximum per failure type avoids counting that event twice.
            collisions = max(
                (
                    int(record.collision_outcome_counts[action])
                    for record in records
                ),
                default=0,
            )
            out_of_road = max(
                (
                    int(record.out_of_road_outcome_counts[action])
                    for record in records
                ),
                default=0,
            )
            score = (
                collisions + out_of_road,
                -float(q_values[action]),
            )
            if best_score is None or score < best_score:
                best_score = score
                candidates = [action]
            elif score == best_score:
                candidates.append(action)
        if not candidates:
            return None
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % len(candidates)
        if count:
            self.least_risk_fallbacks += 1
        return int(candidates[offset])

    def _priority_action(
        self,
        local: Optional[UnifiedStateRecord],
        hazard: Optional[UnifiedStateRecord],
        q_values: np.ndarray,
        base_mask: int,
        key: str,
        count_fallback: bool = True,
    ) -> Tuple[int, str]:
        blocked_mask = self._combined_blocked_mask(local, hazard)
        confirmed_safe_mask = 0 if local is None else int(local.safe_mask)
        confirmed_safe_mask &= ~blocked_mask
        safe_available = base_mask & confirmed_safe_mask
        unknown_available = base_mask & ~(blocked_mask | confirmed_safe_mask)
        if safe_available:
            selected = self._best_action_from_mask(
                q_values, safe_available, self.action_count, key
            )
            category = "confirmed_safe"
        elif unknown_available:
            selected = self._best_action_from_mask(
                q_values, unknown_available, self.action_count, key
            )
            category = "unknown"
        else:
            if count_fallback:
                self.hard_blocked_empty_selection_rejections += 1
            selected = self._least_risk_action(
                [record for record in (local, hazard) if record is not None],
                q_values,
                int(base_mask) & self.full_mask,
                key,
                count=count_fallback,
            )
            category = "all_blocked_least_risk"
        if selected is None:
            raise RuntimeError("Safety-priority selection produced no action.")
        return int(selected), category

    def _hazard_selection_statistics(
        self,
        local: Optional[UnifiedStateRecord],
        hazard: Optional[UnifiedStateRecord],
        q_values: np.ndarray,
        base_mask: int,
        selected: int,
        key: str,
        test: bool = False,
    ) -> None:
        if hazard is None:
            return
        raw_argmax = self._best_action_from_mask(
            q_values, base_mask, self.action_count, key
        )
        if (
            raw_argmax is not None
            and hazard.blocked_mask & (1 << raw_argmax)
            and int(selected) != int(raw_argmax)
        ):
            if test:
                self.test_hazard_actions_filtered += 1
            else:
                self.hazard_actions_filtered += 1

    def select_retired_action(
        self,
        retired_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
        state: np.ndarray,
        safety: np.ndarray,
        hazard_index: Optional[int] = None,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("retired-pool action selection")
        record = self.records[int(retired_index)]
        if record.status != self.RETIRED:
            raise RuntimeError("Retired action targeted a non-retired record.")
        hazard = self.records.get(hazard_index)
        self._inherit_hazard_blocked_evidence(record, hazard)
        selected, category = self._post_first_pass_action(
            record, hazard, q_values, self.full_mask, key
        )
        source = f"retired_pool_{category}"
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        if raw_argmax is not None and selected != raw_argmax:
            self.retired_mask_filtered_actions += 1
        self._hazard_selection_statistics(
            record, hazard, q_values, self.full_mask, selected, key
        )
        self.mark_pending_action_outcome(
            [record.record_id],
            state, safety, episode, step, selected,
        )
        return int(selected), source

    def select_candidate_action(
        self,
        candidate_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
        state: np.ndarray,
        safety: np.ndarray,
        hazard_index: Optional[int] = None,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("candidate action selection")
        record = self.records[int(candidate_index)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Candidate action targeted a non-candidate.")
        hazard = self.records.get(hazard_index)
        self._inherit_hazard_blocked_evidence(record, hazard)
        selected, category = self._priority_action(
            record, hazard, q_values, self.full_mask, key
        )
        if category == "all_blocked_least_risk":
            source = "candidate_all_blocked_least_risk"
        elif category == "confirmed_safe":
            source = "candidate_confirmed_safe"
        else:
            source = "candidate_unknown"
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        if raw_argmax is not None and selected != raw_argmax:
            self.candidate_mask_filtered_actions += 1
        self._hazard_selection_statistics(
            record, hazard, q_values, self.full_mask, selected, key
        )
        self.record_candidate_action(record.record_id, int(selected))
        self.mark_pending_action_outcome(
            [record.record_id],
            state, safety, episode, step, int(selected),
        )
        return int(selected), source

    def select_hazard_action(
        self,
        hazard_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
        state: np.ndarray,
        safety: np.ndarray,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("hazard-memory action selection")
        record = self.records[int(hazard_index)]
        if record.status != self.HAZARD:
            raise RuntimeError("Hazard action targeted a non-hazard record.")
        selected, category = self._priority_action(
            None, record, q_values, self.full_mask, key
        )
        if category == "all_blocked_least_risk":
            source = "hazard_all_blocked_least_risk"
        elif category == "confirmed_safe":
            source = "hazard_confirmed_safe"
        else:
            source = "hazard_unknown"
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        if raw_argmax is not None and selected != raw_argmax:
            self.hazard_mask_filtered_actions += 1
        self._hazard_selection_statistics(
            None, record, q_values, self.full_mask, selected, key
        )
        self.mark_pending_action_outcome(
            (), state, safety, episode, step, int(selected)
        )
        return int(selected), source

    def select_active_action(
        self,
        pool_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
        state: np.ndarray,
        safety: np.ndarray,
        hazard_index: Optional[int] = None,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("active-pool action selection")
        record = self.records[int(pool_index)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Active action targeted a non-active record.")
        hazard = self.records.get(hazard_index)
        self._inherit_hazard_blocked_evidence(record, hazard)
        blocked_mask = self._combined_blocked_mask(record, hazard)

        # The final UNKNOWN-pass action waits for its five-step result before
        # retirement. While waiting, use the ordinary eligible policy without
        # reopening or mutating the scheduled availability mask.
        if record.retirement_pending:
            selected, category = self._post_first_pass_action(
                record, hazard, q_values, self.full_mask, key
            )
            self.mark_pending_action_outcome(
                [record.record_id],
                state,
                safety,
                episode,
                step,
                int(selected),
            )
            return int(selected), f"active_unknown_outcome_pending__{category}"

        # Both scheduled masks exclude BLOCKED actions. Exhausting INITIAL
        # refills the mask once from UNKNOWN; exhausting UNKNOWN retires only
        # after its final delayed safety outcome is available.
        before = int(record.availability_mask)
        record.availability_mask = (
            before & ~blocked_mask & self.full_mask
        )
        self.blocked_actions_skipped_from_scheduled_passes += (
            before & blocked_mask
        ).bit_count()
        if record.availability_mask == 0:
            self._complete_active_scheduled_pass(
                record.record_id,
                episode,
                {
                    "action": -1,
                    "collision": False,
                    "out_of_road": False,
                    "done": False,
                    "step": int(step),
                    "retirement_trigger_type": "status_filtered_pass_end",
                },
                externally_blocked_mask=blocked_mask,
            )
            if record.status == self.RETIRED:
                selected, category = self._post_first_pass_action(
                    record, hazard, q_values, self.full_mask, key
                )
                self.mark_pending_action_outcome(
                    [record.record_id],
                    state,
                    safety,
                    episode,
                    step,
                    int(selected),
                )
                return int(selected), f"active_pass_complete__{category}"
        available_mask = int(record.availability_mask)
        selected, category = self._priority_action(
            record, hazard, q_values, available_mask, key
        )
        if category == "all_blocked_least_risk":
            source = "active_all_blocked_least_risk"
        elif category == "confirmed_safe":
            source = "active_confirmed_safe"
        else:
            source = "active_unknown"
        raw_argmax = self._best_action_from_mask(
            q_values, available_mask, self.action_count, key
        )
        if raw_argmax is not None and selected != raw_argmax:
            self.active_mask_filtered_actions += 1
        self._hazard_selection_statistics(
            record, hazard, q_values, available_mask, selected, key
        )
        self.remove(record.record_id, int(selected))
        complete_pass_after = record.availability_mask == 0
        self.mark_pending_action_outcome(
            [record.record_id],
            state, safety, episode, step, int(selected),
            retire_record_id=(
                record.record_id if complete_pass_after else None
            ),
        )
        if complete_pass_after:
            source = (
                "active_initial_pass_last_action_pending_unknown_pass"
                if record.exploration_pass == 0
                else "active_unknown_pass_last_action_pending_outcome"
            )
        elif record.exploration_pass == 1:
            source = f"active_unknown_pass_{source}"
        return int(selected), source

    def select_frozen_safety_action(
        self,
        raw_state: np.ndarray,
        raw_safety: np.ndarray,
        q_values: np.ndarray,
        key: str,
        episode: int,
        count_as_test: bool = True,
    ) -> Tuple[int, str]:
        """Read-only DQN-plus-safety selection for frozen phases."""
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        safety = np.asarray(raw_safety, dtype=np.float32).reshape(-1)
        if self.auto_calibrate_thresholds:
            if not self.thresholds_frozen:
                raise RuntimeError("Frozen safety selection requires completed calibration.")
            state = self.normalizer.transform(state)
            safety = self.safety_normalizer.transform(safety)

        local: Optional[UnifiedStateRecord] = None
        base_mask = self.full_mask
        local_source = "no_local_match"
        diagnostic_scope = "test" if count_as_test else "train"
        active = self.find_active_match(
            state, safety, diagnostic_scope=diagnostic_scope
        )
        if active[0] is not None:
            local = self.records[int(active[0])]
            local_source = "active"
        else:
            retired = self.find_retired_match(
                state, safety, diagnostic_scope=diagnostic_scope
            )
            if retired[0] is not None:
                local = self.records[int(retired[0])]
                local_source = "retired"
            else:
                candidate = self.find_candidate_match(
                    state, safety, diagnostic_scope=diagnostic_scope
                )
                if candidate[0] is not None:
                    local = self.records[int(candidate[0])]
                    local_source = "candidate"

        hazard_match = (
            self.query_hazard_match(
                state, safety, episode=episode, test=True
            )
            if count_as_test
            else self.find_hazard_match(
                state, safety, diagnostic_scope="train"
            )
        )
        hazard = (
            None if hazard_match[0] is None
            else self.records[int(hazard_match[0])]
        )
        if local is None and hazard is None:
            selected = self._best_action_from_mask(
                q_values, self.full_mask, self.action_count, key
            )
            if selected is None:
                raise RuntimeError("Frozen DQN produced no action.")
            return int(selected), "frozen_dqn_no_safety_match"

        if local is not None and local.status == self.RETIRED:
            # Testing is greedy and read-only, but uses the same post-pass
            # BLOCKED exclusion as training.
            selected, category = self._post_first_pass_action(
                local,
                hazard,
                q_values,
                base_mask,
                key,
                allow_random=False,
                count=False,
            )
        else:
            selected, category = self._priority_action(
                local, hazard, q_values, base_mask, key, count_fallback=False
            )
        self._hazard_selection_statistics(
            local,
            hazard,
            q_values,
            base_mask,
            selected,
            key,
            test=count_as_test,
        )
        return int(selected), f"frozen_{local_source}_{category}"

    def mask(self, pool_index) -> int:
        record = self.records[int(pool_index)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Action mask requested for a non-active pool.")
        return int(record.availability_mask)

    def remove(self, pool_index, action) -> None:
        record = self.records[int(pool_index)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Action removal targeted a non-active pool.")
        bit = 1 << int(action)
        record.availability_mask &= ~bit
        record.action_history_mask |= bit

    def remaining_count(self, pool_index) -> int:
        return self.mask(pool_index).bit_count()
