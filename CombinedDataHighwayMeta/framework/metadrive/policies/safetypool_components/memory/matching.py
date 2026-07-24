"""Calibration and vectorized strict/fallback state matching."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from ..safety import (
    capacity_fallback_valid_mask,
    directional_safety_relative_improvements,
)


class PoolMatchingMixin:
    """Calibration and vectorized strict/fallback state matching."""

    def begin_episode(self) -> None:
        self._require_policy_mutable("pool episode initialization")
        self.calibrator.reset_episode()
        self.safety_calibrator.reset_episode()

    def prepare_matching_state(
        self,
        raw_state: np.ndarray,
        raw_safety: np.ndarray,
        episode: int,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        self._require_policy_mutable("matching-state calibration")
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        safety = np.asarray(raw_safety, dtype=np.float32).reshape(-1)
        if not self.auto_calibrate_thresholds:
            return state, safety, "ready"
        del episode  # Calibration is state-count based and seed independent.
        if self.total_calibration_states < self.normalization_state_count:
            self.normalizer.update(state)
            self.safety_normalizer.update(safety)
            self.total_calibration_states += 1
            return state, safety, "normalizer_warmup"

        if not self.normalizer.frozen:
            self.normalizer.freeze()
            self.safety_normalizer.freeze()

        state_n = self.normalizer.transform(state)
        safety_n = self.safety_normalizer.transform(safety)
        if self.total_calibration_states < self.calibration_state_count:
            self.calibrator.observe(state_n)
            self.safety_calibrator.observe(safety_n)
            self.total_calibration_states += 1
            return state_n, safety_n, "threshold_warmup"

        if not self.thresholds_frozen:
            similarity, distance, candidate_distance = (
                self.calibrator.derive()
            )
            _, _, candidate_s_distance = (
                self.safety_calibrator.derive()
            )

            # Restore the evidence-calibrated Basic Optimal behavior. Safety
            # calibration is conservatively bounded at cosine 0.98 / RMS 0.10.
            self.similarity_threshold = float(similarity)
            self.distance_threshold = float(distance)
            self.safety_similarity_threshold = max(
                0.98, self.configured_safety_similarity_threshold
            )
            self.safety_distance_threshold = min(
                0.10, self.configured_safety_distance_threshold
            )
            # Candidate gates may become tighter, never looser than their
            # configured 0.25 caps.
            self.candidate_distance_threshold = min(
                self.configured_candidate_distance_threshold,
                float(candidate_distance),
            )
            self.candidate_safety_distance_threshold = min(
                self.configured_candidate_safety_distance_threshold,
                max(
                    self.safety_distance_threshold * 1.25,
                    float(candidate_s_distance),
                ),
            )
            self.calibrator.freeze()
            self.safety_calibrator.freeze()
            self.thresholds_frozen = True
        return state_n, safety_n, "ready"

    @staticmethod
    def _cosine(a, an, b, bn) -> float:
        if an == 0.0 or bn == 0.0:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _rms(a, b) -> float:
        return float(np.sqrt(np.mean(np.square(a - b))))

    def _matching_safety_to_raw(self, values: np.ndarray) -> np.ndarray:
        """Invert the frozen safety normalization without changing storage."""
        array = np.asarray(values, dtype=np.float32)
        if not self.auto_calibrate_thresholds:
            return array
        scale = (
            self.safety_normalizer.standard_deviation()
            + self.safety_normalizer.epsilon
        ).astype(np.float32)
        mean = self.safety_normalizer.mean.astype(np.float32)
        return array * scale + mean

    def _best_status_match(
        self,
        state: np.ndarray,
        safety: np.ndarray,
        status: str,
        cosine_threshold: float,
        distance_threshold: float,
        safety_cosine_threshold: float,
        safety_distance_threshold: float,
        record_ids: Optional[Iterable[int]] = None,
        diagnostic_scope: str = "train",
        allow_capacity_fallback: bool = False,
    ) -> Tuple[Optional[int], float, float, float, float]:
        if diagnostic_scope not in self.match_rejection_counters:
            raise ValueError(
                f"Unknown matching diagnostic scope: {diagnostic_scope}"
            )
        diagnostic = self.match_rejection_counters[
            diagnostic_scope
        ][status]
        diagnostic["queries"] += 1
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        safety = np.asarray(safety, dtype=np.float32).reshape(-1)
        state_norm = float(np.linalg.norm(state))
        safety_norm = float(np.linalg.norm(safety))
        if record_ids is None:
            slots = np.asarray(self.status_slots[status], dtype=np.intp)
        else:
            slots = np.asarray(
                [
                    self.record_slots[int(record_id)]
                    for record_id in record_ids
                    if int(record_id) in self.records
                    and self.records[int(record_id)].status == status
                ],
                dtype=np.intp,
            )
        if slots.size == 0:
            diagnostic["empty_queries"] += 1
            diagnostic["no_match_queries"] += 1
            return (
                None, float("-inf"), float("inf"),
                float("-inf"), float("inf"),
            )

        states = self.state_matrix[slots].astype(np.float32, copy=False)
        safeties = self.safety_matrix[slots].astype(np.float32, copy=False)
        state_norms = self.state_norms[slots].astype(np.float32, copy=False)
        safety_norms = self.safety_norms[slots].astype(
            np.float32, copy=False
        )

        state_denominator = state_norms * state_norm
        safety_denominator = safety_norms * safety_norm
        cosines = np.zeros(slots.size, dtype=np.float32)
        safety_cosines = np.zeros(slots.size, dtype=np.float32)
        valid_state_norm = state_denominator > 0.0
        valid_safety_norm = safety_denominator > 0.0
        if np.any(valid_state_norm):
            cosines[valid_state_norm] = (
                states[valid_state_norm] @ state
            ) / state_denominator[valid_state_norm]
        if np.any(~valid_state_norm):
            cosines[~valid_state_norm] = np.all(
                states[~valid_state_norm] == state, axis=1
            ).astype(np.float32)
        if np.any(valid_safety_norm):
            safety_cosines[valid_safety_norm] = (
                safeties[valid_safety_norm] @ safety
            ) / safety_denominator[valid_safety_norm]
        if np.any(~valid_safety_norm):
            safety_cosines[~valid_safety_norm] = np.all(
                safeties[~valid_safety_norm] == safety, axis=1
            ).astype(np.float32)

        distances = np.sqrt(
            np.mean(np.square(states - state), axis=1)
        )
        safety_distances = np.sqrt(
            np.mean(np.square(safeties - safety), axis=1)
        )
        general_cosine_ok = (
            (state_norm < 1e-6)
            | (state_norms < 1e-6)
            | (cosines >= cosine_threshold)
        )
        safety_cosine_ok = (
            (safety_norm < 1e-6)
            | (safety_norms < 1e-6)
            | (safety_cosines >= safety_cosine_threshold)
        )
        general_rms_ok = distances <= distance_threshold
        safety_rms_ok = safety_distances <= safety_distance_threshold
        diagnostic["records_compared"] += int(slots.size)
        diagnostic["general_cosine_failed_records"] += int(
            slots.size - np.count_nonzero(general_cosine_ok)
        )
        diagnostic["general_rms_failed_records"] += int(
            slots.size - np.count_nonzero(general_rms_ok)
        )
        diagnostic["safety_cosine_failed_records"] += int(
            slots.size - np.count_nonzero(safety_cosine_ok)
        )
        diagnostic["safety_rms_failed_records"] += int(
            slots.size - np.count_nonzero(safety_rms_ok)
        )
        valid = (
            general_cosine_ok
            & general_rms_ok
            & safety_cosine_ok
            & safety_rms_ok
        )
        strict_general_ok = general_cosine_ok & general_rms_ok
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            if np.any(
                ~general_cosine_ok
                & general_rms_ok
                & safety_cosine_ok
                & safety_rms_ok
            ):
                diagnostic["general_cosine_sole_block_queries"] += 1
            if np.any(
                general_cosine_ok
                & ~general_rms_ok
                & safety_cosine_ok
                & safety_rms_ok
            ):
                diagnostic["general_rms_sole_block_queries"] += 1
            if np.any(
                general_cosine_ok
                & general_rms_ok
                & ~safety_cosine_ok
                & safety_rms_ok
            ):
                diagnostic["safety_cosine_sole_block_queries"] += 1
            if np.any(
                general_cosine_ok
                & general_rms_ok
                & safety_cosine_ok
                & ~safety_rms_ok
            ):
                diagnostic["safety_rms_sole_block_queries"] += 1

            # Normal lookup gets a deliberately small general-only second
            # chance. Both relaxed general gates and both strict safety gates
            # must pass. This remains one vectorized O(N(D+S)) scan.
            diagnostic["general_relaxation_attempts"] += 1
            relaxed_general_cosine_ok = (
                (state_norm < 1e-6)
                | (state_norms < 1e-6)
                | (
                    cosines
                    >= max(
                        -1.0,
                        cosine_threshold
                        * (1.0 - self.general_cosine_relaxation),
                    )
                )
            )
            relaxed_general_rms_ok = distances <= (
                distance_threshold * (1.0 + self.general_rms_relaxation)
            )
            general_relaxed_valid = (
                relaxed_general_cosine_ok
                & relaxed_general_rms_ok
                & safety_cosine_ok
                & safety_rms_ok
                & ~strict_general_ok
            )
            valid_indices = np.flatnonzero(general_relaxed_valid)
            if valid_indices.size:
                diagnostic["general_relaxation_matches"] += 1

            if (
                valid_indices.size == 0
                and self.close_enough_fallback
                and allow_capacity_fallback
            ):
                diagnostic["fallback_attempts"] += 1
                # Pressure matching keeps the same conservative 2% cosine
                # relaxation, but may allow the configured RMS expansion.
                relaxed_general_cosine_ok = (
                    (state_norm < 1e-6)
                    | (state_norms < 1e-6)
                    | (
                        cosines
                        >= max(
                            -1.0,
                            cosine_threshold
                            * (1.0 - self.general_cosine_relaxation),
                        )
                    )
                )
                relaxed_general_rms_ok = distances <= (
                    distance_threshold
                    * (1.0 + self.capacity_fallback_general_variation)
                )
                raw_current_safety = self._matching_safety_to_raw(safety)
                raw_centroid_safety = self._matching_safety_to_raw(safeties)
                # Directional safety features: larger lane/threat clearance
                # is safer; smaller absolute lane offset/heading error is
                # safer. Speed is context-dependent and stays governed only
                # by the strict safety cosine/RMS gates.
                relative_improvements = (
                    directional_safety_relative_improvements(
                        raw_current_safety, raw_centroid_safety
                    )
                )
                no_directional_worsening = np.all(
                    relative_improvements >= -1e-6, axis=1
                )
                any_directional_improvement = np.any(
                    relative_improvements
                    >= (
                        self.capacity_fallback_safety_improvement
                        - 1e-6
                    ),
                    axis=1,
                )
                directional_safety_ok = (
                    no_directional_worsening
                    & any_directional_improvement
                )
                diagnostic["safety_worsened_records"] += int(
                    slots.size - np.count_nonzero(no_directional_worsening)
                )
                diagnostic[
                    "safety_no_10_percent_improvement_records"
                ] += int(
                    slots.size
                    - np.count_nonzero(any_directional_improvement)
                )
                diagnostic["safety_direction_failed_records"] += int(
                    slots.size - np.count_nonzero(directional_safety_ok)
                )
                # Safety cosine and RMS are never relaxed.
                relaxed_valid = capacity_fallback_valid_mask(
                    relaxed_general_cosine_ok,
                    relaxed_general_rms_ok,
                    safety_cosine_ok,
                    safety_rms_ok,
                    directional_safety_ok,
                    strict_general_ok,
                )
                valid_indices = np.flatnonzero(relaxed_valid)
                if valid_indices.size:
                    diagnostic["fallback_matches"] += 1

            if valid_indices.size == 0:
                diagnostic["no_match_queries"] += 1
                return (
                    None, float("-inf"), float("inf"),
                    float("-inf"), float("inf"),
                )
        else:
            diagnostic["strict_matches"] += 1

        general_components = np.where(
            (state_norm < 1e-6) | (state_norms < 1e-6),
            1.0,
            cosines / max(cosine_threshold, 1e-8),
        )
        safety_components = np.where(
            (safety_norm < 1e-6) | (safety_norms < 1e-6),
            1.0,
            safety_cosines / max(safety_cosine_threshold, 1e-8),
        )
        scores = (
            general_components
            + safety_components
            - distances / max(distance_threshold, 1e-8)
            - safety_distances / max(safety_distance_threshold, 1e-8)
        )
        best_local = int(valid_indices[np.argmax(scores[valid_indices])])
        best_slot = int(slots[best_local])
        best_id = int(self.slot_record_ids[best_slot])
        return (
            best_id,
            float(cosines[best_local]),
            float(distances[best_local]),
            float(safety_cosines[best_local]),
            float(safety_distances[best_local]),
        )

    def find_active_match(self, state, safety, diagnostic_scope="train"):
        return self._best_status_match(
            state, safety, self.ACTIVE,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
            diagnostic_scope=diagnostic_scope,
        )

    def find_retired_match(self, state, safety, diagnostic_scope="train"):
        return self._best_status_match(
            state, safety, self.RETIRED,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
            diagnostic_scope=diagnostic_scope,
        )

    def find_candidate_match(self, state, safety, diagnostic_scope="train"):
        return self._best_status_match(
            state, safety, self.CANDIDATE,
            self.candidate_similarity_threshold,
            self.candidate_distance_threshold,
            self.safety_similarity_threshold,
            self.candidate_safety_distance_threshold,
            diagnostic_scope=diagnostic_scope,
        )

    def find_hazard_match(self, state, safety, diagnostic_scope="train"):
        # Hazard transfer always uses strict permanent-pool gates. Capacity
        # fallback is limited to ACTIVE and RETIRED records.
        return self._best_status_match(
            state, safety, self.HAZARD,
            self.similarity_threshold,
            self.distance_threshold,
            self.safety_similarity_threshold,
            self.safety_distance_threshold,
            diagnostic_scope=diagnostic_scope,
        )

    def _permanent_match_score(
        self, match: Tuple[Optional[int], float, float, float, float]
    ) -> float:
        if match[0] is None:
            return float("-inf")
        return float(
            match[1] / max(self.similarity_threshold, 1e-8)
            + match[3] / max(self.safety_similarity_threshold, 1e-8)
            - match[2] / max(self.distance_threshold, 1e-8)
            - match[4] / max(self.safety_distance_threshold, 1e-8)
        )

    def find_capacity_fallback(
        self,
        state: np.ndarray,
        safety: np.ndarray,
        context: str,
    ) -> Tuple[
        Optional[str],
        Tuple[Optional[int], float, float, float, float],
    ]:
        """Find the best pressure-only permanent match in linear time."""
        if context not in {"candidate_full", "permanent_full", "eviction"}:
            raise ValueError(f"Unknown capacity-fallback context: {context}")
        if context == "candidate_full":
            self.candidate_full_fallback_queries += 1
        elif context == "permanent_full":
            self.permanent_full_fallback_queries += 1
        else:
            self.eviction_fallback_queries += 1

        active = self._best_status_match(
            state,
            safety,
            self.ACTIVE,
            self.similarity_threshold,
            self.distance_threshold,
            self.safety_similarity_threshold,
            self.safety_distance_threshold,
            allow_capacity_fallback=True,
        )
        retired = self._best_status_match(
            state,
            safety,
            self.RETIRED,
            self.similarity_threshold,
            self.distance_threshold,
            self.safety_similarity_threshold,
            self.safety_distance_threshold,
            allow_capacity_fallback=True,
        )
        if active[0] is None and retired[0] is None:
            return None, (
                None,
                float("-inf"),
                float("inf"),
                float("-inf"),
                float("inf"),
            )

        if self._permanent_match_score(active) >= self._permanent_match_score(
            retired
        ):
            status, match = self.ACTIVE, active
            self.capacity_fallback_active_matches += 1
        else:
            status, match = self.RETIRED, retired
            self.capacity_fallback_retired_matches += 1
        if context == "candidate_full":
            self.candidate_full_fallback_matches += 1
        elif context == "permanent_full":
            self.permanent_full_fallback_matches += 1
        else:
            self.eviction_fallback_matches += 1
        return status, match

    def query_hazard_match(self, state, safety, episode: int, test: bool = False):
        """Count and return the best bounded hazard lookup for one state."""
        if test:
            self.test_hazard_queries += 1
        else:
            self.hazard_queries += 1
        match = self.find_hazard_match(
            state,
            safety,
            diagnostic_scope="test" if test else "train",
        )
        if match[0] is None:
            return match
        hazard = self.records[int(match[0])]
        if test:
            self.test_hazard_matches += 1
        else:
            self.hazard_matches += 1
            hazard.hazard_hit_count += 1
            hazard.hazard_last_hit_episode = int(episode)
        return match
