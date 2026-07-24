"""Top-level training action selection independent of experiment loops."""

from __future__ import annotations

import hashlib
import math
from typing import Optional, Tuple

import numpy as np

from .dqn import DQNAgent
from .memory import SimilarStateActionPools


def best_available_action(
    q_values: np.ndarray,
    mask: int,
    action_count: int,
    key: str,
) -> int:
    """Choose maximum-Q among actions whose pool bits remain available."""
    available = [
        action
        for action in range(action_count)
        if mask & (1 << action)
    ]
    if not available:
        raise RuntimeError("best_available_action received an empty mask.")

    maximum = max(float(q_values[action]) for action in available)
    candidates = np.asarray(
        [
            action
            for action in available
            if float(q_values[action]) == maximum
        ],
        dtype=np.int64,
    )
    digest = hashlib.sha256(key.encode()).digest()
    offset = int.from_bytes(digest[:8], "big") % int(candidates.size)
    return int(candidates[offset])

def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    safety_state: Optional[np.ndarray],
    episode: int,
    step: int,
    args,
    action_pools: SimilarStateActionPools,
) -> Tuple[int, str]:
    if experiment != "Karthikeya27adv8":
        raise ValueError(f"Unknown experiment: {experiment}")

    q_values = agent.q_values(state)
    tie_key = f"train|Karthikeya27adv8|{episode}|{step}"
    pooling_limit = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    if episode >= pooling_limit:
        # This is idempotent and also protects standalone callers that do not
        # enter through the main episode loop.
        action_pools.freeze_policy(episode)
        action_pools.total_final_pure_dqn_states += 1
        action = agent._deterministic_extreme_from_q(
            q_values,
            maximum=True,
            key=tie_key,
        )
        return int(action), "final_phase_pure_dqn_argmax"
    if action_pools.policy_frozen:
        raise RuntimeError(
            "A frozen pool policy cannot be re-entered before its boundary."
        )
    if safety_state is None:
        # Fail closed for the pool: no fabricated safety vector, no lookup,
        # no SAFE confirmation, and no candidate mutation for this state.
        action_pools.missing_safety_pure_dqn_states += 1
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        # Keep the bounded delayed horizon advancing, but the empty record set
        # prevents this invalid-safety step from confirming any action SAFE.
        action_pools.mark_pending_action_outcome(
            (), state, None, episode, step, int(action)
        )
        return int(action), "missing_safety_pure_dqn_argmax"

    matching_state, matching_safety, status = (
        action_pools.prepare_matching_state(
            state, safety_state, episode
        )
    )
    if status != "ready":
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            f"{status}_argmax",
        )

    hazard_match = action_pools.query_hazard_match(
        matching_state, matching_safety, episode
    )
    hazard_index = (
        None if hazard_match[0] is None else int(hazard_match[0])
    )

    active_match = action_pools.find_active_match(
        matching_state, matching_safety
    )
    retired_match = (
        None, float("-inf"), float("inf"), float("-inf"), float("inf")
    )
    if active_match[0] is None:
        retired_match = action_pools.find_retired_match(
            matching_state, matching_safety
        )
        if retired_match[0] is not None:
            retired_index, cosine, distance, s_cosine, s_distance = (
                retired_match
            )
            action_pools.record_retired_hit(
                retired_index,
                episode,
                step,
                cosine,
                distance,
                s_cosine,
                s_distance,
            )
            return action_pools.select_retired_action(
                int(retired_index),
                q_values,
                tie_key,
                episode,
                step,
                matching_state,
                matching_safety,
                hazard_index=hazard_index,
            )
    pool_index, candidate_index, pool_status = action_pools.process_state(
        matching_state,
        matching_safety,
        episode,
        active_match=active_match,
        active_match_precomputed=True,
        retired_match=retired_match,
        retired_match_precomputed=active_match[0] is None,
    )
    if pool_index is None:
        if pool_status == "retired_match_requires_filtered_dqn":
            raise RuntimeError(
                "Retired matching must be handled before candidate processing."
            )
        if candidate_index is not None and pool_status == "candidate_absolute_capacity_hazard":
            action, hazard_source = action_pools.select_hazard_action(
                int(candidate_index),
                q_values,
                tie_key,
                episode,
                step,
                matching_state,
                matching_safety,
            )
            return int(action), f"{pool_status}__{hazard_source}"
        if (
            candidate_index is not None
            and pool_status in {
                "candidate_history_exhausted_retired_pool",
                "candidate_suppressed_by_retired_pool",
                "candidate_full_fallback_retired_pool",
                "permanent_full_fallback_retired_pool",
            }
        ):
            action_pools.record_retired_state_hit(
                candidate_index,
                matching_state,
                matching_safety,
                episode,
                step,
            )
            return action_pools.select_retired_action(
                int(candidate_index),
                q_values,
                tie_key,
                episode,
                step,
                matching_state,
                matching_safety,
                hazard_index=hazard_index,
            )
        if candidate_index is not None:
            action, mask_source = action_pools.select_candidate_action(
                int(candidate_index),
                q_values,
                tie_key,
                episode,
                step,
                matching_state,
                matching_safety,
                hazard_index=hazard_index,
            )
            source = (
                pool_status
                if mask_source == "candidate_argmax_allowed"
                else f"{pool_status}__{mask_source}"
            )
            return int(action), source
        if hazard_index is not None:
            action, hazard_source = action_pools.select_hazard_action(
                hazard_index, q_values, tie_key, episode, step,
                matching_state, matching_safety,
            )
            return int(action), f"{pool_status}__{hazard_source}"
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        # Even a capacity fallback contributes a later direct failure to the
        # global hazard memory during the safety-policy phase.
        action_pools.mark_pending_action_outcome(
            (), matching_state, matching_safety, episode, step, int(action)
        )
        return int(action), pool_status

    action, active_source = action_pools.select_active_action(
        int(pool_index), q_values, tie_key, episode, step,
        matching_state, matching_safety, hazard_index=hazard_index,
    )
    base_source = (
        "promoted_pool_first_best_available"
        if pool_status == "candidate_promoted_before_action"
        else (
            "candidate_merged_active_pool_best_available"
            if pool_status == "candidate_merged_into_active_pool"
            else (
                "candidate_full_fallback_active_pool_best_available"
                if pool_status == "candidate_full_fallback_active_pool"
                else (
                    "permanent_full_fallback_active_pool_best_available"
                    if pool_status == "permanent_full_fallback_active_pool"
                    else "permanent_pool_best_available"
                )
            )
        )
    )
    source = (
        active_source
        if active_source == "active_pool_last_action_pending_retirement"
        else (
            base_source
            if active_source == "active_unknown"
            else f"{base_source}__{active_source}"
        )
    )
    return int(action), source
