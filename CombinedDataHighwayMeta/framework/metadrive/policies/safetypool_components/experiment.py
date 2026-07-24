"""Episode rows, frozen evaluation, and the reusable training loop."""

from __future__ import annotations

import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .action_selection import select_training_action
from .constants import SAFETY_VECTOR_NAMES, SHORT_LABELS
from .dqn import DQNAgent
from .environment import (
    AllActionsBlockedEpisode,
    make_env,
    parse_step_info,
    policy_safety_stop_info,
    selected_rmst_event,
)
from .memory import SimilarStateActionPools
from .safety import extract_safety_vector
from .utils import (
    avg,
    flatten_observation,
    observation_sha256,
    set_seed,
)


def episode_row(
    phase: str,
    experiment: str,
    episode: int,
    scenario_seed: int,
    initial_hash: str,
    env_reward: float,
    training_reward: float,
    steps: int,
    parsed: Dict,
    args,
    device: torch.device,
    episode_start: float,
    cpu_start: float,
    agent: DQNAgent,
    losses: Sequence[float] = (),
    action_sources: Optional[Dict[str, int]] = None,
    safety_sums: Optional[np.ndarray] = None,
    safety_minimums: Optional[np.ndarray] = None,
    safety_maximums: Optional[np.ndarray] = None,
    safety_count: int = 0,
) -> Dict:
    event = selected_rmst_event(parsed, args.rmst_event)
    event_definition = (
        "collision"
        if args.rmst_event == "collision"
        else "collision_or_out_of_road_or_policy_safety_stop"
    )
    if safety_count > 0 and safety_sums is not None:
        means = np.asarray(safety_sums, dtype=float) / safety_count
        minimums = np.asarray(safety_minimums, dtype=float)
        maximums = np.asarray(safety_maximums, dtype=float)
    else:
        means = np.full(len(SAFETY_VECTOR_NAMES), math.nan)
        minimums = np.full(len(SAFETY_VECTOR_NAMES), math.nan)
        maximums = np.full(len(SAFETY_VECTOR_NAMES), math.nan)
    return {
        "phase": phase,
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "seed": args.seed,
        "episode": episode,
        "scenario_seed": scenario_seed,
        "initial_observation_sha256": initial_hash,
        "env_reward": float(env_reward),
        "training_reward": float(training_reward),
        "steps": int(steps),
        **parsed,
        "rmst_event_definition": event_definition,
        "rmst_event_observed": bool(event),
        "event_or_censor_time_steps": int(steps),
        "wall_time_seconds": float(time.perf_counter() - episode_start),
        "cpu_time_seconds": float(time.process_time() - cpu_start),
        "average_loss": avg(losses),
        "average_rnd_loss": 0.0,
        "average_rnd_bonus": 0.0,
        "replay_buffer_size": len(agent.replay),
        "learn_steps": agent.learn_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "epsilon": 0.0,
        "rnd_beta": 0.0,
        "noisy_sigma_init": 0.0,
        "network_frozen": phase.startswith("test"),
        "updates_during_test": 0 if phase.startswith("test") else "",
        "action_source_counts": json.dumps(action_sources or {}, sort_keys=True),
        "mean_lane_boundary_clearance": float(means[0]),
        "minimum_lane_boundary_clearance": float(minimums[0]),
        "mean_nearest_collision_hazard_center_distance": float(means[1]),
        "minimum_nearest_collision_hazard_center_distance": float(minimums[1]),
        "mean_absolute_lane_offset": float(means[2]),
        "maximum_absolute_lane_offset": float(maximums[2]),
        "mean_absolute_heading_error": float(means[3]),
        "maximum_absolute_heading_error": float(maximums[3]),
        "mean_ego_speed": float(means[4]),
        "maximum_ego_speed": float(maximums[4]),
    }

def verify_discrete_action_space(env, args) -> int:
    """Verify the configured 3x3 discrete action contract before training."""
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        raise RuntimeError("MetaDrive action space is not Discrete.")
    action_count = int(action_space.n)
    expected = int(args.discrete_steering_dim * args.discrete_throttle_dim)
    if action_count != expected:
        raise RuntimeError(
            f"Configured action grid expects {expected} actions, but MetaDrive exposes {action_count}."
        )
    invalid = [action for action in range(action_count) if not action_space.contains(action)]
    if invalid or action_space.contains(action_count) or action_space.contains(-1):
        raise RuntimeError(
            "MetaDrive discrete action IDs are not the expected contiguous range "
            f"0..{action_count - 1}; invalid in-range IDs: {invalid}."
        )
    config = getattr(env, "config", {})
    for key, expected_value in (
        ("discrete_action", True),
        ("use_multi_discrete", False),
        ("discrete_steering_dim", int(args.discrete_steering_dim)),
        ("discrete_throttle_dim", int(args.discrete_throttle_dim)),
    ):
        if key in config and config[key] != expected_value:
            raise RuntimeError(
                f"MetaDrive action configuration mismatch for {key}: "
                f"expected {expected_value!r}, got {config[key]!r}."
            )
    return action_count

def run_frozen_test_phase(
    agent: DQNAgent,
    args,
    device: torch.device,
    experiment: str,
) -> Tuple[List[Dict], float, float]:
    """Evaluate the frozen final model with pure-DQN argmax only."""
    phase = "test"
    policy_label = "pure DQN"
    env = make_env(args, "test")
    rows: List[Dict] = []
    phase_start = time.perf_counter()
    phase_cpu_start = time.process_time()
    print(
        f"\n===== TESTING START: {SHORT_LABELS[experiment]} "
        f"({policy_label}) =====",
        flush=True,
    )
    try:
        with torch.no_grad():
            for episode in range(args.test_episodes):
                # Match the canonical baseline: per-episode timing includes reset.
                episode_start = time.perf_counter()
                cpu_start = time.process_time()
                scenario_seed = args.test_seed + episode
                state_raw, _ = env.reset(seed=scenario_seed)
                state = flatten_observation(state_raw)
                initial_hash = observation_sha256(state_raw)
                reward_total = 0.0
                parsed = parse_step_info({}, False, False)
                action_sources: Dict[str, int] = {}
                safety_sums = np.zeros(
                    len(SAFETY_VECTOR_NAMES), dtype=np.float64
                )
                safety_minimums = np.full(
                    len(SAFETY_VECTOR_NAMES), np.inf, dtype=np.float64
                )
                safety_maximums = np.full(
                    len(SAFETY_VECTOR_NAMES), -np.inf, dtype=np.float64
                )
                safety_count = 0
                episode_steps = 0
                for step in range(args.max_episode_steps):
                    q_values = agent.q_values(state)
                    tie_key = f"test|{args.seed}|{episode}|{step}"
                    action = agent._deterministic_extreme_from_q(
                        q_values, maximum=True, key=tie_key
                    )
                    source = "frozen_pure_dqn_argmax"
                    action_sources[source] = action_sources.get(source, 0) + 1
                    next_raw, reward, terminated, truncated, info = env.step(
                        int(action)
                    )
                    episode_steps += 1
                    state = flatten_observation(next_raw)
                    reward_total += float(reward)
                    parsed = parse_step_info(
                        info, bool(terminated), bool(truncated)
                    )
                    if terminated or truncated:
                        break
                row = episode_row(
                    phase,
                    experiment,
                    episode,
                    scenario_seed,
                    initial_hash,
                    reward_total,
                    reward_total,
                    episode_steps,
                    parsed,
                    args,
                    device,
                    episode_start,
                    cpu_start,
                    agent,
                    action_sources=action_sources,
                    safety_sums=safety_sums,
                    safety_minimums=safety_minimums,
                    safety_maximums=safety_maximums,
                    safety_count=safety_count,
                )
                rows.append(row)
                if (episode + 1) % args.progress_every == 0:
                    print(
                        f"TEST  {SHORT_LABELS[experiment]:16s} "
                        f"policy={policy_label:15s} ep={episode + 1:03d} "
                        f"reward={reward_total:9.3f} steps={episode_steps:3d} "
                        f"term={parsed['termination_reason']:11s} "
                        f"collision={str(parsed['collision']):5s} "
                        f"wall={row['wall_time_seconds']:.2f}s",
                        flush=True,
                    )
    finally:
        env.close()
    duration = time.perf_counter() - phase_start
    cpu_duration = time.process_time() - phase_cpu_start
    print(
        f"===== TESTING END: {SHORT_LABELS[experiment]} "
        f"({policy_label}) =====",
        flush=True,
    )
    print(f"Testing duration: {duration:.2f}s", flush=True)
    return rows, duration, cpu_duration

def run_experiment(
    experiment: str, args, device: torch.device, output_dir: Path
) -> Tuple[List[Dict], List[Dict], SimilarStateActionPools]:
    if args.deterministic:
        set_seed(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    rows: List[Dict] = []

    train_env = make_env(args, "train")
    initial_observation, _ = train_env.reset(seed=args.seed)
    observation_size = int(flatten_observation(initial_observation).size)
    try:
        action_count = verify_discrete_action_space(train_env, args)
    except Exception:
        train_env.close()
        raise
    agent = DQNAgent(observation_size, action_count, args, device)
    if action_count != 9:
        train_env.close()
        raise RuntimeError(
            f"Karthikeya27adv8 expects exactly 9 discrete actions; environment has {action_count}."
        )
    action_pools = SimilarStateActionPools(
        max_pools=args.max_state_pools,
        max_candidates=args.max_state_candidates,
        candidate_hard_limit=args.candidate_hard_limit,
        candidate_batch_evict_count=args.candidate_batch_evict_count,
        candidate_promotion_visits=args.candidate_promotion_visits,
        hazard_memory_capacity=args.hazard_memory_capacity,
        safe_confirmation_visits=args.safe_confirmation_visits,
        safety_horizon_steps=args.safety_horizon_steps,
        minimum_progress_reward=args.minimum_progress_reward,
        warning_block_threshold=args.warning_block_threshold,
        action_count=action_count,
        observation_size=observation_size,
        safety_size=len(SAFETY_VECTOR_NAMES),
        maximum_pool_capacity=args.maximum_pool_capacity,
        candidate_recent_protection_episodes=(
            args.candidate_recent_protection_episodes
        ),
        capacity_review_interval=args.capacity_review_interval,
        similarity_threshold=args.state_similarity_threshold,
        distance_threshold=args.state_distance_threshold,
        candidate_similarity_threshold=args.candidate_similarity_threshold,
        candidate_distance_threshold=args.candidate_distance_threshold,
        safety_similarity_threshold=args.safety_similarity_threshold,
        safety_distance_threshold=args.safety_distance_threshold,
        candidate_safety_distance_threshold=(
            args.candidate_safety_distance_threshold
        ),
        close_enough_fallback=args.close_enough_fallback,
        general_cosine_relaxation=args.general_cosine_relaxation,
        general_rms_relaxation=args.general_rms_relaxation,
        capacity_fallback_general_variation=(
            args.capacity_fallback_general_variation
        ),
        capacity_fallback_safety_improvement=(
            args.capacity_fallback_safety_improvement
        ),
        auto_calibrate_thresholds=args.auto_calibrate_thresholds,
        calibration_state_count=args.calibration_state_count,
        calibration_max_pairs=args.calibration_max_pairs,
        seed=args.seed,
        candidate_centroid_shift_threshold=(
            args.candidate_centroid_shift_threshold
        ),
        candidate_stable_updates=args.candidate_stable_updates,
        max_candidate_centroid_updates=args.max_candidate_centroid_updates,
        centroid_shift_threshold=args.centroid_shift_threshold,
        centroid_stable_updates=args.centroid_stable_updates,
        max_centroid_updates=args.max_centroid_updates,
        centroid_stability_distance_threshold=(
            args.centroid_stability_distance_threshold
        ),
        pool_storage_dtype=args.pool_storage_dtype,
    )

    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.perf_counter()
    training_cpu_start = time.process_time()
    pooling_limit = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    try:
        for episode in range(args.train_episodes):
            pool_policy_active = episode < pooling_limit
            if pool_policy_active:
                action_pools.review_capacity(episode)
            else:
                action_pools.freeze_policy(episode)
            # Match the canonical baseline: per-episode timing includes reset.
            episode_start = time.perf_counter()
            cpu_start = time.process_time()
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            if pool_policy_active:
                action_pools.begin_episode()
            state = flatten_observation(state_raw)
            initial_hash = observation_sha256(state_raw)
            env_reward_total = 0.0
            training_reward_total = 0.0
            losses: List[float] = []
            action_sources: Dict[str, int] = {}
            safety_sums = np.zeros(len(SAFETY_VECTOR_NAMES), dtype=np.float64)
            safety_minimums = np.full(
                len(SAFETY_VECTOR_NAMES), np.inf, dtype=np.float64
            )
            safety_maximums = np.full(
                len(SAFETY_VECTOR_NAMES), -np.inf, dtype=np.float64
            )
            safety_count = 0
            episode_steps = 0
            parsed = parse_step_info({}, False, False)
            for step in range(args.max_episode_steps):
                if pool_policy_active:
                    safety_state = extract_safety_vector(train_env, args)
                    if safety_state is not None:
                        safety_sums += safety_state
                        safety_minimums = np.minimum(
                            safety_minimums, safety_state
                        )
                        safety_maximums = np.maximum(
                            safety_maximums, safety_state
                        )
                        safety_count += 1
                else:
                    # The final 20% is observation-only pure DQN. Avoid even
                    # reading engine safety objects in this phase.
                    safety_state = None
                try:
                    action, source = select_training_action(
                        experiment,
                        agent,
                        state,
                        safety_state,
                        episode,
                        step,
                        args,
                        action_pools,
                    )
                except AllActionsBlockedEpisode:
                    parsed = policy_safety_stop_info()
                    action_sources["policy_safety_stop"] = (
                        action_sources.get("policy_safety_stop", 0) + 1
                    )
                    break
                action_sources[source] = action_sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                episode_steps += 1
                next_state = flatten_observation(next_raw)
                episode_done = bool(terminated or truncated)
                # Gymnasium time-limit truncation is censoring, not a terminal
                # MDP state. Bootstrap through it because MetaDrive is configured
                # with truncate_as_terminate=False.
                bootstrap_terminal = bool(terminated)
                training_reward = float(env_reward)
                agent.replay.add(
                    state,
                    action,
                    training_reward,
                    next_state,
                    bootstrap_terminal,
                )
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                env_reward_total += float(env_reward)
                training_reward_total += training_reward
                state = next_state
                parsed = parse_step_info(info, bool(terminated), bool(truncated))
                # Safety summaries intentionally contain pre-action values only.
                if pool_policy_active:
                    action_pools.finalize_pending_action_outcome(
                        reward=float(env_reward),
                        parsed=parsed,
                        done=episode_done,
                    )
                elif action_pools.pending_action_outcome is not None:
                    raise RuntimeError(
                        "Frozen pool policy acquired a pending outcome."
                    )
                if episode_done:
                    break

            if pool_policy_active:
                action_pools.end_episode_safety_window(episode)

            row = episode_row(
                "train",
                experiment,
                episode,
                scenario_seed,
                initial_hash,
                env_reward_total,
                training_reward_total,
                episode_steps,
                parsed,
                args,
                device,
                episode_start,
                cpu_start,
                agent,
                losses,
                action_sources,
                safety_sums,
                safety_minimums,
                safety_maximums,
                safety_count,
            )
            rows.append(row)
            if (episode + 1) % args.progress_every == 0:
                print(
                    f"TRAIN {SHORT_LABELS[experiment]:16s} "
                    f"ep={episode + 1:03d} "
                    f"reward={env_reward_total:9.3f} steps={episode_steps:3d} "
                    f"term={parsed['termination_reason']:11s} "
                    f"collision={str(parsed['collision']):5s} "
                    f"wall={row['wall_time_seconds']:.2f}s "
                    f"loss={row['average_loss']:.6f}",
                    flush=True,
                )
    finally:
        train_env.close()

    training_duration = time.perf_counter() - training_start
    training_cpu_duration = time.process_time() - training_cpu_start

    # Save the fully trained final DQN and frozen safety memory. Testing uses
    # only the final DQN; earlier training states are never eligible.
    if not action_pools.policy_frozen:
        action_pools.freeze_policy(args.train_episodes)
    model_path = output_dir / "models" / f"{experiment}_model.pt"
    safety_memory_path = output_dir / "models" / f"{experiment}_safety_memory.pkl"
    agent.save(model_path, args)
    with safety_memory_path.open("wb") as handle:
        pickle.dump(action_pools, handle, protocol=pickle.HIGHEST_PROTOCOL)
    agent.freeze()
    print(f"===== TRAINING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Training duration: {training_duration:.2f}s", flush=True)
    print(
        f"Testing final episode-{args.train_episodes} model with pure DQN",
        flush=True,
    )

    # MetaDrive uses a singleton engine. The single frozen evaluation uses the
    # same final model and canonical test scenarios as the baselines.
    test_rows, testing_duration, testing_cpu_duration = run_frozen_test_phase(
        agent, args, device, experiment
    )
    rows.extend(test_rows)
    runtime_rows: List[Dict] = []
    for phase, phase_seconds, phase_cpu_seconds in (
        ("train", training_duration, training_cpu_duration),
        ("test", testing_duration, testing_cpu_duration),
    ):
        phase_rows = [row for row in rows if row["phase"] == phase]
        runtime_rows.append(
            {
                "method": experiment,
                "method_label": SHORT_LABELS[experiment],
                "phase": phase,
                "episodes": len(phase_rows),
                "phase_wall_time_seconds": float(phase_seconds),
                "phase_cpu_time_seconds": float(phase_cpu_seconds),
                "summed_episode_wall_time_seconds": float(
                    sum(row["wall_time_seconds"] for row in phase_rows)
                ),
                "summed_episode_cpu_time_seconds": float(
                    sum(row["cpu_time_seconds"] for row in phase_rows)
                ),
                "average_wall_time_seconds_per_episode": (
                    float(sum(row["wall_time_seconds"] for row in phase_rows))
                    / len(phase_rows)
                    if phase_rows
                    else math.nan
                ),
                "average_cpu_time_seconds_per_episode": (
                    float(sum(row["cpu_time_seconds"] for row in phase_rows))
                    / len(phase_rows)
                    if phase_rows
                    else math.nan
                ),
            }
        )
    return rows, runtime_rows, action_pools
