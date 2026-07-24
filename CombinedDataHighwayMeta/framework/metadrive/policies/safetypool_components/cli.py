"""Thin orchestration layer connecting configuration and domain services."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .configuration import (
    parse_args,
    prepare_output_dir,
    resolve_output_dir,
    validate_args,
)
from .constants import (
    EXPERIMENTS,
    SAFETY_VECTOR_NAMES,
    SHORT_LABELS,
)
from .environment import metadrive_version
from .experiment import run_experiment
from .memory import SimilarStateActionPools
from .persistence import (
    save_outputs,
    update_baseline_index,
    write_completion_manifest,
)
from .pool_reporting import save_pool_statistics
from .utils import choose_device, set_seed


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.hybrid_design_version = (
        "karthikeya27adv8_two_pass_fail_closed_pure_dqn_final20_v3"
    )
    # Permanent capacity is fixed by making the current and absolute limits
    # identical. This policy-specific capacity is excluded from the baseline's
    # shared DQN/environment compatibility contract.
    args.maximum_pool_capacity = args.max_state_pools
    args.hazard_memory_capacity = max(
        1,
        int(math.ceil(
            args.maximum_pool_capacity * args.hazard_memory_fraction
        )),
    )
    validate_args(args)
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        print(
            f"WARNING: launch with PYTHONHASHSEED={args.seed} for complete "
            "process reproducibility",
            file=sys.stderr,
        )
    if args.deterministic:
        set_seed(args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args.force)
    args.output_dir = str(output_dir)
    device = choose_device(args.device)
    args.device = str(device)

    print("=" * 76)
    print("METADRIVE KARTHIKEYA27ADV8 RETAINED-EVIDENCE SAFETY POLICY")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("MetaDrive:", metadrive_version())
    print("Device:", device)
    print("Experiments:", ", ".join(SHORT_LABELS[e] for e in EXPERIMENTS))
    print("Plain DQN + Target + Replay + Adam: yes")
    print("RND intrinsic reward: no")
    print("Count-based intrinsic reward: no")
    print("Frozen Pure-DQN testing only: yes")
    print("Train/Test/Max steps:", args.train_episodes, args.test_episodes, args.max_episode_steps)
    print("Training seeds:", args.seed, "through", args.seed + args.train_episodes - 1)
    print("Testing seeds:", args.test_seed, "through", args.test_seed + args.test_episodes - 1)
    print("Discrete actions:", args.discrete_steering_dim * args.discrete_throttle_dim)
    print("Learning rate:", args.learning_rate)
    print("Output directory:", output_dir)
    print("Policy name: Karthikeya27adv8")
    print("Initial combined active + retired pools:", args.max_state_pools)
    print("Maximum permanent capacity:", args.maximum_pool_capacity)
    print("Adaptive capacity formula applied:", False)
    print("Pool general representation: full flattened observation")
    print("Pool safety representation:", ", ".join(SAFETY_VECTOR_NAMES))
    print("Unified record statuses: CANDIDATE | ACTIVE | RETIRED | HAZARD")
    print("Lookup: safety-strict HAZARD + ACTIVE -> RETIRED -> CANDIDATE")
    print("Testing policy: frozen pure DQN")
    print("Auto threshold calibration:", args.auto_calibrate_thresholds)
    print("Calibration state target:", args.calibration_state_count)
    print("Candidate soft limit:", args.max_state_candidates)
    print("Candidate hard limit:", args.candidate_hard_limit)
    print("Candidate batch eviction count:", args.candidate_batch_evict_count)
    print("Candidate promotion visits:", args.candidate_promotion_visits)
    print("Hazard-memory capacity:", args.hazard_memory_capacity)
    print("Hazard-memory fraction:", args.hazard_memory_fraction)
    print("Clean outcomes required for SAFE:", args.safe_confirmation_visits)
    print("Delayed safety horizon:", args.safety_horizon_steps)
    print("Minimum progress reward for SAFE:", args.minimum_progress_reward)
    print("Warnings required to block:", args.warning_block_threshold)
    print("First-pass priority: highest-Q SAFE, then UNKNOWN; BLOCKED excluded")
    print(
        "Candidate centroid shift threshold:",
        args.candidate_centroid_shift_threshold,
    )
    print("Candidate stable updates:", args.candidate_stable_updates)
    print(
        "Maximum candidate centroid updates:",
        args.max_candidate_centroid_updates,
    )
    print("Overwrite existing completed run:", args.force)
    print("State similarity threshold:", args.state_similarity_threshold)
    print("Permanent RMS threshold:", args.state_distance_threshold)
    print("Candidate cosine threshold:", args.candidate_similarity_threshold)
    print("Candidate RMS threshold:", args.candidate_distance_threshold)
    print("General matching: RMS plus cosine when norms are stable")
    print(
        "Normal general-match cosine/RMS relaxation:",
        args.general_cosine_relaxation,
        args.general_rms_relaxation,
    )
    print("Relaxed general matching keeps both safety gates strict: yes")
    print("Safety matching gate: RMS distance plus cosine when norms are stable")
    print("Capacity-pressure fallback:", args.close_enough_fallback)
    print(
        "Capacity fallback scope: candidate hard limit, permanent capacity, "
        "or eviction"
    )
    print(
        "Capacity fallback cosine/RMS relaxation:",
        args.general_cosine_relaxation,
        args.capacity_fallback_general_variation,
    )
    print(
        "Capacity fallback safety improvement:",
        args.capacity_fallback_safety_improvement,
    )
    print(
        "Directional safety: lane/threat larger; offset/heading smaller"
    )
    print("Speed directional fallback requirement: no; strict gates only")
    print("Capacity-fallback safety gates relaxed: no")
    print("Pool representatives use adaptive centroid freezing: yes")
    print("Centroid shift threshold:", args.centroid_shift_threshold)
    print("Consecutive stable updates required:", args.centroid_stable_updates)
    print("Maximum centroid updates per pool:", args.max_centroid_updates)
    print(
        "Centroid stability distance threshold:",
        args.centroid_stability_distance_threshold,
    )
    print("Pool-active training fraction:", args.pool_training_fraction)
    print(
        "Final pure-DQN argmax training fraction:",
        1.0 - args.pool_training_fraction,
    )
    print(
        "Final phase: pure DQN argmax; no pool lookup, safety extraction, "
        "or safety masks"
    )
    print(
        "Pool mutations disabled from episode:",
        int(math.ceil(args.train_episodes * args.pool_training_fraction)),
    )
    print("Similarity metrics: cosine similarity + RMS distance")
    print("Pool lookup: preallocated vectorized linear matching")
    print("Representative storage dtype:", args.pool_storage_dtype)
    print("Replay sampling: indexed ring buffer, O(batch_size)")
    print("Test safety extraction: no (pure-DQN argmax only)")
    print("Available actions per new pool: 9")
    print("Candidate visits use outcome-filtered DQN ranking: yes")
    print("Candidate actions are recorded for mask initialization: yes")
    print("Promotion occurs before promotion-visit action selection: yes")
    print("Active initial pass: every eligible non-BLOCKED action once")
    print("Scheduled UNKNOWN-only passes: 1")
    print("UNKNOWN pass excludes BLOCKED actions: yes")
    print("Final UNKNOWN action waits for its five-step outcome: yes")
    print("Missing safety: skip pools and use fresh DQN argmax")
    print("Retired post-pass selection: 80% greedy, 20% UNKNOWN-only random")
    print("Post-pass greedy: highest-Q across SAFE and UNKNOWN")
    print("Post-pass random eligibility: UNKNOWN only; never SAFE or BLOCKED")
    print("No UNKNOWN on random draw: fall back to eligible DQN maximum")
    print("BLOCKED actions excluded while SAFE or UNKNOWN is available")
    print("All-BLOCKED fallback: least failures, then highest fresh DQN Q")
    print("Collision/out-of-road penalties:", args.collision_penalty, args.out_of_road_penalty)
    print("Time-limit truncations bootstrap in replay: yes")
    print("Stored retired action values or rankings: no")
    print("Removal: clear selected bit in O(1)")
    print("Promotion and retirement: O(1) status-tag changes")
    print("Pass advance/retirement: after observing final transition")
    print("Absolute permanent capacity full: do not create new candidates; use maximum-Q")
    print("Availability mask refills: once, for the UNKNOWN-only pass")
    print("RMST event/tau:", args.rmst_event, args.rmst_tau)
    print("=" * 76)

    all_rows: List[Dict] = []
    runtimes: List[Dict] = []
    final_action_pools: Optional[SimilarStateActionPools] = None
    for experiment in EXPERIMENTS:
        experiment_rows, runtime_rows, action_pools = run_experiment(
            experiment, args, device, output_dir
        )
        all_rows.extend(experiment_rows)
        runtimes.extend(runtime_rows)
        final_action_pools = action_pools

    save_outputs(all_rows, runtimes, args, output_dir)
    if final_action_pools is not None:
        save_pool_statistics(final_action_pools, args, output_dir)
    required_outputs_before_manifest = (
        output_dir / "all_episode_results.csv",
        output_dir / "all_episode_results_detailed.csv",
        output_dir / "config.json",
        output_dir / "runtime_statistics.csv",
        output_dir / "collision_metrics.csv",
        output_dir / "policy_safety_stop_metrics.csv",
        output_dir / "model.pt",
        output_dir / "models" / "Karthikeya27adv8_safety_memory.pkl",
        output_dir / "state_pool_global_summary.csv",
        output_dir / "state_candidate_statistics.csv",
        output_dir / "state_retired_pool_statistics.csv",
        output_dir / "state_hazard_memory_statistics.csv",
        output_dir / "state_pool_statistics.csv",
        output_dir / "state_matching_calibration.csv",
        output_dir / "state_matching_rejection_statistics.csv",
        output_dir / "state_pool_capacity_growth.csv",
    )
    missing_outputs = [
        path
        for path in required_outputs_before_manifest
        if not path.is_file()
    ]
    if missing_outputs:
        raise RuntimeError(
            "Experiment finished but required outputs are missing: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    if final_action_pools is None:
        raise RuntimeError("No completed policy pool state is available.")
    manifest = write_completion_manifest(
        all_rows,
        runtimes,
        args,
        output_dir,
        final_action_pools,
    )
    if not (output_dir / "manifest.json").is_file():
        raise RuntimeError("Completion manifest was not created.")
    update_baseline_index(args, output_dir, manifest)
    print("\nExperiment completed successfully.")
    print("Episode results:", output_dir / "all_episode_results.csv")
    print("Frozen test policy: pure-DQN argmax")
    print("Collision metrics:", output_dir / "collision_metrics.csv")
    print(
        "Policy safety-stop metrics:",
        output_dir / "policy_safety_stop_metrics.csv",
    )
    print("Pool statistics:", output_dir / "state_pool_statistics.csv")
    print("Pool summary:", output_dir / "state_pool_global_summary.csv")
    print("Candidate statistics:", output_dir / "state_candidate_statistics.csv")
    print(
        "Evicted candidate history:",
        output_dir / "state_evicted_candidate_history.csv",
    )
    print(
        "Matching calibration:",
        output_dir / "state_matching_calibration.csv",
    )
    print(
        "Matching rejection statistics:",
        output_dir / "state_matching_rejection_statistics.csv",
    )
    print(
        "Capacity growth:",
        output_dir / "state_pool_capacity_growth.csv",
    )
    print(
        "Retired pool statistics:",
        output_dir / "state_retired_pool_statistics.csv",
    )
    print(
        "Hazard memory statistics:",
        output_dir / "state_hazard_memory_statistics.csv",
    )
    print("Runtime statistics:", output_dir / "runtime_statistics.csv")
    print("Manifest:", output_dir / "manifest.json")
    occupancy_plot = output_dir / "plots" / "state_pool_occupancy.png"
    if occupancy_plot.is_file():
        print("Pool occupancy plot:", occupancy_plot)
    else:
        print("Pool occupancy plot: not created (no active pools remained)")
    print("Policy folder name: Karthikeya27adv8")
    print("Results saved to:", output_dir)
