"""CLI parser construction, validation, and output-path preparation."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Optional, Sequence

from .constants import POLICY_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MetaDrive Karthikeya27adv8: Karthikeya20 baseline with retained "
            "blocked evidence and evidence-counted safety masks"
        )
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--max-state-pools",
        type=int,
        default=500,
        help=(
            "Fixed combined active + retired permanent capacity. "
            "Default: 500."
        ),
    )
    parser.add_argument(
        "--max-state-candidates",
        type=int,
        default=125,
        help=(
            "Soft candidate reception limit. Default: 125."
        ),
    )
    parser.add_argument(
        "--candidate-hard-limit",
        type=int,
        default=150,
        help=(
            "Hard candidate limit. Batch eviction occurs when this limit "
            "is reached. Default: 150."
        ),
    )
    parser.add_argument(
        "--candidate-batch-evict-count",
        type=int,
        default=25,
        help=(
            "Number of weakest candidates removed together at the hard limit. "
            "Default: 25."
        ),
    )
    parser.add_argument(
        "--candidate-promotion-visits",
        type=int,
        default=4,
        help=(
            "Visits required before promoting a candidate to a permanent "
            "pool. Default: 4."
        ),
    )
    parser.add_argument(
        "--hazard-memory-fraction",
        type=float,
        default=0.25,
        help=(
            "Hazard capacity as a fraction of maximum permanent capacity. "
            "Default 0.25 gives 125 hazards for 500 pools."
        ),
    )
    parser.add_argument(
        "--safe-confirmation-visits",
        type=int,
        default=2,
        help=(
            "Clean outcomes required before an action becomes confirmed safe. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--safety-horizon-steps",
        type=int,
        default=5,
        help="Future-step window required before a clean action is counted safe.",
    )
    parser.add_argument(
        "--minimum-progress-reward",
        type=float,
        default=0.01,
        help=(
            "Minimum immediate environment reward required for a clean "
            "five-step outcome to count toward SAFE. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--warning-block-threshold",
        type=int,
        default=2,
        help="Precursor warnings required before an action becomes sticky blocked.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--auto-calibrate-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit normalization and bounded matching tolerances in a fixed "
            "normalized space."
        ),
    )
    parser.add_argument(
        "--calibration-state-count",
        type=int,
        default=3000,
        help=(
            "Exact number of observed states using pure DQN argmax before "
            "DQN-plus-safety can start. Default: 3000."
        ),
    )
    parser.add_argument(
        "--calibration-max-pairs",
        type=int,
        default=20000,
        help="Maximum consecutive normalized-state pairs retained for calibration.",
    )
    parser.add_argument(
        "--safety-similarity-threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--safety-distance-threshold",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--candidate-safety-distance-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--safety-nearest-object-cap",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--safety-lane-boundary-cap",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--safety-speed-cap",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--safety-speed-fallback-unit",
        choices=["mps", "kmh"],
        default="mps",
        help=(
            "Unit assumed for vehicle.speed only when speed_km_h is "
            "unavailable."
        ),
    )
    parser.add_argument(
        "--capacity-review-interval",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--candidate-recent-protection-episodes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--state-similarity-threshold",
        type=float,
        default=0.90,
        help="Minimum cosine similarity required to reuse a state pool.",
    )
    parser.add_argument(
        "--state-distance-threshold",
        type=float,
        default=0.20,
        help=(
            "Maximum RMS distance required in addition to cosine "
            "similarity when matching a state pool."
        ),
    )
    parser.add_argument(
        "--candidate-similarity-threshold",
        type=float,
        default=0.90,
        help="Minimum cosine similarity required to reuse a temporary candidate.",
    )
    parser.add_argument(
        "--candidate-distance-threshold",
        type=float,
        default=0.25,
        help="Maximum RMS distance for temporary candidate matching.",
    )
    parser.add_argument(
        "--close-enough-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "At candidate hard-limit, permanent-capacity pressure, or eviction, "
            "allow a general-state fallback while keeping both safety gates "
            "strict."
        ),
    )
    parser.add_argument(
        "--general-cosine-relaxation",
        type=float,
        default=0.02,
        help=(
            "Relative cosine relaxation for the general-state second-stage "
            "match. Both general gates and both strict safety gates must pass. "
            "Default: 0.02."
        ),
    )
    parser.add_argument(
        "--general-rms-relaxation",
        type=float,
        default=0.02,
        help=(
            "Relative RMS relaxation for the normal general-state second-stage "
            "match. Default: 0.02."
        ),
    )
    parser.add_argument(
        "--capacity-fallback-general-variation",
        type=float,
        default=0.10,
        help=(
            "Allowed relative RMS expansion during pressure fallback. Cosine "
            "uses --general-cosine-relaxation. Default: 0.10."
        ),
    )
    parser.add_argument(
        "--capacity-fallback-safety-improvement",
        type=float,
        default=0.10,
        help=(
            "Required improvement in at least one direction-aware raw safety "
            "feature, while none may worsen. Default: 0.10."
        ),
    )
    parser.add_argument(
        "--candidate-centroid-shift-threshold",
        type=float,
        default=0.01,
        help=(
            "Relative candidate-centroid movement below which an update "
            "counts as stable."
        ),
    )
    parser.add_argument(
        "--candidate-stable-updates",
        type=int,
        default=2,
        help=(
            "Consecutive stable candidate-centroid updates required "
            "before freezing."
        ),
    )
    parser.add_argument(
        "--max-candidate-centroid-updates",
        type=int,
        default=4,
        help="Hard maximum centroid updates allowed per candidate.",
    )
    parser.add_argument(
        "--pool-training-fraction",
        type=float,
        default=0.80,
        help=(
            "Fraction of training episodes during which all pool behavior is "
            "active. The remaining episodes use pure maximum-Q actions."
        ),
    )
    parser.add_argument(
        "--centroid-shift-threshold",
        type=float,
        default=0.01,
        help=(
            "Relative centroid movement below which an update counts as stable."
        ),
    )
    parser.add_argument(
        "--centroid-stable-updates",
        type=int,
        default=3,
        help=(
            "Number of consecutive stable centroid updates required before freezing."
        ),
    )
    parser.add_argument(
        "--max-centroid-updates",
        type=int,
        default=10,
        help="Hard maximum centroid updates allowed per pool.",
    )
    parser.add_argument(
        "--centroid-stability-distance-threshold",
        type=float,
        default=0.10,
        help=(
            "Maximum recent mean RMS distance required when deciding "
            "that centroid updates are stable."
        ),
    )
    parser.add_argument(
        "--pool-storage-dtype",
        choices=["float16", "float32"],
        default="float32",
        help=(
            "Representative storage precision. Use float32 for the primary "
            "benchmark; float16 is a memory ablation and may change matches."
        ),
    )
    parser.add_argument("--discrete-steering-dim", type=int, default=3)
    parser.add_argument("--discrete-throttle-dim", type=int, default=3)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--traffic-density", type=float, default=0.20)
    parser.add_argument("--accident-prob", type=float, default=0.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    # MetaDrive expects positive penalty magnitudes and applies the minus sign.
    parser.add_argument("--collision-penalty", type=float, default=50.0)
    parser.add_argument("--out-of-road-penalty", type=float, default=50.0)
    parser.add_argument("--metadrive-log-level", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print training and testing progress every N episodes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing completed Karthikeya run.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training seed; also used in the canonical output folder name.",
    )
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rmst-tau",
        type=int,
        default=500,
        help="Restriction horizon in steps; canonical baseline default is 500.",
    )
    parser.add_argument(
        "--rmst-event",
        choices=["collision", "safety"],
        default="collision",
        help=(
            "collision: vehicle/object crash; safety: collision, off-road, "
            "or policy safety stop"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("policy_results"),
        help=(
            "Comparison output root. Results are written beneath "
            "<output-root>/seed_<seed>/Karthikeya27adv8."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Deprecated leaf-path compatibility option. If supplied, it must "
            "equal <output-root>/seed_<seed>/Karthikeya27adv8."
        ),
    )
    return parser

def validate_args(args) -> None:
    finite_numeric_fields = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "state_similarity_threshold": args.state_similarity_threshold,
        "state_distance_threshold": args.state_distance_threshold,
        "candidate_similarity_threshold": args.candidate_similarity_threshold,
        "candidate_distance_threshold": args.candidate_distance_threshold,
        "safety_similarity_threshold": args.safety_similarity_threshold,
        "safety_distance_threshold": args.safety_distance_threshold,
        "candidate_safety_distance_threshold": (
            args.candidate_safety_distance_threshold
        ),
        "general_cosine_relaxation": args.general_cosine_relaxation,
        "general_rms_relaxation": args.general_rms_relaxation,
        "capacity_fallback_general_variation": (
            args.capacity_fallback_general_variation
        ),
        "capacity_fallback_safety_improvement": (
            args.capacity_fallback_safety_improvement
        ),
        "minimum_progress_reward": args.minimum_progress_reward,
        "traffic_density": args.traffic_density,
        "accident_prob": args.accident_prob,
        "success_reward": args.success_reward,
        "collision_penalty": args.collision_penalty,
        "out_of_road_penalty": args.out_of_road_penalty,
        "pool_training_fraction": args.pool_training_fraction,
        "centroid_shift_threshold": args.centroid_shift_threshold,
        "candidate_centroid_shift_threshold": (
            args.candidate_centroid_shift_threshold
        ),
        "centroid_stability_distance_threshold": (
            args.centroid_stability_distance_threshold
        ),
    }
    invalid_numeric = [
        name
        for name, value in finite_numeric_fields.items()
        if not math.isfinite(float(value))
    ]
    if invalid_numeric:
        raise ValueError(
            "The following numeric arguments must be finite: "
            + ", ".join(invalid_numeric)
        )
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    if args.train_episodes <= 0:
        raise ValueError("--train-episodes must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.replay_capacity < args.batch_size:
        raise ValueError(
            "--replay-capacity must be at least --batch-size."
        )
    if args.target_update_steps <= 0:
        raise ValueError("--target-update-steps must be positive.")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")
    if args.test_episodes <= 0:
        raise ValueError("--test-episodes must be positive.")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive.")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")
    if args.max_state_pools <= 0:
        raise ValueError("--max-state-pools must be positive.")
    if args.maximum_pool_capacity < args.max_state_pools:
        raise ValueError(
            "--maximum-pool-capacity must be at least --max-state-pools."
        )
    if args.max_state_candidates <= 0:
        raise ValueError("--max-state-candidates must be positive.")
    if args.candidate_hard_limit <= args.max_state_candidates:
        raise ValueError(
            "--candidate-hard-limit must exceed --max-state-candidates."
        )
    if args.candidate_batch_evict_count <= 0:
        raise ValueError(
            "--candidate-batch-evict-count must be positive."
        )
    if (
        args.candidate_hard_limit
        - args.candidate_batch_evict_count
        != args.max_state_candidates
    ):
        raise ValueError(
            "--candidate-hard-limit minus --candidate-batch-evict-count "
            "must equal --max-state-candidates."
        )
    if args.candidate_promotion_visits <= 1:
        raise ValueError("--candidate-promotion-visits must be greater than 1.")
    if not 0.0 < args.hazard_memory_fraction <= 1.0:
        raise ValueError("--hazard-memory-fraction must be in (0, 1].")
    if args.hazard_memory_capacity <= 0:
        raise ValueError("Derived hazard-memory capacity must be positive.")
    if args.safe_confirmation_visits <= 0:
        raise ValueError("--safe-confirmation-visits must be positive.")
    if args.safety_horizon_steps <= 0:
        raise ValueError("--safety-horizon-steps must be positive.")
    if args.minimum_progress_reward < 0.0:
        raise ValueError(
            "--minimum-progress-reward must be non-negative."
        )
    if args.warning_block_threshold <= 0:
        raise ValueError("--warning-block-threshold must be positive.")
    if args.hazard_memory_capacity < args.safety_horizon_steps:
        raise ValueError(
            "Derived hazard capacity must be at least the safety horizon."
        )
    action_count = args.discrete_steering_dim * args.discrete_throttle_dim
    if args.candidate_promotion_visits - 1 >= action_count:
        raise ValueError(
            "--candidate-promotion-visits must leave at least one untried action "
            "on the promotion visit."
        )
    if not 0.0 <= args.state_similarity_threshold <= 1.0:
        raise ValueError("--state-similarity-threshold must be between 0 and 1.")
    if args.state_distance_threshold < 0.0:
        raise ValueError("--state-distance-threshold must be non-negative.")
    if not 0.0 <= args.candidate_similarity_threshold <= 1.0:
        raise ValueError(
            "--candidate-similarity-threshold must be between 0 and 1."
        )
    if args.candidate_distance_threshold < 0.0:
        raise ValueError("--candidate-distance-threshold must be non-negative.")
    if not 0.0 <= args.capacity_fallback_general_variation < 1.0:
        raise ValueError(
            "--capacity-fallback-general-variation must be in [0, 1)."
        )
    if not 0.0 <= args.general_cosine_relaxation < 1.0:
        raise ValueError(
            "--general-cosine-relaxation must be in [0, 1)."
        )
    if args.general_rms_relaxation < 0.0:
        raise ValueError(
            "--general-rms-relaxation must be non-negative."
        )
    if args.capacity_fallback_safety_improvement < 0.0:
        raise ValueError(
            "--capacity-fallback-safety-improvement must be non-negative."
        )
    if args.calibration_state_count < 2:
        raise ValueError("--calibration-state-count must be at least 2.")
    if args.calibration_max_pairs <= 0:
        raise ValueError("--calibration-max-pairs must be positive.")
    if not 0.0 <= args.safety_similarity_threshold <= 1.0:
        raise ValueError("--safety-similarity-threshold must be in [0,1].")
    if args.safety_distance_threshold < 0.0:
        raise ValueError("--safety-distance-threshold must be non-negative.")
    if args.candidate_safety_distance_threshold < 0.0:
        raise ValueError(
            "--candidate-safety-distance-threshold must be non-negative."
        )
    if args.capacity_review_interval <= 0:
        raise ValueError("--capacity-review-interval must be positive.")
    if args.candidate_recent_protection_episodes < 0:
        raise ValueError(
            "--candidate-recent-protection-episodes must be non-negative."
        )
    if (
        args.safety_nearest_object_cap <= 0.0
        or args.safety_lane_boundary_cap <= 0.0
        or args.safety_speed_cap <= 0.0
    ):
        raise ValueError("Safety feature caps must be positive.")
    if args.candidate_centroid_shift_threshold < 0.0:
        raise ValueError(
            "--candidate-centroid-shift-threshold must be non-negative."
        )
    if args.candidate_stable_updates <= 0:
        raise ValueError("--candidate-stable-updates must be positive.")
    if args.max_candidate_centroid_updates <= 0:
        raise ValueError(
            "--max-candidate-centroid-updates must be positive."
        )
    if not 0.0 < args.pool_training_fraction < 1.0:
        raise ValueError("--pool-training-fraction must be in the interval (0, 1).")
    pool_policy_boundary = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    if (
        args.auto_calibrate_thresholds
        and pool_policy_boundary * args.max_episode_steps
        < args.calibration_state_count
    ):
        raise ValueError(
            "The configured percentage-based policy phase cannot observe the "
            "fixed calibration-state target even if every episode reaches its "
            "maximum length."
        )
    if args.centroid_shift_threshold < 0.0:
        raise ValueError("--centroid-shift-threshold must be non-negative.")
    if args.centroid_stable_updates <= 0:
        raise ValueError("--centroid-stable-updates must be positive.")
    if args.max_centroid_updates <= 0:
        raise ValueError("--max-centroid-updates must be positive.")
    if args.centroid_stability_distance_threshold < 0.0:
        raise ValueError(
            "--centroid-stability-distance-threshold must be non-negative."
        )
    if args.rmst_tau <= 0:
        raise ValueError("--rmst-tau must be positive.")
    train_start, train_end = args.seed, args.seed + args.train_episodes - 1
    test_start, test_end = args.test_seed, args.test_seed + args.test_episodes - 1
    ranges = (
        ("training", train_start, train_end),
        ("testing", test_start, test_end),
    )
    for index, (name_a, start_a, end_a) in enumerate(ranges):
        for name_b, start_b, end_b in ranges[index + 1:]:
            if max(start_a, start_b) <= min(end_a, end_b):
                raise ValueError(f"{name_a} and {name_b} seed ranges overlap.")
    if args.collision_penalty < 0 or args.out_of_road_penalty < 0:
        raise ValueError(
            "MetaDrive penalty arguments must be non-negative magnitudes."
        )
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1.")
    if not 0.0 <= args.accident_prob <= 1.0:
        raise ValueError("--accident-prob must be between 0 and 1.")

def resolve_output_dir(args) -> Path:
    """Resolve the baseline-compatible seed and policy directory."""
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_dir = (
        output_root.resolve() / f"seed_{args.seed}" / POLICY_NAME
    )
    if args.output_dir is not None:
        supplied = Path(args.output_dir).expanduser()
        if not supplied.is_absolute():
            supplied = Path.cwd() / supplied
        supplied = supplied.resolve()
        if supplied != output_dir:
            raise ValueError(
                "--output-dir must agree with --output-root and use: "
                f"{output_dir}; received: {supplied}"
            )
    return output_dir

def prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Protect completed and partial runs from accidental overwrite."""
    if output_dir.exists():
        has_contents = any(output_dir.iterdir())
        if has_contents and not force:
            raise FileExistsError(
                f"Karthikeya27adv8 output directory is not empty: {output_dir}. "
                "Reuse it or pass --force to remove the previous partial/completed run."
            )
        if force:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse an optional argument list, enabling programmatic reuse and tests."""
    return build_parser().parse_args(argv)
