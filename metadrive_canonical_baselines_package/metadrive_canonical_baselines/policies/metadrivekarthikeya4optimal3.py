#!/usr/bin/env python3
"""MetaDrive Karthikeya4optimal7 DQN with hybrid safety-aware bounded one-pass action coverage.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-18 CEST (+0200)

Training policy
---------------
* Policy name: Karthikeya4optimal7.
* Hybrid design: Karthikeya2-style bounded candidate memory and stable
  centroids combined with safety-aware active/retired matching and adaptive
  permanent-pool capacity.
* Initial active+retired capacity is at least 1,000 and may grow to a maximum
  of 2,000. Capacity may grow every 25 episodes according to new-pool capacity
  pressure and never shrinks.
* Temporary candidates use a soft limit and a larger hard limit; once the
  hard limit is reached, the weakest candidates are evicted in one batch.
* The DQN receives the complete flattened MetaDrive observation.
* Pool matching requires both general-state similarity and safety similarity.
* The safety vector contains pre-action lane-boundary clearance, nearest collision-hazard
  centre distance, absolute lane offset, absolute heading error, and ego speed
  in km/h.
* After an initial pure-DQN learning period, general and safety normalizers/
  calibrators are fitted on training only and frozen.
* Active pools are checked first. Retired pools are checked only when no active
  pool matches.
* Otherwise the state matches or creates a temporary candidate.
* A candidate is promoted after two visits by default.
* Pool behavior starts only after the pure-DQN, normalization, and threshold-
  calibration warm-up is complete, and remains active until the configured
  pool-training boundary (80% of training by default).
* Candidate centroids freeze after three consecutive stable updates or ten
  updates. Active-pool centroids freeze after five consecutive stable updates
  or twenty updates by default.
* During the final 20% of training, pooling is completely disabled and every
  action is selected using normal maximum-Q.
* Candidate visits 1 through promotion-1 record the genuinely executed
  actions. Candidate-created, candidate-matched, and capacity-waiting steps
  use epsilon-greedy with epsilon 0.20 by default:
  80% DQN argmax and 20% uniformly random discrete actions.
* On the promotion visit, the candidate becomes permanent before action
  selection; previously executed candidate actions are removed from the
  new mask, then the highest-Q remaining action is selected and removed.
* At the current permanent-pool limit, promotion-ready candidates remain in
  the candidate zone and use argmax while waiting for the next capacity review.
  A review grows the current capacity, up to 2,000, and promotes waiting
  candidates immediately afterward. At the absolute 2,000-pool limit, new
  unmatched states use argmax without creating candidates.
* Each pool stores one 9-bit action-availability mask.
* Select the maximum-Q action among actions whose availability bits remain set.
* The selected action bit is cleared in O(1).
* Empty masks retire the active pool only after the transition caused by the
  final directly selected action is observed.
* When combined active + retired capacity is full, no new pool is created.
* Candidate eviction checks only the selected eviction shortlist against active
  pools using the normal active-pool thresholds. A matching candidate transfers
  centroid, visits, and action history
  into the active pool before deletion. Eviction
  protects near-promotion and recently-created candidates when possible.
* Retired pools are lookup-only: direct current-state matches suppress repeated
  exploration and use DQN argmax. Candidate-level suppression is tracked
  separately and does not count as a retired current-state hit.
* Direct final-mask actions retire only after their transitions are observed.
  Candidate-history exhaustion is an evidence-based retirement with no action outcome.
* Masks are never refilled.

Complexity
----------
* Active, candidate, and retired lookup uses exact bounded linear matching.
  This guarantees that the globally best valid cosine/RMS match is selected.
* Empty check and bit clearing: O(1).
* Best-available action selection: O(A), where A = 9.
* Matching is O(B(D+S)) on an active hit and O((B+R+C)(D+S)) on a miss.
  Candidate eviction is O(C log C + E*B(D+S) + E*C) with Python-list
  removals; active-pool retirement is O(B). E is the eviction shortlist size.
* Replay insertion and indexed ring-buffer sampling are O(1) and
  O(batch_size), respectively.
* Representatives use float32 storage by default for threshold-stable benchmark
  behavior. Policy storage is O((B+C+R)(D+S) + B), with B + R bounded
  by max-state-pools.

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* Benchmark summaries are collision-focused; raw rewards remain only in the
  episode-level results for auditability.
* No RND and no count-based intrinsic reward.
* Frozen greedy testing with no pool lookup or safety-extraction overhead.
  Safety diagnostic columns in test rows are intentionally NaN.
* Disjoint training and testing scenarios.
* Environment traffic configuration matches the canonical baselines.
* Deterministic PyTorch settings match the canonical baselines.

Example:
    python policies/metadrivekarthikeya4optimal2.py \
      --seed 11 --test-seed 100000 \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda

Results are saved to:
    policy_results/seed_<seed>/Karthikeya4optimal7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# Must be configured before CUDA creates a cuBLAS context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import metadrive
    from metadrive import MetaDriveEnv
except ImportError as exc:
    raise SystemExit(
        "MetaDrive is not installed. Run: python -m pip install metadrive-simulator"
    ) from exc


EXPERIMENTS = ["Karthikeya4optimal7"]
SHORT_LABELS = {
    "Karthikeya4optimal7": "Karthikeya4optimal7",
}
COLORS = {
    "Karthikeya4optimal7": "#2ca02c",
}


CRITICAL_CONFIG_KEYS = (
    "train_episodes",
    "test_episodes",
    "max_episode_steps",
    "max_state_pools",
    "maximum_pool_capacity",
    "max_state_candidates",
    "candidate_hard_limit",
    "candidate_batch_evict_count",
    "candidate_promotion_visits",
    "candidate_epsilon",
    "auto_calibrate_thresholds",
    "calibration_episodes",
    "calibration_start_episode",
    "calibration_max_pairs",
    "safety_similarity_threshold",
    "safety_distance_threshold",
    "candidate_safety_distance_threshold",
    "safety_nearest_object_cap",
    "safety_lane_boundary_cap",
    "safety_speed_cap",
    "safety_speed_fallback_unit",
    "capacity_review_interval",
    "candidate_recent_protection_episodes",
    "state_similarity_threshold",
    "state_distance_threshold",
    "candidate_similarity_threshold",
    "candidate_distance_threshold",
    "candidate_centroid_shift_threshold",
    "candidate_stable_updates",
    "max_candidate_centroid_updates",
    "learning_rate",
    "gamma",
    "batch_size",
    "replay_capacity",
    "target_update_steps",
    "hidden_size",
    "discrete_steering_dim",
    "discrete_throttle_dim",
    "map_blocks",
    "traffic_density",
    "accident_prob",
    "success_reward",
    "collision_penalty",
    "out_of_road_penalty",
    "test_seed",
    "rmst_tau",
    "rmst_event",
    "deterministic",
    "pool_training_fraction",
    "centroid_shift_threshold",
    "centroid_stable_updates",
    "max_centroid_updates",
    "centroid_stability_distance_threshold",
    "pool_storage_dtype",
    "hybrid_design_version",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch using deterministic baseline settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_observation(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(data: Dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def observation_sha256(observation) -> str:
    array = np.ascontiguousarray(flatten_observation(observation))
    return hashlib.sha256(array.tobytes()).hexdigest()


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0



def restricted_mean_survival_time(
    times: Sequence[float], events: Sequence[bool], tau: float
) -> float:
    """Kaplan-Meier RMST integral from step 0 through ``tau``.

    ``times`` contains event or censoring times. ``events`` is True only when
    the selected failure event occurred. Goal and horizon endings are censored.
    """
    tau = float(tau)
    if tau <= 0 or len(times) == 0:
        return 0.0
    t = np.minimum(np.asarray(times, dtype=float), tau)
    e = np.asarray(events, dtype=bool) & (np.asarray(times, dtype=float) <= tau)
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(t[t <= tau]):
        current = float(current)
        area += survival * max(0.0, current - previous)
        at_risk = int(np.sum(t >= current))
        failures = int(np.sum(e & np.isclose(t, current)))
        if at_risk > 0 and failures > 0:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, tau - previous)
    return float(area)










# ---------------------------------------------------------------------------
# Safety-aware pool representation
# ---------------------------------------------------------------------------

SAFETY_VECTOR_NAMES = (
    "lane_boundary_clearance",
    "nearest_collision_hazard_center_distance",
    "absolute_lane_offset",
    "absolute_heading_error",
    "ego_speed_km_h",
)


def _finite_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _angle_difference_radians(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _vehicle_lane(vehicle):
    navigation = getattr(vehicle, "navigation", None)
    lane = getattr(navigation, "current_lane", None)
    return lane if lane is not None else getattr(vehicle, "lane", None)


def _is_collision_hazard(obj) -> bool:
    """Best-effort identification of physical collision hazards.

    This intentionally avoids depending on one MetaDrive release's exact class
    names. It first rejects non-physical runtime helpers, then accepts known
    physical hazards, and finally rejects non-collidable map infrastructure.
    """
    class_name = obj.__class__.__name__.lower()
    module_name = obj.__class__.__module__.lower()
    text = f"{module_name}.{class_name}"

    helper_tokens = (
        "navigation", "camera", "sensor", "engine", "manager",
        "policy", "renderer", "nodepath",
    )
    if any(token in text for token in helper_tokens):
        return False

    included_tokens = (
        "vehicle", "pedestrian", "human", "cyclist", "bicycle",
        "cone", "barrier", "obstacle", "trafficobject", "traffic_object",
        "building", "sidewalk",
    )
    if any(token in text for token in included_tokens):
        return True

    infrastructure_tokens = (
        "lane", "road", "map", "light", "marking", "terrain",
    )
    if any(token in text for token in infrastructure_tokens):
        return False

    # Version-tolerant duck typing. A positioned object with collision geometry
    # or vehicle-like kinematics is treated as a hazard. Static map helpers are
    # filtered above.
    has_position = getattr(obj, "position", None) is not None
    has_collision_geometry = any(
        hasattr(obj, attribute)
        for attribute in (
            "collision_node", "collision_nodes", "body", "chassis",
            "top_down_width", "top_down_length", "WIDTH", "LENGTH",
        )
    )
    has_kinematics = any(
        hasattr(obj, attribute)
        for attribute in ("velocity", "speed", "speed_km_h", "heading_theta")
    )
    return bool(has_position and (has_collision_geometry or has_kinematics))


def _nearest_collision_hazard_distance(
    env, ego_position: np.ndarray, cap: float
) -> float:
    """Best-effort Euclidean distance to a collision-relevant object."""
    engine = getattr(env, "engine", None)
    if engine is None:
        return float(cap)

    objects = None
    getter = getattr(engine, "get_objects", None)
    if callable(getter):
        try:
            objects = getter()
        except TypeError:
            try:
                objects = getter(lambda _: True)
            except Exception:
                objects = None
        except Exception:
            objects = None
    if objects is None:
        objects = getattr(engine, "objects", None)

    if isinstance(objects, dict):
        iterable = objects.values()
    elif objects is None:
        iterable = ()
    else:
        iterable = objects

    ego = getattr(env, "vehicle", None)
    best = float(cap)
    for obj in iterable:
        if obj is ego or not _is_collision_hazard(obj):
            continue
        position = getattr(obj, "position", None)
        if position is None:
            continue
        try:
            point = np.asarray(position, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if point.size < 2:
            continue
        distance = float(np.linalg.norm(point[:2] - ego_position[:2]))
        if 1e-6 < distance < best:
            best = distance
    return float(min(best, cap))


def extract_safety_vector(env, args) -> np.ndarray:
    """Extract five pre-action safety variables without relying on obs indices."""
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return np.asarray(
            [
                args.safety_lane_boundary_cap,
                args.safety_nearest_object_cap,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    try:
        position = np.asarray(vehicle.position, dtype=np.float32).reshape(-1)
    except Exception:
        position = np.zeros(2, dtype=np.float32)
    if position.size < 2:
        position = np.pad(position, (0, 2 - position.size))

    speed_km_h_value = getattr(vehicle, "speed_km_h", None)
    if speed_km_h_value is not None:
        speed = _finite_float(speed_km_h_value, 0.0)
    else:
        fallback_speed = _finite_float(
            getattr(vehicle, "speed", 0.0), 0.0
        )
        if args.safety_speed_fallback_unit == "mps":
            speed = fallback_speed * 3.6
        else:
            speed = fallback_speed
    speed = float(np.clip(abs(speed), 0.0, args.safety_speed_cap))

    lane = _vehicle_lane(vehicle)
    lane_offset = 0.0
    lane_boundary_clearance = float(args.safety_lane_boundary_cap)
    heading_error = 0.0

    if lane is not None:
        longitudinal = 0.0
        local_coordinates = getattr(lane, "local_coordinates", None)
        if callable(local_coordinates):
            try:
                longitudinal, lateral = local_coordinates(position[:2])
                longitudinal = _finite_float(longitudinal, 0.0)
                lane_offset = abs(_finite_float(lateral, 0.0))
            except Exception:
                lane_offset = 0.0

        lane_width = 0.0
        width_at = getattr(lane, "width_at", None)
        if callable(width_at):
            try:
                lane_width = _finite_float(width_at(longitudinal), 0.0)
            except Exception:
                lane_width = 0.0
        if lane_width <= 0.0:
            width_value = getattr(lane, "width", 0.0)
            if callable(width_value):
                try:
                    width_value = width_value(longitudinal)
                except TypeError:
                    try:
                        width_value = width_value()
                    except Exception:
                        width_value = 0.0
                except Exception:
                    width_value = 0.0
            lane_width = _finite_float(width_value, 0.0)

        if lane_width > 0.0:
            lane_boundary_clearance = max(
                0.0, lane_width / 2.0 - lane_offset
            )
            lane_boundary_clearance = min(
                lane_boundary_clearance, args.safety_lane_boundary_cap
            )

        heading_at = getattr(lane, "heading_theta_at", None)
        vehicle_heading = _finite_float(
            getattr(vehicle, "heading_theta", 0.0), 0.0
        )
        if callable(heading_at):
            try:
                lane_heading = _finite_float(
                    heading_at(longitudinal), vehicle_heading
                )
                heading_error = _angle_difference_radians(
                    vehicle_heading, lane_heading
                )
            except Exception:
                heading_error = 0.0

    nearest_hazard = _nearest_collision_hazard_distance(
        env, position, float(args.safety_nearest_object_cap)
    )
    return np.asarray(
        [
            lane_boundary_clearance,
            nearest_hazard,
            lane_offset,
            heading_error,
            speed,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# MetaDrive environment and termination parsing
# ---------------------------------------------------------------------------


def make_env(args, phase: str) -> MetaDriveEnv:
    if phase not in {"train", "test"}:
        raise ValueError("phase must be 'train' or 'test'")
    start_seed = args.seed if phase == "train" else args.test_seed
    num_scenarios = args.train_episodes if phase == "train" else args.test_episodes
    config = {
        "use_render": bool(args.render),
        "image_observation": False,
        "log_level": int(args.metadrive_log_level),
        "discrete_action": True,
        "use_multi_discrete": False,
        "discrete_steering_dim": int(args.discrete_steering_dim),
        "discrete_throttle_dim": int(args.discrete_throttle_dim),
        "horizon": int(args.max_episode_steps),
        "truncate_as_terminate": False,
        "start_seed": int(start_seed),
        "num_scenarios": int(max(1, num_scenarios)),
        "map": int(args.map_blocks),
        "traffic_density": float(args.traffic_density),
        "random_traffic": False,
        "accident_prob": float(args.accident_prob),
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "out_of_road_done": True,
        "success_reward": float(args.success_reward),
        "crash_vehicle_penalty": float(args.collision_penalty),
        "crash_object_penalty": float(args.collision_penalty),
        "out_of_road_penalty": float(args.out_of_road_penalty),
    }
    return MetaDriveEnv(config)


def truthy(info: Dict, *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


def parse_step_info(info: Dict, terminated: bool, truncated: bool) -> Dict:
    crash_vehicle = truthy(info, "crash_vehicle")
    crash_object = truthy(
        info,
        "crash_object",
        "crash_building",
        "crash_human",
        "crash_sidewalk",
    )
    collision = crash_vehicle or crash_object or truthy(info, "crash", "crashed")
    out_of_road = truthy(info, "out_of_road")
    goal_reached = truthy(info, "arrive_dest", "arrived", "success")
    max_steps = truthy(info, "max_step") or bool(truncated)

    if collision:
        reason = "collision"
    elif out_of_road:
        reason = "out_of_road"
    elif goal_reached:
        reason = "goal"
    elif max_steps:
        reason = "max_steps"
    elif terminated:
        reason = "terminated"
    else:
        reason = "running"

    return {
        "termination_reason": reason,
        "collision": collision,
        "crash_vehicle": crash_vehicle,
        "crash_object": crash_object,
        "out_of_road": out_of_road,
        "goal_reached": goal_reached,
        "max_steps_reached": max_steps,
        "step_cost": float(info.get("cost", 0.0) or 0.0),
    }


def selected_rmst_event(parsed: Dict, event_definition: str) -> bool:
    if event_definition == "collision":
        return bool(parsed["collision"])
    if event_definition == "safety":
        return bool(parsed["collision"] or parsed["out_of_road"])
    raise ValueError(f"Unknown RMST event definition: {event_definition}")



# ---------------------------------------------------------------------------
# Training-only normalization and threshold calibration
# ---------------------------------------------------------------------------


class RunningObservationNormalizer:
    """Welford normalizer fitted on training data and then frozen."""

    def __init__(self, dimension: int, epsilon: float = 1e-6):
        self.dimension = int(dimension)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)
        self.frozen = False

    def update(self, state: np.ndarray) -> None:
        if self.frozen:
            return
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("Observation dimension changed during normalization.")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / float(self.count)
        delta2 = vector - self.mean
        self.m2 += delta * delta2

    def standard_deviation(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dimension, dtype=np.float64)
        variance = self.m2 / float(self.count - 1)
        return np.sqrt(np.maximum(variance, self.epsilon))

    def transform(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("Observation dimension changed during normalization.")
        normalized = (vector - self.mean) / (
            self.standard_deviation() + self.epsilon
        )
        return normalized.astype(np.float32)

    def freeze(self) -> None:
        self.frozen = True

    def statistics(self) -> Dict:
        std = self.standard_deviation()
        return {
            "normalizer_observation_count": int(self.count),
            "normalizer_frozen": bool(self.frozen),
            "normalizer_mean_absolute_mean": float(np.mean(np.abs(self.mean))),
            "normalizer_mean_standard_deviation": float(np.mean(std)),
            "normalizer_min_standard_deviation": float(np.min(std)),
            "normalizer_max_standard_deviation": float(np.max(std)),
        }


class SimilarityThresholdCalibrator:
    """Calibrate matching thresholds in one fixed normalized coordinate space."""

    def __init__(
        self,
        max_pairs: int,
        fallback_similarity: float,
        fallback_distance: float,
        fallback_candidate_distance: float,
        seed: int,
    ):
        self.max_pairs = int(max_pairs)
        self.fallback_similarity = float(fallback_similarity)
        self.fallback_distance = float(fallback_distance)
        self.fallback_candidate_distance = float(fallback_candidate_distance)
        self.rng = random.Random(int(seed))
        self.positive_cosines: Deque[float] = deque(maxlen=self.max_pairs)
        self.positive_distances: Deque[float] = deque(maxlen=self.max_pairs)
        self.negative_cosines: Deque[float] = deque(maxlen=self.max_pairs)
        self.negative_distances: Deque[float] = deque(maxlen=self.max_pairs)
        self.previous: Optional[np.ndarray] = None
        self.reference_buffer: Deque[np.ndarray] = deque(maxlen=512)
        self.frozen = False

    @staticmethod
    def pair_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an == 0.0 or bn == 0.0:
            cosine = 1.0 if np.array_equal(a, b) else 0.0
        else:
            cosine = float(np.dot(a, b) / (an * bn))
        # RMS distance is stable in a z-scored space and does not divide by
        # a potentially near-zero representative norm.
        rms_distance = float(np.sqrt(np.mean(np.square(a - b))))
        return cosine, rms_distance

    def reset_episode(self) -> None:
        self.previous = None

    def observe(self, normalized_state: np.ndarray) -> None:
        if self.frozen:
            return
        vector = np.asarray(normalized_state, dtype=np.float32).copy()

        # Adjacent states are treated as likely-positive pairs.
        if self.previous is not None:
            cosine, distance = self.pair_metrics(vector, self.previous)
            if np.isfinite(cosine) and np.isfinite(distance):
                self.positive_cosines.append(cosine)
                self.positive_distances.append(distance)

        # Random older states provide likely-negative contrast pairs.
        if len(self.reference_buffer) >= 8:
            reference = self.reference_buffer[
                self.rng.randrange(len(self.reference_buffer))
            ]
            cosine, distance = self.pair_metrics(vector, reference)
            if np.isfinite(cosine) and np.isfinite(distance):
                self.negative_cosines.append(cosine)
                self.negative_distances.append(distance)

        self.reference_buffer.append(vector)
        self.previous = vector

    def derive(self) -> Tuple[float, float, float]:
        if (
            len(self.positive_cosines) < 20
            or len(self.positive_distances) < 20
        ):
            return (
                self.fallback_similarity,
                self.fallback_distance,
                self.fallback_candidate_distance,
            )

        positive_cosine_q25 = float(
            np.percentile(self.positive_cosines, 25)
        )
        positive_distance_q75 = float(
            np.percentile(self.positive_distances, 75)
        )

        if self.negative_cosines and self.negative_distances:
            negative_cosine_q90 = float(
                np.percentile(self.negative_cosines, 90)
            )
            negative_distance_q10 = float(
                np.percentile(self.negative_distances, 10)
            )
            cosine = max(
                positive_cosine_q25,
                negative_cosine_q90,
            )
            distance = min(
                positive_distance_q75,
                max(positive_distance_q75 * 0.5, negative_distance_q10),
            )
        else:
            cosine = positive_cosine_q25
            distance = positive_distance_q75

        cosine = max(0.85, min(0.995, cosine))
        distance = max(0.05, min(2.0, distance))
        candidate_distance = min(2.5, max(distance, distance * 1.20))
        return cosine, distance, candidate_distance

    def freeze(self) -> None:
        self.frozen = True

    def statistics(self) -> Dict:
        return {
            "calibration_positive_pair_count": len(self.positive_cosines),
            "calibration_negative_pair_count": len(self.negative_cosines),
            "threshold_calibrator_frozen": bool(self.frozen),
            "calibration_positive_cosine_q25": (
                float(np.percentile(self.positive_cosines, 25))
                if self.positive_cosines
                else math.nan
            ),
            "calibration_positive_distance_q75": (
                float(np.percentile(self.positive_distances, 75))
                if self.positive_distances
                else math.nan
            ),
            "calibration_negative_cosine_q90": (
                float(np.percentile(self.negative_cosines, 90))
                if self.negative_cosines
                else math.nan
            ),
            "calibration_negative_distance_q10": (
                float(np.percentile(self.negative_distances, 10))
                if self.negative_distances
                else math.nan
            ),
        }


# ---------------------------------------------------------------------------
# Plain DQN and replay
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity indexed ring buffer with O(batch_size) sampling."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        self.data: List[Optional[Transition]] = [None] * self.capacity
        self.size = 0
        self.next_index = 0
        self.rng = random.Random(int(seed))

    def add(self, state, action, reward, next_state, done) -> None:
        self.data[self.next_index] = Transition(
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )
        self.next_index = (self.next_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        batch_size = int(batch_size)
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than stored.")
        indices = self.rng.sample(range(self.size), batch_size)
        batch = [self.data[index] for index in indices]
        if any(item is None for item in batch):
            raise RuntimeError("Replay buffer contained an uninitialized slot.")
        return [item for item in batch if item is not None]

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_count: int,
        args,
        device: torch.device,
    ):
        self.observation_size = int(observation_size)
        self.action_count = int(action_count)
        self.gamma = float(args.gamma)
        self.batch_size = int(args.batch_size)
        self.target_update_steps = int(args.target_update_steps)
        self.device = device
        self.learn_steps = 0

        self.online = QNetwork(observation_size, action_count, args.hidden_size).to(device)
        self.target = QNetwork(observation_size, action_count, args.hidden_size).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=args.learning_rate)
        self.replay = ReplayBuffer(args.replay_capacity, args.seed + 20_003)

    def tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        was_training = self.online.training
        self.online.eval()
        with torch.no_grad():
            q = self.online(self.tensor(state))[0].detach().cpu().numpy()
        if was_training:
            self.online.train()
        return q.astype(float)

    @staticmethod
    def _deterministic_extreme_from_q(
        q: np.ndarray,
        maximum: bool,
        key: str,
    ) -> int:
        """Choose a tied extreme deterministically, matching the baseline style."""
        extreme = float(np.max(q) if maximum else np.min(q))
        candidates = np.flatnonzero(q == extreme)
        if candidates.size == 0:
            raise RuntimeError("No action was available in the Q-value vector.")
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % int(candidates.size)
        return int(candidates[offset])

    def greedy_action(self, state: np.ndarray, key: str = "greedy") -> int:
        q = self.q_values(state)
        return self._deterministic_extreme_from_q(q, maximum=True, key=key)


    def learn(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        predicted_q = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def freeze(self) -> None:
        for module in (self.online, self.target):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def save(self, path: Path, args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
            },
            path,
        )


# ---------------------------------------------------------------------------
# Karthikeya action-availability masks
# ---------------------------------------------------------------------------


class SimilarStateActionPools:
    """Two-stage general-state pools with a separate safety matching gate."""

    def __init__(
        self,
        max_pools: int,
        maximum_pool_capacity: int,
        max_candidates: int,
        candidate_hard_limit: int,
        candidate_batch_evict_count: int,
        candidate_promotion_visits: int,
        candidate_recent_protection_episodes: int,
        capacity_review_interval: int,
        action_count: int,
        observation_size: int,
        safety_size: int,
        similarity_threshold: float,
        distance_threshold: float,
        candidate_similarity_threshold: float,
        candidate_distance_threshold: float,
        safety_similarity_threshold: float,
        safety_distance_threshold: float,
        candidate_safety_distance_threshold: float,
        auto_calibrate_thresholds: bool,
        calibration_episodes: int,
        calibration_start_episode: int,
        calibration_max_pairs: int,
        seed: int,
        candidate_centroid_shift_threshold: float,
        candidate_stable_updates: int,
        max_candidate_centroid_updates: int,
        centroid_shift_threshold: float,
        centroid_stable_updates: int,
        max_centroid_updates: int,
        centroid_stability_distance_threshold: float,
        pool_storage_dtype: str,
    ):
        if max_pools <= 0 or maximum_pool_capacity < max_pools:
            raise ValueError("Invalid permanent-pool capacities.")
        if max_candidates <= 0 or candidate_hard_limit <= max_candidates:
            raise ValueError("Invalid candidate capacities.")
        if candidate_hard_limit - candidate_batch_evict_count != max_candidates:
            raise ValueError(
                "candidate_hard_limit - candidate_batch_evict_count "
                "must equal max_candidates."
            )

        self.initial_max_pools = int(max_pools)
        self.max_pools = int(max_pools)
        self.maximum_pool_capacity = int(maximum_pool_capacity)
        self.max_candidates = int(max_candidates)
        self.candidate_hard_limit = int(candidate_hard_limit)
        self.candidate_batch_evict_count = int(candidate_batch_evict_count)
        self.candidate_promotion_visits = int(candidate_promotion_visits)
        self.candidate_recent_protection_episodes = int(
            candidate_recent_protection_episodes
        )
        self.capacity_review_interval = int(capacity_review_interval)
        self.action_count = int(action_count)
        self.full_mask = (1 << self.action_count) - 1
        self.storage_dtype = np.float16 if pool_storage_dtype == "float16" else np.float32

        self.similarity_threshold = float(similarity_threshold)
        self.distance_threshold = float(distance_threshold)
        self.candidate_similarity_threshold = float(
            candidate_similarity_threshold
        )
        self.candidate_distance_threshold = float(candidate_distance_threshold)
        self.safety_similarity_threshold = float(safety_similarity_threshold)
        self.safety_distance_threshold = float(safety_distance_threshold)
        self.candidate_safety_distance_threshold = float(
            candidate_safety_distance_threshold
        )

        self.auto_calibrate_thresholds = bool(auto_calibrate_thresholds)
        self.calibration_episodes = int(calibration_episodes)
        self.calibration_start_episode = int(calibration_start_episode)
        self.normalization_episodes = (
            max(1, self.calibration_episodes // 2)
            if self.auto_calibrate_thresholds
            else 0
        )
        self.thresholds_frozen = not self.auto_calibrate_thresholds
        self.normalizer = RunningObservationNormalizer(observation_size)
        self.safety_normalizer = RunningObservationNormalizer(safety_size)
        self.calibrator = SimilarityThresholdCalibrator(
            calibration_max_pairs,
            similarity_threshold,
            distance_threshold,
            candidate_distance_threshold,
            seed + 91_337,
        )
        self.safety_calibrator = SimilarityThresholdCalibrator(
            calibration_max_pairs,
            safety_similarity_threshold,
            safety_distance_threshold,
            candidate_safety_distance_threshold,
            seed + 191_337,
        )

        self.candidate_centroid_shift_threshold = float(
            candidate_centroid_shift_threshold
        )
        self.candidate_stable_updates_required = int(candidate_stable_updates)
        self.max_candidate_centroid_updates = int(
            max_candidate_centroid_updates
        )
        self.centroid_shift_threshold = float(centroid_shift_threshold)
        self.centroid_stable_updates_required = int(centroid_stable_updates)
        self.max_centroid_updates = int(max_centroid_updates)
        self.centroid_stability_distance_threshold = float(
            centroid_stability_distance_threshold
        )

        # Active pools.
        self.representatives: List[np.ndarray] = []
        self.safety_representatives: List[np.ndarray] = []
        self.representative_norms: List[float] = []
        self.safety_representative_norms: List[float] = []
        self.active_pool_ids: List[int] = []
        self.next_pool_id = 0
        self.masks: List[int] = []
        self.visit_counts: List[int] = []
        self.promotion_evidence_visits: List[int] = []
        self.absorbed_candidate_visits: List[int] = []
        self.absorbed_candidate_actions: List[int] = []
        self.first_episode_created: List[int] = []
        self.last_episode_visited: List[int] = []
        self.match_counts: List[int] = []
        self.similarity_sums: List[float] = []
        self.distance_sums: List[float] = []
        self.safety_similarity_sums: List[float] = []
        self.safety_distance_sums: List[float] = []
        self.centroid_update_counts: List[int] = []
        self.centroid_stable_counts: List[int] = []
        self.centroid_last_shifts: List[float] = []
        self.centroid_frozen_by_stability: List[bool] = []
        self.centroid_frozen_by_cap: List[bool] = []
        self.recent_distance_windows: List[Deque[float]] = []

        # Candidate clusters.
        self.candidate_ids: List[int] = []
        self.next_candidate_id = 0
        self.candidate_representatives: List[np.ndarray] = []
        self.candidate_safety_representatives: List[np.ndarray] = []
        self.candidate_norms: List[float] = []
        self.candidate_safety_norms: List[float] = []
        self.candidate_visits: List[int] = []
        self.candidate_first_episode: List[int] = []
        self.candidate_last_episode: List[int] = []
        self.candidate_centroid_update_counts: List[int] = []
        self.candidate_stable_counts: List[int] = []
        self.candidate_last_shifts: List[float] = []
        self.candidate_centroid_frozen: List[bool] = []
        self.candidate_action_masks: List[int] = []

        # Retired lookup-only memory.
        self.retired_representatives: List[np.ndarray] = []
        self.retired_safety_representatives: List[np.ndarray] = []
        self.retired_norms: List[float] = []
        self.retired_safety_norms: List[float] = []
        self.retired_original_pool_ids: List[int] = []
        self.retired_reasons: List[str] = []
        self.retired_episode_created: List[int] = []
        self.retired_episode_retired: List[int] = []
        self.retired_permanent_visits: List[int] = []
        self.retired_actions_explored: List[int] = []
        self.retired_hit_counts: List[int] = []
        self.retired_last_hit_episode: List[int] = []
        self.retired_last_hit_step: List[int] = []
        self.retired_similarity_sums: List[float] = []
        self.retired_distance_sums: List[float] = []
        self.retired_safety_similarity_sums: List[float] = []
        self.retired_safety_distance_sums: List[float] = []
        self.retirement_trigger_action: List[int] = []
        self.retirement_trigger_reward: List[float] = []
        self.retirement_trigger_collision: List[bool] = []
        self.retirement_trigger_out_of_road: List[bool] = []
        self.retirement_trigger_done: List[bool] = []
        self.retirement_trigger_step: List[int] = []
        self.retirement_trigger_type: List[str] = []

        self.pending_retirement: Optional[Dict] = None
        self.evicted_candidate_rows: List[Dict] = []
        self.creation_events: List[Tuple[int, int]] = []
        self.capacity_growth_rows: List[Dict] = []

        # Counts.
        self.total_states_seen = 0
        self.total_calibration_states = 0
        self.total_pool_matches = 0
        self.total_pool_creations = 0
        self.total_retired_pool_hits = 0
        self.total_final_argmax_states = 0
        self.candidates_created = 0
        self.candidates_promoted = 0
        self.candidates_merged_into_active_pool = 0
        self.candidate_action_history_transfers = 0
        self.candidate_action_bits_removed_by_transfer = 0
        self.candidates_evicted = 0
        self.candidates_blocked_by_capacity = 0
        self.candidate_capacity_wait_events = 0
        self.candidates_promoted_after_capacity_growth = 0
        self.absolute_capacity_argmax_states = 0
        self.candidates_suppressed_by_retired_pool = 0
        self.candidate_history_exhaustion_events = 0
        self.candidate_retired_suppression_events = 0
        self.pre_eviction_candidates_checked = 0
        self.pre_eviction_candidates_merged = 0
        self.near_promotion_candidates_protected = 0
        self.recent_candidates_protected = 0
        self.active_pools_retired_by_mask_exhaustion = 0
        self.direct_mask_exhaustion_retirements = 0
        self.candidate_history_exhaustion_retirements = 0
        self.last_review_promoted = 0
        self.last_review_blocked = 0
        self.last_review_capacity_wait_events = 0

    # ---------- Calibration ----------

    def begin_episode(self) -> None:
        self.calibrator.reset_episode()
        self.safety_calibrator.reset_episode()

    def prepare_matching_state(
        self,
        raw_state: np.ndarray,
        raw_safety: np.ndarray,
        episode: int,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
        safety = np.asarray(raw_safety, dtype=np.float32).reshape(-1)
        if not self.auto_calibrate_thresholds:
            return state, safety, "ready"

        if episode < self.calibration_start_episode:
            return state, safety, "pre_calibration_learning"

        calibration_episode = episode - self.calibration_start_episode
        if calibration_episode < self.normalization_episodes:
            self.normalizer.update(state)
            self.safety_normalizer.update(safety)
            self.total_calibration_states += 1
            return state, safety, "normalizer_warmup"

        if not self.normalizer.frozen:
            self.normalizer.freeze()
            self.safety_normalizer.freeze()

        state_n = self.normalizer.transform(state)
        safety_n = self.safety_normalizer.transform(safety)

        if calibration_episode < self.calibration_episodes:
            self.calibrator.observe(state_n)
            self.safety_calibrator.observe(safety_n)
            self.total_calibration_states += 1
            return state_n, safety_n, "threshold_warmup"

        if not self.thresholds_frozen:
            cosine, distance, candidate_distance = self.calibrator.derive()
            s_cosine, s_distance, candidate_s_distance = (
                self.safety_calibrator.derive()
            )
            # Retain the safety gate without allowing calibration to become
            # so strict that nearly every state becomes a new candidate.
            s_cosine = min(float(s_cosine), 0.98)
            s_distance = max(float(s_distance), 0.10)
            candidate_s_distance = max(
                float(candidate_s_distance),
                min(0.20, s_distance * 1.25),
            )
            self.similarity_threshold = cosine
            self.candidate_similarity_threshold = cosine
            self.distance_threshold = distance
            self.candidate_distance_threshold = candidate_distance
            self.safety_similarity_threshold = s_cosine
            self.safety_distance_threshold = s_distance
            self.candidate_safety_distance_threshold = candidate_s_distance
            self.calibrator.freeze()
            self.safety_calibrator.freeze()
            self.thresholds_frozen = True
        return state_n, safety_n, "ready"

    # ---------- Matching ----------

    @staticmethod
    def _cosine(a, an, b, bn) -> float:
        if an == 0.0 or bn == 0.0:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _rms(a, b) -> float:
        return float(np.sqrt(np.mean(np.square(a - b))))

    def _best_match(
        self,
        state: np.ndarray,
        safety: np.ndarray,
        representatives: Sequence[np.ndarray],
        safety_representatives: Sequence[np.ndarray],
        norms: Sequence[float],
        safety_norms: Sequence[float],
        cosine_threshold: float,
        distance_threshold: float,
        safety_cosine_threshold: float,
        safety_distance_threshold: float,
        candidate_indices: Optional[Iterable[int]] = None,
    ) -> Tuple[Optional[int], float, float, float, float]:
        state_norm = float(np.linalg.norm(state))
        safety_norm = float(np.linalg.norm(safety))
        best = None
        best_score = float("-inf")
        best_values = (float("-inf"), float("inf"), float("-inf"), float("inf"))
        indices = range(len(representatives)) if candidate_indices is None else candidate_indices
        for index in indices:
            rep = np.asarray(representatives[index], dtype=np.float32)
            srep = np.asarray(safety_representatives[index], dtype=np.float32)
            rep_norm = norms[index]
            srep_norm = safety_norms[index]
            cosine = self._cosine(state, state_norm, rep, rep_norm)
            distance = self._rms(state, rep)
            safety_cosine = self._cosine(
                safety, safety_norm, srep, srep_norm
            )
            safety_distance = self._rms(safety, srep)
            general_cosine_ok = (
                state_norm < 1e-6
                or rep_norm < 1e-6
                or cosine >= cosine_threshold
            )
            safety_cosine_ok = (
                safety_norm < 1e-6
                or srep_norm < 1e-6
                or safety_cosine >= safety_cosine_threshold
            )
            if (
                general_cosine_ok
                and distance <= distance_threshold
                and safety_cosine_ok
                and safety_distance <= safety_distance_threshold
            ):
                general_cosine_component = (
                    1.0
                    if state_norm < 1e-6 or rep_norm < 1e-6
                    else cosine / max(cosine_threshold, 1e-8)
                )
                safety_cosine_component = (
                    1.0
                    if safety_norm < 1e-6 or srep_norm < 1e-6
                    else safety_cosine / max(safety_cosine_threshold, 1e-8)
                )
                score = (
                    general_cosine_component
                    + safety_cosine_component
                    - distance / max(distance_threshold, 1e-8)
                    - safety_distance / max(safety_distance_threshold, 1e-8)
                )
                if score > best_score:
                    best = index
                    best_score = score
                    best_values = (
                        cosine,
                        distance,
                        safety_cosine,
                        safety_distance,
                    )
        return (best, *best_values)

    def find_active_match(self, state, safety):
        return self._best_match(
            state, safety,
            self.representatives, self.safety_representatives,
            self.representative_norms, self.safety_representative_norms,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
            candidate_indices=None,
        )

    def find_retired_match(self, state, safety):
        return self._best_match(
            state, safety,
            self.retired_representatives, self.retired_safety_representatives,
            self.retired_norms, self.retired_safety_norms,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
            candidate_indices=None,
        )

    def find_candidate_match(self, state, safety):
        return self._best_match(
            state, safety,
            self.candidate_representatives,
            self.candidate_safety_representatives,
            self.candidate_norms,
            self.candidate_safety_norms,
            self.candidate_similarity_threshold,
            self.candidate_distance_threshold,
            self.safety_similarity_threshold,
            self.candidate_safety_distance_threshold,
            candidate_indices=None,
        )



    # ---------- Capacity ----------

    def total_permanent_records(self) -> int:
        return len(self.representatives) + len(self.retired_representatives)

    def permanent_capacity_available(self) -> bool:
        return self.total_permanent_records() < self.max_pools

    def absolute_permanent_capacity_reached(self) -> bool:
        return (
            self.total_permanent_records()
            >= self.maximum_pool_capacity
        )

    def review_capacity(self, episode: int) -> None:
        if (
            episode <= 0
            or episode % self.capacity_review_interval != 0
            or self.max_pools >= self.maximum_pool_capacity
        ):
            return
        promoted_delta = self.candidates_promoted - self.last_review_promoted
        waiting_delta = (
            self.candidate_capacity_wait_events
            - self.last_review_capacity_wait_events
        )
        denominator = promoted_delta + waiting_delta
        pressure = waiting_delta / max(1, denominator)
        has_promotion_ready_waiter = any(
            visits >= self.candidate_promotion_visits
            for visits in self.candidate_visits
        )
        if (
            self.total_permanent_records() >= self.max_pools
            and has_promotion_ready_waiter
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
            old_candidate_soft = self.max_candidates
            old_candidate_hard = self.candidate_hard_limit
            target_candidate_soft = min(
                500,
                max(100, int(math.ceil(0.25 * self.max_pools))),
            )
            self.max_candidates = max(
                self.max_candidates,
                target_candidate_soft,
            )
            self.candidate_hard_limit = max(
                self.max_candidates + 1,
                int(math.ceil(self.max_candidates * 1.20)),
            )
            self.candidate_batch_evict_count = (
                self.candidate_hard_limit - self.max_candidates
            )
            promoted_after_growth = (
                self._promote_waiting_candidates_after_growth(episode)
            )
            self.capacity_growth_rows.append(
                {
                    "episode": int(episode),
                    "capacity_before": int(old_capacity),
                    "capacity_after": int(self.max_pools),
                    "candidate_soft_before": int(old_candidate_soft),
                    "candidate_soft_after": int(self.max_candidates),
                    "candidate_hard_before": int(old_candidate_hard),
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
        self.last_review_blocked = self.candidates_blocked_by_capacity
        self.last_review_capacity_wait_events = (
            self.candidate_capacity_wait_events
        )

    # ---------- Active/retired storage ----------

    def _append_permanent_pool(
        self, state, safety, episode, initial_visits, initial_mask=None
    ) -> int:
        if not self.permanent_capacity_available():
            raise RuntimeError("Combined active + retired capacity is full.")
        state = np.asarray(state, dtype=self.storage_dtype).copy()
        safety = np.asarray(safety, dtype=self.storage_dtype).copy()
        self.representatives.append(state)
        self.safety_representatives.append(safety)
        self.representative_norms.append(float(np.linalg.norm(state)))
        self.safety_representative_norms.append(float(np.linalg.norm(safety)))
        self.active_pool_ids.append(self.next_pool_id)
        self.next_pool_id += 1
        self.masks.append(
            self.full_mask if initial_mask is None else int(initial_mask)
        )
        self.visit_counts.append(0)
        self.promotion_evidence_visits.append(int(initial_visits))
        self.absorbed_candidate_visits.append(0)
        self.absorbed_candidate_actions.append(0)
        self.first_episode_created.append(int(episode))
        self.last_episode_visited.append(int(episode))
        self.match_counts.append(0)
        self.similarity_sums.append(0.0)
        self.distance_sums.append(0.0)
        self.safety_similarity_sums.append(0.0)
        self.safety_distance_sums.append(0.0)
        self.centroid_update_counts.append(0)
        self.centroid_stable_counts.append(0)
        self.centroid_last_shifts.append(0.0)
        self.centroid_frozen_by_stability.append(False)
        self.centroid_frozen_by_cap.append(False)
        self.recent_distance_windows.append(deque(maxlen=5))
        self.total_pool_creations += 1
        self.creation_events.append((int(episode), self.total_pool_creations))
        index = len(self.representatives) - 1
        return index

    def _remove_active_pool(self, index: int) -> None:
        for sequence in (
            self.representatives,
            self.safety_representatives,
            self.representative_norms,
            self.safety_representative_norms,
            self.active_pool_ids,
            self.masks,
            self.visit_counts,
            self.promotion_evidence_visits,
            self.absorbed_candidate_visits,
            self.absorbed_candidate_actions,
            self.first_episode_created,
            self.last_episode_visited,
            self.match_counts,
            self.similarity_sums,
            self.distance_sums,
            self.safety_similarity_sums,
            self.safety_distance_sums,
            self.centroid_update_counts,
            self.centroid_stable_counts,
            self.centroid_last_shifts,
            self.centroid_frozen_by_stability,
            self.centroid_frozen_by_cap,
            self.recent_distance_windows,
        ):
            sequence.pop(index)

    def _retire_active_pool(self, index, episode, reason, trigger=None) -> int:
        retired_index = len(self.retired_representatives)
        self.retired_representatives.append(self.representatives[index].copy())
        self.retired_safety_representatives.append(
            self.safety_representatives[index].copy()
        )
        self.retired_norms.append(self.representative_norms[index])
        self.retired_safety_norms.append(
            self.safety_representative_norms[index]
        )
        self.retired_original_pool_ids.append(self.active_pool_ids[index])
        self.retired_reasons.append(str(reason))
        self.retired_episode_created.append(self.first_episode_created[index])
        self.retired_episode_retired.append(int(episode))
        self.retired_permanent_visits.append(self.visit_counts[index])
        self.retired_actions_explored.append(
            self.action_count - self.remaining_count(index)
        )
        self.retired_hit_counts.append(0)
        self.retired_last_hit_episode.append(-1)
        self.retired_last_hit_step.append(-1)
        self.retired_similarity_sums.append(0.0)
        self.retired_distance_sums.append(0.0)
        self.retired_safety_similarity_sums.append(0.0)
        self.retired_safety_distance_sums.append(0.0)
        trigger = trigger or {}
        self.retirement_trigger_action.append(int(trigger.get("action", -1)))
        self.retirement_trigger_reward.append(
            float(trigger.get("reward", math.nan))
        )
        self.retirement_trigger_collision.append(
            bool(trigger.get("collision", False))
        )
        self.retirement_trigger_out_of_road.append(
            bool(trigger.get("out_of_road", False))
        )
        self.retirement_trigger_done.append(bool(trigger.get("done", False)))
        self.retirement_trigger_step.append(int(trigger.get("step", -1)))
        self.retirement_trigger_type.append(
            str(trigger.get("retirement_trigger_type", "direct_final_action"))
        )
        if reason == "MASK_EXHAUSTED":
            self.direct_mask_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        elif reason == "CANDIDATE_HISTORY_EXHAUSTED":
            self.candidate_history_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        self._remove_active_pool(index)
        return retired_index

    def mark_pending_retirement(self, pool_index, episode, step, action):
        self.pending_retirement = {
            "pool_id": int(self.active_pool_ids[int(pool_index)]),
            "episode": int(episode),
            "step": int(step),
            "action": int(action),
            "retirement_trigger_type": "direct_final_action",
        }

    def finalize_pending_retirement(self, reward, parsed, done):
        if self.pending_retirement is None:
            return
        pool_id = self.pending_retirement["pool_id"]
        if pool_id not in self.active_pool_ids:
            self.pending_retirement = None
            return
        index = self.active_pool_ids.index(pool_id)
        trigger = {
            **self.pending_retirement,
            "reward": float(reward),
            "collision": bool(parsed.get("collision", False)),
            "out_of_road": bool(parsed.get("out_of_road", False)),
            "done": bool(done),
        }
        self._retire_active_pool(
            index,
            self.pending_retirement["episode"],
            "MASK_EXHAUSTED",
            trigger,
        )
        self.pending_retirement = None

    # ---------- Candidates ----------

    def _remove_candidate(self, index: int) -> None:
        for sequence in (
            self.candidate_ids,
            self.candidate_representatives,
            self.candidate_safety_representatives,
            self.candidate_norms,
            self.candidate_safety_norms,
            self.candidate_visits,
            self.candidate_first_episode,
            self.candidate_last_episode,
            self.candidate_centroid_update_counts,
            self.candidate_stable_counts,
            self.candidate_last_shifts,
            self.candidate_centroid_frozen,
            self.candidate_action_masks,
        ):
            sequence.pop(index)

    def absorb_candidate_into_active(
        self,
        candidate_index: int,
        pool_index: int,
        episode: int,
        pre_eviction: bool = False,
    ) -> Optional[int]:
        """Merge a matched candidate into an active pool.

        The candidate-to-active match is the sole eligibility check. Its
        genuinely executed actions are removed from the active availability
        mask, then the candidate is deleted.
        """
        visits = int(self.candidate_visits[candidate_index])
        history = int(self.candidate_action_masks[candidate_index])

        candidate_state = np.asarray(
            self.candidate_representatives[candidate_index],
            dtype=np.float32,
        )
        candidate_safety = np.asarray(
            self.candidate_safety_representatives[candidate_index],
            dtype=np.float32,
        )
        active_state = np.asarray(
            self.representatives[pool_index],
            dtype=np.float32,
        )
        active_safety = np.asarray(
            self.safety_representatives[pool_index],
            dtype=np.float32,
        )

        active_weight = max(
            1,
            self.promotion_evidence_visits[pool_index]
            + self.absorbed_candidate_visits[pool_index]
            + self.visit_counts[pool_index],
        )
        if not (
            self.centroid_frozen_by_stability[pool_index]
            or self.centroid_frozen_by_cap[pool_index]
        ):
            combined_weight = active_weight + visits
            merged_state = (
                active_state * active_weight + candidate_state * visits
            ) / float(combined_weight)
            merged_safety = (
                active_safety * active_weight + candidate_safety * visits
            ) / float(combined_weight)

            relative_shift = float(
                np.linalg.norm(merged_state - active_state)
                / max(float(np.linalg.norm(active_state)), 1e-8)
            )
            absolute_shift = self._rms(merged_state, active_state)
            safety_shift = self._rms(merged_safety, active_safety)
            shift = max(relative_shift, absolute_shift, safety_shift)

            self.representatives[pool_index] = merged_state.astype(
                self.storage_dtype
            )
            self.safety_representatives[pool_index] = merged_safety.astype(
                self.storage_dtype
            )
            self.representative_norms[pool_index] = float(
                np.linalg.norm(self.representatives[pool_index])
            )
            self.safety_representative_norms[pool_index] = float(
                np.linalg.norm(self.safety_representatives[pool_index])
            )

            self.centroid_update_counts[pool_index] += 1
            self.centroid_last_shifts[pool_index] = shift
            candidate_match_distance = self._rms(
                candidate_state, active_state
            )
            self.recent_distance_windows[pool_index].append(
                candidate_match_distance
            )
            recent_distance = (
                float(np.mean(self.recent_distance_windows[pool_index]))
                if self.recent_distance_windows[pool_index]
                else float("inf")
            )
            if (
                shift < self.centroid_shift_threshold
                and recent_distance
                <= self.centroid_stability_distance_threshold
            ):
                self.centroid_stable_counts[pool_index] += 1
            else:
                self.centroid_stable_counts[pool_index] = 0

            if (
                self.centroid_stable_counts[pool_index]
                >= self.centroid_stable_updates_required
            ):
                self.centroid_frozen_by_stability[pool_index] = True
            elif (
                self.centroid_update_counts[pool_index]
                >= self.max_centroid_updates
            ):
                self.centroid_frozen_by_cap[pool_index] = True

        before = int(self.masks[pool_index])
        self.masks[pool_index] = before & ~history
        removed_bits = (before & history).bit_count()

        self.absorbed_candidate_visits[pool_index] += visits
        self.absorbed_candidate_actions[pool_index] += history.bit_count()
        self.candidates_merged_into_active_pool += 1
        self.candidate_action_history_transfers += 1
        self.candidate_action_bits_removed_by_transfer += removed_bits
        if pre_eviction:
            self.pre_eviction_candidates_merged += 1

        self._remove_candidate(candidate_index)

        if self.mask(pool_index) == 0:
            self.candidate_history_exhaustion_events += 1
            return self._retire_active_pool(
                pool_index,
                episode,
                "CANDIDATE_HISTORY_EXHAUSTED",
                {
                    "action": -1,
                    "reward": math.nan,
                    "collision": False,
                    "out_of_road": False,
                    "done": False,
                    "step": -1,
                    "retirement_trigger_type": "candidate_history_evidence",
                },
            )
        return None

    def _promote_waiting_candidates_after_growth(
        self, episode: int
    ) -> int:
        """Promote capacity-waiting candidates immediately after growth."""
        promoted_to_new_pool = 0
        for candidate_index in range(
            len(self.candidate_representatives) - 1, -1, -1
        ):
            if (
                self.candidate_visits[candidate_index]
                < self.candidate_promotion_visits
            ):
                continue

            active_match = self.find_active_match(
                self.candidate_representatives[candidate_index],
                self.candidate_safety_representatives[candidate_index],
            )
            if active_match[0] is not None:
                self.absorb_candidate_into_active(
                    candidate_index,
                    int(active_match[0]),
                    episode,
                )
                continue

            retired_match = self.find_retired_match(
                self.candidate_representatives[candidate_index],
                self.candidate_safety_representatives[candidate_index],
            )
            if retired_match[0] is not None:
                self.candidates_suppressed_by_retired_pool += 1
                self.candidate_retired_suppression_events += 1
                self._remove_candidate(candidate_index)
                continue

            if not self.permanent_capacity_available():
                continue

            history = int(
                self.candidate_action_masks[candidate_index]
            )
            remaining = self.full_mask & ~history
            pool_index = self._append_permanent_pool(
                self.candidate_representatives[candidate_index],
                self.candidate_safety_representatives[candidate_index],
                episode,
                self.candidate_visits[candidate_index],
                remaining,
            )
            self.candidates_promoted += 1
            self.candidates_promoted_after_capacity_growth += 1
            promoted_to_new_pool += 1
            self._remove_candidate(candidate_index)

            if self.mask(pool_index) == 0:
                self.candidate_history_exhaustion_events += 1
                self._retire_active_pool(
                    pool_index,
                    episode,
                    "CANDIDATE_HISTORY_EXHAUSTED",
                    {
                        "action": -1,
                        "reward": math.nan,
                        "collision": False,
                        "out_of_road": False,
                        "done": False,
                        "step": -1,
                        "retirement_trigger_type": (
                            "capacity_growth_candidate_history"
                        ),
                    },
                )
        return promoted_to_new_pool


    def _record_evicted_candidate(self, index: int, episode: int) -> None:
        visits = int(self.candidate_visits[index])
        self.evicted_candidate_rows.append(
            {
                "eviction_episode": int(episode),
                "visit_count": visits,
                "first_episode": int(self.candidate_first_episode[index]),
                "last_episode": int(self.candidate_last_episode[index]),
                "age_episodes": int(
                    episode - self.candidate_first_episode[index]
                ),
                "unique_actions_executed": int(
                    self.candidate_action_masks[index].bit_count()
                ),
                "near_promotion": bool(
                    visits >= self.candidate_promotion_visits - 1
                ),
                "recent_candidate": bool(
                    episode - self.candidate_first_episode[index]
                    < self.candidate_recent_protection_episodes
                ),
            }
        )

    def _batch_evict_candidates(self, episode: int) -> None:
        if len(self.candidate_representatives) < self.candidate_hard_limit:
            return

        removal_needed = max(
            0, len(self.candidate_representatives) - self.max_candidates
        )
        if removal_needed == 0:
            return

        eligible, protected = [], []
        for i, visits in enumerate(self.candidate_visits):
            near = visits >= self.candidate_promotion_visits - 1
            recent = (
                episode - self.candidate_first_episode[i]
                < self.candidate_recent_protection_episodes
            )
            if near or recent:
                protected.append(i)
                self.near_promotion_candidates_protected += int(near)
                self.recent_candidates_protected += int(recent)
            else:
                eligible.append(i)

        key = lambda i: (
            self.candidate_visits[i],
            self.candidate_last_episode[i],
            self.candidate_first_episode[i],
        )
        ranked = sorted(eligible, key=key)
        if len(ranked) < removal_needed:
            ranked.extend(sorted(protected, key=key))

        shortlist_size = min(
            len(ranked),
            max(removal_needed, self.candidate_batch_evict_count),
        )
        shortlist = ranked[:shortlist_size]

        # Store stable pool IDs because candidate-history transfer can retire
        # a pool and shift subsequent active-list indices.
        salvage_pairs: List[Tuple[int, int]] = []
        for candidate_index in shortlist:
            self.pre_eviction_candidates_checked += 1
            match = self.find_active_match(
                self.candidate_representatives[candidate_index],
                self.candidate_safety_representatives[candidate_index],
            )
            if match[0] is not None:
                pool_id = self.active_pool_ids[int(match[0])]
                salvage_pairs.append((candidate_index, pool_id))

        for candidate_index, pool_id in sorted(
            salvage_pairs, reverse=True
        ):
            if pool_id not in self.active_pool_ids:
                continue
            pool_index = self.active_pool_ids.index(pool_id)
            self.absorb_candidate_into_active(
                candidate_index,
                pool_index,
                episode,
                pre_eviction=True,
            )

        # Candidate indices above removed indices shift downward. Re-rank the
        # surviving candidates after salvage before final eviction.
        removal_needed = max(
            0, len(self.candidate_representatives) - self.max_candidates
        )
        if removal_needed == 0:
            return

        eligible, protected = [], []
        for i, visits in enumerate(self.candidate_visits):
            near = visits >= self.candidate_promotion_visits - 1
            recent = (
                episode - self.candidate_first_episode[i]
                < self.candidate_recent_protection_episodes
            )
            (protected if near or recent else eligible).append(i)

        ranked = sorted(eligible, key=key)
        if len(ranked) < removal_needed:
            ranked.extend(sorted(protected, key=key))
        to_remove = ranked[:removal_needed]
        for index in sorted(to_remove, reverse=True):
            self._record_evicted_candidate(index, episode)
            self._remove_candidate(index)
        self.candidates_evicted += len(to_remove)


    def _update_active_centroid(self, index, state, safety, new_visits):
        if (
            self.centroid_frozen_by_stability[index]
            or self.centroid_frozen_by_cap[index]
        ):
            return
        old = np.asarray(self.representatives[index], dtype=np.float32).copy()
        old_safety = np.asarray(self.safety_representatives[index], dtype=np.float32).copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)
        prior = max(
            1,
            self.promotion_evidence_visits[index]
            + self.absorbed_candidate_visits[index]
            + new_visits - 1,
        )
        updated = old + (np.asarray(state, dtype=np.float32) - old) / float(prior + 1)
        updated_safety = old_safety + (
            np.asarray(safety, dtype=np.float32) - old_safety
        ) / float(prior + 1)
        self.representatives[index] = updated.astype(self.storage_dtype)
        self.safety_representatives[index] = updated_safety.astype(self.storage_dtype)
        self.representative_norms[index] = float(
            np.linalg.norm(self.representatives[index])
        )
        self.safety_representative_norms[index] = float(
            np.linalg.norm(self.safety_representatives[index])
        )
        relative_shift = float(np.linalg.norm(updated - old) / old_norm)
        absolute_shift = self._rms(updated, old)
        safety_shift = self._rms(updated_safety, old_safety)
        shift = max(relative_shift, absolute_shift, safety_shift)
        self.centroid_update_counts[index] += 1
        self.centroid_last_shifts[index] = shift
        recent_distance = (
            float(np.mean(self.recent_distance_windows[index]))
            if self.recent_distance_windows[index]
            else float("inf")
        )
        if (
            shift < self.centroid_shift_threshold
            and recent_distance <= self.centroid_stability_distance_threshold
        ):
            self.centroid_stable_counts[index] += 1
        else:
            self.centroid_stable_counts[index] = 0
        if self.centroid_stable_counts[index] >= self.centroid_stable_updates_required:
            self.centroid_frozen_by_stability[index] = True
        elif self.centroid_update_counts[index] >= self.max_centroid_updates:
            self.centroid_frozen_by_cap[index] = True

    def process_state(
        self,
        state,
        safety,
        episode,
        active_match=None,
        active_match_precomputed: bool = False,
    ):
        self.total_states_seen += 1
        active = (
            active_match
            if active_match_precomputed
            else self.find_active_match(state, safety)
        )
        pool_index = active[0]

        if pool_index is not None:
            pool_index = int(pool_index)
            cosine, distance, s_cosine, s_distance = (
                active[1], active[2], active[3], active[4]
            )
            new_visits = self.visit_counts[pool_index] + 1
            self.visit_counts[pool_index] = new_visits
            self.last_episode_visited[pool_index] = episode
            self.match_counts[pool_index] += 1
            self.total_pool_matches += 1
            self.similarity_sums[pool_index] += cosine
            self.distance_sums[pool_index] += distance
            self.safety_similarity_sums[pool_index] += s_cosine
            self.safety_distance_sums[pool_index] += s_distance
            self.recent_distance_windows[pool_index].append(distance)
            self._update_active_centroid(
                pool_index, state, safety, new_visits
            )
            return pool_index, None, "permanent_matched"

        candidate = self.find_candidate_match(state, safety)
        candidate_index = candidate[0]
        if candidate_index is not None:
            i = int(candidate_index)
            new_visits = self.candidate_visits[i] + 1
            if not self.candidate_centroid_frozen[i]:
                old = np.asarray(self.candidate_representatives[i], dtype=np.float32).copy()
                old_safety = np.asarray(
                    self.candidate_safety_representatives[i], dtype=np.float32
                ).copy()
                old_norm = max(float(np.linalg.norm(old)), 1e-8)
                updated = old + (np.asarray(state, dtype=np.float32) - old) / float(new_visits)
                updated_safety = old_safety + (
                    np.asarray(safety, dtype=np.float32) - old_safety
                ) / float(new_visits)
                self.candidate_representatives[i] = updated.astype(self.storage_dtype)
                self.candidate_safety_representatives[i] = updated_safety.astype(self.storage_dtype)
                self.candidate_norms[i] = float(
                    np.linalg.norm(self.candidate_representatives[i])
                )
                self.candidate_safety_norms[i] = float(
                    np.linalg.norm(self.candidate_safety_representatives[i])
                )
                relative_shift = float(np.linalg.norm(updated - old) / old_norm)
                shift = max(
                    relative_shift,
                    self._rms(updated, old),
                    self._rms(updated_safety, old_safety),
                )
                self.candidate_centroid_update_counts[i] += 1
                self.candidate_last_shifts[i] = shift
                self.candidate_stable_counts[i] = (
                    self.candidate_stable_counts[i] + 1
                    if shift < self.candidate_centroid_shift_threshold
                    else 0
                )
                if (
                    self.candidate_stable_counts[i]
                    >= self.candidate_stable_updates_required
                    or self.candidate_centroid_update_counts[i]
                    >= self.max_candidate_centroid_updates
                ):
                    self.candidate_centroid_frozen[i] = True
            self.candidate_visits[i] = new_visits
            self.candidate_last_episode[i] = episode

            if new_visits >= self.candidate_promotion_visits:
                active_candidate_match = self.find_active_match(
                    self.candidate_representatives[i],
                    self.candidate_safety_representatives[i],
                )
                if active_candidate_match[0] is not None:
                    pool_id = self.active_pool_ids[
                        int(active_candidate_match[0])
                    ]
                    retired_index = self.absorb_candidate_into_active(
                        i,
                        int(active_candidate_match[0]),
                        episode,
                    )
                    if retired_index is not None:
                        return (
                            None,
                            None,
                            "candidate_history_exhausted_pool",
                        )
                    return (
                        self.active_pool_ids.index(pool_id),
                        None,
                        "candidate_merged_into_active_pool",
                    )

                retired_candidate_match = self.find_retired_match(
                    self.candidate_representatives[i],
                    self.candidate_safety_representatives[i],
                )
                if retired_candidate_match[0] is not None:
                    self.candidates_suppressed_by_retired_pool += 1
                    self.candidate_retired_suppression_events += 1
                    self._remove_candidate(i)
                    return (
                        None,
                        None,
                        "candidate_suppressed_by_retired_pool",
                    )

                if self.permanent_capacity_available():
                    history = self.candidate_action_masks[i]
                    remaining = self.full_mask & ~history
                    pool_index = self._append_permanent_pool(
                        self.candidate_representatives[i],
                        self.candidate_safety_representatives[i],
                        episode,
                        new_visits,
                        remaining,
                    )
                    self.candidates_promoted += 1
                    self._remove_candidate(i)
                    if self.mask(pool_index) == 0:
                        self.candidate_history_exhaustion_events += 1
                        self._retire_active_pool(
                            pool_index,
                            episode,
                            "CANDIDATE_HISTORY_EXHAUSTED",
                            {
                                "action": -1,
                                "reward": math.nan,
                                "collision": False,
                                "out_of_road": False,
                                "done": False,
                                "step": -1,
                                "retirement_trigger_type": (
                                    "waiting_candidate_history"
                                ),
                            },
                        )
                        return (
                            None,
                            None,
                            "candidate_history_exhausted_pool",
                        )
                    return pool_index, None, "candidate_promoted_before_action"

                if not self.absolute_permanent_capacity_reached():
                    self.candidate_capacity_wait_events += 1
                    return (
                        None,
                        i,
                        "candidate_waiting_for_pool_capacity_argmax",
                    )

                # No future growth is possible at the absolute capacity.
                self.candidates_blocked_by_capacity += 1
                self._remove_candidate(i)
                return (
                    None,
                    None,
                    "candidate_absolute_capacity_argmax",
                )

            return None, i, "candidate_matched_argmax"

        if self.absolute_permanent_capacity_reached():
            self.absolute_capacity_argmax_states += 1
            return None, None, "absolute_pool_capacity_argmax"

        if len(self.candidate_representatives) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)

        stored_state = np.asarray(
            state, dtype=self.storage_dtype
        ).copy()
        stored_safety = np.asarray(
            safety, dtype=self.storage_dtype
        ).copy()
        self.candidate_ids.append(self.next_candidate_id)
        created_candidate_id = self.next_candidate_id
        self.next_candidate_id += 1
        self.candidate_representatives.append(stored_state)
        self.candidate_safety_representatives.append(stored_safety)
        self.candidate_norms.append(float(np.linalg.norm(stored_state)))
        self.candidate_safety_norms.append(
            float(np.linalg.norm(stored_safety))
        )
        self.candidate_visits.append(1)
        self.candidate_first_episode.append(episode)
        self.candidate_last_episode.append(episode)
        self.candidate_centroid_update_counts.append(0)
        self.candidate_stable_counts.append(0)
        self.candidate_last_shifts.append(0.0)
        self.candidate_centroid_frozen.append(False)
        self.candidate_action_masks.append(0)
        self.candidates_created += 1
        created_index = len(self.candidate_representatives) - 1
        if len(self.candidate_representatives) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)
            created_index = (
                self.candidate_ids.index(created_candidate_id)
                if created_candidate_id in self.candidate_ids
                else -1
            )
            if created_index < 0:
                return None, None, "candidate_created_then_evicted_argmax"
        return None, created_index, "candidate_created_argmax"

    def record_candidate_action(self, candidate_index, action):
        self.candidate_action_masks[candidate_index] |= 1 << int(action)

    def record_retired_hit(
        self, retired_index, episode, step,
        similarity, distance, safety_similarity, safety_distance
    ):
        i = int(retired_index)
        self.retired_hit_counts[i] += 1
        self.retired_last_hit_episode[i] = int(episode)
        self.retired_last_hit_step[i] = int(step)
        self.retired_similarity_sums[i] += float(similarity)
        self.retired_distance_sums[i] += float(distance)
        self.retired_safety_similarity_sums[i] += float(safety_similarity)
        self.retired_safety_distance_sums[i] += float(safety_distance)
        self.total_retired_pool_hits += 1

    def mask(self, pool_index): return int(self.masks[pool_index])
    def remove(self, pool_index, action):
        self.masks[pool_index] &= ~(1 << int(action))
    def remaining_count(self, pool_index):
        return int(self.masks[pool_index].bit_count())

    # ---------- Statistics ----------

    def pool_statistics(self):
        rows = []
        for i in range(len(self.representatives)):
            matches = self.match_counts[i]
            remaining = self.remaining_count(i)
            rows.append({
                "pool_id": int(self.active_pool_ids[i]),
                "promotion_evidence_visits": int(self.promotion_evidence_visits[i]),
                "absorbed_candidate_visits": int(self.absorbed_candidate_visits[i]),
                "absorbed_candidate_action_evidence": int(
                    self.absorbed_candidate_actions[i]
                ),
                "active_pool_mask_visits": int(self.visit_counts[i]),
                "matched_state_count": int(matches),
                "actions_tried": int(self.action_count - remaining),
                "remaining_actions": int(remaining),
                "coverage_percent": 100.0 * (self.action_count - remaining) / self.action_count,
                "first_episode_created": int(self.first_episode_created[i]),
                "last_episode_visited": int(self.last_episode_visited[i]),
                "mean_general_cosine_similarity": (
                    self.similarity_sums[i] / matches if matches else 0.0
                ),
                "mean_general_rms_distance": (
                    self.distance_sums[i] / matches if matches else 0.0
                ),
                "mean_safety_cosine_similarity": (
                    self.safety_similarity_sums[i] / matches if matches else 0.0
                ),
                "mean_safety_rms_distance": (
                    self.safety_distance_sums[i] / matches if matches else 0.0
                ),
                "centroid_updates": int(self.centroid_update_counts[i]),
                "centroid_frozen_by_stability": bool(
                    self.centroid_frozen_by_stability[i]
                ),
                "centroid_frozen_by_cap": bool(
                    self.centroid_frozen_by_cap[i]
                ),
            })
        return rows

    def candidate_statistics(self):
        return [{
            "candidate_id": int(self.candidate_ids[i]),
            "visit_count": int(self.candidate_visits[i]),
            "first_episode": int(self.candidate_first_episode[i]),
            "last_episode": int(self.candidate_last_episode[i]),
            "visits_remaining_for_promotion": max(
                0, self.candidate_promotion_visits - self.candidate_visits[i]
            ),
            "unique_actions_executed": int(
                self.candidate_action_masks[i].bit_count()
            ),
        } for i in range(len(self.candidate_representatives))]

    def retired_pool_statistics(self):
        rows = []
        for i in range(len(self.retired_representatives)):
            hits = self.retired_hit_counts[i]
            rows.append({
                "retired_pool_id": i,
                "original_pool_id": int(self.retired_original_pool_ids[i]),
                "retirement_reason": self.retired_reasons[i],
                "episode_created": int(self.retired_episode_created[i]),
                "episode_retired": int(self.retired_episode_retired[i]),
                "permanent_pool_visits": int(self.retired_permanent_visits[i]),
                "actions_explored": int(self.retired_actions_explored[i]),
                "hits_after_retirement": int(hits),
                "last_hit_episode": int(self.retired_last_hit_episode[i]),
                "last_hit_step": int(self.retired_last_hit_step[i]),
                "mean_general_similarity": (
                    self.retired_similarity_sums[i] / hits if hits else 0.0
                ),
                "mean_general_rms_distance": (
                    self.retired_distance_sums[i] / hits if hits else 0.0
                ),
                "mean_safety_similarity": (
                    self.retired_safety_similarity_sums[i] / hits if hits else 0.0
                ),
                "mean_safety_rms_distance": (
                    self.retired_safety_distance_sums[i] / hits if hits else 0.0
                ),
                "retirement_trigger_action": int(
                    self.retirement_trigger_action[i]
                ),
                "retirement_trigger_reward": float(
                    self.retirement_trigger_reward[i]
                ),
                "retirement_trigger_collision": bool(
                    self.retirement_trigger_collision[i]
                ),
                "retirement_trigger_out_of_road": bool(
                    self.retirement_trigger_out_of_road[i]
                ),
                "retirement_trigger_done": bool(
                    self.retirement_trigger_done[i]
                ),
                "retirement_trigger_type": self.retirement_trigger_type[i],
            })
        return rows

    def global_statistics(self):
        accounted = (
            self.candidates_promoted
            + self.candidates_merged_into_active_pool
            + self.candidates_evicted
            + self.candidates_blocked_by_capacity
            + self.candidates_suppressed_by_retired_pool
            + len(self.candidate_representatives)
        )
        pressure = (
            self.candidate_capacity_wait_events
            / max(
                1,
                self.candidates_promoted
                + self.candidate_capacity_wait_events,
            )
        )
        return {
            "initial_pool_capacity": self.initial_max_pools,
            "current_pool_capacity": self.max_pools,
            "maximum_pool_capacity": self.maximum_pool_capacity,
            "capacity_growth_events": len(self.capacity_growth_rows),
            "new_pool_capacity_pressure": float(pressure),
            "active_permanent_pools": len(self.representatives),
            "retired_permanent_pools": len(self.retired_representatives),
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
            "candidates_blocked_by_permanent_capacity": (
                self.candidates_blocked_by_capacity
            ),
            "candidate_capacity_wait_events": (
                self.candidate_capacity_wait_events
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
            "candidates_remaining_at_end": len(
                self.candidate_representatives
            ),
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
            "recent_candidate_protection_occurrences": self.recent_candidates_protected,
            "states_matched_to_active_pools": self.total_pool_matches,
            "states_matched_to_retired_pools": self.total_retired_pool_hits,
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
                f"general_{k}": v
                for k, v in self.normalizer.statistics().items()
            },
            **{
                f"safety_{k}": v
                for k, v in self.safety_normalizer.statistics().items()
            },
        }

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


def deterministic_candidate_epsilon_action(
    q_values: np.ndarray,
    action_count: int,
    epsilon: float,
    key: str,
) -> Tuple[int, bool]:
    """Return a reproducible candidate-only epsilon-greedy action."""
    digest = hashlib.sha256(f"{key}|candidate-epsilon".encode()).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if draw < float(epsilon):
        action = int.from_bytes(digest[8:16], "big") % int(action_count)
        return int(action), True
    return (
        DQNAgent._deterministic_extreme_from_q(
            q_values,
            maximum=True,
            key=key,
        ),
        False,
    )


def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    safety_state: np.ndarray,
    episode: int,
    step: int,
    args,
    action_pools: SimilarStateActionPools,
) -> Tuple[int, str]:
    if experiment != "Karthikeya4optimal7":
        raise ValueError(f"Unknown experiment: {experiment}")

    q_values = agent.q_values(state)
    tie_key = f"train|Karthikeya4optimal7|{episode}|{step}"
    pooling_limit = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    if episode >= pooling_limit:
        action_pools.total_final_argmax_states += 1
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            "final_phase_argmax",
        )

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

    active_match = action_pools.find_active_match(
        matching_state, matching_safety
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
            return (
                agent._deterministic_extreme_from_q(
                    q_values, maximum=True, key=tie_key
                ),
                "retired_pool_argmax",
            )
    pool_index, candidate_index, pool_status = action_pools.process_state(
        matching_state,
        matching_safety,
        episode,
        active_match=active_match,
        active_match_precomputed=True,
    )
    if pool_index is None:
        if candidate_index is not None:
            action, explored = deterministic_candidate_epsilon_action(
                q_values,
                agent.action_count,
                args.candidate_epsilon,
                tie_key,
            )
            action_pools.record_candidate_action(candidate_index, action)
            source_base = (
                pool_status[:-7]
                if pool_status.endswith("_argmax")
                else pool_status
            )
            source = (
                f"{source_base}_epsilon_random"
                if explored
                else f"{source_base}_epsilon_argmax"
            )
            return int(action), source
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        return int(action), pool_status

    mask = action_pools.mask(pool_index)
    if mask == 0:
        raise RuntimeError(
            "An active pool with an empty action mask reached action selection."
        )

    action = best_available_action(
        q_values, mask, agent.action_count, tie_key
    )
    action_pools.remove(pool_index, action)
    source = (
        "promoted_pool_first_best_available"
        if pool_status == "candidate_promoted_before_action"
        else (
            "candidate_merged_active_pool_best_available"
            if pool_status == "candidate_merged_into_active_pool"
            else "permanent_pool_best_available"
        )
    )
    if action_pools.mask(pool_index) == 0:
        action_pools.mark_pending_retirement(
            pool_index, episode, step, action
        )
        source = "active_pool_last_action_pending_retirement"
    return int(action), source


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
        "rmst_event_definition": args.rmst_event,
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
        "epsilon": (
            float(args.candidate_epsilon)
            if phase == "train"
            else 0.0
        ),
        "rnd_beta": 0.0,
        "noisy_sigma_init": 0.0,
        "network_frozen": phase == "test",
        "updates_during_test": 0 if phase == "test" else "",
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
            f"Karthikeya4optimal7 expects exactly 9 discrete actions; environment has {action_count}."
        )
    action_pools = SimilarStateActionPools(
        max_pools=args.max_state_pools,
        max_candidates=args.max_state_candidates,
        candidate_hard_limit=args.candidate_hard_limit,
        candidate_batch_evict_count=args.candidate_batch_evict_count,
        candidate_promotion_visits=args.candidate_promotion_visits,
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
        auto_calibrate_thresholds=args.auto_calibrate_thresholds,
        calibration_episodes=args.calibration_episodes,
        calibration_start_episode=args.calibration_start_episode,
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
    try:
        for episode in range(args.train_episodes):
            action_pools.review_capacity(episode)
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
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
            parsed = parse_step_info({}, False, False)
            episode_start = time.perf_counter()
            cpu_start = time.process_time()
            for step in range(args.max_episode_steps):
                safety_state = extract_safety_vector(train_env, args)
                safety_sums += safety_state
                safety_minimums = np.minimum(safety_minimums, safety_state)
                safety_maximums = np.maximum(safety_maximums, safety_state)
                safety_count += 1
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
                action_sources[source] = action_sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                done = bool(terminated or truncated)
                training_reward = float(env_reward)
                agent.replay.add(state, action, training_reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                env_reward_total += float(env_reward)
                training_reward_total += training_reward
                state = next_state
                parsed = parse_step_info(info, bool(terminated), bool(truncated))
                # Safety summaries intentionally contain pre-action values only.
                action_pools.finalize_pending_retirement(
                    reward=float(env_reward),
                    parsed=parsed,
                    done=done,
                )
                if done:
                    break

            row = episode_row(
                "train",
                experiment,
                episode,
                scenario_seed,
                initial_hash,
                env_reward_total,
                training_reward_total,
                step + 1,
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
            print(
                f"TRAIN {SHORT_LABELS[experiment]:16s} ep={episode:03d} "
                f"reward={env_reward_total:9.3f} steps={step + 1:3d} "
                f"term={parsed['termination_reason']:11s} "
                f"collision={str(parsed['collision']):5s} "
                f"wall={row['wall_time_seconds']:.2f}s loss={row['average_loss']:.6f}",
                flush=True,
            )
    finally:
        train_env.close()

    training_duration = time.perf_counter() - training_start
    training_cpu_duration = time.process_time() - training_cpu_start
    model_path = output_dir / "models" / f"{experiment}_model.pt"
    agent.save(model_path, args)
    agent.freeze()
    print(f"===== TRAINING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Training duration: {training_duration:.2f}s", flush=True)

    # MetaDrive uses a singleton engine. The training environment is closed
    # before the test environment is constructed.
    test_env = make_env(args, "test")
    print(f"\n===== TESTING START: {SHORT_LABELS[experiment]} =====", flush=True)
    testing_start = time.perf_counter()
    testing_cpu_start = time.process_time()
    try:
        with torch.no_grad():
            for episode in range(args.test_episodes):
                scenario_seed = args.test_seed + episode
                state_raw, _ = test_env.reset(seed=scenario_seed)
                state = flatten_observation(state_raw)
                initial_hash = observation_sha256(state_raw)
                reward_total = 0.0
                parsed = parse_step_info({}, False, False)
                episode_start = time.perf_counter()
                cpu_start = time.process_time()
                for step in range(args.max_episode_steps):
                    action = agent.greedy_action(
                        state,
                        key=f"test|{args.seed}|{episode}|{step}",
                    )
                    next_raw, reward, terminated, truncated, info = test_env.step(action)
                    state = flatten_observation(next_raw)
                    reward_total += float(reward)
                    parsed = parse_step_info(info, bool(terminated), bool(truncated))
                    if terminated or truncated:
                        break
                row = episode_row(
                    "test",
                    experiment,
                    episode,
                    scenario_seed,
                    initial_hash,
                    reward_total,
                    reward_total,
                    step + 1,
                    parsed,
                    args,
                    device,
                    episode_start,
                    cpu_start,
                    agent,
                    action_sources={"frozen_greedy": step + 1},
                )
                rows.append(row)
                print(
                    f"TEST  {SHORT_LABELS[experiment]:16s} ep={episode:03d} "
                    f"reward={reward_total:9.3f} steps={step + 1:3d} "
                    f"term={parsed['termination_reason']:11s} "
                    f"collision={str(parsed['collision']):5s} "
                    f"wall={row['wall_time_seconds']:.2f}s",
                    flush=True,
                )
    finally:
        test_env.close()

    testing_duration = time.perf_counter() - testing_start
    testing_cpu_duration = time.process_time() - testing_cpu_start
    print(f"===== TESTING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Testing duration: {testing_duration:.2f}s", flush=True)
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


# ---------------------------------------------------------------------------
# Summaries and figures
# ---------------------------------------------------------------------------



def make_summary(rows: List[Dict], args) -> List[Dict]:
    summary: List[Dict] = []
    for experiment in EXPERIMENTS:
        train = [
            row for row in rows
            if row["experiment"] == experiment and row["phase"] == "train"
        ]
        test = [
            row for row in rows
            if row["experiment"] == experiment and row["phase"] == "test"
        ]
        test_times = [float(row["event_or_censor_time_steps"]) for row in test]
        test_events = [bool(row["rmst_event_observed"]) for row in test]
        rmst = restricted_mean_survival_time(test_times, test_events, args.rmst_tau)
        total_train_time = sum(
            float(row["wall_time_seconds"]) for row in train
        )
        summary.append(
            {
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "learning_rate": args.learning_rate,
                "selected_event_rmst_steps": rmst,
                "RMST_tau_steps": args.rmst_tau,
                "RMST_event_definition": args.rmst_event,
                "total_training_wall_time_seconds": total_train_time,
                "train_collision_rate": float(np.mean([bool(r["collision"]) for r in train])),
                "test_collision_rate": float(np.mean([bool(r["collision"]) for r in test])),
                "train_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in train])),
                "test_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in test])),
                "test_off_road_rate": float(np.mean([bool(r["out_of_road"]) for r in test])),
                "train_collisions_per_1000_steps": (
                    1000.0 * sum(bool(r["collision"]) for r in train)
                    / max(1, sum(int(r["steps"]) for r in train))
                ),
                "test_collisions_per_1000_steps": (
                    1000.0 * sum(bool(r["collision"]) for r in test)
                    / max(1, sum(int(r["steps"]) for r in test))
                ),
                "train_mean_minimum_lane_boundary_clearance": float(
                    np.nanmean([
                        r["minimum_lane_boundary_clearance"] for r in train
                    ])
                ),
                "test_mean_minimum_lane_boundary_clearance": math.nan,
                "train_mean_minimum_collision_hazard_center_distance": float(
                    np.nanmean([
                        r["minimum_nearest_collision_hazard_center_distance"] for r in train
                    ])
                ),
                "test_mean_minimum_collision_hazard_center_distance": math.nan,
                "network_frozen_during_testing": True,
            }
        )
    return summary


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, figure_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(rows: List[Dict], summary: List[Dict], output_dir: Path, args) -> None:
    """Create collision-focused benchmark plots; reward plots are intentionally omitted."""
    apply_ieee_style()
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary).set_index("experiment").loc[EXPERIMENTS]
    labels = [SHORT_LABELS[e] for e in EXPERIMENTS]
    colors = [COLORS[e] for e in EXPERIMENTS]
    x = np.arange(len(EXPERIMENTS))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.bar(x, summary_df["selected_event_rmst_steps"], color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(
        float(summary_df["RMST_tau_steps"].iloc[0]),
        color="black", linestyle="--", linewidth=1, label="Restriction tau",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel(f"{args.rmst_event.capitalize()} RMST (steps)")
    ax.set_title("MetaDrive Restricted Mean Survival")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_selected_event_rmst")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    width = 0.36
    ax.bar(
        x - width / 2, 100 * summary_df["train_collision_rate"], width,
        label="Train", edgecolor="black",
    )
    ax.bar(
        x + width / 2, 100 * summary_df["test_collision_rate"], width,
        label="Test", edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Episodes with collision (%)")
    ax.set_title("MetaDrive Collision Rate")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collision_rates")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(
        x - width / 2, summary_df["train_collisions_per_1000_steps"], width,
        label="Train", edgecolor="black",
    )
    ax.bar(
        x + width / 2, summary_df["test_collisions_per_1000_steps"], width,
        label="Test", edgecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Collisions per 1,000 steps")
    ax.set_title("MetaDrive Collision Exposure")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collisions_per_1000_steps")



def save_pool_statistics(
    action_pools: SimilarStateActionPools,
    args,
    output_dir: Path,
) -> None:
    pool_rows = action_pools.pool_statistics()
    candidate_rows = action_pools.candidate_statistics()
    retired_rows = action_pools.retired_pool_statistics()
    global_row = action_pools.global_statistics()

    def save_rows(path: Path, rows: List[Dict], columns: List[str]) -> None:
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    pool_columns = [
        "pool_id", "promotion_evidence_visits",
        "absorbed_candidate_visits", "absorbed_candidate_action_evidence",
        "active_pool_mask_visits", "matched_state_count",
        "actions_tried", "remaining_actions", "coverage_percent",
        "first_episode_created", "last_episode_visited",
        "mean_general_cosine_similarity", "mean_general_rms_distance",
        "mean_safety_cosine_similarity", "mean_safety_rms_distance",
        "centroid_updates", "centroid_frozen_by_stability",
        "centroid_frozen_by_cap",
    ]
    candidate_columns = [
        "candidate_id", "visit_count", "first_episode", "last_episode",
        "visits_remaining_for_promotion", "unique_actions_executed",
    ]
    retired_columns = [
        "retired_pool_id", "original_pool_id", "retirement_reason",
        "episode_created", "episode_retired", "permanent_pool_visits",
        "actions_explored", "hits_after_retirement",
        "last_hit_episode", "last_hit_step",
        "mean_general_similarity", "mean_general_rms_distance",
        "mean_safety_similarity", "mean_safety_rms_distance",
        "retirement_trigger_action", "retirement_trigger_reward",
        "retirement_trigger_collision",
        "retirement_trigger_out_of_road",
        "retirement_trigger_done",
        "retirement_trigger_type",
    ]
    evicted_columns = [
        "eviction_episode", "visit_count", "first_episode",
        "last_episode", "age_episodes", "unique_actions_executed",
        "near_promotion", "recent_candidate",
    ]
    save_rows(
        output_dir / "state_pool_statistics.csv", pool_rows, pool_columns
    )
    save_rows(
        output_dir / "state_candidate_statistics.csv",
        candidate_rows,
        candidate_columns,
    )
    save_rows(
        output_dir / "state_retired_pool_statistics.csv",
        retired_rows,
        retired_columns,
    )
    save_rows(
        output_dir / "state_evicted_candidate_history.csv",
        action_pools.evicted_candidate_rows,
        evicted_columns,
    )
    pd.DataFrame([global_row]).to_csv(
        output_dir / "state_pool_global_summary.csv", index=False
    )
    pd.DataFrame(
        action_pools.creation_events,
        columns=["episode", "cumulative_pools_created"],
    ).to_csv(output_dir / "state_pool_creation_timeline.csv", index=False)
    capacity_growth_columns = [
        "episode", "capacity_before", "capacity_after",
        "candidate_soft_before", "candidate_soft_after",
        "candidate_hard_before", "candidate_hard_after",
        "new_pool_capacity_pressure", "promoted_since_last_review",
        "waiting_events_since_last_review",
        "waiting_candidates_promoted_after_growth",
    ]
    pd.DataFrame(
        action_pools.capacity_growth_rows,
        columns=capacity_growth_columns,
    ).to_csv(
        output_dir / "state_pool_capacity_growth.csv", index=False
    )

    calibration_row = {
        "pool_representation": "general observation AND safety vector",
        "safety_vector_names": "|".join(SAFETY_VECTOR_NAMES),
        "general_similarity_threshold": action_pools.similarity_threshold,
        "general_rms_threshold": action_pools.distance_threshold,
        "candidate_general_rms_threshold": (
            action_pools.candidate_distance_threshold
        ),
        "safety_similarity_threshold": (
            action_pools.safety_similarity_threshold
        ),
        "safety_rms_threshold": action_pools.safety_distance_threshold,
        "candidate_safety_rms_threshold": (
            action_pools.candidate_safety_distance_threshold
        ),
        **{
            f"general_{k}": v
            for k, v in action_pools.normalizer.statistics().items()
        },
        **{
            f"safety_{k}": v
            for k, v in action_pools.safety_normalizer.statistics().items()
        },
    }
    pd.DataFrame([calibration_row]).to_csv(
        output_dir / "state_matching_calibration.csv", index=False
    )

    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_ieee_style()
    if pool_rows:
        pool_ids = [int(row["pool_id"]) for row in pool_rows]
        visits = [int(row["active_pool_mask_visits"]) for row in pool_rows]
        actions = [int(row["actions_tried"]) for row in pool_rows]

        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.bar(pool_ids, visits)
        ax.set_xlabel("Active pool ID")
        ax.set_ylabel("Mask-controlled visits")
        ax.set_title("Active State Pool Occupancy")
        save_figure(fig, figure_dir, "state_pool_occupancy")

        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        ax.bar(pool_ids, actions)
        ax.axhline(
            action_pools.action_count,
            linestyle="--",
            linewidth=1.0,
            label=f"Complete coverage ({action_pools.action_count} actions)",
        )
        ax.set_xlabel("Active pool ID")
        ax.set_ylabel("Actions attempted")
        ax.set_ylim(0, action_pools.action_count + 0.5)
        ax.set_title("Action Coverage per Active Pool")
        ax.legend(frameon=False)
        save_figure(fig, figure_dir, "state_pool_action_coverage")


def bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def collision_summary(rows: Sequence[Dict], tau: int) -> Dict[str, float]:
    collisions = sum(bool_value(row["collision"]) for row in rows)
    steps = sum(int(row["steps"]) for row in rows)
    episodes = len(rows)
    return {
        "episodes": episodes,
        "collision_count": collisions,
        "total_steps": steps,
        "collision_rmst_event_definition": "collision",
        "collision_rmst": restricted_mean_survival_time(
            [int(row["event_or_censor_time_steps"]) for row in rows],
            [bool_value(row["collision"]) for row in rows],
            tau,
        ),
        "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def critical_config(args) -> Dict:
    values = vars(args)
    return {key: values[key] for key in CRITICAL_CONFIG_KEYS}


def save_framework_compatibility_outputs(
    rows: List[Dict],
    runtimes: List[Dict],
    args,
    output_dir: Path,
) -> None:
    """Write the files expected by baseline comparison/timing workflows."""
    runtime_path = output_dir / "runtime_statistics.csv"
    write_csv(runtime_path, runtimes)

    metric_rows = []
    for phase in ("train", "test"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        metric_rows.append(
            {
                "method": "Karthikeya4optimal7",
                "method_label": "Karthikeya4optimal7",
                "phase": phase,
                **collision_summary(phase_rows, args.rmst_tau),
            }
        )
    metrics_path = output_dir / "collision_metrics.csv"
    write_csv(metrics_path, metric_rows)

    # Keep the existing model location and also expose the baseline-style name.
    nested_model = output_dir / "models" / "Karthikeya4optimal7_model.pt"
    root_model = output_dir / "model.pt"
    if nested_model.is_file():
        root_model.write_bytes(nested_model.read_bytes())

    # The completion manifest is written only after every required output,
    # including pool statistics and plots, has succeeded.



def write_completion_manifest(
    rows: List[Dict],
    runtimes: List[Dict],
    args,
    output_dir: Path,
) -> None:
    results_path = output_dir / "all_episode_results.csv"
    metrics_path = output_dir / "collision_metrics.csv"
    runtime_path = output_dir / "runtime_statistics.csv"
    root_model = output_dir / "model.pt"
    config_path = output_dir / "config.json"
    pool_stats_path = output_dir / "state_pool_statistics.csv"
    candidate_stats_path = output_dir / "state_candidate_statistics.csv"
    retired_stats_path = output_dir / "state_retired_pool_statistics.csv"
    calibration_path = output_dir / "state_matching_calibration.csv"
    capacity_growth_path = output_dir / "state_pool_capacity_growth.csv"
    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": getattr(metadrive, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "seed": args.seed,
        "method": "Karthikeya4optimal7",
        "method_label": "Karthikeya4optimal7",
        "critical_config": critical_config(args),
        "critical_config_sha256": canonical_json_sha256(
            critical_config(args)
        ),
        "model_sha256": sha256_file(root_model),
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_statistics_sha256": sha256_file(runtime_path),
        "config_sha256": sha256_file(config_path),
        "state_pool_statistics_sha256": sha256_file(pool_stats_path),
        "state_candidate_statistics_sha256": sha256_file(candidate_stats_path),
        "state_retired_pool_statistics_sha256": sha256_file(
            retired_stats_path
        ),
        "state_matching_calibration_sha256": sha256_file(calibration_path),
        "state_pool_capacity_growth_sha256": sha256_file(
            capacity_growth_path
        ),
        "phase_runtime": runtimes,
        "created_at_unix": time.time(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def save_outputs(rows: List[Dict], runtimes: List[Dict], args, output_dir: Path) -> None:
    summary = make_summary(rows, args)
    pd.DataFrame(rows).to_csv(output_dir / "all_episode_results.csv", index=False)
    pd.DataFrame(runtimes).to_csv(
        output_dir / "all_experiments_runtime_logs.csv", index=False
    )
    (output_dir / "config.json").write_text(
        json.dumps(json_safe(vars(args)), indent=2), encoding="utf-8"
    )
    make_figures(rows, summary, output_dir, args)
    save_framework_compatibility_outputs(rows, runtimes, args, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaDrive Karthikeya4optimal7: hybrid safety-aware bounded one-pass action coverage"
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--max-state-pools",
        type=int,
        default=None,
        help=(
            "Initial combined active + retired capacity. Default: "
            "min(2000, max(1000, ceil(sqrt(train_episodes*max_steps))))."
        ),
    )
    parser.add_argument(
        "--max-state-candidates",
        type=int,
        default=None,
        help=(
            "Soft candidate limit. Default is 25% of initial permanent "
            "capacity, bounded to [100, 500]."
        ),
    )
    parser.add_argument(
        "--candidate-hard-limit",
        type=int,
        default=None,
        help=(
            "Hard candidate limit. Batch eviction occurs when this limit "
            "is reached."
        ),
    )
    parser.add_argument(
        "--candidate-batch-evict-count",
        type=int,
        default=None,
        help=(
            "Number of weakest candidates removed together at the hard limit."
        ),
    )
    parser.add_argument(
        "--candidate-promotion-visits",
        type=int,
        default=2,
        help="Visits required before promoting a candidate to a permanent pool.",
    )
    parser.add_argument(
        "--candidate-epsilon",
        type=float,
        default=0.20,
        help=(
            "Candidate-stage epsilon-greedy probability. Active pools, "
            "retired pools, the final training phase, and testing remain greedy."
        ),
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
        help="Fit normalization, then calibrate thresholds in a fixed normalized space.",
    )
    parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=15,
        help="Total warm-up episodes, split between normalization and threshold calibration.",
    )
    parser.add_argument(
        "--calibration-start-episode",
        type=int,
        default=10,
        help=(
            "Pure DQN learning episodes completed before fitting the "
            "pool normalizers and thresholds."
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
    parser.add_argument("--out-of-road-penalty", type=float, default=10.0)
    parser.add_argument("--metadrive-log-level", type=int, default=50)
    parser.add_argument("--render", action="store_true")
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
        help="collision: vehicle/object crash; safety: collision or off-road",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional runner-supplied path. It must equal "
            "<project>/policy_results/seed_<seed>/Karthikeya4optimal7."
        ),
    )
    return parser.parse_args()


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
    if not 0.0 <= args.candidate_epsilon <= 1.0:
        raise ValueError("--candidate-epsilon must be in [0, 1].")
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
    if args.calibration_episodes < 0:
        raise ValueError("--calibration-episodes must be non-negative.")
    if args.auto_calibrate_thresholds and args.calibration_episodes < 2:
        raise ValueError(
            "--calibration-episodes must be at least 2 when auto calibration is enabled."
        )
    if args.calibration_start_episode < 0:
        raise ValueError("--calibration-start-episode must be non-negative.")
    if (
        args.calibration_start_episode + args.calibration_episodes
        >= args.train_episodes
    ):
        raise ValueError(
            "Calibration start plus calibration episodes must be less "
            "than --train-episodes."
        )
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
    if not 0.0 < args.pool_training_fraction <= 1.0:
        raise ValueError("--pool-training-fraction must be in the interval (0, 1].")
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
    if max(train_start, test_start) <= min(train_end, test_end):
        raise ValueError("Testing seed range must not overlap the training seed range.")
    if args.collision_penalty < 0 or args.out_of_road_penalty < 0:
        raise ValueError(
            "MetaDrive penalty arguments must be non-negative magnitudes."
        )
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1.")
    if not 0.0 <= args.accident_prob <= 1.0:
        raise ValueError("--accident-prob must be between 0 and 1.")


def resolve_output_dir(args) -> Path:
    """Return the only permitted seed-specific policy output directory."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    output_dir = (
        project_dir / "policy_results" / f"seed_{args.seed}" / "Karthikeya4optimal7"
    ).resolve()
    if args.output_dir is not None:
        supplied = Path(args.output_dir).expanduser()
        if not supplied.is_absolute():
            supplied = Path.cwd() / supplied
        supplied = supplied.resolve()
        if supplied != output_dir:
            raise ValueError(
                "--output-dir must use the canonical seed structure: "
                f"{output_dir}; received: {supplied}"
            )
    return output_dir



def prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Protect completed and partial runs from accidental overwrite."""
    if output_dir.exists():
        has_contents = any(output_dir.iterdir())
        if has_contents and not force:
            raise FileExistsError(
                f"Karthikeya4optimal7 output directory is not empty: {output_dir}. "
                "Reuse it or pass --force to remove the previous partial/completed run."
            )
        if force:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    args.hybrid_design_version = "k2_memory_k3_safety_v1"
    args.maximum_pool_capacity = 2000
    if args.max_state_pools is None:
        args.max_state_pools = min(
            args.maximum_pool_capacity,
            max(
                1000,
                math.ceil(
                    math.sqrt(
                        args.train_episodes * args.max_episode_steps
                    )
                ),
            ),
        )
    if args.max_state_candidates is None:
        args.max_state_candidates = min(
            500,
            max(100, int(math.ceil(0.25 * args.max_state_pools))),
        )
    if args.candidate_hard_limit is None:
        args.candidate_hard_limit = int(
            math.ceil(args.max_state_candidates * 1.20)
        )
    if args.candidate_batch_evict_count is None:
        args.candidate_batch_evict_count = (
            args.candidate_hard_limit - args.max_state_candidates
        )
    validate_args(args)
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
    print("METADRIVE KARTHIKEYA OPTIMAL HYBRID SAFETY-AWARE COVERAGE")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("MetaDrive:", getattr(metadrive, "__version__", "installed"))
    print("Device:", device)
    print("Experiments:", ", ".join(SHORT_LABELS[e] for e in EXPERIMENTS))
    print("Plain DQN + Target + Replay + Adam: yes")
    print("RND intrinsic reward: no")
    print("Count-based intrinsic reward: no")
    print("Frozen greedy testing: yes")
    print("Train/Test/Max steps:", args.train_episodes, args.test_episodes, args.max_episode_steps)
    print("Training seeds:", args.seed, "through", args.seed + args.train_episodes - 1)
    print("Testing seeds:", args.test_seed, "through", args.test_seed + args.test_episodes - 1)
    print("Discrete actions:", args.discrete_steering_dim * args.discrete_throttle_dim)
    print("Learning rate:", args.learning_rate)
    print("Output directory:", output_dir)
    print("Policy name: Karthikeya4optimal7")
    print("Initial combined active + retired pools:", args.max_state_pools)
    print("Maximum adaptive permanent capacity:", args.maximum_pool_capacity)
    print("Adaptive capacity formula applied:", True)
    print("Pool general representation: full flattened observation")
    print("Pool safety representation:", ", ".join(SAFETY_VECTOR_NAMES))
    print("Testing policy: frozen DQN argmax only")
    print("Auto threshold calibration:", args.auto_calibrate_thresholds)
    print("Calibration start episode:", args.calibration_start_episode)
    print("Calibration episodes:", args.calibration_episodes)
    print("Candidate soft limit:", args.max_state_candidates)
    print("Candidate hard limit:", args.candidate_hard_limit)
    print("Candidate batch eviction count:", args.candidate_batch_evict_count)
    print("Candidate promotion visits:", args.candidate_promotion_visits)
    print("Candidate-stage epsilon:", args.candidate_epsilon)
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
    print("Safety matching gate: RMS distance plus cosine when norms are stable")
    print("Pool representatives use adaptive centroid freezing: yes")
    print("Centroid shift threshold:", args.centroid_shift_threshold)
    print("Consecutive stable updates required:", args.centroid_stable_updates)
    print("Maximum centroid updates per pool:", args.max_centroid_updates)
    print(
        "Centroid stability distance threshold:",
        args.centroid_stability_distance_threshold,
    )
    print("Final 20% uses pure argmax with no pooling: yes")
    print("Pool-active training fraction:", args.pool_training_fraction)
    print(
        "Pooling disabled from episode:",
        int(math.ceil(args.train_episodes * args.pool_training_fraction)),
    )
    print("Similarity metrics: cosine similarity + RMS distance")
    print("Pool lookup: exact bounded linear cosine + RMS matching")
    print("Representative storage dtype:", args.pool_storage_dtype)
    print("Replay sampling: indexed ring buffer, O(batch_size)")
    print("Test safety extraction included in timing: no")
    print("Available actions per new pool: 9")
    print("Candidate visits before promotion use argmax: yes")
    print("Candidate actions are recorded for mask initialization: yes")
    print("Promotion occurs before promotion-visit action selection: yes")
    print("Inside permanent pool: highest-Q action among available actions")
    print("Removal: clear selected bit in O(1)")
    print("Final mask action retirement: after observing transition")
    print("Combined capacity full: retry normal active match, then block and use maximum-Q")
    print("Masks are refilled: no")
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
        output_dir / "config.json",
        output_dir / "runtime_statistics.csv",
        output_dir / "collision_metrics.csv",
        output_dir / "model.pt",
        output_dir / "state_pool_global_summary.csv",
        output_dir / "state_candidate_statistics.csv",
        output_dir / "state_retired_pool_statistics.csv",
        output_dir / "state_pool_statistics.csv",
        output_dir / "state_matching_calibration.csv",
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
    write_completion_manifest(all_rows, runtimes, args, output_dir)
    if not (output_dir / "manifest.json").is_file():
        raise RuntimeError("Completion manifest was not created.")
    print("\nExperiment completed successfully.")
    print("Episode results:", output_dir / "all_episode_results.csv")
    print("Collision metrics:", output_dir / "collision_metrics.csv")
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
        "Capacity growth:",
        output_dir / "state_pool_capacity_growth.csv",
    )
    print(
        "Retired pool statistics:",
        output_dir / "state_retired_pool_statistics.csv",
    )
    print("Runtime statistics:", output_dir / "runtime_statistics.csv")
    print("Manifest:", output_dir / "manifest.json")
    occupancy_plot = output_dir / "plots" / "state_pool_occupancy.png"
    if occupancy_plot.is_file():
        print("Pool occupancy plot:", occupancy_plot)
    else:
        print("Pool occupancy plot: not created (no active pools remained)")
    print("Policy folder name: Karthikeya4optimal7")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
