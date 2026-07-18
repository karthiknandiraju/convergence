#!/usr/bin/env python3
"""MetaDrive Karthikeya20 DQN with conservative unified lifecycle state memory.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-18 CEST (+0200)

Training policy
---------------
* Policy name: Karthikeya20.
* A single canonical record collection uses a CANDIDATE, ACTIVE, or RETIRED
  status tag. Promotion and retirement change only that tag and preserve the
  record ID and centroid.
* Combined active+retired capacity is fixed at the Basic Optimal baseline of
  500. Capacity reviews therefore cannot grow the permanent pool.
* Candidate reception uses the Basic Optimal limits: 125 soft, 150 hard, and
  bounded batch eviction of 25 by default.
* The DQN receives the complete flattened MetaDrive observation.
* Pool matching requires both general-state similarity and safety similarity.
* The safety vector contains pre-action lane-boundary clearance, nearest collision-hazard
  centre distance, absolute lane offset, absolute heading error, and ego speed
  in km/h.
* After an initial pure-DQN learning period, general and safety normalizers/
  calibrators are fitted on training only and frozen. General gates are learned
  from evidence; safety calibration is conservatively bounded at cosine 0.98
  and RMS 0.10. Candidate RMS gates may tighten but never exceed 0.25.
* Lookup order is ACTIVE centroids, then RETIRED centroids, then CANDIDATE
  centroids. A candidate is created only if no permanent or candidate record
  matches.
* A candidate is promoted after five visits by default.
* Pool behavior starts only after the pure-DQN, normalization, and threshold-
  calibration warm-up is complete, and remains active until the configured
  pool-training boundary (80% of training by default).
* Candidate centroids freeze after two consecutive stable updates or four
  updates. Active-pool centroids freeze after three consecutive stable updates
  or ten updates by default, matching Basic Optimal.
* During the final 20% of training, pooling is completely disabled and every
  action is selected using normal maximum-Q. The complete pool policy is
  frozen: no matching, visits, masks, candidates, centroids, promotion,
  retirement, eviction, calibration, or capacity review is performed.
* Candidate visits 1 through promotion-1 record the genuinely executed
  actions. Candidate-created, candidate-matched, and capacity-waiting steps
  use fresh DQN ranking, skip blocked actions, and otherwise use argmax.
* On the promotion visit, the candidate becomes permanent before action
  selection; previously executed candidate actions are removed from the
  new mask, then the highest-Q remaining action is selected and removed.
* At the fixed 500-pool limit, no further permanent pools or unmatched-state
  candidates are created; deterministic DQN argmax is used instead during the
  pool-training phase.
* Every record tracks 9-bit unknown, safe, and blocked outcome masks. No DQN
  Q-values or action rankings are stored in the pool memory.
* An ACTIVE record additionally uses one 9-bit action-availability mask for
  bounded one-pass exploration. It selects the maximum-Q remaining action that
  is not blocked. Candidates and retired pools select the maximum-Q unknown or
  safe action across all nine actions.
* The selected action bit is cleared in O(1).
* Empty masks retire the active pool only after the transition caused by the
  final directly selected action is observed.
* When combined active + retired capacity is full, no new pool is created.
* Candidate eviction checks only the selected eviction shortlist against active
  pools using the normal active-pool thresholds. A matching candidate transfers
  centroid, visits, and action history
  into the active pool before deletion. Eviction
  protects near-promotion and recently-created candidates when possible.
* Every CANDIDATE, ACTIVE, or RETIRED match computes fresh DQN Q-values. A
  candidate or retired match scans all nine actions in
  descending-Q effect: every blocked action is skipped and the highest-Q
  unknown or safe action is selected. If all nine actions are blocked, normal
  DQN argmax is used. Candidate-level suppression is tracked separately and
  does not count as a retired current-state hit.
* Time-limit truncations end an episode but remain bootstrap-enabled replay
  transitions; only true MetaDrive termination suppresses the DQN target.
* Capacity review is disabled with pooling during the final pure-argmax phase.
* Direct final-mask actions retire only after their transitions are observed.
  Candidate-history exhaustion is an evidence-based retirement with no action outcome.
* Masks are never refilled.

Complexity
----------
* Active, candidate, and retired lookup uses exact vectorized linear matching
  over preallocated recyclable centroid slots.
  This guarantees that the globally best valid cosine/RMS match is selected.
* Empty checks, bit clearing, candidate recycling, promotion, and retirement:
  O(1) average time.
* Outcome-aware action filtering is O(A), effectively O(1) for fixed A=9.
* Matching is O(B(D+S)) on an active hit and O((B+R+C)(D+S)) on a miss.
  Candidate eviction is O(C log E + E*B(D+S)), where E is the bounded
  eviction shortlist size. With configured bounded E, eviction is linear in
  candidate and pool storage.
* Capacity review uses an O(1) ready-waiter queue and generation cache, checking
  only permanent centroids changed since each waiter's last exact scan.
* Replay insertion and indexed ring-buffer sampling are O(1) and
  O(batch_size), respectively.
* Representatives use float32 storage by default for threshold-stable benchmark
  behavior. Policy storage is O((B+C+R)(D+S+A)), with B + R bounded by
  max-state-pools.

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
    python policies/metadrivekarthikeya20.py \
      --seed 11 --test-seed 100000 \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda

Results are saved to:
    policy_results/seed_<seed>/Karthikeya20
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import platform
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
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


EXPERIMENTS = ["Karthikeya20"]
SHORT_LABELS = {
    "Karthikeya20": "Karthikeya20",
}
COLORS = {
    "Karthikeya20": "#2ca02c",
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
        # Calibration is allowed to tighten a candidate gate, never expand it
        # beyond the configured fallback/cap (0.25 with the policy defaults).
        candidate_distance = min(
            self.fallback_candidate_distance,
            max(distance, distance * 1.20),
        )
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
# Karthikeya20 unified lifecycle and action-status masks
# ---------------------------------------------------------------------------


@dataclass
class UnifiedStateRecord:
    """One centroid record whose lifecycle is controlled by a status tag."""

    record_id: int
    status: str
    state: np.ndarray
    safety: np.ndarray
    state_norm: float
    safety_norm: float
    unknown_mask: int
    safe_mask: int = 0
    blocked_mask: int = 0
    action_history_mask: int = 0
    availability_mask: int = 0

    candidate_visits: int = 0
    candidate_first_episode: int = -1
    candidate_last_episode: int = -1
    candidate_centroid_updates: int = 0
    candidate_stable_updates: int = 0
    candidate_last_shift: float = 0.0
    candidate_centroid_frozen: bool = False
    last_permanent_match_generation: int = -1
    permanent_generation: int = -1

    promotion_evidence_visits: int = 0
    absorbed_candidate_visits: int = 0
    absorbed_candidate_actions: int = 0
    active_mask_visits: int = 0
    match_count: int = 0
    first_episode_created: int = -1
    last_episode_visited: int = -1
    similarity_sum: float = 0.0
    distance_sum: float = 0.0
    safety_similarity_sum: float = 0.0
    safety_distance_sum: float = 0.0
    centroid_updates: int = 0
    centroid_stable_updates: int = 0
    centroid_last_shift: float = 0.0
    centroid_frozen_by_stability: bool = False
    centroid_frozen_by_cap: bool = False
    recent_distance_window: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5)
    )

    retirement_reason: str = ""
    episode_retired: int = -1
    retired_permanent_visits: int = 0
    retired_actions_explored: int = 0
    retired_hit_count: int = 0
    retired_last_hit_episode: int = -1
    retired_last_hit_step: int = -1
    retired_similarity_sum: float = 0.0
    retired_distance_sum: float = 0.0
    retired_safety_similarity_sum: float = 0.0
    retired_safety_distance_sum: float = 0.0
    retirement_trigger_action: int = -1
    retirement_trigger_collision: bool = False
    retirement_trigger_out_of_road: bool = False
    retirement_trigger_done: bool = False
    retirement_trigger_step: int = -1
    retirement_trigger_type: str = ""


class SimilarStateActionPools:
    """Unified CANDIDATE/ACTIVE/RETIRED centroid memory for Karthikeya20."""

    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"

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
        self.observation_size = int(observation_size)
        self.safety_size = int(safety_size)
        self.full_mask = (1 << self.action_count) - 1
        self.storage_dtype = (
            np.float16 if pool_storage_dtype == "float16" else np.float32
        )

        self.configured_similarity_threshold = float(similarity_threshold)
        self.configured_distance_threshold = float(distance_threshold)
        self.configured_safety_similarity_threshold = float(
            safety_similarity_threshold
        )
        self.configured_safety_distance_threshold = float(
            safety_distance_threshold
        )
        self.similarity_threshold = self.configured_similarity_threshold
        self.distance_threshold = self.configured_distance_threshold
        self.candidate_similarity_threshold = float(
            candidate_similarity_threshold
        )
        self.configured_candidate_distance_threshold = float(
            candidate_distance_threshold
        )
        self.candidate_distance_threshold = (
            self.configured_candidate_distance_threshold
        )
        self.safety_similarity_threshold = (
            self.configured_safety_similarity_threshold
        )
        self.safety_distance_threshold = (
            self.configured_safety_distance_threshold
        )
        self.configured_candidate_safety_distance_threshold = float(
            candidate_safety_distance_threshold
        )
        self.candidate_safety_distance_threshold = (
            self.configured_candidate_safety_distance_threshold
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

        # One canonical collection plus O(1) status membership indexes.
        self.records: Dict[int, UnifiedStateRecord] = {}
        self.status_ids: Dict[str, Dict[int, None]] = {
            self.CANDIDATE: {},
            self.ACTIVE: {},
            self.RETIRED: {},
        }
        # Stable record IDs are mapped onto recyclable dense slots. Exact
        # matching is vectorized over these preallocated centroid matrices.
        self.slot_capacity = int(
            self.maximum_pool_capacity + self.candidate_hard_limit
        )
        self.state_matrix = np.zeros(
            (self.slot_capacity, self.observation_size),
            dtype=self.storage_dtype,
        )
        self.safety_matrix = np.zeros(
            (self.slot_capacity, self.safety_size),
            dtype=self.storage_dtype,
        )
        self.state_norms = np.zeros(self.slot_capacity, dtype=np.float32)
        self.safety_norms = np.zeros(self.slot_capacity, dtype=np.float32)
        self.slot_record_ids = np.full(
            self.slot_capacity, -1, dtype=np.int64
        )
        self.record_slots: Dict[int, int] = {}
        self.free_slots: List[int] = list(
            range(self.slot_capacity - 1, -1, -1)
        )
        self.status_slots: Dict[str, List[int]] = {
            self.CANDIDATE: [],
            self.ACTIVE: [],
            self.RETIRED: [],
        }
        self.slot_status_positions = np.full(
            self.slot_capacity, -1, dtype=np.int64
        )

        # Only promotion-ready capacity waiters enter this O(1) queue. The
        # generation log lets a waiter recheck only permanent centroids that
        # changed after its last exact full scan.
        self.ready_waiting_ids: Dict[int, None] = {}
        self.permanent_generation = 0
        self.permanent_change_log: List[int] = []
        self.next_record_id = 0
        self.pending_action_outcome: Optional[Dict] = None
        self.policy_frozen = False
        self.policy_frozen_at_episode: Optional[int] = None
        self.policy_freeze_events = 0
        self.evicted_candidate_rows: List[Dict] = []
        self.creation_events: List[Tuple[int, int]] = []
        self.capacity_growth_rows: List[Dict] = []

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
        self.retired_mask_filtered_actions = 0
        self.retired_all_blocked_fallbacks = 0
        self.active_mask_filtered_actions = 0
        self.active_all_blocked_fallbacks = 0
        self.candidate_mask_filtered_actions = 0
        self.candidate_all_blocked_fallbacks = 0
        self.safe_action_outcomes = 0
        self.blocked_action_outcomes = 0
        self.last_review_promoted = 0
        self.last_review_capacity_wait_events = 0

    # ---------- Unified storage ----------

    def _ids(self, status: str) -> Iterable[int]:
        return self.status_ids[status].keys()

    def _require_policy_mutable(self, operation: str) -> None:
        if self.policy_frozen:
            raise RuntimeError(
                f"Pool policy is frozen; cannot perform {operation}."
            )

    def freeze_policy(self, episode: int) -> None:
        """Permanently freeze all pool-policy state at the argmax boundary."""
        if self.policy_frozen:
            return
        if self.pending_action_outcome is not None:
            raise RuntimeError(
                "Cannot freeze pool policy with an unresolved action outcome."
            )
        # Calibration must not resume or derive new gates in the final phase.
        self.normalizer.freeze()
        self.safety_normalizer.freeze()
        self.calibrator.freeze()
        self.safety_calibrator.freeze()
        self.thresholds_frozen = True
        self.policy_frozen = True
        self.policy_frozen_at_episode = int(episode)
        self.policy_freeze_events += 1

    def _attach_status_slot(self, status: str, slot: int) -> None:
        position = len(self.status_slots[status])
        self.status_slots[status].append(int(slot))
        self.slot_status_positions[int(slot)] = position

    def _detach_status_slot(self, status: str, slot: int) -> None:
        slot = int(slot)
        position = int(self.slot_status_positions[slot])
        slots = self.status_slots[status]
        if position < 0 or position >= len(slots) or slots[position] != slot:
            raise RuntimeError("Dense status-slot index is inconsistent.")
        last_slot = slots[-1]
        slots[position] = last_slot
        self.slot_status_positions[last_slot] = position
        slots.pop()
        self.slot_status_positions[slot] = -1

    def _allocate_record_slot(self, record: UnifiedStateRecord) -> int:
        if not self.free_slots:
            raise RuntimeError("Preallocated unified centroid storage is full.")
        slot = int(self.free_slots.pop())
        self.record_slots[record.record_id] = slot
        self.slot_record_ids[slot] = record.record_id
        self._attach_status_slot(record.status, slot)
        self._sync_record_vectors(record)
        return slot

    def _release_record_slot(self, record: UnifiedStateRecord) -> None:
        slot = int(self.record_slots.pop(record.record_id))
        self._detach_status_slot(record.status, slot)
        self.slot_record_ids[slot] = -1
        self.state_norms[slot] = 0.0
        self.safety_norms[slot] = 0.0
        self.free_slots.append(slot)

    def _sync_record_vectors(self, record: UnifiedStateRecord) -> None:
        slot = self.record_slots.get(record.record_id)
        if slot is None:
            return
        self.state_matrix[slot] = np.asarray(
            record.state, dtype=self.storage_dtype
        )
        self.safety_matrix[slot] = np.asarray(
            record.safety, dtype=self.storage_dtype
        )
        self.state_norms[slot] = float(record.state_norm)
        self.safety_norms[slot] = float(record.safety_norm)

    def _record_permanent_change(self, record: UnifiedStateRecord) -> None:
        if record.status not in {self.ACTIVE, self.RETIRED}:
            return
        self.permanent_generation += 1
        record.permanent_generation = self.permanent_generation
        self.permanent_change_log.append(record.record_id)

    def _changed_permanent_ids_since(self, generation: int) -> List[int]:
        if generation < 0:
            return list(self._ids(self.ACTIVE)) + list(
                self._ids(self.RETIRED)
            )
        changed = self.permanent_change_log[int(generation):]
        # Dict insertion order provides deterministic O(delta) de-duplication.
        return list(dict.fromkeys(changed))

    def _set_status(self, record_id: int, status: str) -> None:
        record = self.records[int(record_id)]
        if status not in self.status_ids:
            raise ValueError(f"Unknown record status: {status}")
        if record.status == status:
            return
        old_status = record.status
        slot = int(self.record_slots[record.record_id])
        self._detach_status_slot(old_status, slot)
        del self.status_ids[record.status][record.record_id]
        self.status_ids[status][record.record_id] = None
        record.status = status
        self._attach_status_slot(status, slot)
        if old_status == self.CANDIDATE:
            self.ready_waiting_ids.pop(record.record_id, None)
        if status in {self.ACTIVE, self.RETIRED}:
            self._record_permanent_change(record)

    def _delete_candidate(self, record_id: int) -> None:
        record = self.records[int(record_id)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Only candidate records can be recycled.")
        self.ready_waiting_ids.pop(record.record_id, None)
        self._release_record_slot(record)
        del self.status_ids[self.CANDIDATE][record.record_id]
        del self.records[record.record_id]

    def _new_candidate(self, state, safety, episode: int) -> int:
        state = np.asarray(state, dtype=self.storage_dtype).copy()
        safety = np.asarray(safety, dtype=self.storage_dtype).copy()
        record_id = self.next_record_id
        self.next_record_id += 1
        record = UnifiedStateRecord(
            record_id=record_id,
            status=self.CANDIDATE,
            state=state,
            safety=safety,
            state_norm=float(np.linalg.norm(state)),
            safety_norm=float(np.linalg.norm(safety)),
            unknown_mask=self.full_mask,
            candidate_visits=1,
            candidate_first_episode=int(episode),
            candidate_last_episode=int(episode),
        )
        self.records[record_id] = record
        self.status_ids[self.CANDIDATE][record_id] = None
        self._allocate_record_slot(record)
        self.candidates_created += 1
        return record_id

    def validate_invariants(self) -> None:
        if len(self.records) != len(self.record_slots):
            raise RuntimeError("Record-to-slot cardinality invariant failed.")
        occupied_slots = set(self.record_slots.values())
        if occupied_slots & set(self.free_slots):
            raise RuntimeError("A centroid slot is both occupied and free.")
        if len(occupied_slots) + len(self.free_slots) != self.slot_capacity:
            raise RuntimeError("Centroid slot accounting invariant failed.")
        for status in (self.CANDIDATE, self.ACTIVE, self.RETIRED):
            if len(self.status_ids[status]) != len(self.status_slots[status]):
                raise RuntimeError("Status membership cardinality failed.")
            for position, slot in enumerate(self.status_slots[status]):
                record_id = int(self.slot_record_ids[slot])
                record = self.records.get(record_id)
                if (
                    record is None
                    or record.status != status
                    or self.record_slots.get(record_id) != slot
                    or int(self.slot_status_positions[slot]) != position
                ):
                    raise RuntimeError("Dense status membership is inconsistent.")
        for record in self.records.values():
            if record.unknown_mask & record.safe_mask:
                raise RuntimeError("Unknown and safe masks overlap.")
            if record.unknown_mask & record.blocked_mask:
                raise RuntimeError("Unknown and blocked masks overlap.")
            if record.safe_mask & record.blocked_mask:
                raise RuntimeError("Safe and blocked masks overlap.")
            if (
                record.unknown_mask
                | record.safe_mask
                | record.blocked_mask
            ) != self.full_mask:
                raise RuntimeError("Outcome masks do not cover all actions.")
        for record_id in self.ready_waiting_ids:
            record = self.records.get(record_id)
            if (
                record is None
                or record.status != self.CANDIDATE
                or record.candidate_visits < self.candidate_promotion_visits
            ):
                raise RuntimeError("Ready-waiter queue invariant failed.")
        if self.total_permanent_records() > self.max_pools:
            raise RuntimeError("Permanent capacity invariant failed.")
        if len(self.status_ids[self.CANDIDATE]) > self.candidate_hard_limit:
            raise RuntimeError("Candidate hard-limit invariant failed.")

    # ---------- Calibration ----------

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

    # ---------- Exact linear matching ----------

    @staticmethod
    def _cosine(a, an, b, bn) -> float:
        if an == 0.0 or bn == 0.0:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.dot(a, b) / (an * bn))

    @staticmethod
    def _rms(a, b) -> float:
        return float(np.sqrt(np.mean(np.square(a - b))))

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
    ) -> Tuple[Optional[int], float, float, float, float]:
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
        valid = (
            general_cosine_ok
            & (distances <= distance_threshold)
            & safety_cosine_ok
            & (safety_distances <= safety_distance_threshold)
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return (
                None, float("-inf"), float("inf"),
                float("-inf"), float("inf"),
            )

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

    def find_active_match(self, state, safety):
        return self._best_status_match(
            state, safety, self.ACTIVE,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
        )

    def find_retired_match(self, state, safety):
        return self._best_status_match(
            state, safety, self.RETIRED,
            self.similarity_threshold, self.distance_threshold,
            self.safety_similarity_threshold, self.safety_distance_threshold,
        )

    def find_candidate_match(self, state, safety):
        return self._best_status_match(
            state, safety, self.CANDIDATE,
            self.candidate_similarity_threshold,
            self.candidate_distance_threshold,
            self.safety_similarity_threshold,
            self.candidate_safety_distance_threshold,
        )

    # ---------- Capacity ----------

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

    # ---------- Lifecycle and outcome masks ----------

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
            self._retire_active_pool(
                record.record_id,
                episode,
                "CANDIDATE_HISTORY_EXHAUSTED",
                {
                    "action": -1,
                    "collision": False,
                    "out_of_road": False,
                    "done": False,
                    "step": -1,
                    "retirement_trigger_type": "candidate_history_evidence",
                },
            )
            return None
        return record.record_id

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
            self.action_count - record.availability_mask.bit_count()
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
        self._set_status(record.record_id, self.RETIRED)
        if reason == "MASK_EXHAUSTED":
            self.direct_mask_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        elif reason == "CANDIDATE_HISTORY_EXHAUSTED":
            self.candidate_history_exhaustion_retirements += 1
            self.active_pools_retired_by_mask_exhaustion += 1
        return record.record_id

    def _mark_action_outcome(
        self, record_id: int, action: int, parsed: Dict
    ) -> None:
        record = self.records.get(int(record_id))
        if record is None:
            return
        bit = 1 << int(action)
        record.unknown_mask &= ~bit
        unsafe = bool(
            parsed.get("collision", False)
            or parsed.get("out_of_road", False)
        )
        if unsafe:
            record.blocked_mask |= bit
            record.safe_mask &= ~bit
            self.blocked_action_outcomes += 1
        elif not (record.blocked_mask & bit):
            record.safe_mask |= bit
            self.safe_action_outcomes += 1
        record.unknown_mask = (
            self.full_mask & ~(record.safe_mask | record.blocked_mask)
        )

    def mark_pending_action_outcome(
        self,
        record_id: int,
        episode: int,
        step: int,
        action: int,
        retire_after_outcome: bool = False,
    ) -> None:
        if self.pending_action_outcome is not None:
            raise RuntimeError("A prior pool action outcome is still pending.")
        self.pending_action_outcome = {
            "record_id": int(record_id),
            "episode": int(episode),
            "step": int(step),
            "action": int(action),
            "retire_after_outcome": bool(retire_after_outcome),
        }

    def mark_pending_retirement(
        self, pool_index: int, episode: int, step: int, action: int
    ) -> None:
        self.mark_pending_action_outcome(
            pool_index, episode, step, action, retire_after_outcome=True
        )

    def finalize_pending_action_outcome(self, reward, parsed, done) -> None:
        self._require_policy_mutable("action-outcome mask update")
        del reward  # No Q-values or reward values are stored in pool records.
        if self.pending_action_outcome is None:
            return
        pending = self.pending_action_outcome
        self.pending_action_outcome = None
        record = self.records.get(pending["record_id"])
        if record is None:
            return
        self._mark_action_outcome(
            record.record_id, pending["action"], parsed
        )
        if pending["retire_after_outcome"] and record.status == self.ACTIVE:
            self._retire_active_pool(
                record.record_id,
                pending["episode"],
                "MASK_EXHAUSTED",
                {
                    **pending,
                    "collision": bool(parsed.get("collision", False)),
                    "out_of_road": bool(parsed.get("out_of_road", False)),
                    "done": bool(done),
                    "retirement_trigger_type": "direct_final_action",
                },
            )

    def finalize_pending_retirement(self, reward, parsed, done) -> None:
        """Compatibility alias for older training-loop integrations."""
        self.finalize_pending_action_outcome(reward, parsed, done)

    # ---------- Candidate lifecycle ----------

    def absorb_candidate_into_active(
        self,
        candidate_index: int,
        pool_index: int,
        episode: int,
        pre_eviction: bool = False,
    ) -> Optional[int]:
        candidate = self.records[int(candidate_index)]
        active = self.records[int(pool_index)]
        if candidate.status != self.CANDIDATE or active.status != self.ACTIVE:
            raise RuntimeError("Candidate-to-active merge received bad tags.")

        visits = int(candidate.candidate_visits)
        history = int(candidate.action_history_mask)
        active_weight = max(
            1,
            active.promotion_evidence_visits
            + active.absorbed_candidate_visits
            + active.active_mask_visits,
        )
        if not (
            active.centroid_frozen_by_stability
            or active.centroid_frozen_by_cap
        ):
            old_state = np.asarray(active.state, dtype=np.float32).copy()
            old_safety = np.asarray(active.safety, dtype=np.float32).copy()
            combined_weight = active_weight + visits
            merged_state = (
                old_state * active_weight
                + np.asarray(candidate.state, dtype=np.float32) * visits
            ) / float(combined_weight)
            merged_safety = (
                old_safety * active_weight
                + np.asarray(candidate.safety, dtype=np.float32) * visits
            ) / float(combined_weight)
            relative_shift = float(
                np.linalg.norm(merged_state - old_state)
                / max(float(np.linalg.norm(old_state)), 1e-8)
            )
            shift = max(
                relative_shift,
                self._rms(merged_state, old_state),
                self._rms(merged_safety, old_safety),
            )
            active.state = merged_state.astype(self.storage_dtype)
            active.safety = merged_safety.astype(self.storage_dtype)
            active.state_norm = float(np.linalg.norm(active.state))
            active.safety_norm = float(np.linalg.norm(active.safety))
            active.centroid_updates += 1
            active.centroid_last_shift = shift
            active.recent_distance_window.append(
                self._rms(candidate.state, old_state)
            )
            recent_distance = float(
                np.mean(active.recent_distance_window)
            )
            if (
                shift < self.centroid_shift_threshold
                and recent_distance
                <= self.centroid_stability_distance_threshold
            ):
                active.centroid_stable_updates += 1
            else:
                active.centroid_stable_updates = 0
            if (
                active.centroid_stable_updates
                >= self.centroid_stable_updates_required
            ):
                active.centroid_frozen_by_stability = True
            elif active.centroid_updates >= self.max_centroid_updates:
                active.centroid_frozen_by_cap = True
            self._sync_record_vectors(active)
            self._record_permanent_change(active)

        before = int(active.availability_mask)
        active.availability_mask = before & ~history
        removed_bits = (before & history).bit_count()
        active.action_history_mask |= history
        active.blocked_mask |= candidate.blocked_mask
        active.safe_mask = (
            (active.safe_mask | candidate.safe_mask)
            & ~active.blocked_mask
        )
        active.unknown_mask = (
            self.full_mask & ~(active.safe_mask | active.blocked_mask)
        )
        active.absorbed_candidate_visits += visits
        active.absorbed_candidate_actions += history.bit_count()
        self.candidates_merged_into_active_pool += 1
        self.candidate_action_history_transfers += 1
        self.candidate_action_bits_removed_by_transfer += removed_bits
        if pre_eviction:
            self.pre_eviction_candidates_merged += 1
        self._delete_candidate(candidate.record_id)

        if active.availability_mask == 0:
            self.candidate_history_exhaustion_events += 1
            return self._retire_active_pool(
                active.record_id,
                episode,
                "CANDIDATE_HISTORY_EXHAUSTED",
                {
                    "action": -1,
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
        promoted_to_new_pool = 0
        for candidate_id in list(self.ready_waiting_ids):
            candidate = self.records.get(candidate_id)
            if (
                candidate is None
                or candidate.status != self.CANDIDATE
                or candidate.candidate_visits
                < self.candidate_promotion_visits
            ):
                self.ready_waiting_ids.pop(candidate_id, None)
                continue
            changed_ids = self._changed_permanent_ids_since(
                candidate.last_permanent_match_generation
            )
            active_match = self._best_status_match(
                candidate.state,
                candidate.safety,
                self.ACTIVE,
                self.similarity_threshold,
                self.distance_threshold,
                self.safety_similarity_threshold,
                self.safety_distance_threshold,
                record_ids=changed_ids,
            )
            if active_match[0] is not None:
                self.absorb_candidate_into_active(
                    candidate.record_id, int(active_match[0]), episode
                )
                continue
            retired_match = self._best_status_match(
                candidate.state,
                candidate.safety,
                self.RETIRED,
                self.similarity_threshold,
                self.distance_threshold,
                self.safety_similarity_threshold,
                self.safety_distance_threshold,
                record_ids=changed_ids,
            )
            if retired_match[0] is not None:
                self.candidates_suppressed_by_retired_pool += 1
                self.candidate_retired_suppression_events += 1
                self._delete_candidate(candidate.record_id)
                continue
            candidate.last_permanent_match_generation = (
                self.permanent_generation
            )
            if not self.permanent_capacity_available():
                break
            if self._promote_candidate(
                candidate.record_id, episode, after_growth=True
            ) is not None:
                promoted_to_new_pool += 1
        return promoted_to_new_pool

    def _record_evicted_candidate(
        self, record_id: int, episode: int
    ) -> None:
        record = self.records[int(record_id)]
        visits = int(record.candidate_visits)
        self.evicted_candidate_rows.append(
            {
                "eviction_episode": int(episode),
                "visit_count": visits,
                "first_episode": int(record.candidate_first_episode),
                "last_episode": int(record.candidate_last_episode),
                "age_episodes": int(
                    episode - record.candidate_first_episode
                ),
                "unique_actions_executed": int(
                    record.action_history_mask.bit_count()
                ),
                "near_promotion": bool(
                    visits >= self.candidate_promotion_visits - 1
                ),
                "recent_candidate": bool(
                    episode - record.candidate_first_episode
                    < self.candidate_recent_protection_episodes
                ),
            }
        )

    def _weakest_candidates(
        self,
        eligible: Sequence[int],
        protected: Sequence[int],
        removal_needed: int,
        limit: int,
    ) -> List[int]:
        key = lambda record_id: (
            self.records[record_id].candidate_visits,
            self.records[record_id].candidate_last_episode,
            self.records[record_id].candidate_first_episode,
            record_id,
        )
        if len(eligible) >= removal_needed:
            return heapq.nsmallest(
                min(limit, len(eligible)), eligible, key=key
            )
        selected = heapq.nsmallest(len(eligible), eligible, key=key)
        selected.extend(
            heapq.nsmallest(
                min(limit - len(selected), len(protected)),
                protected,
                key=key,
            )
        )
        return selected

    def _batch_evict_candidates(self, episode: int) -> None:
        candidate_count = len(self.status_ids[self.CANDIDATE])
        if candidate_count < self.candidate_hard_limit:
            return
        removal_needed = max(0, candidate_count - self.max_candidates)
        if removal_needed == 0:
            return

        eligible, protected = [], []
        for record_id in self._ids(self.CANDIDATE):
            record = self.records[record_id]
            near = (
                record.candidate_visits
                >= self.candidate_promotion_visits - 1
            )
            recent = (
                episode - record.candidate_first_episode
                < self.candidate_recent_protection_episodes
            )
            if near or recent:
                protected.append(record_id)
                self.near_promotion_candidates_protected += int(near)
                self.recent_candidates_protected += int(recent)
            else:
                eligible.append(record_id)

        shortlist = self._weakest_candidates(
            eligible,
            protected,
            removal_needed,
            max(removal_needed, self.candidate_batch_evict_count),
        )
        for candidate_id in shortlist:
            candidate = self.records.get(candidate_id)
            if candidate is None or candidate.status != self.CANDIDATE:
                continue
            self.pre_eviction_candidates_checked += 1
            active_match = self.find_active_match(
                candidate.state, candidate.safety
            )
            if active_match[0] is not None:
                self.absorb_candidate_into_active(
                    candidate.record_id,
                    int(active_match[0]),
                    episode,
                    pre_eviction=True,
                )

        removal_needed = max(
            0, len(self.status_ids[self.CANDIDATE]) - self.max_candidates
        )
        if removal_needed == 0:
            return
        eligible, protected = [], []
        for record_id in self._ids(self.CANDIDATE):
            record = self.records[record_id]
            near = (
                record.candidate_visits
                >= self.candidate_promotion_visits - 1
            )
            recent = (
                episode - record.candidate_first_episode
                < self.candidate_recent_protection_episodes
            )
            (protected if near or recent else eligible).append(record_id)
        to_remove = self._weakest_candidates(
            eligible, protected, removal_needed, removal_needed
        )
        for record_id in to_remove:
            if record_id not in self.records:
                continue
            self._record_evicted_candidate(record_id, episode)
            self._delete_candidate(record_id)
        self.candidates_evicted += len(to_remove)

    # ---------- Centroid updates and ordered state processing ----------

    def _update_active_centroid(
        self, record: UnifiedStateRecord, state, safety
    ) -> None:
        if (
            record.centroid_frozen_by_stability
            or record.centroid_frozen_by_cap
        ):
            return
        old = np.asarray(record.state, dtype=np.float32).copy()
        old_safety = np.asarray(record.safety, dtype=np.float32).copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)
        prior = max(
            1,
            record.promotion_evidence_visits
            + record.absorbed_candidate_visits
            + record.active_mask_visits - 1,
        )
        updated = old + (
            np.asarray(state, dtype=np.float32) - old
        ) / float(prior + 1)
        updated_safety = old_safety + (
            np.asarray(safety, dtype=np.float32) - old_safety
        ) / float(prior + 1)
        record.state = updated.astype(self.storage_dtype)
        record.safety = updated_safety.astype(self.storage_dtype)
        record.state_norm = float(np.linalg.norm(record.state))
        record.safety_norm = float(np.linalg.norm(record.safety))
        shift = max(
            float(np.linalg.norm(updated - old) / old_norm),
            self._rms(updated, old),
            self._rms(updated_safety, old_safety),
        )
        record.centroid_updates += 1
        record.centroid_last_shift = shift
        recent_distance = (
            float(np.mean(record.recent_distance_window))
            if record.recent_distance_window
            else float("inf")
        )
        if (
            shift < self.centroid_shift_threshold
            and recent_distance <= self.centroid_stability_distance_threshold
        ):
            record.centroid_stable_updates += 1
        else:
            record.centroid_stable_updates = 0
        if (
            record.centroid_stable_updates
            >= self.centroid_stable_updates_required
        ):
            record.centroid_frozen_by_stability = True
        elif record.centroid_updates >= self.max_centroid_updates:
            record.centroid_frozen_by_cap = True
        self._sync_record_vectors(record)
        self._record_permanent_change(record)

    def _update_candidate_centroid(
        self, record: UnifiedStateRecord, state, safety, new_visits: int
    ) -> None:
        if record.candidate_centroid_frozen:
            return
        old = np.asarray(record.state, dtype=np.float32).copy()
        old_safety = np.asarray(record.safety, dtype=np.float32).copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)
        updated = old + (
            np.asarray(state, dtype=np.float32) - old
        ) / float(new_visits)
        updated_safety = old_safety + (
            np.asarray(safety, dtype=np.float32) - old_safety
        ) / float(new_visits)
        record.state = updated.astype(self.storage_dtype)
        record.safety = updated_safety.astype(self.storage_dtype)
        record.state_norm = float(np.linalg.norm(record.state))
        record.safety_norm = float(np.linalg.norm(record.safety))
        shift = max(
            float(np.linalg.norm(updated - old) / old_norm),
            self._rms(updated, old),
            self._rms(updated_safety, old_safety),
        )
        record.candidate_centroid_updates += 1
        record.candidate_last_shift = shift
        if shift < self.candidate_centroid_shift_threshold:
            record.candidate_stable_updates += 1
        else:
            record.candidate_stable_updates = 0
        if (
            record.candidate_stable_updates
            >= self.candidate_stable_updates_required
            or record.candidate_centroid_updates
            >= self.max_candidate_centroid_updates
        ):
            record.candidate_centroid_frozen = True
        self._sync_record_vectors(record)

    def process_state(
        self,
        state,
        safety,
        episode,
        active_match=None,
        active_match_precomputed: bool = False,
        retired_match=None,
        retired_match_precomputed: bool = False,
    ):
        self._require_policy_mutable("state-pool lifecycle processing")
        self.total_states_seen += 1
        active = (
            active_match
            if active_match_precomputed
            else self.find_active_match(state, safety)
        )
        active_id = active[0]
        if active_id is not None:
            record = self.records[int(active_id)]
            record.active_mask_visits += 1
            record.last_episode_visited = int(episode)
            record.match_count += 1
            self.total_pool_matches += 1
            record.similarity_sum += float(active[1])
            record.distance_sum += float(active[2])
            record.safety_similarity_sum += float(active[3])
            record.safety_distance_sum += float(active[4])
            record.recent_distance_window.append(float(active[2]))
            self._update_active_centroid(record, state, safety)
            return record.record_id, None, "permanent_matched"

        # Retired lookup is deliberately before candidate lookup. The caller
        # normally handles the retired hit to apply its DQN action mask.
        retired = (
            retired_match
            if retired_match_precomputed
            else self.find_retired_match(state, safety)
        )
        if retired[0] is not None:
            return None, None, "retired_match_requires_filtered_dqn"

        candidate_match = self.find_candidate_match(state, safety)
        if candidate_match[0] is not None:
            candidate = self.records[int(candidate_match[0])]
            new_visits = candidate.candidate_visits + 1
            self._update_candidate_centroid(
                candidate, state, safety, new_visits
            )
            candidate.candidate_visits = new_visits
            candidate.candidate_last_episode = int(episode)
            if new_visits >= self.candidate_promotion_visits:
                active_candidate_match = self.find_active_match(
                    candidate.state, candidate.safety
                )
                if active_candidate_match[0] is not None:
                    active_id = int(active_candidate_match[0])
                    retired_id = self.absorb_candidate_into_active(
                        candidate.record_id, active_id, episode
                    )
                    if retired_id is not None:
                        return (
                            None,
                            retired_id,
                            "candidate_history_exhausted_retired_pool",
                        )
                    return (
                        active_id,
                        None,
                        "candidate_merged_into_active_pool",
                    )

                retired_candidate_match = self.find_retired_match(
                    candidate.state, candidate.safety
                )
                if retired_candidate_match[0] is not None:
                    self.candidates_suppressed_by_retired_pool += 1
                    self.candidate_retired_suppression_events += 1
                    self._delete_candidate(candidate.record_id)
                    return (
                        None,
                        int(retired_candidate_match[0]),
                        "candidate_suppressed_by_retired_pool",
                    )

                if self.permanent_capacity_available():
                    promoted_id = self._promote_candidate(
                        candidate.record_id, episode
                    )
                    if promoted_id is None:
                        return (
                            None,
                            None,
                            "candidate_history_exhausted_pool",
                        )
                    return (
                        promoted_id,
                        None,
                        "candidate_promoted_before_action",
                    )

                if not self.absolute_permanent_capacity_reached():
                    self.candidate_capacity_wait_events += 1
                    candidate.last_permanent_match_generation = (
                        self.permanent_generation
                    )
                    self.ready_waiting_ids[candidate.record_id] = None
                    return (
                        None,
                        candidate.record_id,
                        "candidate_waiting_for_pool_capacity_argmax",
                    )

                self.candidates_blocked_by_capacity += 1
                self._delete_candidate(candidate.record_id)
                return (
                    None, None, "candidate_absolute_capacity_argmax"
                )
            return None, candidate.record_id, "candidate_matched_argmax"

        if self.absolute_permanent_capacity_reached():
            self.absolute_capacity_argmax_states += 1
            return None, None, "absolute_pool_capacity_argmax"

        if len(self.status_ids[self.CANDIDATE]) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)
        created_id = self._new_candidate(state, safety, episode)
        if len(self.status_ids[self.CANDIDATE]) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)
            if created_id not in self.records:
                return (
                    None, None, "candidate_created_then_evicted_argmax"
                )
        return None, created_id, "candidate_created_argmax"

    # ---------- Action masks ----------

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
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % len(candidates)
        return int(candidates[offset])

    def select_retired_action(
        self,
        retired_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("retired-pool action selection")
        record = self.records[int(retired_index)]
        if record.status != self.RETIRED:
            raise RuntimeError("Retired action targeted a non-retired record.")
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        if raw_argmax is None:
            raise RuntimeError("DQN produced no selectable action.")
        allowed_mask = self.full_mask & ~record.blocked_mask
        selected = self._best_action_from_mask(
            q_values, allowed_mask, self.action_count, key
        )
        if selected is None:
            selected = raw_argmax
            source = "retired_pool_all_blocked_argmax"
            self.retired_all_blocked_fallbacks += 1
        else:
            if record.blocked_mask & (1 << raw_argmax):
                self.retired_mask_filtered_actions += 1
                source = "retired_pool_next_best_nonblocked"
            else:
                source = "retired_pool_argmax_allowed"
        self.mark_pending_action_outcome(
            record.record_id, episode, step, selected
        )
        return int(selected), source

    def select_candidate_action(
        self,
        candidate_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("candidate action selection")
        record = self.records[int(candidate_index)]
        if record.status != self.CANDIDATE:
            raise RuntimeError("Candidate action targeted a non-candidate.")
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        allowed_mask = self.full_mask & ~record.blocked_mask
        selected = self._best_action_from_mask(
            q_values, allowed_mask, self.action_count, key
        )
        if selected is None:
            selected = raw_argmax
            source = "candidate_all_blocked_argmax"
            self.candidate_all_blocked_fallbacks += 1
        elif record.blocked_mask & (1 << int(raw_argmax)):
            source = "candidate_next_best_nonblocked"
            self.candidate_mask_filtered_actions += 1
        else:
            source = "candidate_argmax_allowed"
        self.record_candidate_action(record.record_id, int(selected))
        self.mark_pending_action_outcome(
            record.record_id, episode, step, int(selected)
        )
        return int(selected), source

    def select_active_action(
        self,
        pool_index: int,
        q_values: np.ndarray,
        key: str,
        episode: int,
        step: int,
    ) -> Tuple[int, str]:
        self._require_policy_mutable("active-pool action selection")
        record = self.records[int(pool_index)]
        if record.status != self.ACTIVE:
            raise RuntimeError("Active action targeted a non-active record.")
        available_mask = int(record.availability_mask)
        if available_mask == 0:
            raise RuntimeError(
                "An active pool with an empty action mask reached selection."
            )
        raw_argmax = self._best_action_from_mask(
            q_values, self.full_mask, self.action_count, key
        )
        allowed_mask = available_mask & ~record.blocked_mask
        selected = self._best_action_from_mask(
            q_values, allowed_mask, self.action_count, key
        )
        if selected is None:
            # Preserve one-pass progress even if an inconsistent record marks
            # every remaining available action blocked.
            selected = self._best_action_from_mask(
                q_values, available_mask, self.action_count, key
            )
            source = "active_all_available_blocked_fallback"
            self.active_all_blocked_fallbacks += 1
        elif not (allowed_mask & (1 << int(raw_argmax))):
            source = "active_next_best_available_nonblocked"
            self.active_mask_filtered_actions += 1
        else:
            source = "active_argmax_available_allowed"
        if selected is None:
            raise RuntimeError("Active pool produced no available action.")
        self.remove(record.record_id, int(selected))
        retire_after = record.availability_mask == 0
        self.mark_pending_action_outcome(
            record.record_id,
            episode,
            step,
            int(selected),
            retire_after_outcome=retire_after,
        )
        if retire_after:
            source = "active_pool_last_action_pending_retirement"
        return int(selected), source

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

    # ---------- Statistics ----------

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
                    "actions_tried": int(self.action_count - remaining),
                    "remaining_actions": int(remaining),
                    "coverage_percent": float(
                        100.0
                        * (self.action_count - remaining)
                        / self.action_count
                    ),
                    "unknown_actions": int(record.unknown_mask.bit_count()),
                    "safe_actions": int(record.safe_mask.bit_count()),
                    "blocked_actions": int(record.blocked_mask.bit_count()),
                    "unknown_mask": int(record.unknown_mask),
                    "safe_mask": int(record.safe_mask),
                    "blocked_mask": int(record.blocked_mask),
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

    def global_statistics(self):
        remaining_candidates = len(self.status_ids[self.CANDIDATE])
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
            "matching_backend": "preallocated_vectorized_exact_linear",
            "pool_policy_frozen": bool(self.policy_frozen),
            "pool_policy_frozen_at_episode": self.policy_frozen_at_episode,
            "pool_policy_freeze_events": self.policy_freeze_events,
            "final_phase_pool_mutations_permitted": False,
            "preallocated_centroid_slots": self.slot_capacity,
            "occupied_centroid_slots": len(self.record_slots),
            "candidate_status_tag": self.CANDIDATE,
            "active_status_tag": self.ACTIVE,
            "retired_status_tag": self.RETIRED,
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
            "states_matched_to_active_pools": self.total_pool_matches,
            "states_matched_to_retired_pools": self.total_retired_pool_hits,
            "retired_mask_filtered_actions": (
                self.retired_mask_filtered_actions
            ),
            "retired_all_blocked_fallbacks": (
                self.retired_all_blocked_fallbacks
            ),
            "active_mask_filtered_actions": (
                self.active_mask_filtered_actions
            ),
            "active_all_blocked_fallbacks": (
                self.active_all_blocked_fallbacks
            ),
            "candidate_mask_filtered_actions": (
                self.candidate_mask_filtered_actions
            ),
            "candidate_all_blocked_fallbacks": (
                self.candidate_all_blocked_fallbacks
            ),
            "safe_action_outcomes": self.safe_action_outcomes,
            "blocked_action_outcomes": self.blocked_action_outcomes,
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
    safety_state: np.ndarray,
    episode: int,
    step: int,
    args,
    action_pools: SimilarStateActionPools,
) -> Tuple[int, str]:
    if experiment != "Karthikeya20":
        raise ValueError(f"Unknown experiment: {experiment}")

    q_values = agent.q_values(state)
    tie_key = f"train|Karthikeya20|{episode}|{step}"
    pooling_limit = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    if episode >= pooling_limit:
        # This is idempotent and also protects standalone callers that do not
        # enter through the main episode loop.
        action_pools.freeze_policy(episode)
        action_pools.total_final_argmax_states += 1
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            "final_phase_argmax",
        )
    if action_pools.policy_frozen:
        raise RuntimeError(
            "A frozen pool policy cannot be re-entered before its boundary."
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
        if (
            candidate_index is not None
            and pool_status in {
                "candidate_history_exhausted_retired_pool",
                "candidate_suppressed_by_retired_pool",
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
            )
        if candidate_index is not None:
            action, mask_source = action_pools.select_candidate_action(
                int(candidate_index),
                q_values,
                tie_key,
                episode,
                step,
            )
            source = (
                pool_status
                if mask_source == "candidate_argmax_allowed"
                else f"{pool_status}__{mask_source}"
            )
            return int(action), source
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        return int(action), pool_status

    action, active_source = action_pools.select_active_action(
        int(pool_index), q_values, tie_key, episode, step
    )
    base_source = (
        "promoted_pool_first_best_available"
        if pool_status == "candidate_promoted_before_action"
        else (
            "candidate_merged_active_pool_best_available"
            if pool_status == "candidate_merged_into_active_pool"
            else "permanent_pool_best_available"
        )
    )
    source = (
        active_source
        if active_source == "active_pool_last_action_pending_retirement"
        else (
            base_source
            if active_source == "active_argmax_available_allowed"
            else f"{base_source}__{active_source}"
        )
    )
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
        "epsilon": 0.0,
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
            f"Karthikeya20 expects exactly 9 discrete actions; environment has {action_count}."
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
    action_pools.validate_invariants()
    pool_rows = action_pools.pool_statistics()
    candidate_rows = action_pools.candidate_statistics()
    retired_rows = action_pools.retired_pool_statistics()
    global_row = action_pools.global_statistics()

    def save_rows(path: Path, rows: List[Dict], columns: List[str]) -> None:
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    pool_columns = [
        "pool_id", "status", "promotion_evidence_visits",
        "absorbed_candidate_visits", "absorbed_candidate_action_evidence",
        "active_pool_mask_visits", "matched_state_count",
        "actions_tried", "remaining_actions", "coverage_percent",
        "unknown_actions", "safe_actions", "blocked_actions",
        "unknown_mask", "safe_mask", "blocked_mask",
        "first_episode_created", "last_episode_visited",
        "mean_general_cosine_similarity", "mean_general_rms_distance",
        "mean_safety_cosine_similarity", "mean_safety_rms_distance",
        "centroid_updates", "centroid_frozen_by_stability",
        "centroid_frozen_by_cap",
    ]
    candidate_columns = [
        "candidate_id", "status", "visit_count", "first_episode", "last_episode",
        "visits_remaining_for_promotion", "unique_actions_executed",
        "unknown_mask", "safe_mask", "blocked_mask",
    ]
    retired_columns = [
        "retired_pool_id", "original_pool_id", "status", "retirement_reason",
        "episode_created", "episode_retired", "permanent_pool_visits",
        "actions_explored", "hits_after_retirement",
        "unknown_actions", "safe_actions", "blocked_actions",
        "unknown_mask", "safe_mask", "blocked_mask",
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
                "method": "Karthikeya20",
                "method_label": "Karthikeya20",
                "phase": phase,
                **collision_summary(phase_rows, args.rmst_tau),
            }
        )
    metrics_path = output_dir / "collision_metrics.csv"
    write_csv(metrics_path, metric_rows)

    # Keep the existing model location and also expose the baseline-style name.
    nested_model = output_dir / "models" / "Karthikeya20_model.pt"
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
        "method": "Karthikeya20",
        "method_label": "Karthikeya20",
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
        description="MetaDrive Karthikeya20: conservative unified lifecycle safety-aware action coverage"
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--max-state-pools",
        type=int,
        default=500,
        help=(
            "Fixed combined active + retired Basic Optimal capacity. "
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
        default=5,
        help="Visits required before promoting a candidate to a permanent pool.",
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
            "pool normalizers and candidate tolerances."
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
            "<project>/policy_results/seed_<seed>/Karthikeya20."
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
    pool_policy_boundary = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )
    if (
        args.auto_calibrate_thresholds
        and args.calibration_start_episode + args.calibration_episodes
        >= pool_policy_boundary
    ):
        raise ValueError(
            "Calibration must finish before the final pure-argmax "
            "pool-policy freeze boundary."
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
        project_dir / "policy_results" / f"seed_{args.seed}" / "Karthikeya20"
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
                f"Karthikeya20 output directory is not empty: {output_dir}. "
                "Reuse it or pass --force to remove the previous partial/completed run."
            )
        if force:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    args.hybrid_design_version = "karthikeya20_conservative_unified_v1"
    # Baseline configuration: permanent capacity is fixed at 500 by making
    # the current and absolute limits identical. No lifecycle logic changes.
    args.maximum_pool_capacity = args.max_state_pools
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
    print("METADRIVE KARTHIKEYA20 CONSERVATIVE SAFETY-AWARE COVERAGE")
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
    print("Policy name: Karthikeya20")
    print("Initial combined active + retired pools:", args.max_state_pools)
    print("Maximum adaptive permanent capacity:", args.maximum_pool_capacity)
    print("Adaptive capacity formula applied:", True)
    print("Pool general representation: full flattened observation")
    print("Pool safety representation:", ", ".join(SAFETY_VECTOR_NAMES))
    print("Unified record statuses: CANDIDATE | ACTIVE | RETIRED")
    print("Lookup order: ACTIVE -> RETIRED -> CANDIDATE -> create")
    print("Testing policy: frozen DQN argmax only")
    print("Auto threshold calibration:", args.auto_calibrate_thresholds)
    print("Calibration start episode:", args.calibration_start_episode)
    print("Calibration episodes:", args.calibration_episodes)
    print("Candidate soft limit:", args.max_state_candidates)
    print("Candidate hard limit:", args.candidate_hard_limit)
    print("Candidate batch eviction count:", args.candidate_batch_evict_count)
    print("Candidate promotion visits:", args.candidate_promotion_visits)
    print("Candidate-stage policy: highest-Q unknown/safe; skip blocked")
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
    print("Final phase: pure DQN argmax; complete pool policy frozen")
    print(
        "Pooling disabled from episode:",
        int(math.ceil(args.train_episodes * args.pool_training_fraction)),
    )
    print("Similarity metrics: cosine similarity + RMS distance")
    print("Pool lookup: preallocated vectorized exact linear matching")
    print("Representative storage dtype:", args.pool_storage_dtype)
    print("Replay sampling: indexed ring buffer, O(batch_size)")
    print("Test safety extraction included in timing: no")
    print("Available actions per new pool: 9")
    print("Candidate visits use outcome-filtered DQN ranking: yes")
    print("Candidate actions are recorded for mask initialization: yes")
    print("Promotion occurs before promotion-visit action selection: yes")
    print("Active match: highest-Q remaining nonblocked action")
    print("Candidate/retired match: highest-Q unknown/safe action")
    print("Candidate/retired all-blocked fallback: ordinary DQN argmax")
    print("Time-limit truncations bootstrap in replay: yes")
    print("Stored retired action values or rankings: no")
    print("Removal: clear selected bit in O(1)")
    print("Promotion and retirement: O(1) status-tag changes")
    print("Final mask action retirement: after observing transition")
    print("Absolute permanent capacity full: do not create new candidates; use maximum-Q")
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
    print("Policy folder name: Karthikeya20")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
