#!/usr/bin/env python3
"""MetaDrive Karthikeya2 DQN with bounded similar-state one-pass action coverage.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-16 CEST (+0200)

Training policy
---------------
* Policy name: karthikeya2.
* Combined active and retired capacity is derived from sqrt(train episodes × max steps),
  bounded to [100, 1000], unless explicitly supplied.
* Temporary candidates use a soft limit and a larger hard limit; once the
  hard limit is reached, the weakest candidates are evicted in one batch.
* Automatic mode first fits training-only normalization, freezes it, then calibrates
  thresholds in that fixed normalized space. Manual mode uses raw observations.
* Active and retired matches are compared jointly; active wins near ties.
* Otherwise the state matches or creates a temporary candidate.
* A candidate is promoted to a permanent pool only after repeated visits.
* During the first 80% of training, retired-pool lookup, active-pool matching,
  candidate accumulation/promotion, adaptive centroids, action masks, and
  one-pass action coverage are active.
* Each pool representative freezes after three consecutive centroid shifts
  below 1%, or after ten centroid updates, whichever occurs first.
* During the final 20% of training, pooling is completely disabled and every
  action is selected using normal maximum-Q.
* Candidate visits 1 through promotion-1 use normal maximum-Q and record
  the genuinely executed actions.
* On the promotion visit, the candidate becomes permanent before action
  selection; previously executed candidate actions are removed from the
  new mask, then the highest-Q remaining action is selected and removed.
* If permanent capacity is full, unpromoted candidates continue using argmax.
* Each pool stores one 9-bit action-availability mask.
* Select the maximum-Q action among actions whose availability bits remain set.
* The selected action bit is cleared in O(1).
* Empty masks retire the active pool. Absorption-caused exhaustion retires immediately;
  final-action exhaustion retires after its real transition is observed.
* When combined active + retired capacity is full, no new pool is created.
* Candidate eviction first consolidates strong active-pool duplicates and protects
  near-promotion candidates whenever possible.
* A pool is retired only after the transition caused by its final mask action is observed.
* Masks are never refilled.

Complexity
----------
* Active, candidate, and retired matching are linear in their bounded
  representative counts: O((B+C+R)D).
* Empty check and bit clearing: O(1).
* Best-available action selection: O(A), where A = 9.
* Normal matching is O((B+C+R)D). Pre-eviction consolidation is O(CBD), but
  occurs only at the hard candidate limit. Ranking is O(C log C).
* Replay insertion is O(1); replay sampling is O(batch_size).
* Policy storage is O((B+C+R)D + B), with B + R bounded by max-state-pools.

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* No RND and no count-based intrinsic reward.
* Frozen greedy testing.
* Disjoint training and testing scenarios.
* Environment traffic configuration matches the canonical baselines.
* Deterministic PyTorch settings match the canonical baselines.

Example:
    python policies/metadrive_karthikeya2.py \
      --seed 11 --test-seed 100000 \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda

Results are saved to:
    policy_results/seed_<seed>/karthikeya2
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


EXPERIMENTS = ["karthikeya2"]
SHORT_LABELS = {
    "karthikeya2": "Karthikeya2",
}
COLORS = {
    "karthikeya2": "#2ca02c",
}


CRITICAL_CONFIG_KEYS = (
    "train_episodes",
    "test_episodes",
    "max_episode_steps",
    "max_state_pools",
    "max_state_candidates",
    "candidate_hard_limit",
    "candidate_batch_evict_count",
    "candidate_promotion_visits",
    "auto_calibrate_thresholds",
    "calibration_episodes",
    "calibration_max_pairs",
    "state_similarity_threshold",
    "state_distance_threshold",
    "candidate_similarity_threshold",
    "candidate_distance_threshold",
    "duplicate_similarity_threshold",
    "duplicate_distance_threshold",
    "active_tie_similarity_tolerance",
    "active_tie_distance_tolerance",
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
    "pool_training_fraction",
    "centroid_shift_threshold",
    "centroid_stable_updates",
    "max_centroid_updates",
    "centroid_stability_distance_threshold",
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


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def interquartile_mean(values: Sequence[float]) -> float:
    """Mean after removing the lowest and highest 25% of observations."""
    data = np.sort(np.asarray(values, dtype=float))
    if data.size == 0:
        return 0.0
    trim = int(math.floor(0.25 * data.size))
    middle = data[trim : data.size - trim] if trim > 0 else data
    return float(np.mean(middle))


def reward_mode(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    rounded = [round(float(x), 6) for x in values]
    counts: Dict[float, int] = {}
    for value in rounded:
        counts[value] = counts.get(value, 0) + 1
    maximum = max(counts.values())
    return float(max(key for key, count in counts.items() if count == maximum))


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
    """List-backed bounded replay with O(1) insertion and O(batch) sampling."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.position = 0
        self.rng = random.Random(int(seed))

    def add(self, state, action, reward, next_state, done) -> None:
        transition = Transition(
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        indices = self.rng.sample(range(len(self.data)), int(batch_size))
        return [self.data[index] for index in indices]

    def __len__(self) -> int:
        return len(self.data)


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
    """Normalized two-stage abstraction with active and retired action pools."""

    def __init__(
        self,
        max_pools: int,
        max_candidates: int,
        candidate_hard_limit: int,
        candidate_batch_evict_count: int,
        candidate_promotion_visits: int,
        action_count: int,
        observation_size: int,
        similarity_threshold: float,
        distance_threshold: float,
        candidate_similarity_threshold: float,
        candidate_distance_threshold: float,
        duplicate_similarity_threshold: float,
        duplicate_distance_threshold: float,
        active_tie_similarity_tolerance: float,
        active_tie_distance_tolerance: float,
        auto_calibrate_thresholds: bool,
        calibration_episodes: int,
        calibration_max_pairs: int,
        seed: int,
        candidate_centroid_shift_threshold: float,
        candidate_stable_updates: int,
        max_candidate_centroid_updates: int,
        centroid_shift_threshold: float,
        centroid_stable_updates: int,
        max_centroid_updates: int,
        centroid_stability_distance_threshold: float,
    ):
        if max_pools <= 0 or max_candidates <= 0:
            raise ValueError("Pool and candidate capacities must be positive.")
        if candidate_hard_limit <= max_candidates:
            raise ValueError(
                "candidate_hard_limit must be greater than max_candidates."
            )
        if (
            candidate_hard_limit - candidate_batch_evict_count
            != max_candidates
        ):
            raise ValueError(
                "candidate_hard_limit - candidate_batch_evict_count "
                "must equal max_candidates."
            )
        if candidate_promotion_visits <= 1:
            raise ValueError("candidate_promotion_visits must exceed one.")

        self.max_pools = int(max_pools)
        self.max_candidates = int(max_candidates)
        self.candidate_hard_limit = int(candidate_hard_limit)
        self.candidate_batch_evict_count = int(candidate_batch_evict_count)
        self.candidate_promotion_visits = int(candidate_promotion_visits)
        self.action_count = int(action_count)
        self.full_mask = (1 << self.action_count) - 1

        self.similarity_threshold = float(similarity_threshold)
        self.distance_threshold = float(distance_threshold)
        self.candidate_similarity_threshold = float(
            candidate_similarity_threshold
        )
        self.candidate_distance_threshold = float(candidate_distance_threshold)
        self.duplicate_similarity_threshold = float(
            duplicate_similarity_threshold
        )
        self.duplicate_distance_threshold = float(duplicate_distance_threshold)
        self.active_tie_similarity_tolerance = float(
            active_tie_similarity_tolerance
        )
        self.active_tie_distance_tolerance = float(
            active_tie_distance_tolerance
        )

        self.auto_calibrate_thresholds = bool(auto_calibrate_thresholds)
        self.calibration_episodes = int(calibration_episodes)
        self.normalization_episodes = (
            max(1, self.calibration_episodes // 2)
            if self.auto_calibrate_thresholds
            else 0
        )
        self.threshold_calibration_start_episode = self.normalization_episodes
        self.thresholds_frozen = not self.auto_calibrate_thresholds
        self.matching_uses_raw_observations = not self.auto_calibrate_thresholds
        self.normalizer = RunningObservationNormalizer(observation_size)
        self.calibrator = SimilarityThresholdCalibrator(
            max_pairs=calibration_max_pairs,
            fallback_similarity=similarity_threshold,
            fallback_distance=distance_threshold,
            fallback_candidate_distance=candidate_distance_threshold,
            seed=seed + 91_337,
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
        self.representative_norms: List[float] = []
        self.active_pool_ids: List[int] = []
        self.next_pool_id = 0
        self.masks: List[int] = []
        self.visit_counts: List[int] = []
        self.absorbed_candidate_visits: List[int] = []
        self.absorbed_candidate_actions: List[int] = []
        self.promotion_evidence_visits: List[int] = []
        self.first_episode_created: List[int] = []
        self.last_episode_visited: List[int] = []
        self.similarity_sums: List[float] = []
        self.similarity_mins: List[float] = []
        self.similarity_maxs: List[float] = []
        self.match_counts: List[int] = []
        self.distance_sums: List[float] = []
        self.distance_mins: List[float] = []
        self.distance_maxs: List[float] = []
        self.centroid_update_counts: List[int] = []
        self.centroid_stable_counts: List[int] = []
        self.centroid_last_shifts: List[float] = []
        self.centroid_frozen_by_stability: List[bool] = []
        self.centroid_frozen_by_cap: List[bool] = []
        self.recent_distance_windows: List[Deque[float]] = []

        # Temporary candidates.
        self.candidate_representatives: List[np.ndarray] = []
        self.candidate_norms: List[float] = []
        self.candidate_visits: List[int] = []
        self.candidate_first_episode: List[int] = []
        self.candidate_last_episode: List[int] = []
        self.candidate_centroid_update_counts: List[int] = []
        self.candidate_stable_counts: List[int] = []
        self.candidate_last_shifts: List[float] = []
        self.candidate_centroid_frozen: List[bool] = []
        self.candidate_action_masks: List[int] = []

        # One retired structure.
        self.retired_representatives: List[np.ndarray] = []
        self.retired_norms: List[float] = []
        self.retired_original_pool_ids: List[int] = []
        self.retired_reasons: List[str] = []
        self.retired_episode_created: List[int] = []
        self.retired_episode_retired: List[int] = []
        self.retired_candidate_evidence_visits: List[int] = []
        self.retired_absorbed_candidate_visits: List[int] = []
        self.retired_permanent_visits: List[int] = []
        self.retired_actions_explored: List[int] = []
        self.retired_hit_counts: List[int] = []
        self.retired_last_hit_episode: List[int] = []
        self.retired_last_hit_step: List[int] = []
        self.retired_similarity_sums: List[float] = []
        self.retired_distance_sums: List[float] = []
        self.retirement_trigger_action: List[int] = []
        self.retirement_trigger_reward: List[float] = []
        self.retirement_trigger_collision: List[bool] = []
        self.retirement_trigger_out_of_road: List[bool] = []
        self.retirement_trigger_done: List[bool] = []
        self.retirement_trigger_step: List[int] = []

        # Pending retirement occurs after the real transition.
        self.pending_retirement: Optional[Dict] = None
        self.pending_candidate_retired_index: Optional[int] = None

        # Detailed evicted candidate audit trail.
        self.evicted_candidate_rows: List[Dict] = []

        # Global statistics.
        self.total_states_seen = 0
        self.total_calibration_states = 0
        self.total_pool_matches = 0
        self.total_pool_creations = 0
        self.total_no_match_at_capacity = 0
        self.total_final_argmax_states = 0
        self.total_candidate_states_argmax = 0
        self.candidates_created = 0
        self.candidates_promoted = 0
        self.candidates_evicted = 0
        self.candidates_blocked_by_capacity = 0
        self.candidates_merged_into_active_pool = 0
        self.candidate_history_actions_checked = 0
        self.candidate_history_actions_removed = 0
        self.candidate_history_actions_already_absent = 0
        self.absorbed_candidate_visits_total = 0
        self.absorbed_candidate_actions_total = 0
        self.pre_eviction_candidates_checked = 0
        self.pre_eviction_candidates_merged = 0
        self.pre_eviction_candidate_visits_absorbed = 0
        self.pre_eviction_candidate_actions_absorbed = 0
        self.pre_eviction_strong_matches_rejected = 0
        self.near_promotion_candidates_protected = 0
        self.candidates_blocked_by_retired_duplicate = 0
        self.active_pools_retired_by_mask_exhaustion = 0
        self.active_pools_retired_by_absorption_exhaustion = 0
        self.total_retired_pool_hits = 0
        self.creation_events: List[Tuple[int, int]] = []
        self.promotion_visit_counts: List[int] = []

    # ---------- Normalization and calibration ----------

    def begin_episode(self) -> None:
        """Prevent consecutive-pair calibration from crossing episode boundaries."""
        self.calibrator.reset_episode()

    def prepare_matching_state(
        self, raw_state: np.ndarray, episode: int
    ) -> Tuple[np.ndarray, str]:
        vector = np.asarray(raw_state, dtype=np.float32).reshape(-1)

        if not self.auto_calibrate_thresholds:
            # Manual mode deliberately uses raw observations rather than a
            # one-sample, meaningless normalizer.
            return vector, "ready"

        if episode < self.normalization_episodes:
            self.normalizer.update(vector)
            self.total_calibration_states += 1
            return vector, "normalizer_warmup"

        if not self.normalizer.frozen:
            self.normalizer.freeze()

        normalized = self.normalizer.transform(vector)

        if episode < self.calibration_episodes:
            self.calibrator.observe(normalized)
            self.total_calibration_states += 1
            return normalized, "threshold_warmup"

        if not self.thresholds_frozen:
            cosine, distance, candidate_distance = self.calibrator.derive()
            self.similarity_threshold = cosine
            self.candidate_similarity_threshold = cosine
            self.distance_threshold = distance
            self.candidate_distance_threshold = candidate_distance
            self.calibrator.freeze()
            self.thresholds_frozen = True

        return normalized, "ready"

    # ---------- Geometry ----------

    @staticmethod
    def _cosine_similarity(
        state: np.ndarray,
        state_norm: float,
        representative: np.ndarray,
        representative_norm: float,
    ) -> float:
        if state_norm == 0.0 or representative_norm == 0.0:
            return 1.0 if np.array_equal(state, representative) else 0.0
        return float(
            np.dot(state, representative) / (state_norm * representative_norm)
        )

    @staticmethod
    def _relative_l2_distance(
        state: np.ndarray,
        representative: np.ndarray,
        representative_norm: float,
    ) -> float:
        del representative_norm
        return float(np.sqrt(np.mean(np.square(state - representative))))

    def _best_match(
        self,
        vector: np.ndarray,
        representatives: Sequence[np.ndarray],
        norms: Sequence[float],
        similarity_threshold: float,
        distance_threshold: float,
    ) -> Tuple[Optional[int], float, float]:
        vector_norm = float(np.linalg.norm(vector))
        best_index: Optional[int] = None
        best_similarity = float("-inf")
        best_distance = float("inf")
        for index, (representative, representative_norm) in enumerate(
            zip(representatives, norms)
        ):
            similarity = self._cosine_similarity(
                vector, vector_norm, representative, representative_norm
            )
            distance = self._relative_l2_distance(
                vector, representative, representative_norm
            )
            if (
                similarity >= similarity_threshold
                and distance <= distance_threshold
                and (
                    best_index is None
                    or similarity > best_similarity
                    or (
                        math.isclose(similarity, best_similarity)
                        and distance < best_distance
                    )
                )
            ):
                best_index = index
                best_similarity = similarity
                best_distance = distance
        return best_index, best_similarity, best_distance

    def find_active_match(
        self, normalized_state: np.ndarray
    ) -> Tuple[Optional[int], float, float]:
        return self._best_match(
            normalized_state,
            self.representatives,
            self.representative_norms,
            self.similarity_threshold,
            self.distance_threshold,
        )

    def find_retired_match(
        self, normalized_state: np.ndarray
    ) -> Tuple[Optional[int], float, float]:
        return self._best_match(
            normalized_state,
            self.retired_representatives,
            self.retired_norms,
            self.similarity_threshold,
            self.distance_threshold,
        )

    def find_candidate_match(
        self, normalized_state: np.ndarray
    ) -> Tuple[Optional[int], float, float]:
        return self._best_match(
            normalized_state,
            self.candidate_representatives,
            self.candidate_norms,
            self.candidate_similarity_threshold,
            self.candidate_distance_threshold,
        )

    def strong_active_match(
        self, normalized_state: np.ndarray
    ) -> Tuple[Optional[int], float, float]:
        return self._best_match(
            normalized_state,
            self.representatives,
            self.representative_norms,
            max(self.duplicate_similarity_threshold, self.similarity_threshold),
            min(self.duplicate_distance_threshold, self.distance_threshold),
        )

    def choose_active_or_retired(
        self, normalized_state: np.ndarray
    ) -> Tuple[str, Optional[int], float, float]:
        ai, asi, adi = self.find_active_match(normalized_state)
        ri, rsi, rdi = self.find_retired_match(normalized_state)

        if ai is None and ri is None:
            return "none", None, float("-inf"), float("inf")
        if ai is not None and ri is None:
            return "active", ai, asi, adi
        if ri is not None and ai is None:
            return "retired", ri, rsi, rdi

        active_dominates = asi >= rsi and adi <= rdi
        retired_dominates = rsi >= asi and rdi <= adi
        if active_dominates and not retired_dominates:
            return "active", ai, asi, adi
        if retired_dominates and not active_dominates:
            return "retired", ri, rsi, rdi

        # Incomparable or nearly tied matches prefer active exploration.
        if (
            abs(asi - rsi) <= self.active_tie_similarity_tolerance
            or abs(adi - rdi) <= self.active_tie_distance_tolerance
        ):
            return "active", ai, asi, adi

        # Normalize each metric by its configured threshold and compare.
        active_score = (
            asi / max(self.similarity_threshold, 1e-8)
            - adi / max(self.distance_threshold, 1e-8)
        )
        retired_score = (
            rsi / max(self.similarity_threshold, 1e-8)
            - rdi / max(self.distance_threshold, 1e-8)
        )
        if active_score >= retired_score:
            return "active", ai, asi, adi
        return "retired", ri, rsi, rdi

    # ---------- Capacity and storage ----------

    def total_permanent_records(self) -> int:
        return len(self.representatives) + len(self.retired_representatives)

    def permanent_capacity_available(self) -> bool:
        return self.total_permanent_records() < self.max_pools

    def _append_permanent_pool(
        self,
        representative: np.ndarray,
        episode: int,
        initial_visits: int,
        initial_mask: Optional[int] = None,
    ) -> int:
        if not self.permanent_capacity_available():
            raise RuntimeError("Combined active + retired capacity is full.")
        vector = np.asarray(representative, dtype=np.float32).copy()
        self.representatives.append(vector)
        self.representative_norms.append(float(np.linalg.norm(vector)))
        self.active_pool_ids.append(self.next_pool_id)
        self.next_pool_id += 1
        self.masks.append(
            self.full_mask if initial_mask is None else int(initial_mask)
        )
        self.visit_counts.append(0)
        self.absorbed_candidate_visits.append(0)
        self.absorbed_candidate_actions.append(0)
        self.promotion_evidence_visits.append(int(initial_visits))
        self.first_episode_created.append(int(episode))
        self.last_episode_visited.append(int(episode))
        self.similarity_sums.append(0.0)
        self.similarity_mins.append(float("inf"))
        self.similarity_maxs.append(float("-inf"))
        self.match_counts.append(0)
        self.distance_sums.append(0.0)
        self.distance_mins.append(float("inf"))
        self.distance_maxs.append(0.0)
        self.centroid_update_counts.append(0)
        self.centroid_stable_counts.append(0)
        self.centroid_last_shifts.append(0.0)
        self.centroid_frozen_by_stability.append(False)
        self.centroid_frozen_by_cap.append(False)
        self.recent_distance_windows.append(deque(maxlen=5))
        self.total_pool_creations += 1
        self.creation_events.append(
            (int(episode), int(self.total_pool_creations))
        )
        return len(self.representatives) - 1

    def _remove_active_pool(self, index: int) -> None:
        for sequence in (
            self.representatives,
            self.representative_norms,
            self.active_pool_ids,
            self.masks,
            self.visit_counts,
            self.absorbed_candidate_visits,
            self.absorbed_candidate_actions,
            self.promotion_evidence_visits,
            self.first_episode_created,
            self.last_episode_visited,
            self.similarity_sums,
            self.similarity_mins,
            self.similarity_maxs,
            self.match_counts,
            self.distance_sums,
            self.distance_mins,
            self.distance_maxs,
            self.centroid_update_counts,
            self.centroid_stable_counts,
            self.centroid_last_shifts,
            self.centroid_frozen_by_stability,
            self.centroid_frozen_by_cap,
            self.recent_distance_windows,
        ):
            sequence.pop(index)

    def _retire_active_pool(
        self,
        index: int,
        episode: int,
        reason: str,
        trigger: Optional[Dict] = None,
    ) -> int:
        retired_id = len(self.retired_representatives)
        self.retired_representatives.append(self.representatives[index].copy())
        self.retired_norms.append(float(self.representative_norms[index]))
        self.retired_original_pool_ids.append(int(self.active_pool_ids[index]))
        self.retired_reasons.append(str(reason))
        self.retired_episode_created.append(
            int(self.first_episode_created[index])
        )
        self.retired_episode_retired.append(int(episode))
        self.retired_candidate_evidence_visits.append(
            int(self.promotion_evidence_visits[index])
        )
        self.retired_absorbed_candidate_visits.append(
            int(self.absorbed_candidate_visits[index])
        )
        self.retired_permanent_visits.append(int(self.visit_counts[index]))
        self.retired_actions_explored.append(
            int(self.action_count - self.remaining_count(index))
        )
        self.retired_hit_counts.append(0)
        self.retired_last_hit_episode.append(-1)
        self.retired_last_hit_step.append(-1)
        self.retired_similarity_sums.append(0.0)
        self.retired_distance_sums.append(0.0)

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

        if reason == "MASK_EXHAUSTED":
            self.active_pools_retired_by_mask_exhaustion += 1
        self._remove_active_pool(index)
        return retired_id

    def mark_pending_retirement(
        self,
        pool_index: int,
        episode: int,
        step: int,
        action: int,
    ) -> None:
        self.pending_retirement = {
            "pool_id": int(self.active_pool_ids[int(pool_index)]),
            "episode": int(episode),
            "step": int(step),
            "action": int(action),
        }

    def finalize_pending_retirement(
        self,
        reward: float,
        parsed: Dict,
        done: bool,
    ) -> None:
        if self.pending_retirement is None:
            return
        pool_id = int(self.pending_retirement["pool_id"])
        try:
            index = self.active_pool_ids.index(pool_id)
        except ValueError:
            self.pending_retirement = None
            return
        trigger = {
            **self.pending_retirement,
            "reward": float(reward),
            "collision": bool(parsed.get("collision", False)),
            "out_of_road": bool(parsed.get("out_of_road", False)),
            "done": bool(done),
        }
        self._retire_active_pool(
            index=index,
            episode=int(self.pending_retirement["episode"]),
            reason="MASK_EXHAUSTED",
            trigger=trigger,
        )
        self.pending_retirement = None

    # ---------- Candidate consolidation and eviction ----------

    def _remove_candidate(self, index: int) -> None:
        for sequence in (
            self.candidate_representatives,
            self.candidate_norms,
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

    def apply_action_history_to_mask(
        self, pool_index: int, action_history_mask: int
    ) -> Tuple[int, int]:
        before = int(self.masks[int(pool_index)])
        history = int(action_history_mask) & self.full_mask
        removable = before & history
        already_absent = history & (~before) & self.full_mask
        self.masks[int(pool_index)] = before & ~history
        return removable.bit_count(), already_absent.bit_count()

    def absorb_candidate_into_active(
        self,
        candidate_index: int,
        pool_index: int,
        pre_eviction: bool,
        episode: int,
    ) -> Optional[int]:
        visits = int(self.candidate_visits[candidate_index])
        history = int(self.candidate_action_masks[candidate_index])
        unique_actions = int(history.bit_count())

        # Preserve spatial evidence with a weighted centroid merge unless the
        # active representative has already frozen.
        if not (
            self.centroid_frozen_by_stability[pool_index]
            or self.centroid_frozen_by_cap[pool_index]
        ):
            active_weight = max(
                1,
                int(self.promotion_evidence_visits[pool_index])
                + int(self.absorbed_candidate_visits[pool_index])
                + int(self.visit_counts[pool_index]),
            )
            candidate_centroid = self.candidate_representatives[candidate_index]
            combined_weight = active_weight + visits
            merged = (
                self.representatives[pool_index] * active_weight
                + candidate_centroid * visits
            ) / float(combined_weight)
            self.representatives[pool_index] = merged.astype(np.float32)
            self.representative_norms[pool_index] = float(
                np.linalg.norm(self.representatives[pool_index])
            )

        removed, already_absent = self.apply_action_history_to_mask(
            pool_index, history
        )
        self.absorbed_candidate_visits[pool_index] += visits
        self.absorbed_candidate_actions[pool_index] += unique_actions
        self.absorbed_candidate_visits_total += visits
        self.absorbed_candidate_actions_total += unique_actions
        self.candidate_history_actions_checked += unique_actions
        self.candidate_history_actions_removed += removed
        self.candidate_history_actions_already_absent += already_absent
        self.candidates_merged_into_active_pool += 1
        if pre_eviction:
            self.pre_eviction_candidates_merged += 1
            self.pre_eviction_candidate_visits_absorbed += visits
            self.pre_eviction_candidate_actions_absorbed += unique_actions

        self._remove_candidate(candidate_index)

        if self.mask(pool_index) == 0:
            retired_index = self._retire_active_pool(
                pool_index,
                episode=episode,
                reason="ABSORPTION_EXHAUSTED",
                trigger={
                    "action": -1,
                    "reward": math.nan,
                    "collision": False,
                    "out_of_road": False,
                    "done": False,
                    "step": -1,
                },
            )
            self.active_pools_retired_by_absorption_exhaustion += 1
            return retired_index
        return None

    def _record_evicted_candidate(self, index: int, episode: int) -> None:
        self.evicted_candidate_rows.append(
            {
                "eviction_episode": int(episode),
                "visit_count": int(self.candidate_visits[index]),
                "first_episode": int(self.candidate_first_episode[index]),
                "last_episode": int(self.candidate_last_episode[index]),
                "unique_actions_executed": int(
                    self.candidate_action_masks[index].bit_count()
                ),
                "executed_action_mask": int(
                    self.candidate_action_masks[index]
                ),
                "near_promotion": bool(
                    self.candidate_visits[index]
                    >= self.candidate_promotion_visits - 1
                ),
                "centroid_updates": int(
                    self.candidate_centroid_update_counts[index]
                ),
            }
        )

    def _batch_evict_candidates(self, episode: int) -> None:
        if len(self.candidate_representatives) < self.candidate_hard_limit:
            return

        # Consolidate strong active duplicates before discarding evidence.
        index = len(self.candidate_representatives) - 1
        while index >= 0:
            self.pre_eviction_candidates_checked += 1
            pool_index, _, _ = self.strong_active_match(
                self.candidate_representatives[index]
            )
            if pool_index is not None:
                self.absorb_candidate_into_active(
                    candidate_index=index,
                    pool_index=int(pool_index),
                    pre_eviction=True,
                    episode=episode,
                )
            else:
                self.pre_eviction_strong_matches_rejected += 1
            index -= 1

        removal_needed = max(
            0, len(self.candidate_representatives) - self.max_candidates
        )
        if removal_needed == 0:
            return

        normal = [
            i for i, visits in enumerate(self.candidate_visits)
            if visits < self.candidate_promotion_visits - 1
        ]
        protected = [
            i for i, visits in enumerate(self.candidate_visits)
            if visits >= self.candidate_promotion_visits - 1
        ]
        self.near_promotion_candidates_protected += len(protected)

        key = lambda i: (
            self.candidate_visits[i],
            self.candidate_last_episode[i],
            self.candidate_first_episode[i],
        )
        ranked = sorted(normal, key=key)
        if len(ranked) < removal_needed:
            # Last resort only: include near-promotion candidates.
            ranked.extend(sorted(protected, key=key))
        to_remove = ranked[:removal_needed]
        for candidate_index in sorted(to_remove, reverse=True):
            self._record_evicted_candidate(candidate_index, episode)
            self._remove_candidate(candidate_index)
        self.candidates_evicted += len(to_remove)

    # ---------- Centroids ----------

    def _update_permanent_centroid(
        self, index: int, vector: np.ndarray, new_visits: int
    ) -> None:
        if (
            self.centroid_frozen_by_stability[index]
            or self.centroid_frozen_by_cap[index]
        ):
            return

        centroid = self.representatives[index]
        old = centroid.copy()
        old_norm = max(float(np.linalg.norm(old)), 1e-8)

        # Include promotion and absorbed evidence so the first permanent match
        # does not overwrite the candidate-derived representative.
        previous_weight = max(
            1,
            int(self.promotion_evidence_visits[index])
            + int(self.absorbed_candidate_visits[index])
            + int(new_visits) - 1,
        )
        centroid += (vector - centroid) / float(previous_weight + 1)
        self.representative_norms[index] = float(np.linalg.norm(centroid))

        shift = float(np.linalg.norm(centroid - old) / old_norm)
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

        if (
            self.centroid_stable_counts[index]
            >= self.centroid_stable_updates_required
        ):
            self.centroid_frozen_by_stability[index] = True
        elif self.centroid_update_counts[index] >= self.max_centroid_updates:
            self.centroid_frozen_by_cap[index] = True

    # ---------- Main state processing ----------

    def process_state(
        self,
        normalized_state: np.ndarray,
        episode: int,
        active_hint: Optional[int] = None,
    ) -> Tuple[Optional[int], Optional[int], str, float]:
        vector = np.asarray(normalized_state, dtype=np.float32).reshape(-1)
        self.total_states_seen += 1

        if active_hint is None:
            pool_index, similarity, distance = self.find_active_match(vector)
        else:
            pool_index = int(active_hint)
            similarity = self._cosine_similarity(
                vector,
                float(np.linalg.norm(vector)),
                self.representatives[pool_index],
                self.representative_norms[pool_index],
            )
            distance = self._relative_l2_distance(
                vector,
                self.representatives[pool_index],
                self.representative_norms[pool_index],
            )

        candidate_index, candidate_similarity, _ = self.find_candidate_match(
            vector
        )

        if pool_index is not None:
            pool_index = int(pool_index)

            # Absorb only under the stricter duplicate criterion.
            if candidate_index is not None:
                strong_index, _, _ = self.strong_active_match(
                    self.candidate_representatives[int(candidate_index)]
                )
                if strong_index == pool_index:
                    pool_id = int(self.active_pool_ids[pool_index])
                    retired_index = self.absorb_candidate_into_active(
                        int(candidate_index),
                        pool_index,
                        pre_eviction=False,
                        episode=episode,
                    )
                    candidate_index = None
                    if retired_index is not None:
                        self.pending_candidate_retired_index = retired_index
                        return (
                            None,
                            None,
                            "candidate_merge_exhausted_active_pool",
                            float(similarity),
                        )
                    pool_index = self.active_pool_ids.index(pool_id)

            old_visits = self.visit_counts[pool_index]
            new_visits = old_visits + 1
            self.visit_counts[pool_index] = new_visits
            self.last_episode_visited[pool_index] = int(episode)
            self.total_pool_matches += 1
            self.match_counts[pool_index] += 1
            self.similarity_sums[pool_index] += float(similarity)
            self.similarity_mins[pool_index] = min(
                self.similarity_mins[pool_index], float(similarity)
            )
            self.similarity_maxs[pool_index] = max(
                self.similarity_maxs[pool_index], float(similarity)
            )
            self.distance_sums[pool_index] += float(distance)
            self.distance_mins[pool_index] = min(
                self.distance_mins[pool_index], float(distance)
            )
            self.distance_maxs[pool_index] = max(
                self.distance_maxs[pool_index], float(distance)
            )
            self.recent_distance_windows[pool_index].append(float(distance))
            self._update_permanent_centroid(
                pool_index, vector, new_visits
            )
            return pool_index, None, "permanent_matched", float(similarity)

        if candidate_index is not None:
            index = int(candidate_index)
            new_visits = self.candidate_visits[index] + 1
            centroid = self.candidate_representatives[index]
            if not self.candidate_centroid_frozen[index]:
                old = centroid.copy()
                old_norm = max(float(np.linalg.norm(old)), 1e-8)
                centroid += (vector - centroid) / float(new_visits)
                self.candidate_norms[index] = float(np.linalg.norm(centroid))
                shift = float(np.linalg.norm(centroid - old) / old_norm)
                self.candidate_centroid_update_counts[index] += 1
                self.candidate_last_shifts[index] = shift
                if shift < self.candidate_centroid_shift_threshold:
                    self.candidate_stable_counts[index] += 1
                else:
                    self.candidate_stable_counts[index] = 0
                if (
                    self.candidate_stable_counts[index]
                    >= self.candidate_stable_updates_required
                    or self.candidate_centroid_update_counts[index]
                    >= self.max_candidate_centroid_updates
                ):
                    self.candidate_centroid_frozen[index] = True
            self.candidate_visits[index] = new_visits
            self.candidate_last_episode[index] = int(episode)

            if new_visits >= self.candidate_promotion_visits:
                # Recheck active and retired structures using the candidate centroid.
                strong_active, _, _ = self.strong_active_match(centroid)
                if strong_active is not None:
                    active_pool_id = int(
                        self.active_pool_ids[int(strong_active)]
                    )
                    retired_index = self.absorb_candidate_into_active(
                        index,
                        int(strong_active),
                        pre_eviction=False,
                        episode=episode,
                    )
                    if retired_index is not None:
                        self.pending_candidate_retired_index = retired_index
                        return (
                            None,
                            None,
                            "candidate_merge_exhausted_active_pool",
                            float(candidate_similarity),
                        )
                    active_index = self.active_pool_ids.index(active_pool_id)
                    return (
                        active_index,
                        None,
                        "candidate_merged_into_active_pool",
                        float(candidate_similarity),
                    )

                retired_index, retired_similarity, retired_distance = (
                    self.find_retired_match(centroid)
                )
                if retired_index is not None:
                    self.pending_candidate_retired_index = int(retired_index)
                    self.candidates_blocked_by_retired_duplicate += 1
                    self._remove_candidate(index)
                    return (
                        None,
                        None,
                        "candidate_matches_retired_argmax",
                        float(retired_similarity),
                    )

                if self.permanent_capacity_available():
                    history = int(self.candidate_action_masks[index])
                    remaining = self.full_mask & ~history
                    permanent_index = self._append_permanent_pool(
                        representative=centroid.copy(),
                        episode=episode,
                        initial_visits=new_visits - 1,
                        initial_mask=remaining,
                    )
                    self.promotion_visit_counts.append(new_visits)
                    self.candidates_promoted += 1
                    self._remove_candidate(index)
                    return (
                        permanent_index,
                        None,
                        "candidate_promoted_before_action",
                        float(candidate_similarity),
                    )

                self.total_no_match_at_capacity += 1
                self.total_candidate_states_argmax += 1
                self.candidates_blocked_by_capacity += 1
                self._remove_candidate(index)
                return (
                    None,
                    None,
                    "candidate_capacity_argmax",
                    float(candidate_similarity),
                )

            self.total_candidate_states_argmax += 1
            return (
                None,
                index,
                "candidate_matched_argmax",
                float(candidate_similarity),
            )

        if len(self.candidate_representatives) >= self.candidate_hard_limit:
            self._batch_evict_candidates(episode)

        self.candidate_representatives.append(vector.copy())
        self.candidate_norms.append(float(np.linalg.norm(vector)))
        self.candidate_visits.append(1)
        self.candidate_first_episode.append(int(episode))
        self.candidate_last_episode.append(int(episode))
        self.candidate_centroid_update_counts.append(0)
        self.candidate_stable_counts.append(0)
        self.candidate_last_shifts.append(0.0)
        self.candidate_centroid_frozen.append(False)
        self.candidate_action_masks.append(0)
        self.candidates_created += 1
        self.total_candidate_states_argmax += 1
        created_index = len(self.candidate_representatives) - 1

        # Enforce the documented hard-limit trigger immediately.
        if len(self.candidate_representatives) >= self.candidate_hard_limit:
            created_identity = id(self.candidate_representatives[created_index])
            self._batch_evict_candidates(episode)
            created_index = next(
                (
                    i
                    for i, representative in enumerate(
                        self.candidate_representatives
                    )
                    if id(representative) == created_identity
                ),
                -1,
            )
            if created_index < 0:
                # The just-created candidate was evicted. Its current argmax
                # action still executes, but there is no candidate history left.
                return None, None, "candidate_created_then_evicted_argmax", 1.0

        return (
            None,
            created_index,
            "candidate_created_argmax",
            1.0,
        )

    def record_candidate_action(self, candidate_index: int, action: int) -> None:
        self.candidate_action_masks[int(candidate_index)] |= 1 << int(action)

    def record_retired_hit(
        self,
        retired_index: int,
        episode: int,
        step: int,
        similarity: float,
        distance: float,
    ) -> None:
        index = int(retired_index)
        self.retired_hit_counts[index] += 1
        self.retired_last_hit_episode[index] = int(episode)
        self.retired_last_hit_step[index] = int(step)
        self.retired_similarity_sums[index] += float(similarity)
        self.retired_distance_sums[index] += float(distance)
        self.total_retired_pool_hits += 1

    def mask(self, pool_index: int) -> int:
        return int(self.masks[int(pool_index)])

    def remove(self, pool_index: int, action: int) -> None:
        self.masks[int(pool_index)] &= ~(1 << int(action))

    def remaining_count(self, pool_index: int) -> int:
        return int(self.masks[int(pool_index)].bit_count())

    # ---------- Statistics ----------

    def pool_statistics(self) -> List[Dict]:
        rows: List[Dict] = []
        for index in range(len(self.representatives)):
            visits = int(self.visit_counts[index])
            remaining = self.remaining_count(index)
            actions_tried = self.action_count - remaining
            matches = int(self.match_counts[index])
            rows.append(
                {
                    "pool_id": int(self.active_pool_ids[index]),
                    "promotion_evidence_visits": int(
                        self.promotion_evidence_visits[index]
                    ),
                    "absorbed_candidate_visits": int(
                        self.absorbed_candidate_visits[index]
                    ),
                    "absorbed_candidate_actions": int(
                        self.absorbed_candidate_actions[index]
                    ),
                    "active_pool_mask_visits": visits,
                    "total_observed_visits": int(
                        self.promotion_evidence_visits[index]
                        + self.absorbed_candidate_visits[index]
                        + visits
                    ),
                    "matched_state_count": matches,
                    "actions_tried": int(actions_tried),
                    "remaining_actions": int(remaining),
                    "coverage_percent": 100.0
                    * actions_tried
                    / self.action_count,
                    "first_episode_created": int(
                        self.first_episode_created[index]
                    ),
                    "last_episode_visited": int(
                        self.last_episode_visited[index]
                    ),
                    "mean_match_similarity": (
                        self.similarity_sums[index] / matches
                        if matches
                        else 0.0
                    ),
                    "minimum_match_similarity": (
                        self.similarity_mins[index] if matches else 0.0
                    ),
                    "maximum_match_similarity": (
                        self.similarity_maxs[index] if matches else 0.0
                    ),
                    "mean_relative_l2_distance": (
                        self.distance_sums[index] / matches
                        if matches
                        else 0.0
                    ),
                    "minimum_relative_l2_distance": (
                        self.distance_mins[index] if matches else 0.0
                    ),
                    "maximum_relative_l2_distance": (
                        self.distance_maxs[index] if matches else 0.0
                    ),
                    "centroid_updates": int(
                        self.centroid_update_counts[index]
                    ),
                    "centroid_last_relative_shift": float(
                        self.centroid_last_shifts[index]
                    ),
                    "centroid_frozen_by_stability": bool(
                        self.centroid_frozen_by_stability[index]
                    ),
                    "centroid_frozen_by_cap": bool(
                        self.centroid_frozen_by_cap[index]
                    ),
                }
            )
        return rows

    def candidate_statistics(self) -> List[Dict]:
        return [
            {
                "candidate_id": index,
                "visit_count": int(self.candidate_visits[index]),
                "first_episode": int(self.candidate_first_episode[index]),
                "last_episode": int(self.candidate_last_episode[index]),
                "visits_remaining_for_promotion": max(
                    0,
                    self.candidate_promotion_visits
                    - int(self.candidate_visits[index]),
                ),
                "centroid_updates": int(
                    self.candidate_centroid_update_counts[index]
                ),
                "centroid_last_relative_shift": float(
                    self.candidate_last_shifts[index]
                ),
                "centroid_consecutive_stable_updates": int(
                    self.candidate_stable_counts[index]
                ),
                "centroid_is_frozen": bool(
                    self.candidate_centroid_frozen[index]
                ),
                "unique_actions_executed": int(
                    self.candidate_action_masks[index].bit_count()
                ),
                "executed_action_mask": int(
                    self.candidate_action_masks[index]
                ),
            }
            for index in range(len(self.candidate_representatives))
        ]

    def retired_pool_statistics(self) -> List[Dict]:
        rows: List[Dict] = []
        for index in range(len(self.retired_representatives)):
            hits = int(self.retired_hit_counts[index])
            rows.append(
                {
                    "retired_pool_id": index,
                    "original_pool_id": int(
                        self.retired_original_pool_ids[index]
                    ),
                    "retirement_reason": self.retired_reasons[index],
                    "episode_created": int(
                        self.retired_episode_created[index]
                    ),
                    "episode_retired": int(
                        self.retired_episode_retired[index]
                    ),
                    "candidate_evidence_visits": int(
                        self.retired_candidate_evidence_visits[index]
                    ),
                    "absorbed_candidate_visits": int(
                        self.retired_absorbed_candidate_visits[index]
                    ),
                    "permanent_pool_visits": int(
                        self.retired_permanent_visits[index]
                    ),
                    "actions_explored": int(
                        self.retired_actions_explored[index]
                    ),
                    "hits_after_retirement": hits,
                    "last_hit_episode": int(
                        self.retired_last_hit_episode[index]
                    ),
                    "last_hit_step": int(
                        self.retired_last_hit_step[index]
                    ),
                    "mean_retired_hit_similarity": (
                        self.retired_similarity_sums[index] / hits
                        if hits
                        else 0.0
                    ),
                    "mean_retired_hit_relative_l2_distance": (
                        self.retired_distance_sums[index] / hits
                        if hits
                        else 0.0
                    ),
                    "retirement_trigger_action": int(
                        self.retirement_trigger_action[index]
                    ),
                    "retirement_trigger_reward": float(
                        self.retirement_trigger_reward[index]
                    ),
                    "retirement_trigger_collision": bool(
                        self.retirement_trigger_collision[index]
                    ),
                    "retirement_trigger_out_of_road": bool(
                        self.retirement_trigger_out_of_road[index]
                    ),
                    "retirement_trigger_done": bool(
                        self.retirement_trigger_done[index]
                    ),
                    "retirement_trigger_step": int(
                        self.retirement_trigger_step[index]
                    ),
                }
            )
        return rows

    def global_statistics(self) -> Dict:
        active_rows = self.pool_statistics()
        active_visits = [
            int(row["active_pool_mask_visits"]) for row in active_rows
        ]
        retired_visits = [int(v) for v in self.retired_permanent_visits]
        all_visits = active_visits + retired_visits
        created_accounted = (
            self.candidates_promoted
            + self.candidates_merged_into_active_pool
            + self.candidates_evicted
            + self.candidates_blocked_by_capacity
            + self.candidates_blocked_by_retired_duplicate
            + len(self.candidate_representatives)
        )
        all_pool_total_visits = sum(all_visits)
        return {
            "maximum_combined_permanent_capacity": self.max_pools,
            "permanent_pools_created_total": int(self.total_pool_creations),
            "active_permanent_pools": len(self.representatives),
            "retired_permanent_pools": len(self.retired_representatives),
            "combined_permanent_records": self.total_permanent_records(),
            "combined_capacity_invariant_holds": (
                self.total_permanent_records() <= self.max_pools
            ),
            "unused_combined_permanent_capacity": (
                self.max_pools - self.total_permanent_records()
            ),
            "combined_permanent_capacity_usage_percent": (
                100.0 * self.total_permanent_records() / self.max_pools
            ),
            "matching_uses_raw_observations": (
                self.matching_uses_raw_observations
            ),
            "normalization_episodes": self.normalization_episodes,
            "threshold_calibration_start_episode": (
                self.threshold_calibration_start_episode
            ),
            "distance_metric": "RMS distance",
            "calibrated_state_similarity_threshold": (
                self.similarity_threshold
            ),
            "calibrated_state_distance_threshold": self.distance_threshold,
            "calibrated_candidate_distance_threshold": (
                self.candidate_distance_threshold
            ),
            **self.normalizer.statistics(),
            **self.calibrator.statistics(),
            "candidate_soft_limit": self.max_candidates,
            "candidate_hard_limit": self.candidate_hard_limit,
            "active_candidates_at_end": len(
                self.candidate_representatives
            ),
            "candidates_created": int(self.candidates_created),
            "candidates_promoted_to_new_pool": int(
                self.candidates_promoted
            ),
            "candidates_merged_into_active_pool": int(
                self.candidates_merged_into_active_pool
            ),
            "candidates_evicted": int(self.candidates_evicted),
            "candidates_blocked_by_permanent_capacity": int(
                self.candidates_blocked_by_capacity
            ),
            "candidates_blocked_by_retired_duplicate": int(
                self.candidates_blocked_by_retired_duplicate
            ),
            "candidates_remaining_at_end": len(
                self.candidate_representatives
            ),
            "candidate_accounting_total": int(created_accounted),
            "candidate_accounting_matches_created": (
                created_accounted == self.candidates_created
            ),
            "candidates_never_created_new_pool": int(
                self.candidates_created - self.candidates_promoted
            ),
            "absorbed_candidate_visits": int(
                self.absorbed_candidate_visits_total
            ),
            "absorbed_candidate_actions": int(
                self.absorbed_candidate_actions_total
            ),
            "pre_eviction_candidates_checked": int(
                self.pre_eviction_candidates_checked
            ),
            "pre_eviction_candidates_merged": int(
                self.pre_eviction_candidates_merged
            ),
            "pre_eviction_candidate_visits_absorbed": int(
                self.pre_eviction_candidate_visits_absorbed
            ),
            "pre_eviction_candidate_actions_absorbed": int(
                self.pre_eviction_candidate_actions_absorbed
            ),
            "pre_eviction_strong_matches_rejected": int(
                self.pre_eviction_strong_matches_rejected
            ),
            "near_promotion_candidates_protected": int(
                self.near_promotion_candidates_protected
            ),
            "candidate_history_actions_checked": int(
                self.candidate_history_actions_checked
            ),
            "candidate_history_actions_removed": int(
                self.candidate_history_actions_removed
            ),
            "candidate_history_actions_already_absent": int(
                self.candidate_history_actions_already_absent
            ),
            "pool_active_training_states": int(self.total_states_seen),
            "calibration_training_states": int(
                self.total_calibration_states
            ),
            "final_argmax_training_states": int(
                self.total_final_argmax_states
            ),
            "states_matched_to_active_pools": int(
                self.total_pool_matches
            ),
            "states_matched_to_retired_pools": int(
                self.total_retired_pool_hits
            ),
            "states_blocked_by_permanent_capacity": int(
                self.total_no_match_at_capacity
            ),
            "mean_active_pool_mask_visits": (
                float(np.mean(active_visits)) if active_visits else 0.0
            ),
            "median_active_pool_mask_visits": (
                float(np.median(active_visits)) if active_visits else 0.0
            ),
            "largest_active_pool_visit_share_percent": (
                100.0 * max(active_visits) / sum(active_visits)
                if active_visits and sum(active_visits)
                else 0.0
            ),
            "mean_all_permanent_pool_visits": (
                float(np.mean(all_visits)) if all_visits else 0.0
            ),
            "median_all_permanent_pool_visits": (
                float(np.median(all_visits)) if all_visits else 0.0
            ),
            "largest_all_permanent_pool_visit_share_percent": (
                100.0 * max(all_visits) / all_pool_total_visits
                if all_visits and all_pool_total_visits
                else 0.0
            ),
            "retired_due_to_mask_exhaustion": int(
                self.active_pools_retired_by_mask_exhaustion
            ),
            "retired_due_to_absorption_exhaustion": int(
                self.active_pools_retired_by_absorption_exhaustion
            ),
            "total_retired_pool_hits": int(
                self.total_retired_pool_hits
            ),
            "mean_hits_per_retired_pool": (
                float(np.mean(self.retired_hit_counts))
                if self.retired_hit_counts
                else 0.0
            ),
            "median_hits_per_retired_pool": (
                float(np.median(self.retired_hit_counts))
                if self.retired_hit_counts
                else 0.0
            ),
            "maximum_hits_for_one_retired_pool": (
                max(self.retired_hit_counts)
                if self.retired_hit_counts
                else 0
            ),
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
    episode: int,
    step: int,
    args,
    action_pools: SimilarStateActionPools,
) -> Tuple[int, str]:
    """Select an action and defer any final-action retirement until transition."""
    if experiment != "karthikeya2":
        raise ValueError(f"Unknown experiment: {experiment}")

    q_values = agent.q_values(state)
    tie_key = f"train|karthikeya2|{episode}|{step}"
    pooling_episode_limit = int(
        math.ceil(args.train_episodes * args.pool_training_fraction)
    )

    if episode >= pooling_episode_limit:
        action_pools.total_final_argmax_states += 1
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            "final_phase_argmax",
        )

    matching_state, preparation_status = (
        action_pools.prepare_matching_state(state, episode)
    )
    if preparation_status != "ready":
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            f"{preparation_status}_argmax",
        )

    decision, match_index, similarity, distance = (
        action_pools.choose_active_or_retired(matching_state)
    )
    if decision == "retired":
        action_pools.record_retired_hit(
            retired_index=int(match_index),
            episode=episode,
            step=step,
            similarity=similarity,
            distance=distance,
        )
        return (
            agent._deterministic_extreme_from_q(
                q_values, maximum=True, key=tie_key
            ),
            "retired_pool_argmax",
        )

    pool_index, candidate_index, pool_status, status_similarity = (
        action_pools.process_state(
            normalized_state=matching_state,
            episode=episode,
            active_hint=(
                int(match_index) if decision == "active" else None
            ),
        )
    )

    if pool_index is None:
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        if candidate_index is not None:
            action_pools.record_candidate_action(candidate_index, action)
        if (
            pool_status in {"candidate_matches_retired_argmax", "candidate_merge_exhausted_active_pool"}
            and action_pools.pending_candidate_retired_index is not None
        ):
            retired_index = action_pools.pending_candidate_retired_index
            action_pools.record_retired_hit(
                retired_index=retired_index,
                episode=episode,
                step=step,
                similarity=status_similarity,
                distance=0.0,
            )
            action_pools.pending_candidate_retired_index = None
        return int(action), pool_status

    mask = action_pools.mask(pool_index)
    if mask == 0:
        action = agent._deterministic_extreme_from_q(
            q_values, maximum=True, key=tie_key
        )
        action_pools.mark_pending_retirement(
            pool_index=pool_index,
            episode=episode,
            step=step,
            action=action,
        )
        return int(action), "empty_mask_argmax_pending_retirement"

    action = best_available_action(
        q_values=q_values,
        mask=mask,
        action_count=agent.action_count,
        key=tie_key,
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
            pool_index=pool_index,
            episode=episode,
            step=step,
            action=action,
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
) -> Dict:
    event = selected_rmst_event(parsed, args.rmst_event)
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
    }


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
    if not hasattr(train_env.action_space, "n"):
        train_env.close()
        raise RuntimeError("MetaDrive action space is not Discrete; check discrete_action config.")
    action_count = int(train_env.action_space.n)
    agent = DQNAgent(observation_size, action_count, args, device)
    if action_count != 9:
        train_env.close()
        raise RuntimeError(
            f"Karthikeya2 expects exactly 9 discrete actions; environment has {action_count}."
        )
    action_pools = SimilarStateActionPools(
        max_pools=args.max_state_pools,
        max_candidates=args.max_state_candidates,
        candidate_hard_limit=args.candidate_hard_limit,
        candidate_batch_evict_count=args.candidate_batch_evict_count,
        candidate_promotion_visits=args.candidate_promotion_visits,
        action_count=action_count,
        observation_size=observation_size,
        similarity_threshold=args.state_similarity_threshold,
        distance_threshold=args.state_distance_threshold,
        candidate_similarity_threshold=args.candidate_similarity_threshold,
        candidate_distance_threshold=args.candidate_distance_threshold,
        duplicate_similarity_threshold=args.duplicate_similarity_threshold,
        duplicate_distance_threshold=args.duplicate_distance_threshold,
        active_tie_similarity_tolerance=(
            args.active_tie_similarity_tolerance
        ),
        active_tie_distance_tolerance=args.active_tie_distance_tolerance,
        auto_calibrate_thresholds=args.auto_calibrate_thresholds,
        calibration_episodes=args.calibration_episodes,
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
    )

    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.perf_counter()
    training_cpu_start = time.process_time()
    try:
        for episode in range(args.train_episodes):
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            action_pools.begin_episode()
            state = flatten_observation(state_raw)
            initial_hash = observation_sha256(state_raw)
            env_reward_total = 0.0
            training_reward_total = 0.0
            losses: List[float] = []
            action_sources: Dict[str, int] = {}
            parsed = parse_step_info({}, False, False)
            episode_start = time.perf_counter()
            cpu_start = time.process_time()
            for step in range(args.max_episode_steps):
                action, source = select_training_action(
                    experiment, agent, state, episode, step, args, action_pools
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


def convergence_episode(
    rewards: Sequence[float], target_reward: float, threshold_fraction: float, window: int
) -> int:
    if not rewards:
        return 0
    window = max(1, min(int(window), len(rewards)))
    rolling = np.convolve(
        np.asarray(rewards, dtype=float), np.ones(window) / window, mode="valid"
    )
    threshold = float(threshold_fraction) * float(target_reward)
    for index, value in enumerate(rolling):
        if float(value) >= threshold:
            return int(index + window)
    return int(len(rewards))


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
        train_rewards = [float(row["env_reward"]) for row in train]
        train_total_rewards = [float(row["training_reward"]) for row in train]
        test_rewards = [float(row["env_reward"]) for row in test]
        test_times = [float(row["event_or_censor_time_steps"]) for row in test]
        test_events = [bool(row["rmst_event_observed"]) for row in test]
        mean_r = float(np.mean(test_rewards))
        median_r = float(np.median(test_rewards))
        iqmr = interquartile_mean(test_rewards)
        rmst = restricted_mean_survival_time(test_times, test_events, args.rmst_tau)
        conv_episode = convergence_episode(
            train_rewards,
            mean_r,
            args.convergence_threshold_fraction,
            args.convergence_window,
        )
        total_train_time = sum(float(row["wall_time_seconds"]) for row in train)
        conv_time = conv_episode / max(args.train_episodes, 1) * total_train_time
        summary.append(
            {
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "learning_rate": args.learning_rate,
                "mean_R": mean_r,
                "median_R": median_r,
                "IQMR": iqmr,
                "RMST_steps": rmst,
                "RMST_tau_steps": args.rmst_tau,
                "RMST_event_definition": args.rmst_event,
                "average_train_env_reward": float(np.mean(train_rewards)),
                "median_train_env_reward": float(np.median(train_rewards)),
                "std_train_env_reward": float(np.std(train_rewards)),
                "average_train_total_reward": float(np.mean(train_total_rewards)),
                "std_test_reward": float(np.std(test_rewards)),
                "mode_test_reward": reward_mode(test_rewards),
                "min_test_reward": float(np.min(test_rewards)),
                "max_test_reward": float(np.max(test_rewards)),
                "q1_test_reward": percentile(test_rewards, 25),
                "q3_test_reward": percentile(test_rewards, 75),
                "convergence_episode": conv_episode,
                "convergence_time_seconds": conv_time,
                "total_training_wall_time_seconds": total_train_time,
                "train_collision_rate": float(np.mean([bool(r["collision"]) for r in train])),
                "test_collision_rate": float(np.mean([bool(r["collision"]) for r in test])),
                "train_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in train])),
                "test_goal_rate": float(np.mean([bool(r["goal_reached"]) for r in test])),
                "test_off_road_rate": float(np.mean([bool(r["out_of_road"]) for r in test])),
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


def make_figures(rows: List[Dict], summary: List[Dict], output_dir: Path) -> None:
    apply_ieee_style()
    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame([row for row in rows if row["phase"] == "train"])
    test_df = pd.DataFrame([row for row in rows if row["phase"] == "test"])
    summary_df = pd.DataFrame(summary).set_index("experiment").loc[EXPERIMENTS]
    labels = [SHORT_LABELS[e] for e in EXPERIMENTS]
    colors = [COLORS[e] for e in EXPERIMENTS]
    x = np.arange(len(EXPERIMENTS))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for experiment in EXPERIMENTS:
        data = train_df[train_df["experiment"] == experiment].sort_values("episode")
        ax.plot(
            data["episode"],
            data["env_reward"].rolling(20, min_periods=1).mean(),
            linewidth=1.5,
            label=SHORT_LABELS[experiment],
            color=COLORS[experiment],
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("20-episode mean environment reward")
    ax.set_title("MetaDrive DQN Training Reward")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_training_reward")

    groups = [
        test_df[test_df["experiment"] == experiment]["env_reward"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    boxplot_kwargs = dict(showmeans=True, patch_artist=True)
    try:
        box = ax.boxplot(groups, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        # Matplotlib < 3.9 uses ``labels`` instead of ``tick_labels``.
        box = ax.boxplot(groups, labels=labels, **boxplot_kwargs)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Frozen-test environment reward")
    ax.set_title("MetaDrive Test Reward Distribution")
    save_figure(fig, figure_dir, "ieee_test_reward_boxplot")

    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    width = 0.25
    for offset, column, label in [(-width, "mean_R", "Mean R"), (0, "median_R", "Median R"), (width, "IQMR", "IQMR")]:
        ax.bar(x + offset, summary_df[column], width=width, label=label, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Frozen-test environment reward")
    ax.set_title("MetaDrive Reward Metrics")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_mean_median_iqmr")

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.bar(x, summary_df["RMST_steps"], color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(float(summary_df["RMST_tau_steps"].iloc[0]), color="black", linestyle="--", linewidth=1, label="Restriction tau")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("RMST (steps)")
    ax.set_title("MetaDrive Event-Free Restricted Mean Survival")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_rmst")

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    width = 0.36
    ax.bar(x - width / 2, 100 * summary_df["test_collision_rate"], width, label="Collision", edgecolor="black")
    ax.bar(x + width / 2, 100 * summary_df["test_goal_rate"], width, label="Goal", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Frozen-test episodes (%)")
    ax.set_title("MetaDrive Collision and Goal Rates")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collision_goal_rates")



def save_pool_statistics(
    action_pools: SimilarStateActionPools,
    args,
    output_dir: Path,
) -> None:
    """Save diagnostics with stable headers even when a table is empty."""
    pool_rows = action_pools.pool_statistics()
    candidate_rows = action_pools.candidate_statistics()
    retired_rows = action_pools.retired_pool_statistics()
    global_row = action_pools.global_statistics()

    pool_columns = [
        "pool_id", "promotion_evidence_visits",
        "absorbed_candidate_visits", "absorbed_candidate_actions",
        "active_pool_mask_visits", "total_observed_visits",
        "matched_state_count", "actions_tried", "remaining_actions",
        "coverage_percent", "first_episode_created",
        "last_episode_visited", "mean_match_similarity",
        "minimum_match_similarity", "maximum_match_similarity",
        "mean_relative_l2_distance", "minimum_relative_l2_distance",
        "maximum_relative_l2_distance", "centroid_updates",
        "centroid_last_relative_shift", "centroid_frozen_by_stability",
        "centroid_frozen_by_cap",
    ]
    candidate_columns = [
        "candidate_id", "visit_count", "first_episode", "last_episode",
        "visits_remaining_for_promotion", "centroid_updates",
        "centroid_last_relative_shift",
        "centroid_consecutive_stable_updates", "centroid_is_frozen",
        "unique_actions_executed", "executed_action_mask",
    ]
    retired_columns = [
        "retired_pool_id", "original_pool_id", "retirement_reason",
        "episode_created", "episode_retired",
        "candidate_evidence_visits", "absorbed_candidate_visits",
        "permanent_pool_visits", "actions_explored",
        "hits_after_retirement", "last_hit_episode", "last_hit_step",
        "mean_retired_hit_similarity",
        "mean_retired_hit_relative_l2_distance",
        "retirement_trigger_action", "retirement_trigger_reward",
        "retirement_trigger_collision",
        "retirement_trigger_out_of_road",
        "retirement_trigger_done", "retirement_trigger_step",
    ]
    evicted_columns = [
        "eviction_episode", "visit_count", "first_episode",
        "last_episode", "unique_actions_executed",
        "executed_action_mask", "near_promotion", "centroid_updates",
    ]
    creation_columns = ["episode", "cumulative_pools_created"]

    pd.DataFrame(pool_rows, columns=pool_columns).to_csv(
        output_dir / "state_pool_statistics.csv", index=False
    )
    pd.DataFrame(retired_rows, columns=retired_columns).to_csv(
        output_dir / "state_retired_pool_statistics.csv", index=False
    )
    pd.DataFrame(candidate_rows, columns=candidate_columns).to_csv(
        output_dir / "state_candidate_statistics.csv", index=False
    )
    pd.DataFrame(
        action_pools.evicted_candidate_rows, columns=evicted_columns
    ).to_csv(
        output_dir / "state_evicted_candidate_history.csv", index=False
    )
    pd.DataFrame([global_row]).to_csv(
        output_dir / "state_pool_global_summary.csv", index=False
    )

    creation_rows = [
        {"episode": int(episode), "cumulative_pools_created": int(count)}
        for episode, count in action_pools.creation_events
    ]
    pd.DataFrame(creation_rows, columns=creation_columns).to_csv(
        output_dir / "state_pool_creation_timeline.csv", index=False
    )

    calibration_row = {
        **action_pools.normalizer.statistics(),
        **action_pools.calibrator.statistics(),
        "matching_uses_raw_observations": (
            action_pools.matching_uses_raw_observations
        ),
        "normalization_episodes": action_pools.normalization_episodes,
        "threshold_calibration_start_episode": (
            action_pools.threshold_calibration_start_episode
        ),
        "state_similarity_threshold": action_pools.similarity_threshold,
        "state_distance_threshold": action_pools.distance_threshold,
        "candidate_similarity_threshold": (
            action_pools.candidate_similarity_threshold
        ),
        "candidate_distance_threshold": (
            action_pools.candidate_distance_threshold
        ),
        "duplicate_similarity_threshold": (
            action_pools.duplicate_similarity_threshold
        ),
        "duplicate_distance_threshold": (
            action_pools.duplicate_distance_threshold
        ),
        "distance_metric": "RMS distance",
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

    if creation_rows:
        timeline = pd.DataFrame(creation_rows)
        creation_limit = int(
            math.ceil(args.train_episodes * args.pool_training_fraction)
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.step(
            timeline["episode"],
            timeline["cumulative_pools_created"],
            where="post",
            linewidth=1.5,
        )
        ax.axvline(
            creation_limit,
            linestyle="--",
            linewidth=1.0,
            label="Pooling disabled",
        )
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Cumulative pools ever created")
        ax.set_title("State Pool Creation During Pooling Phase")
        ax.legend(frameon=False)
        save_figure(fig, figure_dir, "state_pool_creation_timeline")


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
                "method": "karthikeya2",
                "method_label": "Karthikeya2",
                "phase": phase,
                **collision_summary(phase_rows, args.rmst_tau),
            }
        )
    metrics_path = output_dir / "collision_metrics.csv"
    write_csv(metrics_path, metric_rows)

    # Keep the existing model location and also expose the baseline-style name.
    nested_model = output_dir / "models" / "karthikeya2_model.pt"
    root_model = output_dir / "model.pt"
    if nested_model.is_file():
        root_model.write_bytes(nested_model.read_bytes())

    results_path = output_dir / "all_episode_results.csv"
    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": getattr(metadrive, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "seed": args.seed,
        "method": "karthikeya2",
        "method_label": "Karthikeya2",
        "critical_config": critical_config(args),
        "critical_config_sha256": canonical_json_sha256(critical_config(args)),
        "model_sha256": sha256_file(root_model) if root_model.is_file() else "",
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_statistics_sha256": sha256_file(runtime_path),
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
    pd.DataFrame([r for r in rows if r["phase"] == "train"]).to_csv(
        output_dir / "all_experiments_train_episode_rewards.csv", index=False
    )
    pd.DataFrame([r for r in rows if r["phase"] == "test"]).to_csv(
        output_dir / "all_experiments_test_episode_rewards.csv", index=False
    )
    pd.DataFrame(summary).to_csv(
        output_dir / "all_experiments_learning_rate_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "method": row["method"],
                "Mean R": row["mean_R"],
                "Median R": row["median_R"],
                "IQMR": row["IQMR"],
                "RMST": row["RMST_steps"],
            }
            for row in summary
        ]
    ).to_csv(output_dir / "four_primary_test_metrics.csv", index=False)
    pd.DataFrame(runtimes).to_csv(
        output_dir / "all_experiments_runtime_logs.csv", index=False
    )
    (output_dir / "config.json").write_text(
        json.dumps(json_safe(vars(args)), indent=2), encoding="utf-8"
    )
    make_figures(rows, summary, output_dir)
    save_framework_compatibility_outputs(rows, runtimes, args, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaDrive Karthikeya2: bounded similar-state one-pass action coverage"
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument(
        "--max-state-pools",
        type=int,
        default=None,
        help=(
            "Combined active + retired capacity. Default: "
            "min(1000, max(100, ceil(sqrt(train_episodes * max_episode_steps))))."
        ),
    )
    parser.add_argument(
        "--max-state-candidates",
        type=int,
        default=100,
        help="Soft limit for temporary state candidates.",
    )
    parser.add_argument(
        "--candidate-hard-limit",
        type=int,
        default=120,
        help=(
            "Hard candidate limit. Batch eviction occurs when this limit "
            "is reached."
        ),
    )
    parser.add_argument(
        "--candidate-batch-evict-count",
        type=int,
        default=20,
        help=(
            "Number of weakest candidates removed together at the hard limit."
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
        help="Fit normalization, then calibrate thresholds in a fixed normalized space.",
    )
    parser.add_argument(
        "--calibration-episodes",
        type=int,
        default=25,
        help="Total warm-up episodes, split between normalization and threshold calibration.",
    )
    parser.add_argument(
        "--calibration-max-pairs",
        type=int,
        default=20000,
        help="Maximum consecutive normalized-state pairs retained for calibration.",
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
        "--duplicate-similarity-threshold",
        type=float,
        default=0.97,
        help="Stricter cosine threshold for candidate-to-active absorption.",
    )
    parser.add_argument(
        "--duplicate-distance-threshold",
        type=float,
        default=0.08,
        help="Stricter RMS threshold for candidate-to-active absorption.",
    )
    parser.add_argument(
        "--active-tie-similarity-tolerance",
        type=float,
        default=0.005,
        help="Similarity difference within which active wins over retired.",
    )
    parser.add_argument(
        "--active-tie-distance-tolerance",
        type=float,
        default=0.01,
        help="Distance difference within which active wins over retired.",
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
    parser.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument(
        "--rmst-tau",
        type=int,
        default=None,
        help="Restriction horizon in steps; defaults to --max-episode-steps",
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
            "<project>/policy_results/seed_<seed>/karthikeya2."
        ),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    if args.train_episodes <= 0:
        raise ValueError("--train-episodes must be positive.")
    if args.test_episodes <= 0:
        raise ValueError("--test-episodes must be positive.")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive.")
    if args.max_state_pools <= 0:
        raise ValueError("--max-state-pools must be positive.")
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
    if not 0.0 <= args.duplicate_similarity_threshold <= 1.0:
        raise ValueError(
            "--duplicate-similarity-threshold must be between 0 and 1."
        )
    if args.duplicate_distance_threshold < 0.0:
        raise ValueError(
            "--duplicate-distance-threshold must be non-negative."
        )
    if args.active_tie_similarity_tolerance < 0.0:
        raise ValueError(
            "--active-tie-similarity-tolerance must be non-negative."
        )
    if args.active_tie_distance_tolerance < 0.0:
        raise ValueError(
            "--active-tie-distance-tolerance must be non-negative."
        )
    if args.calibration_episodes < 0:
        raise ValueError("--calibration-episodes must be non-negative.")
    if args.calibration_episodes >= args.train_episodes:
        raise ValueError(
            "--calibration-episodes must be less than --train-episodes."
        )
    if args.calibration_max_pairs <= 0:
        raise ValueError("--calibration-max-pairs must be positive.")
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
    if args.rmst_tau > args.max_episode_steps:
        raise ValueError("--rmst-tau cannot exceed --max-episode-steps.")
    train_start, train_end = args.seed, args.seed + args.train_episodes - 1
    test_start, test_end = args.test_seed, args.test_seed + args.test_episodes - 1
    if max(train_start, test_start) <= min(train_end, test_end):
        raise ValueError("Testing seed range must not overlap the training seed range.")
    if args.collision_penalty < 0 or args.out_of_road_penalty < 0:
        raise ValueError("MetaDrive penalty arguments must be non-negative magnitudes.")


def resolve_output_dir(args) -> Path:
    """Return the only permitted seed-specific policy output directory."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    output_dir = (
        project_dir / "policy_results" / f"seed_{args.seed}" / "karthikeya2"
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
                f"Karthikeya2 output directory is not empty: {output_dir}. "
                "Reuse it or pass --force to remove the previous partial/completed run."
            )
        if force:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.rmst_tau is None:
        args.rmst_tau = args.max_episode_steps
    if args.max_state_pools is None:
        args.max_state_pools = min(
            1000,
            max(
                100,
                math.ceil(
                    math.sqrt(
                        args.train_episodes * args.max_episode_steps
                    )
                ),
            ),
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
    print("METADRIVE KARTHIKEYA2 SIMILAR-STATE ONE-PASS ACTION COVERAGE")
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
    print("Policy name: karthikeya2")
    print("Maximum combined active + retired pools:", args.max_state_pools)
    print("Adaptive capacity formula applied:", True)
    print("Auto threshold calibration:", args.auto_calibrate_thresholds)
    print("Calibration episodes:", args.calibration_episodes)
    print("Duplicate cosine threshold:", args.duplicate_similarity_threshold)
    print("Duplicate RMS threshold:", args.duplicate_distance_threshold)
    print("Candidate soft limit:", args.max_state_candidates)
    print("Candidate hard limit:", args.candidate_hard_limit)
    print("Candidate batch eviction count:", args.candidate_batch_evict_count)
    print("Candidate promotion visits:", args.candidate_promotion_visits)
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
    print("Pool matching requires both thresholds: yes")
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
    print("Available actions per new pool: 9")
    print("Candidate visits before promotion use argmax: yes")
    print("Candidate actions are recorded for mask initialization: yes")
    print("Promotion occurs before promotion-visit action selection: yes")
    print("Inside permanent pool: highest-Q action among available actions")
    print("Removal: clear selected bit in O(1)")
    print("Final mask action retirement: after observing transition")
    print("Combined permanent capacity full: reject promotion and use maximum-Q")
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
    required_outputs = (
        output_dir / "all_episode_results.csv",
        output_dir / "config.json",
        output_dir / "runtime_statistics.csv",
        output_dir / "collision_metrics.csv",
        output_dir / "manifest.json",
        output_dir / "model.pt",
    )
    missing_outputs = [path for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            "Experiment finished but required outputs are missing: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    print("\nExperiment completed successfully.")
    print("Episode results:", output_dir / "all_episode_results.csv")
    print("Primary metrics:", output_dir / "four_primary_test_metrics.csv")
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
        "Retired pool statistics:",
        output_dir / "state_retired_pool_statistics.csv",
    )
    print("Runtime statistics:", output_dir / "runtime_statistics.csv")
    print("Collision metrics:", output_dir / "collision_metrics.csv")
    print("Manifest:", output_dir / "manifest.json")
    print("Pool occupancy plot:", output_dir / "plots" / "state_pool_occupancy.png")
    print("Policy folder name: karthikeya2")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
