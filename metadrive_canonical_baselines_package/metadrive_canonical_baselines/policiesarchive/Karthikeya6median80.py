#!/usr/bin/env python3
"""MetaDrive DQN with four safety-risk action pools.

Author: Sai Durga Karthik Nandiraju
Design revision: four retired risk pools using only MetaDrive rewards, 2026-07-17

Training policy
---------------
* The complete flattened observation is always supplied to the DQN.
* A separate pre-action safety vector produces one scalar risk coordinate.
* Four fixed risk intervals are classified: low, medium, high, and critical.
* Every risk interval can store actions in a lazily created pool.
* There are no candidates, promotions, evictions, similarity searches, or
  adaptive capacity changes.
* During the first 80% of training episodes:
    - a newly created pool uniformly samples one remaining action;
    - every active pool uniformly samples one remaining action and removes it;
    - an empty mask retires only after the final action transition is learned;
    - a retired pool uniformly samples an action whose stored value is greater
      than or equal to the pool's 80th-percentile (P80) action value;
    - ties at P80 are eligible; a maximum-value fallback guarantees a valid
      action if numerical edge cases leave the eligible set empty;
    - every pool action value stores only the latest exact MetaDrive
      environment reward observed for that action;
    - pool rewards are never normalized, rescaled, clipped, or supplemented
      with additional collision/off-road deductions;
    - the DQN replay buffer and network are updated for every transition.
* During the final 20% of training, every action is DQN argmax.
* Testing is frozen DQN argmax only. No pool action is used during testing.

The pool safety gate combines collision and off-road indicators. Collision
risk uses nearest hazard surface distance and estimated time-to-collision.
Off-road risk uses lane-boundary clearance, normalized lane offset, heading
error, and one-second projected lane-boundary clearance.

Complexity
----------
* Risk classification and active/retired lookup are O(1).
* Mask clearing, empty checks, pool creation, and retirement are O(1).
* Random remaining-action selection scans exactly nine actions, so it is O(1).
* Retired-pool P80 and upper-percentile selection inspect exactly nine stored
  action values, so they are O(1).
* Pool memory is four pools x nine actions, so it is O(1).
* Safety extraction is O(H), where H is the number of MetaDrive objects
  inspected at the current step; no quadratic matching is used.
* Replay sampling is O(batch_size); DQN computation follows the fixed network
  and batch dimensions.

Example
-------
python -u policies/Karthikeya6Median80_80_20.py \
  --seed 3 --test-seed 100000 \
  --train-episodes 500 --test-episodes 300 \
  --max-episode-steps 500 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


METHOD = "Karthikeya6Median80_80_20"
METHOD_LABEL = "Karthikeya6Median80_80_20"
RISK_LEVELS = ("low", "medium", "high", "critical")
RISK_COLORS = {
    "low": "#2ca02c",
    "medium": "#ffbf00",
    "high": "#ff7f0e",
    "critical": "#d62728",
}
SAFETY_VECTOR_NAMES = (
    "lane_boundary_clearance",
    "nearest_collision_hazard_surface_distance",
    "estimated_time_to_collision",
    "absolute_lane_offset",
    "normalized_lane_offset",
    "absolute_heading_error",
    "projected_lane_boundary_clearance_1s",
    "ego_speed_km_h",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
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


def observation_sha256(observation) -> str:
    array = np.ascontiguousarray(flatten_observation(observation))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else math.nan


def deterministic_argmax(values: np.ndarray, key: str) -> int:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise RuntimeError("No finite action value exists for argmax selection.")
    comparable = np.where(finite, values, -np.inf)
    maximum = float(np.max(comparable))
    tied_actions = np.flatnonzero(comparable == maximum)
    if tied_actions.size == 0:
        raise RuntimeError("No argmax action exists.")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int(
        tied_actions[int.from_bytes(digest[:8], "big") % tied_actions.size]
    )


def restricted_mean_survival_time(
    times: Sequence[int], events: Sequence[bool], tau: int
) -> float:
    if not times:
        return math.nan
    pairs = sorted(
        (min(float(time_value), float(tau)), bool(event))
        for time_value, event in zip(times, events)
    )
    unique_event_times = sorted({t for t, event in pairs if event and t <= tau})
    survival = 1.0
    previous = 0.0
    area = 0.0
    for current in unique_event_times:
        area += survival * max(0.0, current - previous)
        at_risk = sum(t >= current for t, _ in pairs)
        failures = sum(t == current and event for t, event in pairs)
        if at_risk:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, tau - previous)
    return float(area)


# ---------------------------------------------------------------------------
# Safety extraction and risk coordinate
# ---------------------------------------------------------------------------


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


def _vector2(value) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(2, dtype=np.float32)
    if vector.size < 2:
        vector = np.pad(vector, (0, 2 - vector.size))
    return vector[:2]


def _is_collision_hazard(obj) -> bool:
    class_name = obj.__class__.__name__.lower()
    module_name = obj.__class__.__module__.lower()
    text = f"{module_name}.{class_name}"
    if any(
        token in text
        for token in (
            "navigation", "camera", "sensor", "engine", "manager",
            "policy", "renderer", "nodepath",
        )
    ):
        return False
    if any(
        token in text
        for token in (
            "vehicle", "pedestrian", "human", "cyclist", "bicycle",
            "cone", "barrier", "obstacle", "trafficobject", "traffic_object",
            "building", "sidewalk",
        )
    ):
        return True
    if any(token in text for token in ("lane", "road", "map", "light", "marking", "terrain")):
        return False
    has_position = getattr(obj, "position", None) is not None
    has_geometry = any(
        hasattr(obj, attr)
        for attr in (
            "collision_node", "collision_nodes", "body", "chassis",
            "top_down_width", "top_down_length", "WIDTH", "LENGTH",
        )
    )
    has_motion = any(
        hasattr(obj, attr)
        for attr in ("velocity", "speed", "speed_km_h", "heading_theta")
    )
    return bool(has_position and (has_geometry or has_motion))


def _object_radius(obj, default: float = 0.75) -> float:
    width = _finite_float(
        getattr(obj, "top_down_width", getattr(obj, "WIDTH", 0.0)), 0.0
    )
    length = _finite_float(
        getattr(obj, "top_down_length", getattr(obj, "LENGTH", 0.0)), 0.0
    )
    if width <= 0.0 and length <= 0.0:
        return float(default)
    return float(max(default, 0.5 * math.hypot(width, length)))


def _hazard_metrics(env, ego_position: np.ndarray, ego_velocity: np.ndarray, args):
    distance_cap = float(args.safety_nearest_object_cap)
    ttc_cap = float(args.safety_ttc_cap)
    engine = getattr(env, "engine", None)
    if engine is None:
        return distance_cap, ttc_cap
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
    iterable = objects.values() if isinstance(objects, dict) else (objects or ())
    ego = getattr(env, "vehicle", None)
    ego_radius = _object_radius(ego, 1.0) if ego is not None else 1.0
    best_surface = distance_cap
    best_ttc = ttc_cap
    for obj in iterable:
        if obj is ego or not _is_collision_hazard(obj):
            continue
        position = getattr(obj, "position", None)
        if position is None:
            continue
        relative_position = _vector2(position) - ego_position[:2]
        center_distance = float(np.linalg.norm(relative_position))
        if center_distance <= 1e-6:
            continue
        surface_distance = max(
            0.0,
            center_distance - ego_radius - _object_radius(obj),
        )
        best_surface = min(best_surface, surface_distance)
        direction = relative_position / center_distance
        object_velocity = _vector2(getattr(obj, "velocity", (0.0, 0.0)))
        relative_velocity = object_velocity - ego_velocity[:2]
        closing_speed = -float(np.dot(relative_velocity, direction))
        if closing_speed > 0.1:
            best_ttc = min(best_ttc, surface_distance / closing_speed)
    return float(min(best_surface, distance_cap)), float(min(best_ttc, ttc_cap))


def extract_safety_vector(env, args) -> np.ndarray:
    """Extract collision and off-road measures before taking an action."""
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                math.pi,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    position = _vector2(getattr(vehicle, "position", (0.0, 0.0)))
    velocity = _vector2(getattr(vehicle, "velocity", (0.0, 0.0)))
    speed_km_h_value = getattr(vehicle, "speed_km_h", None)
    if speed_km_h_value is not None:
        speed_km_h = _finite_float(speed_km_h_value, 0.0)
    else:
        fallback = _finite_float(getattr(vehicle, "speed", 0.0), 0.0)
        speed_km_h = fallback * 3.6 if args.safety_speed_fallback_unit == "mps" else fallback
    speed_km_h = float(np.clip(abs(speed_km_h), 0.0, args.safety_speed_cap))
    speed_mps = speed_km_h / 3.6

    lane = _vehicle_lane(vehicle)
    lane_offset = 0.0
    lane_clearance = 0.0 if lane is None else float(args.safety_lane_boundary_cap)
    normalized_offset = 1.0 if lane is None else 0.0
    heading_error = 0.0
    if lane is not None:
        longitudinal = 0.0
        local_coordinates = getattr(lane, "local_coordinates", None)
        if callable(local_coordinates):
            try:
                longitudinal, lateral = local_coordinates(position)
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
            half_width = max(lane_width / 2.0, 1e-6)
            lane_clearance = min(
                float(args.safety_lane_boundary_cap),
                max(0.0, half_width - lane_offset),
            )
            normalized_offset = float(np.clip(lane_offset / half_width, 0.0, 2.0))
        vehicle_heading = _finite_float(getattr(vehicle, "heading_theta", 0.0), 0.0)
        heading_at = getattr(lane, "heading_theta_at", None)
        if callable(heading_at):
            try:
                lane_heading = _finite_float(heading_at(longitudinal), vehicle_heading)
                heading_error = _angle_difference_radians(vehicle_heading, lane_heading)
            except Exception:
                heading_error = 0.0

    lateral_speed = abs(speed_mps * math.sin(heading_error))
    projected_clearance = max(
        0.0,
        lane_clearance - lateral_speed * float(args.safety_projection_seconds),
    )
    hazard_distance, ttc = _hazard_metrics(env, position, velocity, args)
    return np.asarray(
        [
            lane_clearance,
            hazard_distance,
            ttc,
            lane_offset,
            normalized_offset,
            heading_error,
            projected_clearance,
            speed_km_h,
        ],
        dtype=np.float32,
    )


def safety_risk_components(safety: np.ndarray, args) -> Tuple[float, float, float]:
    """Return collision, off-road, and combined risk in [0, 1]."""
    (
        lane_clearance,
        hazard_distance,
        ttc,
        _lane_offset,
        normalized_offset,
        heading_error,
        projected_clearance,
        speed_km_h,
    ) = np.asarray(safety, dtype=float)
    if hazard_distance >= 0.999 * args.safety_nearest_object_cap:
        distance_risk = 0.0
    else:
        distance_risk = 1.0 - np.clip(
            hazard_distance / args.collision_safe_distance, 0.0, 1.0
        )
    if ttc >= 0.999 * args.safety_ttc_cap:
        ttc_risk = 0.0
    else:
        ttc_risk = 1.0 - np.clip(ttc / args.collision_safe_ttc, 0.0, 1.0)
    # Speed may increase collision urgency, but it must never reduce a risk
    # already indicated by distance or time-to-collision.
    speed_factor = 1.0 + 0.25 * np.clip(
        speed_km_h / args.safety_speed_cap, 0.0, 1.0
    )
    collision_risk = float(np.clip(max(distance_risk, ttc_risk) * speed_factor, 0.0, 1.0))

    clearance_risk = 1.0 - np.clip(
        lane_clearance / args.offroad_safe_clearance, 0.0, 1.0
    )
    projected_risk = 1.0 - np.clip(
        projected_clearance / args.offroad_safe_clearance, 0.0, 1.0
    )
    offset_risk = np.clip(normalized_offset, 0.0, 1.0)
    heading_risk = np.clip(
        heading_error / args.offroad_safe_heading_error, 0.0, 1.0
    )
    offroad_risk = float(
        np.clip(max(clearance_risk, projected_risk, offset_risk, heading_risk), 0.0, 1.0)
    )
    return collision_risk, offroad_risk, float(max(collision_risk, offroad_risk))


# ---------------------------------------------------------------------------
# Environment and termination parsing
# ---------------------------------------------------------------------------


def make_env(args, phase: str) -> MetaDriveEnv:
    if phase not in {"train", "test"}:
        raise ValueError("phase must be train or test")
    start_seed = args.seed if phase == "train" else args.test_seed
    count = args.train_episodes if phase == "train" else args.test_episodes
    return MetaDriveEnv(
        {
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
            "num_scenarios": int(max(1, count)),
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
    )


def truthy(info: Dict, *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


def parse_step_info(info: Dict, terminated: bool, truncated: bool) -> Dict:
    crash_vehicle = truthy(info, "crash_vehicle")
    crash_object = truthy(
        info,
        "crash_object", "crash_building", "crash_human", "crash_sidewalk",
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
        "collision": bool(collision),
        "crash_vehicle": bool(crash_vehicle),
        "crash_object": bool(crash_object),
        "out_of_road": bool(out_of_road),
        "goal_reached": bool(goal_reached),
        "max_steps_reached": bool(max_steps),
        "step_cost": 1.0,
    }


def selected_rmst_event(parsed: Dict, definition: str) -> bool:
    if definition == "collision":
        return bool(parsed["collision"])
    return bool(parsed["collision"] or parsed["out_of_road"])


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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.storage: List[Optional[Transition]] = [None] * self.capacity
        self.size = 0
        self.position = 0
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        self.storage[self.position] = Transition(
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            bool(done),
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = self.rng.sample(range(self.size), batch_size)
        batch = [self.storage[index] for index in indices]
        if any(item is None for item in batch):
            raise RuntimeError("Replay buffer contained an uninitialized transition.")
        return [item for item in batch if item is not None]

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    def __init__(self, observation_size: int, action_count: int, args, device):
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
        return torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        was_training = self.online.training
        self.online.eval()
        with torch.no_grad():
            values = self.online(self.tensor(state))[0].detach().cpu().numpy()
        if was_training:
            self.online.train()
        return values.astype(float)

    def greedy_action(self, state: np.ndarray, key: str) -> int:
        return deterministic_argmax(self.q_values(state), key)

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
        predicted = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_values = self.target(next_states).max(dim=1, keepdim=True).values
            targets = rewards + (1.0 - dones) * self.gamma * next_values
        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def freeze(self) -> None:
        for network in (self.online, self.target):
            network.eval()
            for parameter in network.parameters():
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
# Four fixed safety-risk pools
# ---------------------------------------------------------------------------


@dataclass
class RiskPool:
    pool_id: int
    risk_level: str
    lower_bound_exclusive: float
    upper_bound_inclusive: float
    action_count: int
    episode_created: int
    mask: int = field(init=False)
    episode_retired: int = -1
    state_visits: int = 0
    active_random_actions: int = 0
    retired_percentile_random_actions: int = 0
    retired_empty_percentile_fallbacks: int = 0
    last_retired_percentile_value: float = math.nan
    last_episode_visited: int = -1
    last_step_visited: int = -1
    risk_score_sum: float = 0.0
    collision_risk_sum: float = 0.0
    offroad_risk_sum: float = 0.0
    action_selection_counts: np.ndarray = field(init=False)
    action_update_counts: np.ndarray = field(init=False)
    action_values: np.ndarray = field(init=False)
    collision_counts: np.ndarray = field(init=False)
    out_of_road_counts: np.ndarray = field(init=False)
    stored_argmax_action: int = -1

    def __post_init__(self):
        self.mask = (1 << self.action_count) - 1
        self.action_selection_counts = np.zeros(self.action_count, dtype=np.int64)
        self.action_update_counts = np.zeros(self.action_count, dtype=np.int64)
        self.action_values = np.full(self.action_count, -np.inf, dtype=np.float64)
        self.collision_counts = np.zeros(self.action_count, dtype=np.int64)
        self.out_of_road_counts = np.zeros(self.action_count, dtype=np.int64)

    @property
    def retired(self) -> bool:
        return self.episode_retired >= 0

    @property
    def remaining_actions(self) -> int:
        return int(self.mask.bit_count())

    @property
    def actions_tried(self) -> int:
        return self.action_count - self.remaining_actions

    @property
    def risk_centroid(self) -> float:
        return self.risk_score_sum / self.state_visits if self.state_visits else math.nan


class FourRiskActionPools:
    """Four risk intervals with at most four stored action pools."""

    def __init__(self, action_count: int, args):
        if action_count != 9:
            raise ValueError("FourRiskActionPools requires exactly nine actions.")
        self.action_count = int(action_count)
        self.args = args
        self.thresholds = (
            float(args.risk_low_max),
            float(args.risk_medium_max),
            float(args.risk_high_max),
        )
        self.active: Dict[str, RiskPool] = {}
        self.retired: Dict[str, RiskPool] = {}
        self.rng = random.Random(int(args.seed) + 40_009)
        self.creation_events: List[Dict] = []
        self.total_states_gated = 0
        self.total_active_random_actions = 0
        self.total_retired_percentile_random_actions = 0
        self.total_retired_empty_percentile_fallbacks = 0
        self.total_new_pool_random_actions = 0
        self.total_final_phase_argmax_actions = 0

    def classify(self, safety: np.ndarray) -> Tuple[str, float, float, float]:
        collision_risk, offroad_risk, risk_score = safety_risk_components(
            safety, self.args
        )
        low, medium, high = self.thresholds
        if risk_score <= low:
            level = "low"
        elif risk_score <= medium:
            level = "medium"
        elif risk_score <= high:
            level = "high"
        else:
            level = "critical"
        return level, risk_score, collision_risk, offroad_risk

    def _bounds(self, level: str) -> Tuple[float, float]:
        low, medium, high = self.thresholds
        return {
            "low": (-math.inf, low),
            "medium": (low, medium),
            "high": (medium, high),
            "critical": (high, math.inf),
        }[level]

    def create_pool(self, level: str, episode: int) -> RiskPool:
        if level not in RISK_LEVELS:
            raise ValueError(f"Unknown risk level: {level}.")
        if level in self.active or level in self.retired:
            raise RuntimeError(f"Risk pool {level} already exists.")
        lower, upper = self._bounds(level)
        pool = RiskPool(
            pool_id=RISK_LEVELS.index(level),
            risk_level=level,
            lower_bound_exclusive=lower,
            upper_bound_inclusive=upper,
            action_count=self.action_count,
            episode_created=int(episode),
        )
        self.active[level] = pool
        self.creation_events.append(
            {
                "episode": int(episode),
                "risk_level": level,
                "pool_id": pool.pool_id,
                "cumulative_pools_created": len(self.active) + len(self.retired),
            }
        )
        return pool

    def available_actions(self, pool: RiskPool) -> List[int]:
        return [
            action
            for action in range(self.action_count)
            if pool.mask & (1 << action)
        ]

    def observe_state(
        self,
        pool: RiskPool,
        risk_score: float,
        collision_risk: float,
        offroad_risk: float,
        episode: int,
        step: int,
    ) -> None:
        pool.state_visits += 1
        pool.risk_score_sum += float(risk_score)
        pool.collision_risk_sum += float(collision_risk)
        pool.offroad_risk_sum += float(offroad_risk)
        pool.last_episode_visited = int(episode)
        pool.last_step_visited = int(step)

    def choose_active_random(self, pool: RiskPool) -> int:
        available = self.available_actions(pool)
        if not available:
            raise RuntimeError("Active risk pool has an empty mask.")
        action = int(self.rng.choice(available))
        pool.mask &= ~(1 << action)
        pool.active_random_actions += 1
        self.total_active_random_actions += 1
        return action

    def choose_new_pool_random(self, pool: RiskPool) -> int:
        action = self.choose_active_random(pool)
        self.total_new_pool_random_actions += 1
        return int(action)

    def choose_retired_percentile_random(self, pool: RiskPool) -> int:
        """Uniformly sample a stored action at or above the configured percentile."""
        if not pool.retired:
            raise RuntimeError("P80 selection requires a retired pool.")
        values = np.asarray(pool.action_values, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            raise RuntimeError("Retired pool has no finite environment-reward values.")
        percentile_value = float(
            np.percentile(values[finite], float(self.args.retired_value_percentile))
        )
        pool.last_retired_percentile_value = percentile_value
        eligible = np.flatnonzero(finite & (values >= percentile_value))
        if eligible.size == 0:
            maximum_value = float(np.max(values[finite]))
            eligible = np.flatnonzero(finite & (values == maximum_value))
            pool.retired_empty_percentile_fallbacks += 1
            self.total_retired_empty_percentile_fallbacks += 1
        action = int(self.rng.choice(eligible.tolist()))
        pool.retired_percentile_random_actions += 1
        self.total_retired_percentile_random_actions += 1
        return action

    def record_selection(self, pool: RiskPool, action: int) -> None:
        pool.action_selection_counts[int(action)] += 1

    def update_after_transition(
        self,
        level: Optional[str],
        action: int,
        reward: float,
        parsed: Dict,
        episode: int,
    ) -> None:
        if level is None:
            return
        if level not in RISK_LEVELS:
            raise ValueError(f"Unknown risk level: {level}.")
        pool = self.retired.get(level) or self.active.get(level)
        if pool is None:
            raise RuntimeError(f"Selected risk pool {level} disappeared.")
        environment_reward = float(reward)
        if not math.isfinite(environment_reward):
            raise ValueError("MetaDrive returned a non-finite environment reward.")
        action = int(action)
        pool.action_values[action] = environment_reward
        pool.action_update_counts[action] += 1
        pool.collision_counts[action] += int(bool(parsed["collision"]))
        pool.out_of_road_counts[action] += int(bool(parsed["out_of_road"]))
        finite = np.isfinite(pool.action_values)
        if finite.any():
            pool.stored_argmax_action = deterministic_argmax(
                pool.action_values,
                f"pool|{level}|{episode}|{pool.action_update_counts.sum()}",
            )
        if level in self.active and pool.mask == 0:
            pool.episode_retired = int(episode)
            self.retired[level] = self.active.pop(level)

    def all_pools(self) -> List[RiskPool]:
        pools = list(self.active.values()) + list(self.retired.values())
        return sorted(pools, key=lambda pool: pool.pool_id)

    def pool_statistics(self) -> List[Dict]:
        rows = []
        for pool in self.all_pools():
            visits = max(pool.state_visits, 1)
            row = {
                "pool_id": pool.pool_id,
                "risk_level": pool.risk_level,
                "status": "retired" if pool.retired else "active",
                "lower_bound_exclusive": pool.lower_bound_exclusive,
                "upper_bound_inclusive": pool.upper_bound_inclusive,
                "risk_score_centroid": pool.risk_centroid,
                "mean_collision_risk": pool.collision_risk_sum / visits,
                "mean_offroad_risk": pool.offroad_risk_sum / visits,
                "state_visits": pool.state_visits,
                "active_random_actions": pool.active_random_actions,
                "retired_percentile_random_actions": pool.retired_percentile_random_actions,
                "retired_empty_percentile_fallbacks": pool.retired_empty_percentile_fallbacks,
                "last_retired_percentile_value": pool.last_retired_percentile_value,
                "actions_tried": pool.actions_tried,
                "remaining_actions": pool.remaining_actions,
                "coverage_percent": 100.0 * pool.actions_tried / pool.action_count,
                "episode_created": pool.episode_created,
                "episode_retired": pool.episode_retired,
                "last_episode_visited": pool.last_episode_visited,
                "last_step_visited": pool.last_step_visited,
                "stored_argmax_action": pool.stored_argmax_action,
            }
            for action in range(self.action_count):
                row[f"action_{action}_selections"] = int(pool.action_selection_counts[action])
                row[f"action_{action}_updates"] = int(pool.action_update_counts[action])
                value = pool.action_values[action]
                row[f"action_{action}_pool_value"] = float(value) if math.isfinite(value) else math.nan
                row[f"action_{action}_latest_environment_reward"] = (
                    float(value) if math.isfinite(value) else math.nan
                )
                row[f"action_{action}_collisions"] = int(pool.collision_counts[action])
                row[f"action_{action}_out_of_road"] = int(pool.out_of_road_counts[action])
            rows.append(row)
        return rows

    def retired_pool_statistics(self) -> List[Dict]:
        return [row for row in self.pool_statistics() if row["status"] == "retired"]

    def global_statistics(self) -> Dict:
        return {
            "risk_intervals": 4,
            "maximum_total_pools": len(RISK_LEVELS),
            "pooled_risk_levels": ",".join(RISK_LEVELS),
            "pools_created": len(self.active) + len(self.retired),
            "active_pools": len(self.active),
            "retired_pools": len(self.retired),
            "candidate_process_present": False,
            "promotion_process_present": False,
            "eviction_process_present": False,
            "total_states_gated": self.total_states_gated,
            "new_pool_random_actions": self.total_new_pool_random_actions,
            "active_pool_random_actions": self.total_active_random_actions,
            "retired_percentile_random_actions": self.total_retired_percentile_random_actions,
            "retired_empty_percentile_fallbacks": self.total_retired_empty_percentile_fallbacks,
            "final_phase_argmax_actions": self.total_final_phase_argmax_actions,
            "pool_training_fraction": float(self.args.pool_training_fraction),
            "low_risk_max": self.thresholds[0],
            "medium_risk_max": self.thresholds[1],
            "high_risk_max": self.thresholds[2],
            "pool_reward_source": "unmodified_metadrive_environment_reward",
            "pool_value_update": "latest_environment_reward_overwrite",
            "retired_value_percentile": float(self.args.retired_value_percentile),
            "retired_selection_rule": "uniform_random_greater_than_or_equal_to_configured_percentile",
            "pool_reward_normalization_present": False,
            "pool_reward_averaging_present": False,
            "pool_additional_safety_penalties_present": False,
        }


def select_training_action(
    agent: DQNAgent,
    state: np.ndarray,
    safety: np.ndarray,
    episode: int,
    step: int,
    args,
    pools: FourRiskActionPools,
) -> Tuple[int, str, Optional[str]]:
    key = f"train|{args.seed}|{episode}|{step}"
    pool_episode_limit = int(math.ceil(args.train_episodes * args.pool_training_fraction))
    if episode >= pool_episode_limit:
        pools.total_final_phase_argmax_actions += 1
        return agent.greedy_action(state, key), "final_phase_argmax", None

    level, risk_score, collision_risk, offroad_risk = pools.classify(safety)
    pools.total_states_gated += 1
    if level in pools.active:
        pool = pools.active[level]
        pools.observe_state(
            pool, risk_score, collision_risk, offroad_risk, episode, step
        )
        action = pools.choose_active_random(pool)
        pools.record_selection(pool, action)
        return action, f"active_{level}_pool_random", level

    if level in pools.retired:
        pool = pools.retired[level]
        pools.observe_state(
            pool, risk_score, collision_risk, offroad_risk, episode, step
        )
        action = pools.choose_retired_percentile_random(pool)
        pools.record_selection(pool, action)
        return action, f"retired_{level}_pool_percentile_random", level

    pool = pools.create_pool(level, episode)
    pools.observe_state(pool, risk_score, collision_risk, offroad_risk, episode, step)
    action = pools.choose_new_pool_random(pool)
    pools.record_selection(pool, action)
    return action, f"new_{level}_pool_random", level


# ---------------------------------------------------------------------------
# Training, frozen testing, and result rows
# ---------------------------------------------------------------------------


def verify_discrete_action_space(env, args) -> int:
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        raise RuntimeError("MetaDrive action space is not Discrete.")
    count = int(action_space.n)
    expected = int(args.discrete_steering_dim * args.discrete_throttle_dim)
    if count != expected or count != 9:
        raise RuntimeError(
            f"The policy requires exactly 9 actions; MetaDrive exposes {count}."
        )
    invalid = [action for action in range(count) if not action_space.contains(action)]
    if invalid or action_space.contains(count) or action_space.contains(-1):
        raise RuntimeError("Discrete action IDs must be contiguous from 0 through 8.")
    return count


def episode_row(
    phase: str,
    episode: int,
    scenario_seed: int,
    initial_hash: str,
    reward_total: float,
    steps: int,
    parsed: Dict,
    args,
    episode_start: float,
    cpu_start: float,
    agent: DQNAgent,
    losses: Sequence[float],
    action_sources: Dict[str, int],
    safety_sums: np.ndarray,
    safety_minimums: np.ndarray,
    safety_maximums: np.ndarray,
    safety_count: int,
) -> Dict:
    if safety_count:
        means = safety_sums / safety_count
        minimums = safety_minimums
        maximums = safety_maximums
    else:
        means = minimums = maximums = np.full(len(SAFETY_VECTOR_NAMES), math.nan)
    event = selected_rmst_event(parsed, args.rmst_event)
    return {
        "phase": phase,
        "experiment": METHOD,
        "method": METHOD_LABEL,
        "seed": args.seed,
        "episode": episode,
        "scenario_seed": scenario_seed,
        "initial_observation_sha256": initial_hash,
        "env_reward": float(reward_total),
        "training_reward": float(reward_total),
        "steps": int(steps),
        **parsed,
        "rmst_event_definition": args.rmst_event,
        "rmst_event_observed": bool(event),
        "event_or_censor_time_steps": int(steps),
        "wall_time_seconds": float(time.perf_counter() - episode_start),
        "cpu_time_seconds": float(time.process_time() - cpu_start),
        "average_loss": avg(losses),
        "replay_buffer_size": len(agent.replay),
        "learn_steps": agent.learn_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "network_frozen": phase == "test",
        "updates_during_test": 0 if phase == "test" else "",
        "action_source_counts": json.dumps(action_sources, sort_keys=True),
        "mean_lane_boundary_clearance": float(means[0]),
        "minimum_lane_boundary_clearance": float(minimums[0]),
        "mean_nearest_collision_hazard_surface_distance": float(means[1]),
        "minimum_nearest_collision_hazard_surface_distance": float(minimums[1]),
        "mean_estimated_time_to_collision": float(means[2]),
        "minimum_estimated_time_to_collision": float(minimums[2]),
        "mean_absolute_lane_offset": float(means[3]),
        "maximum_absolute_lane_offset": float(maximums[3]),
        "mean_normalized_lane_offset": float(means[4]),
        "maximum_normalized_lane_offset": float(maximums[4]),
        "mean_absolute_heading_error": float(means[5]),
        "maximum_absolute_heading_error": float(maximums[5]),
        "mean_projected_lane_boundary_clearance_1s": float(means[6]),
        "minimum_projected_lane_boundary_clearance_1s": float(minimums[6]),
        "mean_ego_speed": float(means[7]),
        "maximum_ego_speed": float(maximums[7]),
    }


def run_experiment(args, device: torch.device, output_dir: Path):
    set_seed(args.seed, args.deterministic)
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
    pools = FourRiskActionPools(action_count, args)

    print(f"\n===== TRAINING START: {METHOD_LABEL} =====", flush=True)
    training_start = time.perf_counter()
    training_cpu_start = time.process_time()
    try:
        for episode in range(args.train_episodes):
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            state = flatten_observation(state_raw)
            initial_hash = observation_sha256(state_raw)
            reward_total = 0.0
            losses: List[float] = []
            sources: Dict[str, int] = {}
            safety_sums = np.zeros(len(SAFETY_VECTOR_NAMES), dtype=np.float64)
            safety_minimums = np.full(len(SAFETY_VECTOR_NAMES), np.inf)
            safety_maximums = np.full(len(SAFETY_VECTOR_NAMES), -np.inf)
            safety_count = 0
            parsed = parse_step_info({}, False, False)
            episode_start = time.perf_counter()
            cpu_start = time.process_time()
            for step in range(args.max_episode_steps):
                safety = extract_safety_vector(train_env, args)
                safety_sums += safety
                safety_minimums = np.minimum(safety_minimums, safety)
                safety_maximums = np.maximum(safety_maximums, safety)
                safety_count += 1
                action, source, level = select_training_action(
                    agent, state, safety, episode, step, args, pools
                )
                sources[source] = sources.get(source, 0) + 1
                next_raw, reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                done = bool(terminated or truncated)
                parsed = parse_step_info(info, bool(terminated), bool(truncated))

                # DQN learning always receives MetaDrive's unmodified reward.
                environment_reward = float(reward)
                agent.replay.add(
                    state, action, environment_reward, next_state, done
                )
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

                # The chosen action's pool value is overwritten with only the
                # latest exact MetaDrive environment reward. No averaging,
                # normalization, pool penalty, or DQN Q-value enters it.
                if level is not None:
                    pools.update_after_transition(
                        level,
                        action,
                        environment_reward,
                        parsed,
                        episode,
                    )
                reward_total += environment_reward
                state = next_state
                if done:
                    break
            row = episode_row(
                "train", episode, scenario_seed, initial_hash, reward_total,
                step + 1, parsed, args, episode_start, cpu_start, agent, losses,
                sources, safety_sums, safety_minimums, safety_maximums, safety_count,
            )
            rows.append(row)
            print(
                f"TRAIN {METHOD_LABEL:27s} ep={episode:03d} "
                f"reward={reward_total:9.3f} steps={step + 1:3d} "
                f"term={parsed['termination_reason']:11s} "
                f"collision={str(parsed['collision']):5s} "
                f"wall={row['wall_time_seconds']:.2f}s loss={row['average_loss']:.6f}",
                flush=True,
            )
    finally:
        train_env.close()

    training_wall = time.perf_counter() - training_start
    training_cpu = time.process_time() - training_cpu_start
    nested_model = output_dir / "models" / f"{METHOD}_model.pt"
    agent.save(nested_model, args)
    shutil.copyfile(nested_model, output_dir / "model.pt")
    agent.freeze()
    print(f"===== TRAINING END: {training_wall:.2f}s =====", flush=True)

    test_env = make_env(args, "test")
    print(f"\n===== FROZEN TESTING START: {METHOD_LABEL} =====", flush=True)
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
                safety_sums = np.zeros(len(SAFETY_VECTOR_NAMES), dtype=np.float64)
                safety_minimums = np.full(len(SAFETY_VECTOR_NAMES), np.inf)
                safety_maximums = np.full(len(SAFETY_VECTOR_NAMES), -np.inf)
                safety_count = 0
                episode_start = time.perf_counter()
                cpu_start = time.process_time()
                for step in range(args.max_episode_steps):
                    # Safety is diagnostic only during testing; pools are never queried.
                    safety = extract_safety_vector(test_env, args)
                    safety_sums += safety
                    safety_minimums = np.minimum(safety_minimums, safety)
                    safety_maximums = np.maximum(safety_maximums, safety)
                    safety_count += 1
                    action = agent.greedy_action(
                        state, f"test|{args.seed}|{episode}|{step}"
                    )
                    next_raw, reward, terminated, truncated, info = test_env.step(action)
                    state = flatten_observation(next_raw)
                    reward_total += float(reward)
                    parsed = parse_step_info(info, bool(terminated), bool(truncated))
                    if terminated or truncated:
                        break
                row = episode_row(
                    "test", episode, scenario_seed, initial_hash, reward_total,
                    step + 1, parsed, args, episode_start, cpu_start, agent, (),
                    {"frozen_dqn_argmax": step + 1},
                    safety_sums, safety_minimums, safety_maximums, safety_count,
                )
                rows.append(row)
                print(
                    f"TEST  {METHOD_LABEL:27s} ep={episode:03d} "
                    f"reward={reward_total:9.3f} steps={step + 1:3d} "
                    f"term={parsed['termination_reason']:11s} "
                    f"collision={str(parsed['collision']):5s} "
                    f"wall={row['wall_time_seconds']:.2f}s",
                    flush=True,
                )
    finally:
        test_env.close()
    testing_wall = time.perf_counter() - testing_start
    testing_cpu = time.process_time() - testing_cpu_start
    print(f"===== FROZEN TESTING END: {testing_wall:.2f}s =====", flush=True)

    runtime_rows = []
    for phase, wall, cpu in (
        ("train", training_wall, training_cpu),
        ("test", testing_wall, testing_cpu),
    ):
        phase_rows = [row for row in rows if row["phase"] == phase]
        runtime_rows.append(
            {
                "method": METHOD,
                "method_label": METHOD_LABEL,
                "phase": phase,
                "episodes": len(phase_rows),
                "phase_wall_time_seconds": float(wall),
                "phase_cpu_time_seconds": float(cpu),
                "summed_episode_wall_time_seconds": float(sum(row["wall_time_seconds"] for row in phase_rows)),
                "summed_episode_cpu_time_seconds": float(sum(row["cpu_time_seconds"] for row in phase_rows)),
                "average_wall_time_seconds_per_episode": float(sum(row["wall_time_seconds"] for row in phase_rows) / len(phase_rows)),
                "average_cpu_time_seconds_per_episode": float(sum(row["cpu_time_seconds"] for row in phase_rows) / len(phase_rows)),
            }
        )
    return rows, runtime_rows, pools


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def phase_collision_metrics(rows: Sequence[Dict], phase: str, args) -> Dict:
    selected = [row for row in rows if row["phase"] == phase]
    collisions = sum(bool_value(row["collision"]) for row in selected)
    offroad = sum(bool_value(row["out_of_road"]) for row in selected)
    steps = sum(int(row["steps"]) for row in selected)
    times = [int(row["event_or_censor_time_steps"]) for row in selected]
    events = [bool_value(row["rmst_event_observed"]) for row in selected]
    return {
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "phase": phase,
        "episodes": len(selected),
        "collision_count": collisions,
        "out_of_road_count": offroad,
        "total_steps": steps,
        "collision_rmst_event_definition": args.rmst_event,
        "collision_rmst": restricted_mean_survival_time(times, events, args.rmst_tau),
        "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else math.nan,
        "collision_rate": collisions / len(selected) if selected else math.nan,
        "out_of_road_rate": offroad / len(selected) if selected else math.nan,
        "goal_rate": sum(bool_value(row["goal_reached"]) for row in selected) / len(selected) if selected else math.nan,
    }


def save_pool_outputs(pools: FourRiskActionPools, output_dir: Path) -> None:
    pool_rows = pools.pool_statistics()
    retired_rows = pools.retired_pool_statistics()
    pd.DataFrame(pool_rows).to_csv(output_dir / "state_pool_statistics.csv", index=False)
    pd.DataFrame(retired_rows).to_csv(
        output_dir / "state_retired_pool_statistics.csv", index=False
    )
    pd.DataFrame([pools.global_statistics()]).to_csv(
        output_dir / "state_pool_global_summary.csv", index=False
    )
    pd.DataFrame(pools.creation_events).to_csv(
        output_dir / "state_pool_creation_timeline.csv", index=False
    )
    threshold_rows = [
        {
            "risk_level": "low",
            "lower_bound_exclusive": -math.inf,
            "upper_bound_inclusive": pools.thresholds[0],
        },
        {
            "risk_level": "medium",
            "lower_bound_exclusive": pools.thresholds[0],
            "upper_bound_inclusive": pools.thresholds[1],
        },
        {
            "risk_level": "high",
            "lower_bound_exclusive": pools.thresholds[1],
            "upper_bound_inclusive": pools.thresholds[2],
        },
        {
            "risk_level": "critical",
            "lower_bound_exclusive": pools.thresholds[2],
            "upper_bound_inclusive": math.inf,
        },
    ]
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "state_risk_thresholds.csv", index=False
    )

    figure_dir = output_dir / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if pool_rows:
        labels = [row["risk_level"].title() for row in pool_rows]
        visits = [row["state_visits"] for row in pool_rows]
        coverage = [row["coverage_percent"] for row in pool_rows]
        colors = [RISK_COLORS[row["risk_level"]] for row in pool_rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(labels, visits, color=colors)
        ax.set_ylabel("Training state visits")
        ax.set_title("Four Risk Pool Occupancy")
        fig.tight_layout()
        fig.savefig(figure_dir / "four_risk_pool_occupancy.png", dpi=300)
        fig.savefig(figure_dir / "four_risk_pool_occupancy.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(labels, coverage, color=colors)
        ax.axhline(100.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylim(0.0, 105.0)
        ax.set_ylabel("Action-mask coverage (%)")
        ax.set_title("Four Risk Pool Action Coverage")
        fig.tight_layout()
        fig.savefig(figure_dir / "four_risk_pool_action_coverage.png", dpi=300)
        fig.savefig(figure_dir / "four_risk_pool_action_coverage.pdf")
        plt.close(fig)


def save_outputs(rows, runtimes, pools, args, output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "all_episode_results.csv", index=False)
    pd.DataFrame(runtimes).to_csv(
        output_dir / "all_experiments_runtime_logs.csv", index=False
    )
    pd.DataFrame(runtimes).to_csv(output_dir / "runtime_statistics.csv", index=False)
    metrics = [
        phase_collision_metrics(rows, "train", args),
        phase_collision_metrics(rows, "test", args),
    ]
    pd.DataFrame(metrics).to_csv(output_dir / "collision_metrics.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps(json_safe(vars(args)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    save_pool_outputs(pools, output_dir)

    manifest_paths = {
        "model_sha256": output_dir / "model.pt",
        "results_sha256": output_dir / "all_episode_results.csv",
        "metrics_sha256": output_dir / "collision_metrics.csv",
        "runtime_statistics_sha256": output_dir / "runtime_statistics.csv",
        "config_sha256": output_dir / "config.json",
        "state_pool_statistics_sha256": output_dir / "state_pool_statistics.csv",
        "state_retired_pool_statistics_sha256": output_dir / "state_retired_pool_statistics.csv",
        "state_risk_thresholds_sha256": output_dir / "state_risk_thresholds.csv",
    }
    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": getattr(metadrive, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(args.device),
        "seed": args.seed,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "design": "four_retired_pools_p80_random_latest_env_reward_80_20_v8",
        "candidate_process_present": False,
        "promotion_process_present": False,
        "eviction_process_present": False,
        "phase_runtime": runtimes,
        "created_at_unix": time.time(),
        **{name: sha256_file(path) for name, path in manifest_paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI and validation
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MetaDrive DQN with four active/retired safety-risk action pools"
        )
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--pool-training-fraction", type=float, default=0.80)
    parser.add_argument("--retired-value-percentile", type=float, default=80.0)
    parser.add_argument("--risk-low-max", type=float, default=0.25)
    parser.add_argument("--risk-medium-max", type=float, default=0.50)
    parser.add_argument("--risk-high-max", type=float, default=0.75)

    parser.add_argument("--collision-safe-distance", type=float, default=30.0)
    parser.add_argument("--collision-safe-ttc", type=float, default=5.0)
    parser.add_argument("--offroad-safe-clearance", type=float, default=1.5)
    parser.add_argument("--offroad-safe-heading-error", type=float, default=0.60)
    parser.add_argument("--safety-projection-seconds", type=float, default=1.0)
    parser.add_argument("--safety-nearest-object-cap", type=float, default=100.0)
    parser.add_argument("--safety-ttc-cap", type=float, default=20.0)
    parser.add_argument("--safety-lane-boundary-cap", type=float, default=10.0)
    parser.add_argument("--safety-speed-cap", type=float, default=200.0)
    parser.add_argument(
        "--safety-speed-fallback-unit", choices=["mps", "kmh"], default="mps"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--discrete-steering-dim", type=int, default=3)
    parser.add_argument("--discrete-throttle-dim", type=int, default=3)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--traffic-density", type=float, default=0.20)
    parser.add_argument("--accident-prob", type=float, default=0.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=50.0)
    parser.add_argument("--out-of-road-penalty", type=float, default=10.0)
    parser.add_argument("--metadrive-log-level", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--rmst-tau", type=int, default=500)
    parser.add_argument(
        "--rmst-event", choices=["collision", "safety"], default="collision"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args) -> None:
    positive = {
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "target_update_steps": args.target_update_steps,
        "hidden_size": args.hidden_size,
        "collision_safe_distance": args.collision_safe_distance,
        "collision_safe_ttc": args.collision_safe_ttc,
        "offroad_safe_clearance": args.offroad_safe_clearance,
        "offroad_safe_heading_error": args.offroad_safe_heading_error,
        "safety_projection_seconds": args.safety_projection_seconds,
        "safety_nearest_object_cap": args.safety_nearest_object_cap,
        "safety_ttc_cap": args.safety_ttc_cap,
        "safety_lane_boundary_cap": args.safety_lane_boundary_cap,
        "safety_speed_cap": args.safety_speed_cap,
        "rmst_tau": args.rmst_tau,
    }
    invalid = [name for name, value in positive.items() if not math.isfinite(float(value)) or value <= 0]
    if invalid:
        raise ValueError("These arguments must be finite and positive: " + ", ".join(invalid))
    if args.replay_capacity < args.batch_size:
        raise ValueError("--replay-capacity must be at least --batch-size.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1.")
    if not 0.0 < args.pool_training_fraction < 1.0:
        raise ValueError("--pool-training-fraction must be strictly between 0 and 1.")
    if not math.isfinite(args.retired_value_percentile) or not (
        0.0 <= args.retired_value_percentile <= 100.0
    ):
        raise ValueError("--retired-value-percentile must be between 0 and 100.")
    if not (
        0.0 <= args.risk_low_max
        < args.risk_medium_max
        < args.risk_high_max
        <= 1.0
    ):
        raise ValueError(
            "Risk thresholds must satisfy 0 <= low < medium < high <= 1."
        )
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("Seeds must be non-negative.")
    train_end = args.seed + args.train_episodes - 1
    test_end = args.test_seed + args.test_episodes - 1
    if max(args.seed, args.test_seed) <= min(train_end, test_end):
        raise ValueError("Training and testing scenario-seed ranges overlap.")
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1.")
    if not 0.0 <= args.accident_prob <= 1.0:
        raise ValueError("--accident-prob must be between 0 and 1.")
    if args.collision_penalty < 0 or args.out_of_road_penalty < 0:
        raise ValueError("Environment penalty magnitudes must be non-negative.")


def resolve_output_dir(args) -> Path:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    canonical = (
        project_dir
        / "policy_results"
        / f"seed_{args.seed}"
        / METHOD
    ).resolve()
    if args.output_dir is None:
        return canonical
    supplied = Path(args.output_dir).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    supplied = supplied.resolve()
    if supplied != canonical:
        raise ValueError(f"--output-dir must equal {canonical}; received {supplied}")
    return canonical


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Pass --force to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed, args.deterministic)
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args.force)
    device = choose_device(args.device)
    args.device = str(device)
    args.output_dir = str(output_dir)
    args.design_version = "four_retired_pools_p80_random_latest_env_reward_80_20_v8"

    pool_limit = int(math.ceil(args.train_episodes * args.pool_training_fraction))
    print("=" * 76)
    print("METADRIVE KARTHIKEYA FOUR SAFETY-RISK POOLS")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("MetaDrive:", getattr(metadrive, "__version__", "installed"))
    print("Device:", device)
    print("Training policy: first", pool_limit, "episodes use the safety gate")
    print("Final argmax phase:", args.train_episodes - pool_limit, "episodes")
    print("Risk thresholds: low <=", args.risk_low_max,
          "; medium <=", args.risk_medium_max,
          "; high <=", args.risk_high_max,
          "; critical above")
    print("New pool first action: uniform random remaining action")
    print("Active pool member: uniform random remaining action")
    print(
        "Retired pool member: uniform random action with value >= P",
        args.retired_value_percentile,
        sep="",
    )
    print("Retired numerical fallback: uniform random among maximum-valued ties")
    print("DQN reward source: unmodified MetaDrive reward")
    print("Pool reward source: unmodified MetaDrive reward")
    print("Pool value update: overwrite with latest environment reward")
    print("Pool reward averaging/normalization/additional deductions: none")
    print("Every training transition updates DQN: yes")
    print("Candidates/promotion/eviction: removed")
    print("Testing: frozen DQN argmax only")
    print("Output directory:", output_dir)
    print("=" * 76)

    rows, runtimes, pools = run_experiment(args, device, output_dir)
    save_outputs(rows, runtimes, pools, args, output_dir)
    print("\nExperiment completed successfully.")
    print("Episode results:", output_dir / "all_episode_results.csv")
    print("Collision metrics:", output_dir / "collision_metrics.csv")
    print("Pool statistics:", output_dir / "state_pool_statistics.csv")
    print("Retired pool statistics:", output_dir / "state_retired_pool_statistics.csv")
    print("Risk thresholds:", output_dir / "state_risk_thresholds.csv")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
