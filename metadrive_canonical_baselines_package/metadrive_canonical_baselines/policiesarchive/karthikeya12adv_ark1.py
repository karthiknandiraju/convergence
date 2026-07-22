#!/usr/bin/env python3
"""MetaDrive karthikeya12adv: risk-gated DQN with exact reward medians.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-17 CEST (+0200)

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* No RND and no count-based intrinsic reward.
* Frozen greedy testing: no optimizer, replay, or target updates.
* Disjoint training and testing scenarios.
* Defaults: 500 train episodes, 300 test episodes, 500 maximum steps.
* Policy name: karthikeya12adv.
* Four dynamically calibrated safety pools: Low, Medium, High, and Critical.
* The first 20% of training uses global DQN argmax and records warm-up
  experience. At the end of warm-up, the 25th, 50th, and 75th percentiles of
  observed pre-action combined risk become the frozen pool boundaries.
* Warm-up transitions are then routed retrospectively through those learned
  boundaries. Exact High/Critical rewards and safety outcomes seed their
  histories, but warm-up actions do not consume mask entries.
* Every pool receives one nine-action mask. Masks are used only after warm-up
  and never reset.
* During the middle 60% of training, every active pool uses the highest-Q
  available action in its mask. Only that selected action is removed.
* Retired Low/Medium pools use global DQN argmax and collect no pool rewards.
* Once a mask is empty, that pool is permanently retired. A retired
  High/Critical pool uses DQN's top three actions. Each
  receives a reward-dominant score: 0.80 * exact reward median minus
  0.20 * (recorded collision rate + off-road rate). The highest score wins.
* The score is used only after all three candidates have at least three exact
  reward observations; otherwise DQN's highest-ranked candidate is used.
* During the final 20%, all states use global DQN argmax, while High/Critical
  exact environment-reward histories and safety rates continue to update.
* Environment rewards are never normalized, clipped, rescaled, or supplemented.
* Streaming exact medians use two heaps: O(log N) insertion and O(N) memory.

Primary test metrics
--------------------
* Mean R: mean environment return.
* Median R: median environment return.
* IQMR: interquartile mean reward (middle 50% of returns).
* RMST: Kaplan-Meier restricted mean event-free survival time up to tau,
  where tau defaults to 500 steps. The event can be collision-only (default)
  or any safety failure (collision or off-road).

Main outputs
------------
* four_primary_test_metrics.csv
* all_episode_results.csv
* all_experiments_train_episode_rewards.csv
* all_experiments_test_episode_rewards.csv
* all_experiments_learning_rate_summary.csv
* all_experiments_runtime_logs.csv
* models/*.pt
* high_critical_pool_statistics.csv
* high_critical_exact_reward_history.csv
* high_critical_pool_100_episode_log.csv
* high_critical_pool_logs/high_critical_tables_episode_*.csv
* dynamic_pool_calibration.csv
* plots/*.png and *.pdf

Install:
    python -m pip install metadrive-simulator gymnasium torch numpy pandas matplotlib psutil

Example (when this file is placed in the project's ``policies`` folder):
    python policies/karthikeya12adv.py \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda \
      --seed 11

Results are saved only to ``policy_results/seed_<seed>/karthikeya12adv`` under the
project root. A runner-supplied ``--output-dir`` is accepted only when it
resolves to that same canonical folder.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import platform
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

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
    import psutil
except Exception:
    psutil = None

try:
    import metadrive
    from metadrive import MetaDriveEnv
except ImportError as exc:
    raise SystemExit(
        "MetaDrive is not installed. Run: python -m pip install metadrive-simulator"
    ) from exc


EXPERIMENTS = ["karthikeya12adv"]
SHORT_LABELS = {
    "karthikeya12adv": "karthikeya12adv",
}
COLORS = {
    "karthikeya12adv": "#2ca02c",
}
RISK_LEVELS = ("low", "medium", "high", "critical")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def process_memory_mb() -> float:
    if psutil is None:
        return 0.0
    return float(psutil.Process().memory_info().rss / (1024**2))


def system_memory_metrics() -> Dict[str, float]:
    if psutil is None:
        return {"ram_used_mb": 0.0, "ram_total_mb": 0.0, "ram_percent": 0.0}
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb": float(vm.used / (1024**2)),
        "ram_total_mb": float(vm.total / (1024**2)),
        "ram_percent": float(vm.percent),
    }


def smi_metrics() -> Dict[str, float]:
    values = {
        "gpu_util_percent": 0.0,
        "gpu_memory_used_mb_smi": 0.0,
        "gpu_memory_total_mb_smi": 0.0,
        "gpu_power_watts": 0.0,
        "gpu_temperature_c": 0.0,
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip().splitlines()[0]
        data = [item.strip() for item in output.split(",")]
        values.update(
            {
                "gpu_util_percent": float(data[0]),
                "gpu_memory_used_mb_smi": float(data[1]),
                "gpu_memory_total_mb_smi": float(data[2]),
                "gpu_power_watts": float(data[3]),
                "gpu_temperature_c": float(data[4]),
            }
        )
    except Exception:
        pass
    return values


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


# ---------------------------------------------------------------------------
# Pre-action collision/off-road risk
# ---------------------------------------------------------------------------


def _finite_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _vector2(value) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(2, dtype=np.float32)
    if vector.size < 2:
        vector = np.pad(vector, (0, 2 - vector.size))
    return vector[:2]


def _vehicle_lane(vehicle):
    navigation = getattr(vehicle, "navigation", None)
    lane = getattr(navigation, "current_lane", None)
    return lane if lane is not None else getattr(vehicle, "lane", None)


def _angle_difference_radians(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


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


def _is_collision_hazard(obj) -> bool:
    text = f"{obj.__class__.__module__}.{obj.__class__.__name__}".lower()
    if any(
        token in text
        for token in (
            "navigation", "camera", "sensor", "engine", "manager", "policy",
            "renderer", "nodepath", "lane", "road", "map", "light",
            "marking", "terrain",
        )
    ):
        return False
    if any(
        token in text
        for token in (
            "vehicle", "pedestrian", "human", "cyclist", "bicycle", "cone",
            "barrier", "obstacle", "trafficobject", "traffic_object",
            "building", "sidewalk",
        )
    ):
        return True
    return getattr(obj, "position", None) is not None and any(
        hasattr(obj, attr)
        for attr in (
            "collision_node", "collision_nodes", "body", "chassis",
            "top_down_width", "top_down_length", "WIDTH", "LENGTH",
            "velocity", "speed", "speed_km_h", "heading_theta",
        )
    )


def _hazard_metrics(
    env,
    ego_position: np.ndarray,
    ego_velocity: np.ndarray,
    args,
) -> Tuple[float, float]:
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
        relative_velocity = (
            _vector2(getattr(obj, "velocity", (0.0, 0.0))) - ego_velocity[:2]
        )
        closing_speed = -float(np.dot(relative_velocity, direction))
        if closing_speed > 0.1:
            best_ttc = min(best_ttc, surface_distance / closing_speed)
    return float(min(best_surface, distance_cap)), float(min(best_ttc, ttc_cap))


def extract_safety_vector(env, args) -> np.ndarray:
    """Return pre-action lane, hazard, heading, projection, and speed data."""
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return np.asarray(
            [0.0, 0.0, 0.0, 0.0, 1.0, math.pi, 0.0, 0.0],
            dtype=np.float32,
        )
    position = _vector2(getattr(vehicle, "position", (0.0, 0.0)))
    velocity = _vector2(getattr(vehicle, "velocity", (0.0, 0.0)))
    speed_value = getattr(vehicle, "speed_km_h", None)
    if speed_value is None:
        speed_value = _finite_float(getattr(vehicle, "speed", 0.0), 0.0)
        if args.safety_speed_fallback_unit == "mps":
            speed_value *= 3.6
    speed_km_h = float(np.clip(abs(_finite_float(speed_value, 0.0)), 0.0, args.safety_speed_cap))
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
        width_at = getattr(lane, "width_at", None)
        lane_width = 0.0
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
    lateral_speed = abs((speed_km_h / 3.6) * math.sin(heading_error))
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


def safety_risk_components(
    safety: np.ndarray,
    args,
) -> Tuple[float, float, float]:
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
    distance_risk = 0.0 if hazard_distance >= 0.999 * args.safety_nearest_object_cap else (
        1.0 - np.clip(hazard_distance / args.collision_safe_distance, 0.0, 1.0)
    )
    ttc_risk = 0.0 if ttc >= 0.999 * args.safety_ttc_cap else (
        1.0 - np.clip(ttc / args.collision_safe_ttc, 0.0, 1.0)
    )
    speed_factor = 1.0 + 0.25 * np.clip(
        speed_km_h / args.safety_speed_cap, 0.0, 1.0
    )
    collision_risk = float(
        np.clip(max(distance_risk, ttc_risk) * speed_factor, 0.0, 1.0)
    )
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
        np.clip(
            max(clearance_risk, projected_risk, offset_risk, heading_risk),
            0.0,
            1.0,
        )
    )
    return collision_risk, offroad_risk, float(max(collision_risk, offroad_risk))


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
        "random_traffic": True,
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
    def __init__(self, capacity: int):
        self.data: Deque[Transition] = deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done) -> None:
        self.data.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, int(batch_size))

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
        self.replay = ReplayBuffer(args.replay_capacity)

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
    def _random_argmax_from_q(q: np.ndarray) -> int:
        """Reservoir-sample tied maxima without allocating candidates."""
        extreme = float(np.max(q))
        chosen = -1
        matches = 0
        for action, value in enumerate(q):
            if float(value) == extreme:
                matches += 1
                if random.randrange(matches) == 0:
                    chosen = action
        if chosen < 0:  # Defensive only; a non-empty Q vector always has an extreme.
            raise RuntimeError("No action was available in the Q-value vector.")
        return int(chosen)

    def greedy_action(self, state: np.ndarray) -> int:
        q = self.q_values(state)
        return self._random_argmax_from_q(q)

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
# Exploration and experiment execution
# ---------------------------------------------------------------------------


@dataclass
class StreamingMedian:
    """Chronological exact rewards plus a two-heap exact median."""

    values: List[float] = field(default_factory=list)
    lower: List[float] = field(default_factory=list)  # Negated max heap.
    upper: List[float] = field(default_factory=list)  # Min heap.

    def add(self, environment_reward: float) -> None:
        value = float(environment_reward)
        if not math.isfinite(value):
            raise ValueError("MetaDrive returned a non-finite environment reward.")
        self.values.append(value)
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def latest(self) -> float:
        return self.values[-1] if self.values else math.nan

    @property
    def median(self) -> float:
        if not self.values:
            return math.nan
        if len(self.lower) > len(self.upper):
            return float(-self.lower[0])
        return float((-self.lower[0] + self.upper[0]) / 2.0)


@dataclass(frozen=True)
class WarmupTransition:
    """One exact transition retained until dynamic boundaries are learned."""

    combined_risk: float
    action: int
    environment_reward: float
    collision: bool
    offroad: bool


@dataclass
class RiskPool:
    level: str
    action_count: int
    mask: int = field(init=False)
    reward_histories: List[StreamingMedian] = field(init=False)
    action_selections: np.ndarray = field(init=False)
    action_updates: np.ndarray = field(init=False)
    collisions: np.ndarray = field(init=False)
    offroads: np.ndarray = field(init=False)
    visits: int = 0
    retired_episode: int = -1
    retired_step: int = -1

    def __post_init__(self) -> None:
        self.reward_histories = [StreamingMedian() for _ in range(self.action_count)]
        self.action_selections = np.zeros(self.action_count, dtype=np.int64)
        self.action_updates = np.zeros(self.action_count, dtype=np.int64)
        self.collisions = np.zeros(self.action_count, dtype=np.int64)
        self.offroads = np.zeros(self.action_count, dtype=np.int64)
        self.mask = (1 << self.action_count) - 1

    @property
    def retired(self) -> bool:
        return self.mask == 0


class Karthikeya12AdvPools:
    """Four dynamically calibrated pools; only High/Critical retain rewards."""

    def __init__(self, action_count: int, args):
        if action_count != 9:
            raise ValueError("karthikeya12adv requires exactly nine discrete actions.")
        self.action_count = int(action_count)
        self.args = args
        self.warmup_episode_count = int(
            math.floor(args.train_episodes * args.warmup_training_fraction)
        )
        self.policy_episode_count = int(
            math.floor(args.train_episodes * args.pool_policy_training_fraction)
        )
        self.policy_episode_end = self.warmup_episode_count + self.policy_episode_count
        self.final_episode_count = args.train_episodes - self.policy_episode_end
        self.thresholds: Optional[Tuple[float, float, float]] = None
        self.warmup_transitions: List[WarmupTransition] = []
        self.calibration_sample_count = 0
        self.calibration_pool_counts = {level: 0 for level in RISK_LEVELS}
        self.pools = {
            level: RiskPool(level, self.action_count) for level in RISK_LEVELS
        }
        self.snapshot_rows: List[Dict] = []
        self.warmup_dqn_actions = 0
        self.masked_dqn_actions = 0
        self.top3_score_actions = 0
        self.insufficient_history_dqn_actions = 0
        self.low_medium_dqn_actions = 0
        self.final_phase_dqn_actions = 0

    def training_phase(self, episode: int) -> str:
        if episode < self.warmup_episode_count:
            return "warmup"
        if episode < self.policy_episode_end:
            return "pool_policy"
        return "final_dqn"

    def risk_components(self, safety: np.ndarray) -> Tuple[float, float, float]:
        return safety_risk_components(safety, self.args)

    def classify_combined(self, combined: float) -> str:
        if self.thresholds is None:
            raise RuntimeError("Dynamic pool boundaries have not been calibrated yet.")
        low, medium, high = self.thresholds
        if combined <= low:
            return "low"
        if combined <= medium:
            return "medium"
        if combined <= high:
            return "high"
        return "critical"

    def classify(self, safety: np.ndarray) -> Tuple[str, float, float, float]:
        collision_risk, offroad_risk, combined = self.risk_components(safety)
        level = self.classify_combined(combined)
        return level, combined, collision_risk, offroad_risk

    def record_warmup_transition(
        self,
        combined_risk: float,
        action: int,
        environment_reward: float,
        parsed: Dict,
    ) -> None:
        risk = float(combined_risk)
        reward = float(environment_reward)
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("Warm-up combined risk must be finite and in [0, 1].")
        if not math.isfinite(reward):
            raise ValueError("MetaDrive returned a non-finite environment reward.")
        self.warmup_transitions.append(
            WarmupTransition(
                combined_risk=risk,
                action=int(action),
                environment_reward=reward,
                collision=bool(parsed["collision"]),
                offroad=bool(parsed["out_of_road"]),
            )
        )

    def finalize_calibration(self) -> Tuple[float, float, float]:
        """Learn frozen quartile boundaries and route warm-up experience."""
        if self.thresholds is not None:
            return self.thresholds
        if not self.warmup_transitions:
            raise RuntimeError("Cannot calibrate four pools without warm-up experience.")
        risks = np.asarray(
            [record.combined_risk for record in self.warmup_transitions],
            dtype=np.float64,
        )
        learned = np.percentile(risks, [25.0, 50.0, 75.0])
        self.thresholds = tuple(float(value) for value in learned)
        self.calibration_sample_count = len(self.warmup_transitions)

        for record in self.warmup_transitions:
            level = self.classify_combined(record.combined_risk)
            self.calibration_pool_counts[level] += 1
            pool = self.pools[level]
            pool.visits += 1
            pool.action_selections[record.action] += 1
            if level in {"high", "critical"}:
                self._record_high_critical_outcome(
                    level,
                    record.action,
                    record.environment_reward,
                    record.collision,
                    record.offroad,
                )

        # All exact High/Critical rewards have been transferred into their
        # streaming histories; release the temporary warm-up buffer.
        self.warmup_transitions.clear()
        return self.thresholds

    @staticmethod
    def _argmax_in_mask(q_values: np.ndarray, mask: int) -> int:
        """Linear scan with deterministic smallest-action tie breaking."""
        best_action = -1
        best_value = -math.inf
        for action, q_value in enumerate(np.asarray(q_values, dtype=float)):
            if mask & (1 << action):
                value = float(q_value)
                if value > best_value:
                    best_value = value
                    best_action = action
        if best_action < 0:
            raise RuntimeError("The action mask contains no available action.")
        return int(best_action)

    @staticmethod
    def _top_three_actions(q_values: np.ndarray) -> List[int]:
        """Return DQN top three with O(A) time and O(1) extra memory."""
        top: List[Tuple[float, int]] = []
        for action, q_value in enumerate(np.asarray(q_values, dtype=float)):
            item = (float(q_value), int(action))
            position = 0
            while position < len(top):
                old_q, old_action = top[position]
                if item[0] > old_q or (item[0] == old_q and item[1] < old_action):
                    break
                position += 1
            top.insert(position, item)
            if len(top) > 3:
                top.pop()
        return [action for _, action in top]

    @staticmethod
    def clear_mask_action(
        pool: RiskPool,
        action: int,
        episode: int,
        step: int,
    ) -> None:
        pool.mask &= ~(1 << int(action))
        if pool.mask == 0 and pool.retired_episode < 0:
            pool.retired_episode = int(episode)
            pool.retired_step = int(step)

    def masked_dqn_action(
        self,
        pool: RiskPool,
        q_values: np.ndarray,
        episode: int,
        step: int,
    ) -> int:
        action = self._argmax_in_mask(q_values, pool.mask)
        self.clear_mask_action(pool, action, episode, step)
        pool.action_selections[action] += 1
        self.masked_dqn_actions += 1
        return action

    @staticmethod
    def action_safety_rates(
        pool: RiskPool,
        action: int,
    ) -> Tuple[float, float, float]:
        updates = int(pool.action_updates[action])
        if updates <= 0:
            return math.nan, math.nan, math.nan
        collision_rate = float(pool.collisions[action]) / updates
        offroad_rate = float(pool.offroads[action]) / updates
        return collision_rate, offroad_rate, collision_rate + offroad_rate

    def action_score(self, pool: RiskPool, action: int) -> float:
        _collision_rate, _offroad_rate, combined_rate = self.action_safety_rates(
            pool, action
        )
        if not math.isfinite(combined_rate):
            return math.nan
        median_reward = pool.reward_histories[action].median
        return float(
            self.args.reward_score_weight * median_reward
            - self.args.safety_score_weight * combined_rate
        )

    def top3_score_action(
        self,
        pool: RiskPool,
        q_values: np.ndarray,
    ) -> Tuple[int, bool]:
        candidates = self._top_three_actions(q_values)
        if any(
            pool.reward_histories[action].count < self.args.minimum_score_samples
            for action in candidates
        ):
            action = candidates[0]
            pool.action_selections[action] += 1
            self.insufficient_history_dqn_actions += 1
            return int(action), False
        best_action = -1
        best_score = -math.inf
        best_q = -math.inf
        for action in candidates:
            score = self.action_score(pool, action)
            if not math.isfinite(score):
                score = -math.inf
            q_value = float(q_values[action])
            if (
                score > best_score
                or (score == best_score and q_value > best_q)
                or (
                    score == best_score
                    and q_value == best_q
                    and (best_action < 0 or action < best_action)
                )
            ):
                best_score = score
                best_q = q_value
                best_action = action
        if best_action < 0:
            raise RuntimeError("DQN top-three reward/safety scoring failed.")
        pool.action_selections[best_action] += 1
        self.top3_score_actions += 1
        return int(best_action), True

    def record_dqn_action(self, pool: RiskPool, action: int, final_phase: bool) -> None:
        pool.action_selections[int(action)] += 1
        if final_phase:
            self.final_phase_dqn_actions += 1
        else:
            self.low_medium_dqn_actions += 1

    def update_after_transition(
        self,
        level: Optional[str],
        action: int,
        environment_reward: float,
        parsed: Dict,
    ) -> None:
        if level is None:
            return
        if level not in {"high", "critical"}:
            raise ValueError("Only High/Critical pools may store reward histories.")
        self._record_high_critical_outcome(
            level,
            int(action),
            float(environment_reward),
            bool(parsed["collision"]),
            bool(parsed["out_of_road"]),
        )

    def _record_high_critical_outcome(
        self,
        level: str,
        action: int,
        environment_reward: float,
        collision: bool,
        offroad: bool,
    ) -> None:
        reward = float(environment_reward)
        if not math.isfinite(reward):
            raise ValueError("MetaDrive returned a non-finite environment reward.")
        pool = self.pools[level]
        action = int(action)
        pool.reward_histories[action].add(reward)
        pool.action_updates[action] += 1
        pool.collisions[action] += int(collision)
        pool.offroads[action] += int(offroad)

    def snapshot(self, completed_episodes: int, output_dir: Path) -> None:
        detailed_rows: List[Dict] = []
        summary_rows: List[Dict] = []
        for level in ("high", "critical"):
            pool = self.pools[level]
            for action, history in enumerate(pool.reward_histories):
                collision_rate, offroad_rate, combined_rate = self.action_safety_rates(
                    pool, action
                )
                summary = {
                    "completed_episodes": int(completed_episodes),
                    "risk_level": level,
                    "action": action,
                    "mask": pool.mask,
                    "status": "retired" if pool.retired else "active",
                    "retired_episode": pool.retired_episode,
                    "retired_step": pool.retired_step,
                    "reward_count": history.count,
                    "median_environment_reward": history.median,
                    "latest_environment_reward": history.latest,
                    "reward_safety_score": self.action_score(pool, action),
                    "collision_rate": collision_rate,
                    "offroad_rate": offroad_rate,
                    "collision_plus_offroad_rate": combined_rate,
                    "collisions": int(pool.collisions[action]),
                    "offroads": int(pool.offroads[action]),
                }
                summary_rows.append(summary)
                detailed_rows.append(
                    {
                        **summary,
                        "exact_environment_reward_list": json.dumps(
                            history.values, separators=(",", ":")
                        ),
                    }
                )
        self.snapshot_rows.extend(summary_rows)
        log_dir = output_dir / "high_critical_pool_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detailed_rows).to_csv(
            log_dir / f"high_critical_tables_episode_{completed_episodes:04d}.csv",
            index=False,
        )
        pd.DataFrame(
            self.snapshot_rows,
            columns=(
                "completed_episodes", "risk_level", "action", "mask",
                "status", "retired_episode", "retired_step",
                "reward_count", "median_environment_reward",
                "latest_environment_reward", "reward_safety_score",
                "collision_rate", "offroad_rate",
                "collision_plus_offroad_rate", "collisions", "offroads",
            ),
        ).to_csv(output_dir / "high_critical_pool_100_episode_log.csv", index=False)

    def save_final_outputs(self, output_dir: Path) -> None:
        if self.thresholds is None:
            raise RuntimeError("Dynamic pool calibration did not complete.")
        reward_rows: List[Dict] = []
        statistics_rows: List[Dict] = []
        for level in ("high", "critical"):
            pool = self.pools[level]
            for action, history in enumerate(pool.reward_histories):
                collision_rate, offroad_rate, combined_rate = self.action_safety_rates(
                    pool, action
                )
                statistics_rows.append(
                    {
                        "risk_level": level,
                        "action": action,
                        "selection_count": int(pool.action_selections[action]),
                        "status": "retired" if pool.retired else "active",
                        "retired_episode": pool.retired_episode,
                        "retired_step": pool.retired_step,
                        "reward_count": history.count,
                        "median_environment_reward": history.median,
                        "latest_environment_reward": history.latest,
                        "reward_safety_score": self.action_score(pool, action),
                        "collision_rate": collision_rate,
                        "offroad_rate": offroad_rate,
                        "collision_plus_offroad_rate": combined_rate,
                        "collisions": int(pool.collisions[action]),
                        "offroads": int(pool.offroads[action]),
                    }
                )
                for reward_index, exact_reward in enumerate(history.values):
                    reward_rows.append(
                        {
                            "risk_level": level,
                            "action": action,
                            "reward_index": reward_index,
                            "exact_environment_reward": exact_reward,
                        }
                    )
        pd.DataFrame(statistics_rows).to_csv(
            output_dir / "high_critical_pool_statistics.csv", index=False
        )
        pd.DataFrame(
            reward_rows,
            columns=(
                "risk_level", "action", "reward_index",
                "exact_environment_reward",
            ),
        ).to_csv(output_dir / "high_critical_exact_reward_history.csv", index=False)
        pd.DataFrame(
            self.snapshot_rows,
            columns=(
                "completed_episodes", "risk_level", "action", "mask",
                "status", "retired_episode", "retired_step",
                "reward_count", "median_environment_reward",
                "latest_environment_reward", "reward_safety_score",
                "collision_rate", "offroad_rate",
                "collision_plus_offroad_rate", "collisions", "offroads",
            ),
        ).to_csv(output_dir / "high_critical_pool_100_episode_log.csv", index=False)
        pd.DataFrame(
            [
                {
                    "policy": "karthikeya12adv",
                    "warmup_training_fraction": self.args.warmup_training_fraction,
                    "pool_policy_training_fraction": self.args.pool_policy_training_fraction,
                    "final_dqn_training_fraction": 1.0
                    - self.args.warmup_training_fraction
                    - self.args.pool_policy_training_fraction,
                    "warmup_episode_count": self.warmup_episode_count,
                    "pool_policy_episode_count": self.policy_episode_count,
                    "final_dqn_episode_count": self.final_episode_count,
                    "warmup_calibration_samples": self.calibration_sample_count,
                    "learned_low_max": self.thresholds[0],
                    "learned_medium_max": self.thresholds[1],
                    "learned_high_max": self.thresholds[2],
                    "warmup_global_dqn_actions": self.warmup_dqn_actions,
                    "masked_dqn_actions": self.masked_dqn_actions,
                    "top3_reward_safety_score_actions": self.top3_score_actions,
                    "insufficient_history_dqn_actions": self.insufficient_history_dqn_actions,
                    "low_medium_global_dqn_actions": self.low_medium_dqn_actions,
                    "final_phase_global_dqn_actions": self.final_phase_dqn_actions,
                    "reward_score_weight": self.args.reward_score_weight,
                    "safety_score_weight": self.args.safety_score_weight,
                    "minimum_score_samples": self.args.minimum_score_samples,
                    "reward_source": "exact_unmodified_metadrive_environment_reward",
                    "reward_normalization_present": False,
                    "reward_clipping_present": False,
                }
            ]
        ).to_csv(output_dir / "karthikeya12adv_pool_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "calibration_samples": self.calibration_sample_count,
                    "learned_low_max": self.thresholds[0],
                    "learned_medium_max": self.thresholds[1],
                    "learned_high_max": self.thresholds[2],
                    "warmup_low_samples": self.calibration_pool_counts["low"],
                    "warmup_medium_samples": self.calibration_pool_counts["medium"],
                    "warmup_high_samples": self.calibration_pool_counts["high"],
                    "warmup_critical_samples": self.calibration_pool_counts["critical"],
                    "boundaries_frozen_after_warmup": True,
                    "reward_source": "exact_unmodified_metadrive_environment_reward",
                }
            ]
        ).to_csv(output_dir / "dynamic_pool_calibration.csv", index=False)


def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    safety: np.ndarray,
    episode: int,
    step: int,
    args,
    pools: Karthikeya12AdvPools,
) -> Tuple[int, str, Optional[str], float]:
    if experiment != "karthikeya12adv":
        raise ValueError(f"Unknown experiment: {experiment}")
    _collision_risk, _offroad_risk, combined_risk = pools.risk_components(safety)
    phase = pools.training_phase(episode)

    if phase == "warmup":
        action = agent.greedy_action(state)
        pools.warmup_dqn_actions += 1
        return action, "warmup_global_dqn_argmax", None, combined_risk

    level = pools.classify_combined(combined_risk)
    pool = pools.pools[level]
    pool.visits += 1

    if phase == "final_dqn":
        action = agent.greedy_action(state)
        pools.record_dqn_action(pool, action, final_phase=True)
        update_level = level if level in {"high", "critical"} else None
        return action, f"final_phase_{level}_global_dqn_argmax", update_level, combined_risk

    if level in {"low", "medium"}:
        if pool.mask:
            q_values = agent.q_values(state)
            action = pools.masked_dqn_action(pool, q_values, episode, step)
            return action, f"{level}_masked_dqn_argmax", None, combined_risk
        action = agent.greedy_action(state)
        pools.record_dqn_action(pool, action, final_phase=False)
        return action, f"retired_{level}_global_dqn_argmax", None, combined_risk

    q_values = agent.q_values(state)
    if pool.mask:
        action = pools.masked_dqn_action(pool, q_values, episode, step)
        return action, f"{level}_masked_dqn_argmax", level, combined_risk
    action, used_score = pools.top3_score_action(pool, q_values)
    if used_score:
        return action, f"{level}_top3_reward_safety_score", level, combined_risk
    return action, f"{level}_top3_insufficient_history_dqn", level, combined_risk


def episode_row(
    phase: str,
    experiment: str,
    episode: int,
    scenario_seed: int,
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
        "episode": episode,
        "scenario_seed": scenario_seed,
        "env_reward": float(env_reward),
        "training_reward": float(training_reward),
        "steps": int(steps),
        **parsed,
        "rmst_event_definition": args.rmst_event,
        "rmst_event_observed": bool(event),
        "event_or_censor_time_steps": int(steps),
        "wall_time_seconds": time.time() - episode_start,
        "cpu_time_seconds": time.process_time() - cpu_start,
        "gpu_memory_mb": gpu_memory_mb(device),
        "process_memory_mb": process_memory_mb(),
        **system_memory_metrics(),
        **smi_metrics(),
        "average_loss": avg(losses),
        "replay_buffer_size": len(agent.replay),
        "learn_steps": agent.learn_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "network_frozen": phase == "test",
        "updates_during_test": 0 if phase == "test" else "",
        "action_source_counts": json.dumps(action_sources or {}, sort_keys=True),
    }


def run_experiment(
    experiment: str, args, device: torch.device, output_dir: Path
) -> Tuple[List[Dict], Dict]:
    set_seed(args.seed)
    rows: List[Dict] = []

    train_env = make_env(args, "train")
    initial_observation, _ = train_env.reset(seed=args.seed)
    observation_size = int(flatten_observation(initial_observation).size)
    if not hasattr(train_env.action_space, "n"):
        train_env.close()
        raise RuntimeError("MetaDrive action space is not Discrete; check discrete_action config.")
    action_count = int(train_env.action_space.n)
    agent = DQNAgent(observation_size, action_count, args, device)
    pools = Karthikeya12AdvPools(action_count, args)

    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.time()
    try:
        for episode in range(args.train_episodes):
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            state = flatten_observation(state_raw)
            env_reward_total = 0.0
            training_reward_total = 0.0
            losses: List[float] = []
            action_sources: Dict[str, int] = {}
            parsed = parse_step_info({}, False, False)
            episode_start = time.time()
            cpu_start = time.process_time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for step in range(args.max_episode_steps):
                safety = extract_safety_vector(train_env, args)
                action, source, pool_level, combined_risk = select_training_action(
                    experiment, agent, state, safety, episode, step, args, pools
                )
                action_sources[source] = action_sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                done = bool(terminated or truncated)
                parsed = parse_step_info(info, bool(terminated), bool(truncated))
                # DQN and High/Critical histories receive the same exact,
                # unmodified MetaDrive environment reward.
                training_reward = float(env_reward)
                agent.replay.add(state, action, training_reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                if pools.training_phase(episode) == "warmup":
                    pools.record_warmup_transition(
                        combined_risk,
                        action,
                        training_reward,
                        parsed,
                    )
                else:
                    pools.update_after_transition(
                        pool_level,
                        action,
                        training_reward,
                        parsed,
                    )
                env_reward_total += float(env_reward)
                training_reward_total += training_reward
                state = next_state
                if done:
                    break

            row = episode_row(
                "train",
                experiment,
                episode,
                scenario_seed,
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
            if episode + 1 == pools.warmup_episode_count:
                learned_thresholds = pools.finalize_calibration()
                print(
                    "DYNAMIC POOLS READY after "
                    f"{pools.warmup_episode_count} warm-up episodes: "
                    f"low<={learned_thresholds[0]:.6f}, "
                    f"medium<={learned_thresholds[1]:.6f}, "
                    f"high<={learned_thresholds[2]:.6f}, "
                    "otherwise critical",
                    flush=True,
                )
            print(
                f"TRAIN {SHORT_LABELS[experiment]:16s} ep={episode:03d} "
                f"reward={env_reward_total:9.3f} steps={step + 1:3d} "
                f"term={parsed['termination_reason']:11s} "
                f"collision={str(parsed['collision']):5s} "
                f"wall={row['wall_time_seconds']:.2f}s loss={row['average_loss']:.6f}",
                flush=True,
            )
            if (episode + 1) % 100 == 0:
                pools.snapshot(episode + 1, output_dir)
    finally:
        train_env.close()

    training_duration = time.time() - training_start
    pools.save_final_outputs(output_dir)
    model_path = output_dir / "models" / f"{experiment}_model.pt"
    agent.save(model_path, args)
    agent.freeze()
    print(f"===== TRAINING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Training duration: {training_duration:.2f}s", flush=True)

    # MetaDrive uses a singleton engine. The training environment is closed
    # before the test environment is constructed.
    test_env = make_env(args, "test")
    print(f"\n===== TESTING START: {SHORT_LABELS[experiment]} =====", flush=True)
    testing_start = time.time()
    try:
        with torch.no_grad():
            for episode in range(args.test_episodes):
                scenario_seed = args.test_seed + episode
                state_raw, _ = test_env.reset(seed=scenario_seed)
                state = flatten_observation(state_raw)
                reward_total = 0.0
                parsed = parse_step_info({}, False, False)
                episode_start = time.time()
                cpu_start = time.process_time()
                for step in range(args.max_episode_steps):
                    action = agent.greedy_action(state)
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

    testing_duration = time.time() - testing_start
    print(f"===== TESTING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Testing duration: {testing_duration:.2f}s", flush=True)
    runtime = {
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "training_duration_seconds": training_duration,
        "testing_duration_seconds": testing_duration,
        "model_path": str(model_path),
    }
    return rows, runtime


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
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    make_figures(rows, summary, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaDrive karthikeya12adv risk-gated DQN policy"
    )
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--warmup-training-fraction", type=float, default=0.20)
    parser.add_argument("--pool-policy-training-fraction", type=float, default=0.60)
    parser.add_argument("--reward-score-weight", type=float, default=0.80)
    parser.add_argument("--safety-score-weight", type=float, default=0.20)
    parser.add_argument("--minimum-score-samples", type=int, default=3)
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
        "--safety-speed-fallback-unit", choices=("mps", "kmh"), default="mps"
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
        "--seed",
        type=int,
        required=True,
        help="Training seed; also used in the canonical output folder name.",
    )
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
            "<project>/policy_results/seed_<seed>/karthikeya12adv."
        ),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    if args.train_episodes <= 0 or args.max_episode_steps <= 0:
        raise ValueError("Training episodes and maximum episode steps must be positive.")
    if args.test_episodes <= 0:
        raise ValueError("Testing episodes must be positive.")
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
    if args.discrete_steering_dim * args.discrete_throttle_dim != 9:
        raise ValueError("karthikeya12adv requires exactly 3 x 3 = 9 discrete actions.")
    if not 0.0 < args.warmup_training_fraction < 1.0:
        raise ValueError("--warmup-training-fraction must be strictly between 0 and 1.")
    if not 0.0 < args.pool_policy_training_fraction < 1.0:
        raise ValueError(
            "--pool-policy-training-fraction must be strictly between 0 and 1."
        )
    if args.warmup_training_fraction + args.pool_policy_training_fraction >= 1.0:
        raise ValueError("Warm-up and pool-policy fractions must leave a final DQN phase.")
    warmup_count = int(
        math.floor(args.train_episodes * args.warmup_training_fraction)
    )
    policy_count = int(
        math.floor(args.train_episodes * args.pool_policy_training_fraction)
    )
    final_count = args.train_episodes - warmup_count - policy_count
    if min(warmup_count, policy_count, final_count) <= 0:
        raise ValueError(
            "Training episodes are too few for non-empty warm-up, pool-policy, "
            "and final-DQN phases."
        )
    if args.reward_score_weight < 0.0 or args.safety_score_weight < 0.0:
        raise ValueError("Reward and safety score weights must be non-negative.")
    if not math.isclose(
        args.reward_score_weight + args.safety_score_weight,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Reward and safety score weights must sum to 1.0.")
    if args.minimum_score_samples <= 0:
        raise ValueError("--minimum-score-samples must be positive.")
    positive_safety = (
        args.collision_safe_distance,
        args.collision_safe_ttc,
        args.offroad_safe_clearance,
        args.offroad_safe_heading_error,
        args.safety_projection_seconds,
        args.safety_nearest_object_cap,
        args.safety_ttc_cap,
        args.safety_lane_boundary_cap,
        args.safety_speed_cap,
    )
    if any(not math.isfinite(float(value)) or value <= 0 for value in positive_safety):
        raise ValueError("All safety-distance, time, heading, and speed values must be positive.")


def resolve_output_dir(args) -> Path:
    """Return the only permitted seed-specific policy output directory."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    output_dir = (
        project_dir / "policy_results" / f"seed_{args.seed}" / "karthikeya12adv"
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


def main() -> None:
    args = parse_args()
    if args.rmst_tau is None:
        args.rmst_tau = args.max_episode_steps
    validate_args(args)
    set_seed(args.seed)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    args.output_dir = str(output_dir)
    device = choose_device(args.device)

    print("=" * 76)
    print("METADRIVE KARTHIKEYA12ADV RISK-GATED DQN POLICY")
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
    print("Policy name: karthikeya12adv")
    warmup_count = int(
        math.floor(args.train_episodes * args.warmup_training_fraction)
    )
    policy_count = int(
        math.floor(args.train_episodes * args.pool_policy_training_fraction)
    )
    final_count = args.train_episodes - warmup_count - policy_count
    print(
        "Training phases:",
        f"{warmup_count} warm-up DQN episodes /",
        f"{policy_count} dynamic-pool policy episodes /",
        f"{final_count} final DQN episodes",
    )
    print("Dynamic boundaries: warm-up combined-risk quartiles; frozen afterward")
    print(
        "Low/Medium: active highest-Q available mask action; "
        "retired global DQN argmax; no pool rewards"
    )
    print("Pool masks: untouched during warm-up; never reset after retirement")
    print(
        "High/Critical policy phase:",
        f"{100.0 * args.pool_policy_training_fraction:.1f}% of training; one mask",
    )
    print("All active pools: remove only the highest-Q available mask action")
    print("Post-mask candidates: DQN top three")
    print("Minimum exact rewards per scored candidate:", args.minimum_score_samples)
    print(
        "Candidate score:",
        args.reward_score_weight,
        "* reward median -",
        args.safety_score_weight,
        "* (collision rate + off-road rate)",
    )
    print("Final 20%: global DQN argmax; High/Critical histories update")
    print("Exact pool reward normalization/modification: none")
    print("Action scan: O(A) time, O(1) extra memory; A=9")
    print("Streaming median: O(log N) insertion, O(N) history memory")
    print("High/Critical table logs: every 100 episodes")
    print("RMST event/tau:", args.rmst_event, args.rmst_tau)
    print("=" * 76)

    all_rows: List[Dict] = []
    runtimes: List[Dict] = []
    for experiment in EXPERIMENTS:
        experiment_rows, runtime = run_experiment(experiment, args, device, output_dir)
        all_rows.extend(experiment_rows)
        runtimes.append(runtime)
    save_outputs(all_rows, runtimes, args, output_dir)
    required_outputs = (
        output_dir / "all_episode_results.csv",
        output_dir / "config.json",
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
    print("High/Critical statistics:", output_dir / "high_critical_pool_statistics.csv")
    print("Exact reward history:", output_dir / "high_critical_exact_reward_history.csv")
    print("100-episode pool log:", output_dir / "high_critical_pool_100_episode_log.csv")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
