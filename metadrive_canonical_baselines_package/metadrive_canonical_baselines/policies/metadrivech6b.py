#!/usr/bin/env python3
"""MetaDrive Chapter 6B Median 50 DQN policy.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-19 CEST (+0200)

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* No RND and no count-based intrinsic reward.
* Frozen greedy testing: no optimizer, replay, or target updates.
* Disjoint training and testing scenarios.
* Defaults: 500 train episodes, 300 test episodes, 500 maximum steps.
* The two middle episodes override the diagonal and episode-block rules:
  the middle episode uses U80 for its first 50% of steps and L20 afterward;
  the next-to-middle episode uses L20 first and U80 afterward.
* Outside those two episodes, the anti-diagonal uses minimum-Q and the main
  diagonal uses maximum-Q. Anti-diagonal minimum-Q has overlap precedence.
* At other positions, the first 50% of training episodes use maximum-Q and the
  remaining 50% use epsilon-greedy with epsilon 0.2 by default (80% maximum-Q,
  20% uniformly random action).
* Testing uses only the final frozen Pure DQN with deterministic argmax.
* Environment, DQN, replay, seed, truncation, CSV, runtime, metric, and manifest
  contracts match the attached canonical baseline. The training schedule is
  the intended experimental difference.
* This file contains only the Median 50 policy. Canonical epsilon, NoisyNet,
  and DQN+RND baselines are produced separately by the baseline package.

Primary test metrics
--------------------
* Mean R: mean environment return.
* Median R: median environment return.
* IQMR: interquartile mean reward (middle 50% of returns).
* RMST: Kaplan-Meier restricted mean collision-free survival time up to tau,
  where tau defaults to 500 steps.

Main outputs
------------
* all_episode_results.csv
* collision_metrics.csv
* runtime_statistics.csv
* config.json
* manifest.json
* model.pt and models/median_50_model.pt

Install:
    python -m pip install metadrive-simulator gymnasium torch numpy pandas matplotlib psutil

Example (when this file is placed in the project's ``policies`` folder):
    python policies/metadrivech6b.py \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda \
      --seed 11

Results are saved only to ``policy_results/seed_<seed>/ch6b`` under the
project root. A runner-supplied ``--output-dir`` is accepted only when it
resolves to that same canonical folder.
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
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Must be configured before CUDA creates a cuBLAS context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-canonical-baselines")

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


EXPERIMENTS = ["median_50"]
SHORT_LABELS = {
    "median_50": "Median 50",
}
COLORS = {
    "median_50": "#2ca02c",
}

POLICY_NAME = "ch6b"
OBSERVATION_SOURCE = "flattened_metadrive_observation_only"
TEST_POLICY = "final_trained_frozen_dqn_argmax"

CANONICAL_EPISODE_COLUMNS = (
    "phase", "experiment", "method", "seed", "episode",
    "scenario_seed", "initial_observation_sha256", "env_reward",
    "training_reward", "steps", "termination_reason", "collision",
    "crash_vehicle", "crash_object", "out_of_road", "goal_reached",
    "max_steps_reached", "rmst_event_definition", "rmst_event_observed",
    "event_or_censor_time_steps", "wall_time_seconds", "cpu_time_seconds",
    "average_loss", "average_rnd_loss", "average_rnd_bonus",
    "replay_buffer_size", "learn_steps", "epsilon", "rnd_beta",
    "noisy_sigma_init", "network_frozen", "updates_during_test",
    "action_source_counts",
)

CRITICAL_CONFIG_KEYS = (
    "seed", "deterministic", "train_episodes", "test_episodes",
    "max_episode_steps", "epsilon", "learning_rate", "gamma",
    "batch_size", "replay_capacity", "target_update_steps", "hidden_size",
    "discrete_steering_dim", "discrete_throttle_dim", "map_blocks",
    "traffic_density", "accident_prob", "success_reward",
    "collision_penalty", "out_of_road_penalty", "test_seed", "rmst_tau",
    "progress_every",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
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
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_json_sha256(data: Dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    original_t = np.asarray(times, dtype=float)
    t = np.minimum(original_t, tau)
    e = np.asarray(events, dtype=bool) & (original_t <= tau)
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(t[e]):
        current = float(current)
        area += survival * max(0.0, current - previous)
        at_risk = int(np.sum(t >= current))
        failures = int(np.sum(e & np.isclose(t, current)))
        if at_risk > 0 and failures > 0:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, tau - previous)
    return float(area)


def bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def collision_summary(rows: Sequence[Dict], tau: int) -> Dict[str, float]:
    collisions = sum(bool_value(row["collision"]) for row in rows)
    offroad = sum(bool_value(row["out_of_road"]) for row in rows)
    goals = sum(bool_value(row["goal_reached"]) for row in rows)
    steps = sum(int(row["steps"]) for row in rows)
    episodes = len(rows)
    combined = sum(
        bool_value(row["collision"]) or bool_value(row["out_of_road"])
        for row in rows
    )
    times = [int(row["event_or_censor_time_steps"]) for row in rows]
    return {
        "episodes": episodes,
        "collision_count": collisions,
        "out_of_road_count": offroad,
        "total_steps": steps,
        "collision_rmst_event_definition": "collision",
        "collision_rmst": restricted_mean_survival_time(
            times, [bool_value(row["collision"]) for row in rows], tau
        ),
        "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "out_of_road_rate": offroad / episodes if episodes else 0.0,
        "goal_count": goals,
        "goal_rate": goals / episodes if episodes else 0.0,
        "combined_safety_failure_count": combined,
        "combined_safety_failure_rate": combined / episodes if episodes else 0.0,
        "combined_safety_failures_per_1000_steps": (
            1000.0 * combined / steps if steps else 0.0
        ),
        "combined_safety_rmst_event_definition": "collision_or_out_of_road",
        "combined_safety_rmst": restricted_mean_survival_time(
            times,
            [
                bool_value(row["collision"]) or bool_value(row["out_of_road"])
                for row in rows
            ],
            tau,
        ),
    }


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
    """Baseline-compatible indexed ring buffer."""

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
        self.policy_rng = random.Random(args.seed + 10_001)

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
    def _deterministic_extreme(q: np.ndarray, maximum: bool, key: str) -> int:
        extreme = float(np.max(q) if maximum else np.min(q))
        candidates = np.flatnonzero(q == extreme)
        if candidates.size == 0:
            raise RuntimeError("No finite action was available.")
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int(candidates[int.from_bytes(digest[:8], "big") % candidates.size])

    def greedy_action(self, state: np.ndarray, key: str) -> int:
        return self._deterministic_extreme(self.q_values(state), True, key)

    def lowest_q_action(self, state: np.ndarray, key: str) -> int:
        return self._deterministic_extreme(self.q_values(state), False, key)

    def percentile_tail_action(
        self, state: np.ndarray, upper: bool, key: str
    ) -> int:
        """Uniformly sample the strict U80/L20 tail in O(A) time."""
        q = self.q_values(state)
        threshold = float(np.percentile(q, 80.0 if upper else 20.0))
        selected: Optional[int] = None
        seen = 0
        for action, value in enumerate(q):
            eligible = value > threshold if upper else value < threshold
            if eligible:
                seen += 1
                if self.policy_rng.randrange(seen) == 0:
                    selected = int(action)
        if selected is not None:
            return selected
        return self._deterministic_extreme(q, upper, key)

    def epsilon_greedy_action(
        self, state: np.ndarray, epsilon: float, key: str
    ) -> Tuple[int, str]:
        if self.policy_rng.random() < float(epsilon):
            return self.policy_rng.randrange(self.action_count), "random"
        return self.greedy_action(state, key), "max_q"

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

        self.online.train()
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
                "method": POLICY_NAME,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "config": json_safe(vars(args)),
            },
            path,
        )


# ---------------------------------------------------------------------------
# Exploration and experiment execution
# ---------------------------------------------------------------------------


def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    episode: int,
    step: int,
    args,
) -> Tuple[int, str]:
    if experiment == "median_50":
        key = f"train|{POLICY_NAME}|{episode}|{step}"
        greedy_episode_count = int(
            math.ceil(args.train_episodes * args.greedy_episode_fraction)
        )
        middle_episode = max(0, greedy_episode_count - 1)
        next_middle_episode = min(args.train_episodes - 1, middle_episode + 1)
        first_half_steps = (args.max_episode_steps + 1) // 2

        # These two complete episode schedules intentionally override the
        # diagonal rules, as requested.
        if episode == middle_episode:
            if step < first_half_steps:
                return (
                    agent.percentile_tail_action(state, True, key),
                    "median_50_middle_first_half_u80",
                )
            return (
                agent.percentile_tail_action(state, False, key),
                "median_50_middle_second_half_l20",
            )
        if episode == next_middle_episode and next_middle_episode != middle_episode:
            if step < first_half_steps:
                return (
                    agent.percentile_tail_action(state, False, key),
                    "median_50_next_middle_first_half_l20",
                )
            return (
                agent.percentile_tail_action(state, True, key),
                "median_50_next_middle_second_half_u80",
            )

        # Scale the rectangular episode/step grid using integer arithmetic.
        # This keeps classification O(1) without allocating a matrix.
        episode_scale = args.max_episode_steps - 1
        step_scale = args.train_episodes - 1
        scaled_episode = episode * episode_scale
        scaled_step = step * step_scale
        main_delta = scaled_episode - scaled_step
        anti_delta = (
            scaled_episode + scaled_step - episode_scale * step_scale
        )
        if anti_delta == 0:
            return (
                agent.lowest_q_action(state, key),
                "median_50_min_q_antidiagonal",
            )
        if main_delta == 0:
            return (
                agent.greedy_action(state, key),
                "median_50_max_q_diagonal",
            )

        if episode < greedy_episode_count:
            return (
                agent.greedy_action(state, key),
                "median_50_first_half_max_q",
            )

        action, branch = agent.epsilon_greedy_action(
            state, args.epsilon, key
        )
        return action, f"median_50_second_half_epsilon_{branch}"
    raise ValueError(f"Unknown experiment: {experiment}")


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
    episode_start: float,
    cpu_start: float,
    agent: DQNAgent,
    losses: Sequence[float] = (),
    action_sources: Optional[Dict[str, int]] = None,
) -> Dict:
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
        "termination_reason": parsed["termination_reason"],
        "collision": bool(parsed["collision"]),
        "crash_vehicle": bool(parsed["crash_vehicle"]),
        "crash_object": bool(parsed["crash_object"]),
        "out_of_road": bool(parsed["out_of_road"]),
        "goal_reached": bool(parsed["goal_reached"]),
        "max_steps_reached": bool(parsed["max_steps_reached"]),
        "rmst_event_definition": "collision",
        "rmst_event_observed": bool(parsed["collision"]),
        "event_or_censor_time_steps": int(steps),
        "wall_time_seconds": time.perf_counter() - episode_start,
        "cpu_time_seconds": time.process_time() - cpu_start,
        "average_loss": avg(losses),
        "average_rnd_loss": 0.0,
        "average_rnd_bonus": 0.0,
        "replay_buffer_size": len(agent.replay),
        "learn_steps": agent.learn_steps,
        "epsilon": args.epsilon,
        "rnd_beta": 0.0,
        "noisy_sigma_init": 0.0,
        "network_frozen": phase == "test",
        "updates_during_test": 0 if phase == "test" else "",
        "action_source_counts": json.dumps(action_sources or {}, sort_keys=True),
    }


def run_experiment(
    experiment: str, args, device: torch.device, output_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    set_seed(args.seed, args.deterministic)
    rows: List[Dict] = []

    train_env = make_env(args, "train")
    initial_observation, _ = train_env.reset(seed=args.seed)
    observation_size = int(flatten_observation(initial_observation).size)
    if not hasattr(train_env.action_space, "n"):
        train_env.close()
        raise RuntimeError("MetaDrive action space is not Discrete; check discrete_action config.")
    action_count = int(train_env.action_space.n)
    expected_actions = int(
        args.discrete_steering_dim * args.discrete_throttle_dim
    )
    if action_count != expected_actions or action_count != 9:
        train_env.close()
        raise RuntimeError(
            "Canonical comparisons require exactly nine discrete actions; "
            f"configured={expected_actions}, exposed={action_count}."
        )
    agent = DQNAgent(observation_size, action_count, args, device)

    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.perf_counter()
    training_cpu_start = time.process_time()
    try:
        for episode in range(args.train_episodes):
            episode_start = time.perf_counter()
            cpu_start = time.process_time()
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            state = flatten_observation(state_raw)
            initial_hash = observation_sha256(state_raw)
            env_reward_total = 0.0
            training_reward_total = 0.0
            losses: List[float] = []
            action_sources: Dict[str, int] = {}
            parsed = parse_step_info({}, False, False)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for step in range(args.max_episode_steps):
                action, source = select_training_action(
                    experiment, agent, state, episode, step, args
                )
                action_sources[source] = action_sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                episode_done = bool(terminated or truncated)
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
                episode_start,
                cpu_start,
                agent,
                losses,
                action_sources,
            )
            rows.append(row)
            if (episode + 1) % args.progress_every == 0:
                print(
                    f"TRAIN {SHORT_LABELS[experiment]:16s} ep={episode + 1:03d} "
                    f"reward={env_reward_total:9.3f} steps={step + 1:3d} "
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
                episode_start = time.perf_counter()
                cpu_start = time.process_time()
                scenario_seed = args.test_seed + episode
                state_raw, _ = test_env.reset(seed=scenario_seed)
                state = flatten_observation(state_raw)
                initial_hash = observation_sha256(state_raw)
                reward_total = 0.0
                parsed = parse_step_info({}, False, False)
                for step in range(args.max_episode_steps):
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
                    episode_start,
                    cpu_start,
                    agent,
                    action_sources={"frozen_argmax": step + 1},
                )
                rows.append(row)
                if (episode + 1) % args.progress_every == 0:
                    print(
                        f"TEST  {SHORT_LABELS[experiment]:16s} "
                        f"ep={episode + 1:03d} reward={reward_total:9.3f} "
                        f"steps={step + 1:3d} "
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
    for phase, wall_seconds, cpu_seconds in (
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
                "phase_wall_time_seconds": float(wall_seconds),
                "phase_cpu_time_seconds": float(cpu_seconds),
                "summed_episode_wall_time_seconds": float(
                    sum(row["wall_time_seconds"] for row in phase_rows)
                ),
                "summed_episode_cpu_time_seconds": float(
                    sum(row["cpu_time_seconds"] for row in phase_rows)
                ),
                "average_wall_time_seconds_per_episode": (
                    float(sum(row["wall_time_seconds"] for row in phase_rows))
                    / len(phase_rows)
                    if phase_rows else math.nan
                ),
                "average_cpu_time_seconds_per_episode": (
                    float(sum(row["cpu_time_seconds"] for row in phase_rows))
                    / len(phase_rows)
                    if phase_rows else math.nan
                ),
            }
        )
    return rows, runtime_rows


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
                "RMST_event_definition": "collision",
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


def prepare_output_directory(path: Path, force: bool) -> None:
    populated = path.exists() and any(path.iterdir())
    if populated and not force:
        raise FileExistsError(
            f"Policy result already exists: {path}. Reuse it or pass --force."
        )
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(parents=True, exist_ok=True)


def update_baseline_index(seed_root: Path, manifest: Dict, args) -> None:
    """Merge this policy into the seed-level comparison index."""
    index_path = seed_root / "baseline_index.json"
    index: Dict = {}
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                index = loaded
        except (OSError, json.JSONDecodeError):
            index = {}
    existing_methods = index.get("methods", [])
    if not isinstance(existing_methods, list):
        existing_methods = []
    methods = [
        item for item in existing_methods
        if not isinstance(item, dict) or item.get("method") != POLICY_NAME
    ]
    methods.append(manifest)
    index.update(
        {
            "seed": args.seed,
            "environment": "MetaDrive",
            "observation_source": OBSERVATION_SOURCE,
            "uses_engine_object_safety_scan": False,
            "methods": methods,
            "critical_config": critical_config(args),
            "critical_config_sha256": canonical_json_sha256(critical_config(args)),
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(index_path)


def save_outputs(
    rows: List[Dict], runtime_rows: List[Dict], args, output_dir: Path
) -> Dict:
    if tuple(rows[0].keys()) != CANONICAL_EPISODE_COLUMNS:
        raise RuntimeError("Episode result columns do not match the canonical baseline.")

    results_path = output_dir / "all_episode_results.csv"
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "collision_metrics.csv"
    runtime_path = output_dir / "runtime_statistics.csv"
    write_csv(results_path, rows)
    write_csv(runtime_path, runtime_rows)

    config = json_safe(vars(args).copy())
    config.update(
        {
            "method": POLICY_NAME,
            "method_label": SHORT_LABELS[EXPERIMENTS[0]],
            "observation_source": OBSERVATION_SOURCE,
            "uses_engine_object_safety_scan": False,
            "test_policy": TEST_POLICY,
            "training_schedule": (
                "middle U80-to-L20; next-middle L20-to-U80; "
                "anti-diagonal min-Q; main diagonal max-Q; first episode half "
                "max-Q; second episode half epsilon-greedy"
            ),
        }
    )
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    metric_rows = []
    for phase in ("train", "test"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        metric_rows.append(
            {
                "method": POLICY_NAME,
                "method_label": SHORT_LABELS[EXPERIMENTS[0]],
                "phase": phase,
                **collision_summary(phase_rows, args.rmst_tau),
            }
        )
    write_csv(metrics_path, metric_rows)

    nested_model_path = output_dir / "models" / f"{EXPERIMENTS[0]}_model.pt"
    model_path = output_dir / "model.pt"
    if not nested_model_path.is_file():
        raise FileNotFoundError(f"Trained model is missing: {nested_model_path}")
    shutil.copy2(nested_model_path, model_path)

    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": getattr(metadrive, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "seed": args.seed,
        "method": POLICY_NAME,
        "method_label": SHORT_LABELS[EXPERIMENTS[0]],
        "observation_source": OBSERVATION_SOURCE,
        "uses_engine_object_safety_scan": False,
        "test_policy": TEST_POLICY,
        "model_selection": "final_training_episode",
        "critical_config": critical_config(args),
        "critical_config_sha256": canonical_json_sha256(critical_config(args)),
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
        "results_sha256": sha256_file(results_path),
        "metrics_sha256": sha256_file(metrics_path),
        "runtime_statistics_sha256": sha256_file(runtime_path),
        "phase_runtime": runtime_rows,
        "created_at_unix": time.time(),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    update_baseline_index(output_dir.parent, manifest, args)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaDrive Chapter 6B: Median 50 only"
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
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Exploration probability in the final episode block.",
    )
    parser.add_argument(
        "--greedy-episode-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of Chapter 6B training episodes that use maximum-Q at "
            "non-diagonal positions; the remainder use epsilon-greedy actions"
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
        "--seed",
        type=int,
        required=True,
        help="Training seed; also used in the canonical output folder name.",
    )
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument(
        "--rmst-tau",
        type=int,
        default=500,
        help="Restriction horizon in steps.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("policy_results"),
        help="Root containing seed_<seed>/ch6b.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional runner-supplied path. It must equal "
            "<project>/policy_results/seed_<seed>/ch6b."
        ),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    positive = {
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "target_update_steps": args.target_update_steps,
        "hidden_size": args.hidden_size,
        "discrete_steering_dim": args.discrete_steering_dim,
        "discrete_throttle_dim": args.discrete_throttle_dim,
        "map_blocks": args.map_blocks,
        "rmst_tau": args.rmst_tau,
        "progress_every": args.progress_every,
        "convergence_window": args.convergence_window,
    }
    invalid = [
        name for name, value in positive.items()
        if not math.isfinite(float(value)) or value <= 0
    ]
    if invalid:
        raise ValueError(
            "These arguments must be finite and positive: " + ", ".join(invalid)
        )
    if args.train_episodes < 2 or args.max_episode_steps < 2:
        raise ValueError(
            "The middle/diagonal schedule requires at least two training "
            "episodes and two maximum episode steps."
        )
    train_start, train_end = args.seed, args.seed + args.train_episodes - 1
    test_start, test_end = args.test_seed, args.test_seed + args.test_episodes - 1
    if max(train_start, test_start) <= min(train_end, test_end):
        raise ValueError("Testing seed range must not overlap the training seed range.")
    if not 0.0 <= args.greedy_episode_fraction <= 1.0:
        raise ValueError("--greedy-episode-fraction must be between 0 and 1.")
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("--epsilon must be between 0 and 1.")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("--gamma must be between 0 and 1.")
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1.")
    if not 0.0 <= args.accident_prob <= 1.0:
        raise ValueError("--accident-prob must be between 0 and 1.")
    if not math.isfinite(args.success_reward):
        raise ValueError("--success-reward must be finite.")
    if (
        not math.isfinite(args.collision_penalty)
        or not math.isfinite(args.out_of_road_penalty)
        or args.collision_penalty < 0
        or args.out_of_road_penalty < 0
    ):
        raise ValueError("MetaDrive penalty arguments must be non-negative magnitudes.")
    if args.replay_capacity < args.batch_size:
        raise ValueError("--replay-capacity must be at least --batch-size.")
    if args.discrete_steering_dim * args.discrete_throttle_dim != 9:
        raise ValueError("Canonical comparisons require exactly nine discrete actions.")


def resolve_output_dir(args) -> Path:
    """Return the only permitted seed-specific policy output directory."""
    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / f"seed_{args.seed}"
        / POLICY_NAME
    )
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
    validate_args(args)
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        print(
            f"WARNING: launch with PYTHONHASHSEED={args.seed} for complete process reproducibility",
            file=sys.stderr,
        )
    set_seed(args.seed, args.deterministic)
    output_dir = resolve_output_dir(args)
    prepare_output_directory(output_dir, args.force)
    args.output_dir = str(output_dir)
    device = choose_device(args.device)
    args.device = str(device)

    print("=" * 76)
    print("METADRIVE CHAPTER 6B MEDIAN 50 DQN POLICY")
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
    greedy_episode_count = int(
        math.ceil(args.train_episodes * args.greedy_episode_fraction)
    )
    middle_episode = max(0, greedy_episode_count - 1)
    print("Policy name/folder:", POLICY_NAME)
    print("Middle episode:", middle_episode + 1, "uses U80 then L20")
    print("Next-to-middle episode:", middle_episode + 2, "uses L20 then U80")
    print("Outside middle overrides: anti-diagonal min-Q, main diagonal max-Q")
    print("Maximum-Q episode block: 1 through", greedy_episode_count)
    print(
        f"Epsilon-greedy (epsilon={args.epsilon:g}) episode block:",
        greedy_episode_count + 1,
        "through",
        args.train_episodes,
    )
    print("RMST event/tau: collision", args.rmst_tau)
    print("=" * 76)

    all_rows: List[Dict] = []
    runtimes: List[Dict] = []
    for experiment in EXPERIMENTS:
        experiment_rows, runtime_rows = run_experiment(experiment, args, device, output_dir)
        all_rows.extend(experiment_rows)
        runtimes.extend(runtime_rows)
    save_outputs(all_rows, runtimes, args, output_dir)
    required_outputs = (
        output_dir / "all_episode_results.csv",
        output_dir / "collision_metrics.csv",
        output_dir / "runtime_statistics.csv",
        output_dir / "config.json",
        output_dir / "model.pt",
        output_dir / "manifest.json",
    )
    missing_outputs = [path for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            "Experiment finished but required outputs are missing: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    print("\nExperiment completed successfully.")
    print("Episode results:", output_dir / "all_episode_results.csv")
    print("Collision metrics:", output_dir / "collision_metrics.csv")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
