#!/usr/bin/env python3
"""MetaDrive genmax DQN experiment.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-15

This is the MetaDrive counterpart of the supplied HighwayEnv experiment.

Shared setup
------------
* Separate DQN, target network, replay buffer, and Adam optimizer per method.
* Neither method uses RND, a count bonus, or intrinsic rewards.
* Frozen argmax testing: no optimizer, replay, or target updates.
* Exact maximum-Q ties use the same reproducible uniform rule for both methods;
  tie occurrences are written to ``action_source_counts``.
* Disjoint training and testing scenarios.
* Defaults: 500 train episodes, 300 test episodes, 500 maximum steps.
* genmax uses upper-tail exploration in the lower-middle episode and
  lower-tail exploration in the following episode. At every other ordinary
  position, the first 50% of episodes use maximum-Q and the remaining 50% use
  80% maximum-Q and 20% uniformly random discrete actions. For even E, the lower middle is E/2;
  for odd E, it is (E + 1)/2 and the first block is rounded up.
* Lower-tail L20 samples uniformly from actions below the 20th Q-value
  percentile, falling back to minimum-Q if that set is empty. Upper-tail U80
  samples uniformly from actions above the 80th percentile, falling back to
  maximum-Q if that set is empty.
* Main-diagonal positions use maximum-Q and anti-diagonal positions use
  minimum-Q. Their intersection uses minimum-Q, matching Chapter 6B.

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
* figures_ieee/*.png and *.pdf
* plots/*.jpeg (generated automatically after the experiment)

Install:
    python -m pip install metadrive-simulator gymnasium torch numpy pandas matplotlib psutil

Example (when this file is placed in the project's ``policies`` folder):
    python policies/metadrive_genmax.py \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda \
      --seed 11

If ``--output-dir`` is omitted, results are saved automatically to
``policy_results/seed_<seed>/genmax`` under the project root. If
``--output-dir`` is supplied by a runner, it must resolve to that same
canonical folder; legacy or arbitrary output folders are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
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


EXPERIMENTS = ["genmax"]
SHORT_LABELS = {
    "genmax": "genmax",
}
COLORS = {
    "genmax": "#2ca02c",
}

PLOT_LABELS = {
    "genmax": "genmax",
}
PLOT_COLORS = {
    "genmax": "#2CA02C",
}
PLOT_ROLLING_WINDOW = 10
PLOT_CONVERGENCE_FRACTION = 0.95
PLOT_BOOTSTRAP_SAMPLES = 20_000
PLOT_RANDOM_SEED = 20260714
PLOT_FILES = (
    "01_mean_frozen_test_reward.jpeg",
    "02_iqm_frozen_test_reward.jpeg",
    "03_metadrive_rmst_collision_free_survival.jpeg",
    "04_time_to_shared_95pct_target.jpeg",
    "05_frozen_test_reward_boxplot.jpeg",
)


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
    """IQM using linear-time partitioning instead of a full sort."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return 0.0
    trim = int(math.floor(0.25 * data.size))
    if trim == 0:
        return float(np.mean(data))
    upper = data.size - trim
    partitioned = np.partition(data, (trim, upper - 1))
    middle = partitioned[trim:upper]
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
    """Kaplan-Meier RMST in O(N + tau) for integer environment steps.

    ``times`` contains event or censoring times. ``events`` is True only when
    the selected failure event occurred. Goal and horizon endings are censored.
    """
    tau_steps = int(tau)
    if tau_steps <= 0 or len(times) == 0:
        return 0.0
    if len(times) != len(events):
        raise ValueError("RMST times and events must have equal lengths.")

    raw_times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(raw_times)):
        raise ValueError("RMST times must be finite.")
    rounded_times = np.rint(raw_times).astype(int)
    t = np.clip(rounded_times, 0, tau_steps)
    e = np.asarray(events, dtype=bool) & (raw_times <= tau_steps)
    all_counts = np.bincount(t, minlength=tau_steps + 1)
    event_counts = np.bincount(t[e], minlength=tau_steps + 1)

    survival = 1.0
    area = 0.0
    previous = 0
    at_risk = len(t)
    for current in np.flatnonzero(all_counts):
        current = int(current)
        area += survival * max(0, current - previous)
        failures = int(event_counts[current])
        if at_risk > 0 and failures > 0:
            survival *= 1.0 - failures / at_risk
        at_risk -= int(all_counts[current])
        previous = current
    area += survival * max(0, tau_steps - previous)
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
# DQN and replay
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
        if q.size == 0:
            raise RuntimeError("DQN produced an empty Q-value vector.")
        if not np.all(np.isfinite(q)):
            raise RuntimeError("DQN produced a non-finite Q-value.")
        return q

    @staticmethod
    def random_extreme_index(q: np.ndarray, find_maximum: bool) -> int:
        """Select uniformly among tied extrema in O(A) time and O(1) space."""
        best_value = float(q[0])
        selected = 0
        tie_count = 1
        for index in range(1, len(q)):
            value = float(q[index])
            is_better = value > best_value if find_maximum else value < best_value
            if is_better:
                best_value = value
                selected = index
                tie_count = 1
            elif value == best_value:
                tie_count += 1
                if random.randrange(tie_count) == 0:
                    selected = index
        return int(selected)

    def greedy_action(self, state: np.ndarray) -> int:
        q = self.q_values(state)
        return self.random_extreme_index(q, find_maximum=True)

    def reproducible_argmax_action(
        self, state: np.ndarray, tie_key: str
    ) -> Tuple[int, int]:
        """Return argmax and exact-max tie count without index-order bias."""
        q = self.q_values(state)
        best = np.flatnonzero(q == np.max(q))
        tie_count = int(best.size)
        if tie_count == 1:
            return int(best[0]), tie_count
        digest = hashlib.sha256(tie_key.encode("utf-8")).digest()
        tie_rng = random.Random(int.from_bytes(digest, byteorder="big"))
        return int(tie_rng.choice(best.tolist())), tie_count

    def lowest_q_action(self, state: np.ndarray) -> int:
        """Choose the action with the lowest predicted long-term return."""
        q = self.q_values(state)
        return self.random_extreme_index(q, find_maximum=False)

    def random_percentile_tail_action(
        self,
        state: np.ndarray,
        percentile: float,
        choose_upper: bool,
    ) -> int:
        """Uniformly sample a strict Q-value percentile tail.

        Percentile selection and the eligibility scan are O(A). Reservoir
        sampling uses O(1) candidate memory beyond the Q-value vector.
        """
        q = self.q_values(state)
        threshold = float(np.percentile(q, percentile))
        selected = -1
        eligible_count = 0

        for index, raw_value in enumerate(q):
            value = float(raw_value)
            eligible = (
                value > threshold
                if choose_upper
                else value < threshold
            )
            if eligible:
                eligible_count += 1
                if random.randrange(eligible_count) == 0:
                    selected = index

        # A strict tail can be empty, for example when every Q-value is equal.
        # U80 then uses argmax and L20 uses argmin, with uniform tie-breaking.
        if selected < 0:
            return self.random_extreme_index(
                q, find_maximum=choose_upper
            )
        return int(selected)

    def upper_80_action(self, state: np.ndarray) -> int:
        """Choose uniformly above P80; fall back to maximum-Q."""
        return self.random_percentile_tail_action(
            state, percentile=80.0, choose_upper=True
        )

    def lower_20_action(self, state: np.ndarray) -> int:
        """Choose uniformly below P20; fall back to minimum-Q."""
        return self.random_percentile_tail_action(
            state, percentile=20.0, choose_upper=False
        )

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


def middle_episode_distribution_role(
    episode_number: int,
    step_number: int,
    episode_count: int,
    step_count: int,
) -> str:
    """Return max/min/U80/L20/epsilon20 for one E x S schedule position.

    Classification is O(1) time and O(1) auxiliary memory per cell.
    """
    if not 1 <= episode_number <= episode_count:
        raise ValueError("episode_number is outside the distribution matrix.")
    if not 1 <= step_number <= step_count:
        raise ValueError("step_number is outside the distribution matrix.")

    is_main_diagonal = episode_number == step_number
    is_antidiagonal = episode_number + step_number == step_count + 1

    # Chapter 6B precedence: minimum-Q wins at a diagonal intersection.
    if is_main_diagonal and is_antidiagonal:
        return "min"
    if is_antidiagonal:
        return "min"
    if is_main_diagonal:
        return "max"

    # Split the two middle rows by step halves.
    # Middle row: first 50% U80, remaining 50% L20.
    # Next-to-middle row: first 50% L20, remaining 50% U80.
    middle_episode = (episode_count + 1) // 2
    first_half_steps = (step_count + 1) // 2
    if episode_number == middle_episode:
        return "upper_80" if step_number <= first_half_steps else "lower_20"
    if episode_number == middle_episode + 1:
        return "lower_20" if step_number <= first_half_steps else "upper_80"

    # Match Chapter 6B at all remaining positions: the first 50% of episode
    # rows use 80% maximum-Q and 20% uniform random actions.
    greedy_episode_count = int(math.ceil(episode_count * 0.5))
    if episode_number <= greedy_episode_count:
        return "max"
    return "epsilon20"


def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    episode: int,
    step: int,
    args,
) -> Tuple[int, str]:
    if experiment == "genmax":
        episode_number = episode + 1
        step_number = step + 1
        role = middle_episode_distribution_role(
            episode_number,
            step_number,
            args.train_episodes,
            args.max_episode_steps,
        )

        if role == "max":
            return agent.greedy_action(state), "genmax_max"
        if role == "min":
            return agent.lowest_q_action(state), "genmax_min"
        if role == "upper_80":
            return (
                agent.upper_80_action(state),
                "genmax_upper_80",
            )
        if role == "lower_20":
            return (
                agent.lower_20_action(state),
                "genmax_lower_20",
            )
        if role == "epsilon20":
            if random.random() < 0.20:
                return (
                    random.randrange(agent.action_count),
                    "genmax_random20",
                )
            return agent.greedy_action(state), "genmax_max80"
        raise RuntimeError(f"Unknown genmax schedule role: {role}")

    raise ValueError(f"Unknown experiment: {experiment}")


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
        "uses_rnd": False,
        "uses_count_bonus": False,
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
                action, source = select_training_action(
                    experiment, agent, state, episode, step, args
                )
                action_sources[source] = action_sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                done = bool(terminated or truncated)
                full_reward = float(env_reward)
                agent.replay.add(state, action, full_reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                env_reward_total += float(env_reward)
                training_reward_total += float(full_reward)
                state = next_state
                parsed = parse_step_info(info, bool(terminated), bool(truncated))
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

    training_duration = time.time() - training_start
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
                test_action_sources: Dict[str, int] = {}
                parsed = parse_step_info({}, False, False)
                episode_start = time.time()
                cpu_start = time.process_time()
                for step in range(args.max_episode_steps):
                    tie_key = f"{args.seed}|{scenario_seed}|{episode}|{step}"
                    action, tie_count = agent.reproducible_argmax_action(
                        state, tie_key
                    )
                    source = (
                        "frozen_argmax_exact_tie"
                        if tie_count > 1
                        else "frozen_argmax_unique"
                    )
                    test_action_sources[source] = (
                        test_action_sources.get(source, 0) + 1
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
                    reward_total,
                    reward_total,
                    step + 1,
                    parsed,
                    args,
                    device,
                    episode_start,
                    cpu_start,
                    agent,
                    action_sources=test_action_sources,
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
    figure_dir = output_dir / "figures_ieee"
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
    boxplot_options = {"showmeans": True, "patch_artist": True}
    try:
        box = ax.boxplot(groups, tick_labels=labels, **boxplot_options)
    except TypeError:
        box = ax.boxplot(groups, labels=labels, **boxplot_options)
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


def configure_chapter_plot_style(dpi: int) -> None:
    """IEEE-sized styling used by the standalone Chapter-9 plotter."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": dpi,
        }
    )


def save_chapter_plot(
    fig: plt.Figure, plot_dir: Path, filename: str, dpi: int
) -> None:
    fig.tight_layout(pad=0.6)
    fig.savefig(
        plot_dir / filename,
        format="jpeg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def set_chapter_ylim(
    ax: plt.Axes, values: Iterable[float], top_extra: float = 0.20
) -> None:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return
    low = min(0.0, float(array.min()))
    high = max(0.0, float(array.max()))
    span = max(high - low, abs(high), abs(low), 1.0)
    ax.set_ylim(low - 0.08 * span, high + top_extra * span)


def add_chapter_bar_labels(
    ax: plt.Axes, bars, values: Iterable[float]
) -> None:
    values = list(values)
    y_low, y_high = ax.get_ylim()
    offset = 0.012 * (y_high - y_low)
    for bar, value in zip(bars, values):
        positive = value >= 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if positive else -offset),
            f"{value:.2f}",
            ha="center",
            va="bottom" if positive else "top",
            fontsize=7,
        )


def bootstrap_iqm_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int = PLOT_BOOTSTRAP_SAMPLES,
    batch_size: int = 1000,
) -> Tuple[float, float]:
    """Batched percentile-bootstrap CI with bounded temporary memory."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    if count == 0:
        return math.nan, math.nan
    lower = int(math.floor(0.25 * count))
    upper = int(math.ceil(0.75 * count))
    distribution = np.empty(samples, dtype=float)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, count, size=(stop - start, count))
        sampled = values[indices]
        if lower == 0 and upper == count:
            distribution[start:stop] = sampled.mean(axis=1)
        else:
            partitioned = np.partition(
                sampled, (lower, upper - 1), axis=1
            )
            distribution[start:stop] = partitioned[:, lower:upper].mean(axis=1)
    ci = np.quantile(distribution, [0.025, 0.975])
    return float(ci[0]), float(ci[1])


def shared_convergence_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[float, Dict[str, Dict]]:
    test_means = test_df.groupby("experiment")["env_reward"].mean().reindex(EXPERIMENTS)
    target = PLOT_CONVERGENCE_FRACTION * float(test_means.max())
    metrics: Dict[str, Dict] = {}
    for experiment in EXPERIMENTS:
        group = train_df[train_df["experiment"].eq(experiment)].sort_values("episode").copy()
        group["rolling_reward"] = group["env_reward"].rolling(
            PLOT_ROLLING_WINDOW, min_periods=PLOT_ROLLING_WINDOW
        ).mean()
        crossed = group[group["rolling_reward"].ge(target)]
        reached = not crossed.empty
        if reached:
            position = int(group.index.get_loc(crossed.index[0]))
            observed = group.iloc[: position + 1]
            episode = int(observed.iloc[-1]["episode"]) + 1
        else:
            observed = group
            episode = None
        metrics[experiment] = {
            "reached": reached,
            "episode": episode,
            "seconds": float(observed["wall_time_seconds"].fillna(0).sum()),
            "interactions": int(observed["steps"].fillna(0).sum()),
        }
    return target, metrics


def make_chapter_plots(
    rows: List[Dict], summary: List[Dict], args, output_dir: Path
) -> None:
    """Create the five Chapter-9-style JPEGs automatically after the run."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for filename in PLOT_FILES:
        path = plot_dir / filename
        if path.exists():
            path.unlink()

    configure_chapter_plot_style(args.plot_dpi)
    train_df = pd.DataFrame([row for row in rows if row["phase"] == "train"])
    test_df = pd.DataFrame([row for row in rows if row["phase"] == "test"])
    summary_by_experiment = {row["experiment"]: row for row in summary}
    labels = [PLOT_LABELS[experiment] for experiment in EXPERIMENTS]
    colors = [PLOT_COLORS[experiment] for experiment in EXPERIMENTS]
    x = np.arange(len(EXPERIMENTS))

    # 1. Mean frozen-test reward.
    means = [
        float(test_df.loc[test_df["experiment"].eq(experiment), "env_reward"].mean())
        for experiment in EXPERIMENTS
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(x, means, width=0.58, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_title("Mean Frozen-Test Reward (Higher Is Better)")
    ax.set_ylabel("Mean episode reward")
    ax.set_xticks(x, labels)
    set_chapter_ylim(ax, means)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    add_chapter_bar_labels(ax, bars, means)
    save_chapter_plot(fig, plot_dir, PLOT_FILES[0], args.plot_dpi)

    # 2. IQM with bootstrap confidence intervals.
    rng = np.random.default_rng(PLOT_RANDOM_SEED)
    iqm_values: List[float] = []
    iqm_intervals: List[Tuple[float, float]] = []
    for experiment in EXPERIMENTS:
        rewards = test_df.loc[
            test_df["experiment"].eq(experiment), "env_reward"
        ].to_numpy(dtype=float)
        iqm_values.append(interquartile_mean(rewards))
        iqm_intervals.append(bootstrap_iqm_ci(rewards, rng))
    heights = np.asarray(iqm_values, dtype=float)
    lower_error = np.maximum(
        0.0, heights - np.asarray([interval[0] for interval in iqm_intervals])
    )
    upper_error = np.maximum(
        0.0, np.asarray([interval[1] for interval in iqm_intervals]) - heights
    )
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(
        x,
        heights,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        yerr=np.vstack([lower_error, upper_error]),
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.set_title("IQM Frozen-Test Reward")
    ax.set_ylabel("Interquartile mean reward")
    ax.set_xticks(x, labels)
    set_chapter_ylim(
        ax,
        np.concatenate([heights - lower_error, heights + upper_error]),
        0.28,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    add_chapter_bar_labels(ax, bars, heights)
    save_chapter_plot(fig, plot_dir, PLOT_FILES[1], args.plot_dpi)

    # 3. Median-50 collision-free RMST.
    median_rmst = float(summary_by_experiment["genmax"]["RMST_steps"])
    rmst_values = [median_rmst]
    rmst_labels = ["genmax RMST"]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    rmst_bars = ax.bar(
        np.arange(1),
        rmst_values,
        width=0.58,
        color=[PLOT_COLORS["genmax"]],
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_title("MetaDrive Restricted Mean Collision-Free Survival")
    ax.set_ylabel(f"RMST (steps; {args.rmst_tau}-step horizon)")
    ax.set_xticks(np.arange(1), rmst_labels)
    set_chapter_ylim(ax, rmst_values, 0.28)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    add_chapter_bar_labels(ax, rmst_bars, rmst_values)
    save_chapter_plot(fig, plot_dir, PLOT_FILES[2], args.plot_dpi)

    # 4. Cumulative training time to 95% of the frozen-test mean reward.
    shared_target, convergence = shared_convergence_metrics(train_df, test_df)
    convergence_values = [convergence[experiment]["seconds"] for experiment in EXPERIMENTS]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bars = ax.bar(
        x,
        convergence_values,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    set_chapter_ylim(ax, convergence_values, 0.32)
    for bar, experiment in zip(bars, EXPERIMENTS):
        record = convergence[experiment]
        if record["reached"]:
            label = f"{record['seconds']:.2f} s\nepisode {record['episode']}"
        else:
            bar.set_hatch("///")
            label = f">{record['seconds']:.2f} s\nnot reached"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.set_title("Training Time to 95% Frozen-Test Reward Target")
    ax.set_ylabel("Cumulative wall-clock time (s)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save_chapter_plot(fig, plot_dir, PLOT_FILES[3], args.plot_dpi)

    # 5. Frozen-test reward boxplot.
    reward_groups = [
        test_df.loc[test_df["experiment"].eq(experiment), "env_reward"].to_numpy(dtype=float)
        for experiment in EXPERIMENTS
    ]
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    boxplot_options = {
        "widths": 0.55,
        "patch_artist": True,
        "showfliers": True,
        "medianprops": {"color": "black", "linewidth": 1.2},
        "boxprops": {"linewidth": 0.8},
        "whiskerprops": {"linewidth": 0.8},
        "capprops": {"linewidth": 0.8},
        "flierprops": {"marker": "o", "markersize": 2.5, "alpha": 0.45},
    }
    try:
        box = ax.boxplot(reward_groups, tick_labels=labels, **boxplot_options)
    except TypeError:
        box = ax.boxplot(reward_groups, labels=labels, **boxplot_options)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    ax.set_title("Frozen-Test Reward Distribution")
    ax.set_ylabel("Episode reward")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    save_chapter_plot(fig, plot_dir, PLOT_FILES[4], args.plot_dpi)

    print(f"Created {len(PLOT_FILES)} JPEG plots in: {plot_dir}")
    for filename in PLOT_FILES:
        print(" ", filename)
    print(f"95% frozen-test reward target: {shared_target:.2f}")


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
    make_chapter_plots(rows, summary, args, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaDrive genmax policy"
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
        help="Training seed. It is also used in the automatic output folder name.",
    )
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=600,
        help="Resolution of JPEG files written to the plots directory.",
    )
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
            "<project>/policy_results/seed_<seed>/genmax."
        ),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    if args.train_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("Training and testing episodes must be positive.")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive.")
    if args.plot_dpi <= 0:
        raise ValueError("--plot-dpi must be positive.")
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
    """Return an absolute, seed-specific output directory.

    The normal installation is ``<project>/policies/metadrive_genmax.py``. For
    convenience, running the file directly from the project root is also
    supported.
    """
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    output_dir = (
        project_dir / "policy_results" / f"seed_{args.seed}" / "genmax"
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
    # Record the actual absolute destination in config.json.
    args.output_dir = str(output_dir)
    device = choose_device(args.device)

    print("=" * 76)
    print("METADRIVE DQN EXPERIMENT")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("MetaDrive:", getattr(metadrive, "__version__", "installed"))
    print("Device:", device)
    print("Experiments:", ", ".join(SHORT_LABELS[e] for e in EXPERIMENTS))
    print("Target network + replay + Adam per method: yes")
    print("RND active: no")
    print("Count bonus active: no")
    print("Frozen argmax testing: yes")
    print("Train/Test/Max steps:", args.train_episodes, args.test_episodes, args.max_episode_steps)
    print("Training seeds:", args.seed, "through", args.seed + args.train_episodes - 1)
    print("Testing seeds:", args.test_seed, "through", args.test_seed + args.test_episodes - 1)
    print("Discrete actions:", args.discrete_steering_dim * args.discrete_throttle_dim)
    print("Learning rate:", args.learning_rate)
    print("Output directory:", output_dir)
    print("genmax diagonal uses maximum-Q action: yes")
    print("genmax anti-diagonal uses lowest-Q action: yes")
    print("genmax diagonal intersection uses lowest-Q action: yes")
    print("genmax middle episode uses U80: yes")
    print("genmax next-to-middle episode uses L20: yes")
    print("genmax first-half ordinary positions use maximum-Q: yes")
    print("genmax remaining ordinary positions use 80% maximum-Q and 20% uniform random actions: yes")
    print("Frozen testing uses argmax only: yes")
    print("Frozen-test exact ties use shared reproducible uniform choice: yes")
    print("Training and testing scenarios are disjoint: yes")
    print("Automatic JPEG plots folder: yes")
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
    print("Plots saved to:", output_dir / "plots")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
