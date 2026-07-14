#!/usr/bin/env python3
"""MetaDrive Chapter 6B DQN experiment: NoisyNet versus block Max/Random.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-14 01:47:19 CEST (+0200)

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* No RND and no count-based intrinsic reward.
* No epsilon-greedy exploration. The comparison method uses factorized
  Gaussian NoisyNet parameter-space exploration.
* Frozen deterministic testing: noise is disabled and exact Q-value ties use
  a shared reproducible uniform selection rule; there are no optimizer,
  replay, or target updates.
* Disjoint training and testing scenarios. Traffic is deterministic for each
  reset seed, and exact initial-observation hashes are verified across methods.
* Defaults: 500 train episodes, 300 test episodes, 500 maximum steps.
* The Chapter 6B schedule forces minimum-Q on the anti-diagonal e+s=S+1 and maximum-Q on
  the main diagonal e=s. Minimum-Q wins if these rules overlap.
* For wide schedules (S>E), p=S-E: the first p steps of the final episode use
  minimum-Q and its last p steps use maximum-Q. Maximum-Q wins if these
  padding regions overlap. Square and tall schedules have p=0.
* At all other positions, the first 50% of training episodes use maximum-Q;
  the remaining 50% use a uniformly random discrete action.
* Positions after an early collision, termination, or truncation are not run.
* NoisyNet DQN is trained independently as the comparison baseline.

Primary test metrics
--------------------
* Mean R: mean environment return.
* Median R: median environment return.
* IQMR: interquartile mean reward (middle 50% of returns).
* VaR5/CVaR5: fifth-percentile reward and mean reward in the worst 5% tail.
* RMST: Kaplan-Meier restricted mean event-free survival time up to tau,
  where tau defaults to 500 steps. The event can be collision-only (default)
  or any safety failure (collision or off-road).
* Collision incidence per 1,000 frozen-test steps.
* Paired probability of longer test duration on identical scenario seeds;
  duration ties contribute one half to each method.

Main outputs
------------
* four_primary_test_metrics.csv
* extended_test_metrics.csv
* paired_test_duration_metrics.csv
* all_episode_results.csv
* all_experiments_train_episode_rewards.csv
* all_experiments_test_episode_rewards.csv
* all_experiments_learning_rate_summary.csv
* all_experiments_runtime_logs.csv
* initial_condition_records.csv
* initial_condition_verification.json
* models/*.pt
* plots/*.png, *.pdf, and *.jpeg, generated automatically by this file

Install:
    python -m pip install metadrive-simulator gymnasium torch numpy pandas matplotlib psutil

Example:
    python metadrivech6b_noisy.py \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda \
      --seed 27 --output-dir ../ch6b_noisy_results_27
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


EXPERIMENTS = ["noisy_dqn", "chapter6b"]
SHORT_LABELS = {
    "noisy_dqn": "NoisyNet",
    "chapter6b": "Chapter 6B",
}
COLORS = {
    "noisy_dqn": "#9467bd",
    "chapter6b": "#2ca02c",
}


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


def observation_sha256(observation) -> str:
    """Exact float32 fingerprint used for paired-reset verification."""
    array = np.ascontiguousarray(flatten_observation(observation))
    return hashlib.sha256(array.tobytes()).hexdigest()


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def interquartile_mean(values: Sequence[float]) -> float:
    """IQM via partitioning, avoiding a full O(n log n) sort."""
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
        # Pairing requires a reset seed to reproduce the same map and initial
        # traffic state regardless of the previous policy trajectory.
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


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer from Fortunato et al."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_noise", torch.empty(out_features, in_features))
        self.register_buffer("bias_noise", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        values = torch.randn(size, device=device)
        return values.sign() * values.abs().sqrt()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(
            self.weight_sigma, self.sigma_init / math.sqrt(self.in_features)
        )
        nn.init.constant_(
            self.bias_sigma, self.sigma_init / math.sqrt(self.out_features)
        )

    def reset_noise(self) -> None:
        noise_in = self._scale_noise(self.in_features, self.weight_mu.device)
        noise_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_noise.copy_(noise_out.outer(noise_in))
        self.bias_noise.copy_(noise_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_noise
            bias = self.bias_mu + self.bias_sigma * self.bias_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_size: int,
        noisy: bool = False,
        noisy_sigma_init: float = 0.5,
    ):
        super().__init__()
        self.noisy = bool(noisy)
        linear = (
            lambda in_size, out_size: NoisyLinear(
                in_size, out_size, noisy_sigma_init
            )
            if self.noisy
            else nn.Linear(in_size, out_size)
        )
        self.network = nn.Sequential(
            linear(observation_size, hidden_size),
            nn.ReLU(),
            linear(hidden_size, hidden_size),
            nn.ReLU(),
            linear(hidden_size, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def reset_noise(self) -> None:
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


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
        experiment: str,
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
        self.noisy = experiment == "noisy_dqn"

        self.online = QNetwork(
            observation_size,
            action_count,
            args.hidden_size,
            noisy=self.noisy,
            noisy_sigma_init=args.noisy_sigma_init,
        ).to(device)
        self.target = QNetwork(
            observation_size,
            action_count,
            args.hidden_size,
            noisy=self.noisy,
            noisy_sigma_init=args.noisy_sigma_init,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        if self.noisy:
            self.target.train()
        else:
            self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=args.learning_rate)
        self.replay = ReplayBuffer(args.replay_capacity)

    def tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def q_values(self, state: np.ndarray, use_noise: bool = False) -> np.ndarray:
        was_training = self.online.training
        if self.noisy and use_noise:
            self.online.train()
            self.online.reset_noise()
        else:
            self.online.eval()
        with torch.no_grad():
            q = self.online(self.tensor(state))[0].detach().cpu().numpy()
        if was_training:
            self.online.train()
        else:
            self.online.eval()
        return q.astype(float)

    @staticmethod
    def _uniform_tie_choice(candidates: np.ndarray, tie_key: str) -> int:
        """Choose uniformly from exact ties without global-RNG side effects."""
        if candidates.size == 0:
            raise RuntimeError("No candidate action was available.")
        digest = hashlib.sha256(str(tie_key).encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return int(candidates[offset % int(candidates.size)])

    def _extreme_action(
        self,
        state: np.ndarray,
        tie_key: str,
        maximize: bool,
        use_noise: bool = False,
    ) -> Tuple[int, int]:
        q = self.q_values(state, use_noise=use_noise)
        extreme = np.max(q) if maximize else np.min(q)
        candidates = np.flatnonzero(q == extreme)
        return self._uniform_tie_choice(candidates, tie_key), int(candidates.size)

    def reproducible_argmax_action(
        self, state: np.ndarray, tie_key: str, use_noise: bool = False
    ) -> Tuple[int, int]:
        return self._extreme_action(state, tie_key, True, use_noise)

    def reproducible_argmin_action(
        self, state: np.ndarray, tie_key: str
    ) -> Tuple[int, int]:
        return self._extreme_action(state, tie_key, False, False)

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
            if self.noisy:
                self.target.reset_noise()
            next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        if self.noisy:
            self.online.reset_noise()
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
                "noisy": self.noisy,
                "noisy_sigma_init": args.noisy_sigma_init,
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
    max_key = f"train|{args.seed}|{episode}|{step}|max"
    min_key = f"train|{args.seed}|{episode}|{step}|min"

    def maximum(source: str, use_noise: bool = False) -> Tuple[int, str]:
        action, ties = agent.reproducible_argmax_action(
            state, max_key, use_noise=use_noise
        )
        return action, source + ("_exact_tie" if ties > 1 else "")

    def minimum(source: str) -> Tuple[int, str]:
        action, ties = agent.reproducible_argmin_action(state, min_key)
        return action, source + ("_exact_tie" if ties > 1 else "")

    if experiment == "noisy_dqn":
        return maximum("noisy_argmax", use_noise=True)

    if experiment == "chapter6b":
        episode_number = episode + 1
        step_number = step + 1

        # Median-50 base geometry.
        is_diagonal_step = episode_number == step_number
        is_antidiagonal_step = (
            episode_number + step_number == args.max_episode_steps + 1
        )

        # Only wide E-by-S schedules receive final-row padding.
        padding = max(0, args.max_episode_steps - args.train_episodes)
        is_final_episode = episode_number == args.train_episodes
        is_min_q_padding = (
            is_final_episode and padding > 0 and step_number <= padding
        )
        is_max_q_padding = (
            is_final_episode
            and padding > 0
            and step_number > args.max_episode_steps - padding
        )

        # Maximum-Q wins when the two wide-matrix padding regions overlap.
        # The anti-/main-diagonal overlap still uses minimum-Q.
        if is_min_q_padding and is_max_q_padding:
            return maximum("chapter6b_max_q_padding_overlap")

        if is_antidiagonal_step or is_min_q_padding:
            return minimum("chapter6b_min_q_antidiagonal")

        if is_diagonal_step or is_max_q_padding:
            return maximum("chapter6b_max_q_diagonal")

        greedy_episode_count = int(
            math.ceil(args.train_episodes * args.greedy_episode_fraction)
        )
        if episode_number <= greedy_episode_count:
            return maximum("chapter6b_first_half_max_q")
        return random.randrange(agent.action_count), "chapter6b_second_half_random"
    raise ValueError(f"Unknown experiment: {experiment}")


def episode_row(
    phase: str,
    experiment: str,
    episode: int,
    scenario_seed: int,
    initial_observation_sha256: str,
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
        "initial_observation_sha256": initial_observation_sha256,
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
        "noisy_parameter_exploration": experiment == "noisy_dqn" and phase == "train",
        "noisy_sigma_init": args.noisy_sigma_init if experiment == "noisy_dqn" else 0.0,
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
    agent = DQNAgent(observation_size, action_count, experiment, args, device)

    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.time()
    try:
        for episode in range(args.train_episodes):
            scenario_seed = args.seed + episode
            state_raw, _ = train_env.reset(seed=scenario_seed)
            state = flatten_observation(state_raw)
            initial_state_hash = observation_sha256(state)
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
                environment_done = bool(terminated or truncated)
                reached_external_limit = step + 1 >= args.max_episode_steps
                replay_done = bool(environment_done or reached_external_limit)
                training_reward = float(env_reward)
                agent.replay.add(
                    state, action, training_reward, next_state, replay_done
                )
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                env_reward_total += float(env_reward)
                training_reward_total += training_reward
                state = next_state
                parsed = parse_step_info(
                    info,
                    bool(terminated),
                    bool(truncated or (reached_external_limit and not terminated)),
                )
                if replay_done:
                    break

            row = episode_row(
                "train",
                experiment,
                episode,
                scenario_seed,
                initial_state_hash,
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
                initial_state_hash = observation_sha256(state)
                reward_total = 0.0
                test_action_sources: Dict[str, int] = {}
                parsed = parse_step_info({}, False, False)
                episode_start = time.time()
                cpu_start = time.process_time()
                for step in range(args.max_episode_steps):
                    tie_key = (
                        f"test|{args.seed}|{scenario_seed}|{episode}|{step}|max"
                    )
                    action, ties = agent.reproducible_argmax_action(
                        state, tie_key
                    )
                    source = (
                        "frozen_argmax_exact_tie"
                        if ties > 1
                        else "frozen_argmax_unique"
                    )
                    test_action_sources[source] = (
                        test_action_sources.get(source, 0) + 1
                    )
                    next_raw, reward, terminated, truncated, info = test_env.step(action)
                    state = flatten_observation(next_raw)
                    reward_total += float(reward)
                    reached_external_limit = step + 1 >= args.max_episode_steps
                    parsed = parse_step_info(
                        info,
                        bool(terminated),
                        bool(truncated or (reached_external_limit and not terminated)),
                    )
                    if terminated or truncated or reached_external_limit:
                        break
                row = episode_row(
                    "test",
                    experiment,
                    episode,
                    scenario_seed,
                    initial_state_hash,
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
    return convergence_details(
        rewards, target_reward, threshold_fraction, window
    )[0]


def convergence_details(
    rewards: Sequence[float],
    target_reward: float,
    threshold_fraction: float,
    window: int,
) -> Tuple[int, bool]:
    if not rewards:
        return 0, False
    window = max(1, min(int(window), len(rewards)))
    rolling = np.convolve(
        np.asarray(rewards, dtype=float), np.ones(window) / window, mode="valid"
    )
    threshold = float(threshold_fraction) * float(target_reward)
    for index, value in enumerate(rolling):
        if float(value) >= threshold:
            return int(index + window), True
    return int(len(rewards)), False


def reward_var_cvar(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Return lower-tail reward VaR and CVaR at probability ``alpha``."""
    rewards = np.asarray(values, dtype=float)
    if rewards.size == 0:
        return 0.0, 0.0
    var_value = float(np.quantile(rewards, alpha))
    tail = rewards[rewards <= var_value]
    cvar_value = float(tail.mean()) if tail.size else var_value
    return var_value, cvar_value


def paired_duration_metrics(rows: List[Dict]) -> Dict[str, float]:
    """Compare test duration on identical scenario seeds with half-credit ties."""
    test_df = pd.DataFrame([row for row in rows if row["phase"] == "test"])
    paired = test_df.pivot_table(
        index="scenario_seed",
        columns="experiment",
        values="steps",
        aggfunc="first",
    ).dropna(subset=EXPERIMENTS)
    if paired.empty:
        raise RuntimeError("No paired frozen-test scenarios were available.")

    noisy_steps = paired["noisy_dqn"].to_numpy(dtype=float)
    chapter_steps = paired["chapter6b"].to_numpy(dtype=float)
    noisy_strict = float(np.mean(noisy_steps > chapter_steps))
    chapter_strict = float(np.mean(chapter_steps > noisy_steps))
    tie_probability = float(np.mean(noisy_steps == chapter_steps))
    return {
        "paired_scenarios": int(len(paired)),
        "noisy_dqn_strictly_longer_probability": noisy_strict,
        "chapter6b_strictly_longer_probability": chapter_strict,
        "equal_duration_probability": tie_probability,
        "noisy_dqn_longer_probability_ties_split": noisy_strict
        + 0.5 * tie_probability,
        "chapter6b_longer_probability_ties_split": chapter_strict
        + 0.5 * tie_probability,
    }


def make_summary(rows: List[Dict], args) -> List[Dict]:
    summary: List[Dict] = []
    test_means = {
        experiment: float(
            np.mean(
                [
                    float(row["env_reward"])
                    for row in rows
                    if row["experiment"] == experiment
                    and row["phase"] == "test"
                ]
            )
        )
        for experiment in EXPERIMENTS
    }
    shared_reference_reward = max(test_means.values())
    shared_convergence_target = (
        float(args.convergence_threshold_fraction) * shared_reference_reward
    )
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
        collision_events = [bool(row["collision"]) for row in test]
        safety_events = [
            bool(row["collision"] or row["out_of_road"]) for row in test
        ]
        mean_r = float(np.mean(test_rewards))
        median_r = float(np.median(test_rewards))
        iqmr = interquartile_mean(test_rewards)
        var5, cvar5 = reward_var_cvar(test_rewards, 0.05)
        rmst = restricted_mean_survival_time(test_times, test_events, args.rmst_tau)
        collision_rmst = restricted_mean_survival_time(
            test_times, collision_events, args.rmst_tau
        )
        safety_rmst = restricted_mean_survival_time(
            test_times, safety_events, args.rmst_tau
        )
        total_test_steps = int(sum(int(row["steps"]) for row in test))
        test_collision_count = int(sum(bool(row["collision"]) for row in test))
        collisions_per_1000_steps = (
            1000.0 * test_collision_count / total_test_steps
            if total_test_steps > 0
            else 0.0
        )
        conv_episode, convergence_reached = convergence_details(
            train_rewards,
            shared_reference_reward,
            args.convergence_threshold_fraction,
            args.convergence_window,
        )
        ordered_train = sorted(train, key=lambda row: int(row["episode"]))
        total_train_time = sum(
            float(row["wall_time_seconds"]) for row in ordered_train
        )
        observed_until_convergence = (
            ordered_train[:conv_episode]
            if convergence_reached
            else ordered_train
        )
        conv_time = sum(
            float(row["wall_time_seconds"])
            for row in observed_until_convergence
        )
        summary.append(
            {
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "learning_rate": args.learning_rate,
                "mean_R": mean_r,
                "median_R": median_r,
                "IQMR": iqmr,
                "VaR5_reward": var5,
                "CVaR5_reward": cvar5,
                "RMST_steps": rmst,
                "collision_RMST_steps": collision_rmst,
                "safety_failure_RMST_steps": safety_rmst,
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
                "convergence_reached": bool(convergence_reached),
                "shared_convergence_reference_reward": shared_reference_reward,
                "shared_convergence_target_reward": shared_convergence_target,
                "convergence_time_seconds": conv_time,
                "total_training_wall_time_seconds": total_train_time,
                "train_collision_rate": float(np.mean([bool(r["collision"]) for r in train])),
                "test_collision_rate": float(np.mean([bool(r["collision"]) for r in test])),
                "test_collision_count": test_collision_count,
                "total_test_steps": total_test_steps,
                "test_collisions_per_1000_steps": collisions_per_1000_steps,
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
    fig.savefig(
        figure_dir / f"{name}.jpeg",
        format="jpeg",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
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
    width = 0.25
    ax.bar(x - width, 100 * summary_df["test_collision_rate"], width, label="Collision", edgecolor="black")
    ax.bar(x, 100 * summary_df["test_off_road_rate"], width, label="Off-road", edgecolor="black")
    ax.bar(x + width, 100 * summary_df["test_goal_rate"], width, label="Goal", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Frozen-test episodes (%)")
    ax.set_title("MetaDrive Collision, Off-Road, and Goal Rates")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collision_goal_rates")

    # Dedicated IQM plot.
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    iqm_values = summary_df["IQMR"].to_numpy(dtype=float)
    bars = ax.bar(x, iqm_values, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Interquartile mean reward")
    ax.set_title("MetaDrive Frozen-Test IQM Reward")
    save_figure(fig, figure_dir, "ieee_iqm_reward")

    # Lower-tail reward risk: VaR 5% and CVaR 5%.
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    width = 0.34
    ax.bar(
        x - width / 2,
        summary_df["VaR5_reward"],
        width,
        label=r"VaR$_{5\%}$",
        color=colors,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        summary_df["CVaR5_reward"],
        width,
        label=r"CVaR$_{5\%}$",
        color=colors,
        alpha=0.55,
        edgecolor="black",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Worst-tail frozen-test reward")
    ax.set_title("MetaDrive Reward Tail Risk")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_reward_var5_cvar5")

    # Collision incidence normalized by exposure.
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    incidence = summary_df["test_collisions_per_1000_steps"].to_numpy(dtype=float)
    bars = ax.bar(x, incidence, color=colors, edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Collisions per 1,000 steps")
    ax.set_title("MetaDrive Frozen-Test Collision Incidence")
    save_figure(fig, figure_dir, "ieee_collisions_per_1000_steps")

    # Paired probability of longer duration on identical scenario seeds.
    paired = paired_duration_metrics(rows)
    paired_probabilities = 100.0 * np.asarray(
        [
            paired["noisy_dqn_longer_probability_ties_split"],
            paired["chapter6b_longer_probability_ties_split"],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    bars = ax.bar(
        x,
        paired_probabilities,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.axhline(50.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, max(100.0, float(paired_probabilities.max()) * 1.12))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Paired probability of longer duration (%)")
    ax.set_title("MetaDrive Paired Longer-Duration Probability")
    save_figure(fig, figure_dir, "ieee_paired_longer_duration_probability")

    # Frozen-test episode-duration distribution.
    duration_groups = [
        test_df[test_df["experiment"] == experiment]["steps"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    duration_box_kwargs = dict(showmeans=True, patch_artist=True)
    try:
        duration_box = ax.boxplot(
            duration_groups, tick_labels=labels, **duration_box_kwargs
        )
    except TypeError:
        duration_box = ax.boxplot(
            duration_groups, labels=labels, **duration_box_kwargs
        )
    for patch, color in zip(duration_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("Frozen-test episode duration (steps)")
    ax.set_title("MetaDrive Test Duration Distribution")
    save_figure(fig, figure_dir, "ieee_test_duration_boxplot")

    # Convergence episode and cumulative time.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    convergence_episodes = summary_df["convergence_episode"].to_numpy(dtype=float)
    convergence_times = summary_df["convergence_time_seconds"].to_numpy(dtype=float)
    convergence_reached = summary_df["convergence_reached"].to_numpy(dtype=bool)
    episode_bars = axes[0].bar(
        x, convergence_episodes, color=colors, edgecolor="black"
    )
    for bar, reached in zip(episode_bars, convergence_reached):
        if not reached:
            bar.set_hatch("///")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].set_ylabel("Episode")
    axes[0].set_title("Shared-Target Convergence Episode")
    time_bars = axes[1].bar(
        x, convergence_times, color=colors, edgecolor="black"
    )
    for bar, reached in zip(time_bars, convergence_reached):
        if not reached:
            bar.set_hatch("///")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].set_ylabel("Cumulative wall time (s)")
    axes[1].set_title("Actual Cumulative Convergence Time")
    save_figure(fig, figure_dir, "ieee_convergence_episode_time")


def verify_initial_conditions(
    rows_df: pd.DataFrame, args, output_dir: Path
) -> Dict:
    """Verify exact paired resets and persist enough detail to diagnose failures."""
    columns = [
        "phase",
        "experiment",
        "episode",
        "scenario_seed",
        "initial_observation_sha256",
    ]
    records = rows_df[columns].copy()
    records.to_csv(output_dir / "initial_condition_records.csv", index=False)

    grouped = records.groupby(
        ["phase", "episode", "scenario_seed"], sort=True
    ).agg(
        method_count=("experiment", "nunique"),
        hash_count=("initial_observation_sha256", "nunique"),
    )
    expected_group_count = int(args.train_episodes + args.test_episodes)
    expected_method_count = int(len(EXPERIMENTS))
    mismatches = grouped[
        grouped["method_count"].ne(expected_method_count)
        | grouped["hash_count"].ne(1)
    ]

    expected_train_seeds = set(range(args.seed, args.seed + args.train_episodes))
    expected_test_seeds = set(
        range(args.test_seed, args.test_seed + args.test_episodes)
    )
    observed_train_seeds = set(
        records.loc[records["phase"].eq("train"), "scenario_seed"]
        .astype(int)
        .unique()
        .tolist()
    )
    observed_test_seeds = set(
        records.loc[records["phase"].eq("test"), "scenario_seed"]
        .astype(int)
        .unique()
        .tolist()
    )
    seed_schedule_verified = (
        observed_train_seeds == expected_train_seeds
        and observed_test_seeds == expected_test_seeds
    )

    mismatch_details: List[Dict] = []
    for phase, episode, scenario_seed in mismatches.index.tolist():
        group = records[
            records["phase"].eq(phase)
            & records["episode"].eq(episode)
            & records["scenario_seed"].eq(scenario_seed)
        ].sort_values("experiment")
        mismatch_details.append(
            {
                "phase": str(phase),
                "episode": int(episode),
                "scenario_seed": int(scenario_seed),
                "methods": group[
                    ["experiment", "initial_observation_sha256"]
                ].to_dict(orient="records"),
            }
        )

    verified = bool(
        len(grouped) == expected_group_count
        and mismatches.empty
        and seed_schedule_verified
    )
    verification = {
        "mode": "matched_scenario_schedule",
        "expected_methods_per_group": expected_method_count,
        "expected_groups": expected_group_count,
        "groups_checked": int(len(grouped)),
        "mismatched_groups": int(len(mismatches)),
        "seed_schedule_verified": bool(seed_schedule_verified),
        "verified": verified,
        "scope": (
            "Exact float32 initial observation must match across methods for "
            "each phase, episode, and scenario seed."
        ),
        "mismatch_details": mismatch_details,
    }
    verification_path = output_dir / "initial_condition_verification.json"
    verification_path.write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    return verification


def save_outputs(rows: List[Dict], runtimes: List[Dict], args, output_dir: Path) -> None:
    # Save irreplaceable raw data before strict paired-condition validation.
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(output_dir / "all_episode_results.csv", index=False)
    rows_df[rows_df["phase"].eq("train")].to_csv(
        output_dir / "all_experiments_train_episode_rewards.csv", index=False
    )
    rows_df[rows_df["phase"].eq("test")].to_csv(
        output_dir / "all_experiments_test_episode_rewards.csv", index=False
    )
    pd.DataFrame(runtimes).to_csv(
        output_dir / "all_experiments_runtime_logs.csv", index=False
    )
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    verification = verify_initial_conditions(rows_df, args, output_dir)
    if not verification["verified"]:
        raise RuntimeError(
            "Initial-condition verification failed after raw results were "
            f"saved. See: {output_dir / 'initial_condition_verification.json'}"
        )

    summary = make_summary(rows, args)
    pd.DataFrame(summary).to_csv(
        output_dir / "all_experiments_learning_rate_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "experiment": row["experiment"],
                "method": row["method"],
                "mean_reward": row["mean_R"],
                "median_reward": row["median_R"],
                "iqm_reward": row["IQMR"],
                "reward_var_5pct": row["VaR5_reward"],
                "reward_cvar_5pct": row["CVaR5_reward"],
                "selected_event_rmst_steps": row["RMST_steps"],
                "selected_rmst_event": row["RMST_event_definition"],
                "collision_free_rmst_steps": row["collision_RMST_steps"],
                "safety_failure_free_rmst_steps": row[
                    "safety_failure_RMST_steps"
                ],
                "rmst_tau_steps": row["RMST_tau_steps"],
                "test_collision_count": row["test_collision_count"],
                "total_test_steps": row["total_test_steps"],
                "collisions_per_1000_steps": row[
                    "test_collisions_per_1000_steps"
                ],
                "collision_rate": row["test_collision_rate"],
                "off_road_rate": row["test_off_road_rate"],
                "goal_rate": row["test_goal_rate"],
                "convergence_episode": row["convergence_episode"],
                "convergence_reached": row["convergence_reached"],
                "shared_convergence_target_reward": row[
                    "shared_convergence_target_reward"
                ],
                "convergence_time_seconds": row["convergence_time_seconds"],
            }
            for row in summary
        ]
    ).to_csv(output_dir / "extended_test_metrics.csv", index=False)
    pd.DataFrame([paired_duration_metrics(rows)]).to_csv(
        output_dir / "paired_test_duration_metrics.csv", index=False
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
    make_figures(rows, summary, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir.parent / "ch6b_noisy_results_27"
    parser = argparse.ArgumentParser(
        description="MetaDrive Chapter 6B DQN: NoisyNet versus Max/Random blocks"
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
        "--noisy-sigma-init",
        type=float,
        default=0.5,
        help="Initial factorized-Gaussian NoisyNet sigma scale",
    )
    parser.add_argument(
        "--greedy-episode-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of Chapter 6B training episodes that use maximum-Q at "
            "non-diagonal positions; the remainder use uniform random actions"
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

    parser.add_argument("--seed", type=int, default=27)
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
        "--output-dir", default=str(default_output_dir)
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.train_episodes <= 0 or args.test_episodes <= 0:
        raise ValueError("Training and testing episodes must be positive.")
    if args.max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be positive.")
    if args.rmst_tau <= 0:
        raise ValueError("--rmst-tau must be positive.")
    if args.rmst_tau > args.max_episode_steps:
        raise ValueError("--rmst-tau cannot exceed --max-episode-steps.")
    train_start, train_end = args.seed, args.seed + args.train_episodes - 1
    test_start, test_end = args.test_seed, args.test_seed + args.test_episodes - 1
    if max(train_start, test_start) <= min(train_end, test_end):
        raise ValueError("Testing seed range must not overlap the training seed range.")
    if args.noisy_sigma_init <= 0.0:
        raise ValueError("--noisy-sigma-init must be positive.")
    if not 0.0 <= args.greedy_episode_fraction <= 1.0:
        raise ValueError("--greedy-episode-fraction must be between 0 and 1.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1].")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.batch_size <= 0 or args.replay_capacity < args.batch_size:
        raise ValueError(
            "--batch-size must be positive and cannot exceed replay capacity."
        )
    if args.target_update_steps <= 0 or args.hidden_size <= 0:
        raise ValueError(
            "--target-update-steps and --hidden-size must be positive."
        )
    if args.discrete_steering_dim <= 0 or args.discrete_throttle_dim <= 0:
        raise ValueError("Discrete action dimensions must be positive.")
    if args.map_blocks <= 0:
        raise ValueError("--map-blocks must be positive.")
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1.")
    if not 0.0 <= args.accident_prob <= 1.0:
        raise ValueError("--accident-prob must be between 0 and 1.")
    if not 0.0 < args.convergence_threshold_fraction <= 1.0:
        raise ValueError(
            "--convergence-threshold-fraction must be in (0, 1]."
        )
    if args.convergence_window <= 0:
        raise ValueError("--convergence-window must be positive.")
    if args.collision_penalty < 0 or args.out_of_road_penalty < 0:
        raise ValueError("MetaDrive penalty arguments must be non-negative magnitudes.")


def main() -> None:
    args = parse_args()
    if args.rmst_tau is None:
        args.rmst_tau = args.max_episode_steps
    validate_args(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    device = choose_device(args.device)

    print("=" * 76)
    print("METADRIVE CHAPTER 6B DQN EXPERIMENT")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("MetaDrive:", getattr(metadrive, "__version__", "installed"))
    print("Device:", device)
    print("Experiments:", ", ".join(SHORT_LABELS[e] for e in EXPERIMENTS))
    print("Plain DQN + Target + Replay + Adam: yes")
    print("RND intrinsic reward: no")
    print("Count-based intrinsic reward: no")
    print("No epsilon-greedy exploration: yes")
    print("Factorized Gaussian NoisyNet baseline: yes")
    print("Frozen deterministic argmax testing with noise disabled: yes")
    print("Train/Test/Max steps:", args.train_episodes, args.test_episodes, args.max_episode_steps)
    print("Training seeds:", args.seed, "through", args.seed + args.train_episodes - 1)
    print("Testing seeds:", args.test_seed, "through", args.test_seed + args.test_episodes - 1)
    print("Discrete actions:", args.discrete_steering_dim * args.discrete_throttle_dim)
    print("Learning rate:", args.learning_rate)
    print("NoisyNet sigma init:", args.noisy_sigma_init)
    greedy_episode_count = int(
        math.ceil(args.train_episodes * args.greedy_episode_fraction)
    )
    print("Shared reproducible uniform exact-tie selection: yes")
    print("Deterministic traffic per scenario seed: yes")
    print("Strict initial-observation hash verification: yes")
    rectangular_padding = max(0, args.max_episode_steps - args.train_episodes)
    print("Chapter 6B main diagonal e=s uses maximum-Q action: yes")
    print("Chapter 6B anti-diagonal e+s=S+1 uses minimum-Q action: yes")
    print("Chapter 6B minimum-Q wins diagonal overlaps: yes")
    print("Chapter 6B wide-matrix final-row padding p=S-E:", rectangular_padding)
    print("Chapter 6B maximum-Q wins padding overlaps: yes")
    print("Chapter 6B maximum-Q episode block: 1 through", greedy_episode_count)
    print(
        "Chapter 6B uniform-random episode block:",
        greedy_episode_count + 1,
        "through",
        args.train_episodes,
    )
    print("RMST event/tau:", args.rmst_event, args.rmst_tau)
    print("Automatic plots folder:", output_dir / "plots")
    print("=" * 76)

    all_rows: List[Dict] = []
    runtimes: List[Dict] = []
    for experiment in EXPERIMENTS:
        experiment_rows, runtime = run_experiment(experiment, args, device, output_dir)
        all_rows.extend(experiment_rows)
        runtimes.append(runtime)
    save_outputs(all_rows, runtimes, args, output_dir)
    print("\nExperiment completed successfully.")
    print("Primary metrics:", output_dir / "four_primary_test_metrics.csv")
    print("Extended metrics:", output_dir / "extended_test_metrics.csv")
    print(
        "Paired duration metrics:",
        output_dir / "paired_test_duration_metrics.csv",
    )
    print(
        "Initial-condition verification:",
        output_dir / "initial_condition_verification.json",
    )
    print("Plots saved to:", output_dir / "plots")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
