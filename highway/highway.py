#!/usr/bin/env python3
"""
HighwayEnv DQN experiment in the same general style as the attached CARLA set4_v2 experiment.

Experiments, in strict order:
1. Epsilon Greedy
2. Median 50
3. Median First 50

Shared setup:
- Separate DQN, target network, replay buffer, Adam optimizer, RND predictor,
  and count table for every experiment.
- DQN + RND + count-based intrinsic reward.
- Frozen greedy testing: no optimizer, replay, target, RND, or count updates.
- The same episode seeds are reused across methods for fair comparison.
- Default: 500 train episodes, 300 test episodes, 500 maximum steps.
- Default DQN learning rate: 5e-5.
- Median First phase: epsilon * train_episodes (100 episodes when
  epsilon=0.2 and train_episodes=500).

Environment:
- HighwayEnv "highway-v0"
- Kinematics observation, flattened before entering the DQN.
- DiscreteMetaAction action space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import highway_env  # noqa: F401  # registers HighwayEnv environments
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


EXPERIMENTS = ["standard_epsilon", "median_50", "median_50_first"]
SHORT_LABELS = {
    "standard_epsilon": "Epsilon",
    "median_50": "Median 50",
    "median_50_first": "Median First 50",
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


def process_memory_mb() -> float:
    if psutil is None:
        return 0.0
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))


def system_memory_metrics() -> Dict[str, float]:
    if psutil is None:
        return {"ram_used_mb": 0.0, "ram_total_mb": 0.0, "ram_percent": 0.0}
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb": float(vm.used / (1024 ** 2)),
        "ram_total_mb": float(vm.total / (1024 ** 2)),
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
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=2,
        ).strip().splitlines()[0]
        data = [item.strip() for item in result.split(",")]
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
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def flatten_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def reward_mode(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    rounded = [round(float(x), 6) for x in values]
    counts: Dict[float, int] = {}
    for value in rounded:
        counts[value] = counts.get(value, 0) + 1
    maximum = max(counts.values())
    return float(max(key for key, count in counts.items() if count == maximum))


def make_env(args) -> gym.Env:
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": args.observed_vehicles,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False,
            "order": "sorted",
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": args.lanes_count,
        "vehicles_count": args.traffic_vehicles,
        "controlled_vehicles": 1,
        "duration": args.max_episode_steps,
        "initial_spacing": args.initial_spacing,
        "simulation_frequency": args.simulation_frequency,
        "policy_frequency": args.policy_frequency,
        "collision_reward": args.collision_reward,
        "right_lane_reward": args.right_lane_reward,
        "high_speed_reward": args.high_speed_reward,
        "lane_change_reward": args.lane_change_reward,
        "reward_speed_range": [args.reward_speed_min, args.reward_speed_max],
        "normalize_reward": args.normalize_reward,
        "offroad_terminal": True,
        "show_trajectories": False,
        "render_agent": False,
    }
    return gym.make(args.env_id, config=config, render_mode=None)


# ---------------------------------------------------------------------------
# DQN, RND and replay
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


class RNDNetwork(nn.Module):
    def __init__(self, observation_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
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

        self.rnd_target = RNDNetwork(
            observation_size, args.hidden_size, args.rnd_output_size
        ).to(device)
        self.rnd_predictor = RNDNetwork(
            observation_size, args.hidden_size, args.rnd_output_size
        ).to(device)
        self.rnd_target.eval()
        for parameter in self.rnd_target.parameters():
            parameter.requires_grad = False
        self.rnd_optimizer = optim.Adam(
            self.rnd_predictor.parameters(), lr=args.rnd_learning_rate
        )

    def tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.online.eval()
        with torch.no_grad():
            q = self.online(self.tensor(state))[0].detach().cpu().numpy()
        self.online.train()
        return q.astype(float)

    def greedy_action(self, state: np.ndarray) -> int:
        q = self.q_values(state)
        best = np.flatnonzero(q == np.max(q))
        return int(random.choice(best.tolist()))

    def median_lower_half_action(self, state: np.ndarray) -> int:
        q = self.q_values(state)
        median_value = float(np.median(q))
        candidates = np.flatnonzero(q <= median_value)
        if len(candidates) == 0:
            return self.greedy_action(state)
        return int(random.choice(candidates.tolist()))

    def intrinsic_reward(self, state: np.ndarray) -> float:
        with torch.no_grad():
            target = self.rnd_target(self.tensor(state))
            prediction = self.rnd_predictor(self.tensor(state))
            return float(F.mse_loss(prediction, target).item())

    def train_rnd(self, state: np.ndarray) -> float:
        x = self.tensor(state)
        with torch.no_grad():
            target = self.rnd_target(x)
        prediction = self.rnd_predictor(x)
        loss = F.mse_loss(prediction, target)
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.rnd_predictor.parameters(), 10.0)
        self.rnd_optimizer.step()
        return float(loss.detach().cpu().item())

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
        for module in (self.online, self.target, self.rnd_target, self.rnd_predictor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def save(self, path: Path, args) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "rnd_target": self.rnd_target.state_dict(),
                "rnd_predictor": self.rnd_predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rnd_optimizer": self.rnd_optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "learning_rate": args.learning_rate,
                "gamma": args.gamma,
                "epsilon": args.epsilon,
            },
            path,
        )


class CountBonus:
    def __init__(self, beta: float, bin_size: float):
        self.beta = float(beta)
        self.bin_size = max(float(bin_size), 1e-8)
        self.counts: Dict[Tuple[int, ...], int] = {}

    def bonus(self, state: np.ndarray) -> Tuple[float, int]:
        key = tuple(np.round(np.asarray(state) / self.bin_size).astype(int).tolist())
        visits = self.counts.get(key, 0) + 1
        self.counts[key] = visits
        return float(self.beta / math.sqrt(visits)), int(visits)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    episode: int,
    args,
) -> Tuple[int, str]:
    if experiment == "standard_epsilon":
        if random.random() < args.epsilon:
            return random.randrange(agent.action_count), "epsilon_random"
        return agent.greedy_action(state), "greedy"

    if experiment == "median_50":
        if random.random() < args.epsilon:
            return agent.median_lower_half_action(state), "median50_explore"
        return agent.greedy_action(state), "greedy"

    if experiment == "median_50_first":
        initial_episodes = int(round(args.epsilon * args.train_episodes))
        if episode < initial_episodes:
            return agent.median_lower_half_action(state), "median_first_phase"
        return agent.greedy_action(state), "greedy"

    raise ValueError(f"Unknown experiment: {experiment}")


# ---------------------------------------------------------------------------
# Training and testing
# ---------------------------------------------------------------------------

def run_experiment(
    experiment: str,
    args,
    device: torch.device,
    output_dir: Path,
) -> Tuple[List[Dict], Dict]:
    set_seed(args.seed)

    env = make_env(args)
    initial_observation, _ = env.reset(seed=args.seed)
    observation_size = int(flatten_observation(initial_observation).size)
    action_count = int(env.action_space.n)

    agent = DQNAgent(observation_size, action_count, args, device)
    count_bonus = CountBonus(args.count_beta, args.count_state_bin_size)
    rows: List[Dict] = []

    # Training
    print(f"\n===== TRAINING START: {SHORT_LABELS[experiment]} =====", flush=True)
    training_start = time.time()

    for episode in range(args.train_episodes):
        # Same seed sequence for every exploration experiment.
        state_raw, _ = env.reset(seed=args.seed + episode)
        state = flatten_observation(state_raw)

        env_reward_total = 0.0
        training_reward_total = 0.0
        losses: List[float] = []
        rnd_losses: List[float] = []
        rnd_values: List[float] = []
        count_values: List[float] = []
        action_sources: Dict[str, int] = {}
        term_reason = "max_steps"
        collision = False
        episode_start = time.time()
        cpu_start = time.process_time()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for step in range(args.max_episode_steps):
            action, source = select_training_action(
                experiment, agent, state, episode, args
            )
            action_sources[source] = action_sources.get(source, 0) + 1

            next_raw, env_reward, terminated, truncated, info = env.step(action)
            next_state = flatten_observation(next_raw)
            done = bool(terminated or truncated)

            rnd_raw = agent.intrinsic_reward(next_state)
            rnd_loss = agent.train_rnd(next_state)
            count_value, _ = count_bonus.bonus(next_state)
            training_reward = (
                float(env_reward)
                + args.rnd_beta * rnd_raw
                + count_value
            )

            agent.replay.add(state, action, training_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            env_reward_total += float(env_reward)
            training_reward_total += float(training_reward)
            rnd_values.append(rnd_raw)
            rnd_losses.append(rnd_loss)
            count_values.append(count_value)
            state = next_state

            collision = bool(info.get("crashed", False))
            if collision:
                term_reason = "collision"
            elif terminated:
                term_reason = "terminated"
            elif truncated:
                term_reason = "max_steps"

            if done:
                break

        metrics = {
            "phase": "train",
            "experiment": experiment,
            "method": SHORT_LABELS[experiment],
            "episode": episode,
            "env_reward": env_reward_total,
            "training_reward": training_reward_total,
            "steps": step + 1,
            "termination_reason": term_reason,
            "collision": collision,
            "wall_time_seconds": time.time() - episode_start,
            "cpu_time_seconds": time.process_time() - cpu_start,
            "gpu_memory_mb": gpu_memory_mb(device),
            "process_memory_mb": process_memory_mb(),
            **system_memory_metrics(),
            **smi_metrics(),
            "average_loss": avg(losses),
            "average_rnd_intrinsic": avg(rnd_values),
            "average_rnd_loss": avg(rnd_losses),
            "average_count_intrinsic": avg(count_values),
            "replay_buffer_size": len(agent.replay),
            "learn_steps": agent.learn_steps,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "rnd_beta": args.rnd_beta,
            "count_beta": args.count_beta,
            "action_source_counts": json.dumps(action_sources, sort_keys=True),
        }
        rows.append(metrics)

        print(
            f"TRAIN {SHORT_LABELS[experiment]:16s} ep={episode:03d} "
            f"env_reward={env_reward_total:8.3f} "
            f"train_reward={training_reward_total:8.3f} "
            f"steps={step + 1:3d} term={term_reason:10s} "
            f"wall={metrics['wall_time_seconds']:.2f}s "
            f"loss={metrics['average_loss']:.6f}",
            flush=True,
        )

    training_duration = time.time() - training_start
    model_path = output_dir / "models" / f"{experiment}_model.pt"
    agent.save(model_path, args)

    # Frozen testing
    agent.freeze()
    print(f"===== TRAINING END: {SHORT_LABELS[experiment]} =====", flush=True)
    print(f"Training duration: {training_duration:.2f}s", flush=True)
    print(f"\n===== TESTING START: {SHORT_LABELS[experiment]} =====", flush=True)

    testing_start = time.time()
    with torch.no_grad():
        for episode in range(args.test_episodes):
            state_raw, _ = env.reset(seed=args.seed + 100000 + episode)
            state = flatten_observation(state_raw)
            total_reward = 0.0
            term_reason = "max_steps"
            collision = False
            episode_start = time.time()
            cpu_start = time.process_time()

            for step in range(args.max_episode_steps):
                action = agent.greedy_action(state)
                next_raw, reward, terminated, truncated, info = env.step(action)
                state = flatten_observation(next_raw)
                total_reward += float(reward)

                collision = bool(info.get("crashed", False))
                if collision:
                    term_reason = "collision"
                elif terminated:
                    term_reason = "terminated"
                elif truncated:
                    term_reason = "max_steps"

                if terminated or truncated:
                    break

            metrics = {
                "phase": "test",
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "episode": episode,
                "env_reward": total_reward,
                "training_reward": total_reward,
                "steps": step + 1,
                "termination_reason": term_reason,
                "collision": collision,
                "wall_time_seconds": time.time() - episode_start,
                "cpu_time_seconds": time.process_time() - cpu_start,
                "gpu_memory_mb": gpu_memory_mb(device),
                "process_memory_mb": process_memory_mb(),
                **system_memory_metrics(),
                **smi_metrics(),
                "average_loss": 0.0,
                "average_rnd_intrinsic": 0.0,
                "average_rnd_loss": 0.0,
                "average_count_intrinsic": 0.0,
                "replay_buffer_size": len(agent.replay),
                "learn_steps": agent.learn_steps,
                "epsilon": 0.0,
                "gamma": args.gamma,
                "learning_rate": args.learning_rate,
                "rnd_beta": 0.0,
                "count_beta": 0.0,
                "network_frozen": True,
                "updates_during_test": 0,
                "action_source_counts": json.dumps({"frozen_greedy": step + 1}),
            }
            rows.append(metrics)

            print(
                f"TEST  {SHORT_LABELS[experiment]:16s} ep={episode:03d} "
                f"reward={total_reward:8.3f} steps={step + 1:3d} "
                f"term={term_reason:10s} wall={metrics['wall_time_seconds']:.2f}s",
                flush=True,
            )

    testing_duration = time.time() - testing_start
    env.close()
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
# Summaries and plots
# ---------------------------------------------------------------------------

def convergence_episode(
    rewards: Sequence[float],
    target_reward: float,
    threshold_fraction: float,
    window: int,
) -> int:
    if not rewards:
        return 0
    window = max(1, min(int(window), len(rewards)))
    rolling = np.convolve(
        np.asarray(rewards, dtype=float),
        np.ones(window) / window,
        mode="valid",
    )
    threshold = threshold_fraction * target_reward
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
        average_test = float(np.mean(test_rewards))

        conv_episode = convergence_episode(
            train_rewards,
            average_test,
            args.convergence_threshold_fraction,
            args.convergence_window,
        )
        total_train_time = sum(float(row["wall_time_seconds"]) for row in train)
        conv_time = (
            conv_episode / max(args.train_episodes, 1)
        ) * total_train_time

        summary.append(
            {
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "learning_rate": args.learning_rate,
                "average_train_env_reward": float(np.mean(train_rewards)),
                "median_train_env_reward": float(np.median(train_rewards)),
                "std_train_env_reward": float(np.std(train_rewards)),
                "average_train_total_reward": float(np.mean(train_total_rewards)),
                "average_test_reward": average_test,
                "median_test_reward": float(np.median(test_rewards)),
                "mode_test_reward": reward_mode(test_rewards),
                "std_test_reward": float(np.std(test_rewards)),
                "min_test_reward": float(np.min(test_rewards)),
                "max_test_reward": float(np.max(test_rewards)),
                "q1_test_reward": percentile(test_rewards, 25),
                "q3_test_reward": percentile(test_rewards, 75),
                "convergence_episode": conv_episode,
                "convergence_time_seconds": conv_time,
                "total_training_wall_time_seconds": total_train_time,
                "train_collision_rate": float(np.mean([bool(row["collision"]) for row in train])),
                "test_collision_rate": float(np.mean([bool(row["collision"]) for row in test])),
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


def make_figures(
    rows: List[Dict],
    summary: List[Dict],
    output_dir: Path,
) -> None:
    apply_ieee_style()
    figure_dir = output_dir / "figures_ieee"
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame([row for row in rows if row["phase"] == "train"])
    test_df = pd.DataFrame([row for row in rows if row["phase"] == "test"])
    summary_df = pd.DataFrame(summary)
    labels = [SHORT_LABELS[experiment] for experiment in EXPERIMENTS]
    x = np.arange(len(EXPERIMENTS))

    # Training moving-average environment reward.
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for experiment in EXPERIMENTS:
        data = train_df[train_df["experiment"] == experiment].sort_values("episode")
        rolling = data["env_reward"].rolling(20, min_periods=1).mean()
        ax.plot(data["episode"], rolling, linewidth=1.5, label=SHORT_LABELS[experiment])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("20-episode mean environment reward")
    ax.set_title("HighwayEnv DQN Training Reward")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_training_reward")

    # Average test reward.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    values = summary_df.set_index("experiment").loc[EXPERIMENTS, "average_test_reward"].to_numpy()
    errors = summary_df.set_index("experiment").loc[EXPERIMENTS, "std_test_reward"].to_numpy()
    ax.bar(x, values, yerr=errors, capsize=3, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Average frozen-test reward")
    ax.set_title("HighwayEnv Average Test Reward")
    save_figure(fig, figure_dir, "ieee_average_test_reward")

    # Test boxplot.
    groups = [
        test_df[test_df["experiment"] == experiment]["env_reward"].to_numpy()
        for experiment in EXPERIMENTS
    ]
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Frozen-test reward")
    ax.set_title("HighwayEnv Test Reward Distribution")
    save_figure(fig, figure_dir, "ieee_test_reward_boxplot")

    # Convergence episode.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    values = summary_df.set_index("experiment").loc[EXPERIMENTS, "convergence_episode"].to_numpy()
    ax.bar(x, values, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Convergence episode")
    ax.set_title("HighwayEnv Training Convergence")
    save_figure(fig, figure_dir, "ieee_convergence_episode")

    # Convergence time.
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    values = summary_df.set_index("experiment").loc[EXPERIMENTS, "convergence_time_seconds"].to_numpy()
    ax.bar(x, values, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Convergence time (s)")
    ax.set_title("HighwayEnv Convergence Time")
    save_figure(fig, figure_dir, "ieee_convergence_time")

    # Collision rates.
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    width = 0.36
    train_rates = summary_df.set_index("experiment").loc[EXPERIMENTS, "train_collision_rate"].to_numpy() * 100
    test_rates = summary_df.set_index("experiment").loc[EXPERIMENTS, "test_collision_rate"].to_numpy() * 100
    ax.bar(x - width / 2, train_rates, width=width, label="Train", edgecolor="black")
    ax.bar(x + width / 2, test_rates, width=width, label="Test", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("HighwayEnv Collision Rate")
    ax.legend(frameon=False)
    save_figure(fig, figure_dir, "ieee_collision_rate")


def save_outputs(
    rows: List[Dict],
    runtimes: List[Dict],
    args,
    output_dir: Path,
) -> None:
    summary = make_summary(rows, args)

    pd.DataFrame(rows).to_csv(
        output_dir / "all_episode_results.csv", index=False
    )
    pd.DataFrame(
        [row for row in rows if row["phase"] == "train"]
    ).to_csv(output_dir / "all_experiments_train_episode_rewards.csv", index=False)
    pd.DataFrame(
        [row for row in rows if row["phase"] == "test"]
    ).to_csv(output_dir / "all_experiments_test_episode_rewards.csv", index=False)
    pd.DataFrame(summary).to_csv(
        output_dir / "all_experiments_learning_rate_summary.csv", index=False
    )
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", default="highway-v0")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--rnd-beta", type=float, default=0.01)
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnd-output-size", type=int, default=64)
    parser.add_argument("--count-beta", type=float, default=0.05)
    parser.add_argument("--count-state-bin-size", type=float, default=0.25)

    parser.add_argument("--lanes-count", type=int, default=4)
    parser.add_argument("--traffic-vehicles", type=int, default=40)
    parser.add_argument("--observed-vehicles", type=int, default=5)
    parser.add_argument("--initial-spacing", type=float, default=2.0)
    parser.add_argument("--simulation-frequency", type=int, default=15)
    parser.add_argument("--policy-frequency", type=int, default=5)

    parser.add_argument("--collision-reward", type=float, default=-1.0)
    parser.add_argument("--right-lane-reward", type=float, default=0.1)
    parser.add_argument("--high-speed-reward", type=float, default=0.4)
    parser.add_argument("--lane-change-reward", type=float, default=0.0)
    parser.add_argument("--reward-speed-min", type=float, default=20.0)
    parser.add_argument("--reward-speed-max", type=float, default=30.0)
    parser.add_argument(
        "--no-normalize-reward",
        action="store_false",
        dest="normalize_reward",
    )
    parser.set_defaults(normalize_reward=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    parser.add_argument("--convergence-window", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="results_highway_dqn_three_experiments",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    device = choose_device(args.device)

    print("=" * 76)
    print("HIGHWAYENV DQN EXPERIMENT")
    print("=" * 76)
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("HighwayEnv:", getattr(highway_env, "__version__", "installed"))
    print("Device:", device)
    print("Experiments:", ", ".join(SHORT_LABELS[x] for x in EXPERIMENTS))
    print("DQN + RND + Count + Target + Replay + Adam: yes")
    print("Separate model/replay/RND/count per experiment: yes")
    print("Frozen greedy testing: yes")
    print(
        "Train/Test/Max steps:",
        args.train_episodes,
        args.test_episodes,
        args.max_episode_steps,
    )
    print("Learning rate:", args.learning_rate)
    print("Epsilon:", args.epsilon)
    print(
        "Median First phase episodes:",
        int(round(args.epsilon * args.train_episodes)),
    )
    print("=" * 76)

    all_rows: List[Dict] = []
    runtimes: List[Dict] = []

    for experiment in EXPERIMENTS:
        experiment_rows, runtime = run_experiment(
            experiment, args, device, output_dir
        )
        all_rows.extend(experiment_rows)
        runtimes.append(runtime)

    save_outputs(all_rows, runtimes, args, output_dir)
    print("\nExperiment completed successfully.")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()

