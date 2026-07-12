#!/usr/bin/env python3
"""
SUMO set4_v2 DQN experiment matched to the CARLA set4_v2 structure.

Experiments, in strict order:
  1. Epsilon Greedy
  2. Median 50

Neural setup for every experiment:
  DQN + RND + Count-Based intrinsic reward + target network
  + replay buffer + Adam optimizer

Testing:
  - Networks frozen
  - Greedy argmax actions during testing
  - No optimizer, replay, RND, count, or target updates
  - Environment reward only

Default run:
  500 training episodes
  300 testing episodes
  500 maximum steps
  learning rate 5e-5
  epsilon 0.2
  gamma 0.99
  target speed 8.333333 m/s (30 km/h)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces

try:
    import psutil
except Exception:
    psutil = None


EXPERIMENTS = ["standard_epsilon", "median_50"]

SHORT_LABELS = {
    "standard_epsilon": "Epsilon Greedy",
    "median_50": "Median 50",
}


# ---------------------------------------------------------------------
# Reproducibility and system metrics
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str) -> torch.device:
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def cuda_ok(device: str) -> bool:
    return str(device).startswith("cuda") and torch.cuda.is_available()


def make_gpu_timer(device: str):
    if not cuda_ok(device):
        return None
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def stop_gpu_timer(timer, device: str) -> float:
    if timer is None or not cuda_ok(device):
        return 0.0
    start, end = timer
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / 1000.0)


def reset_gpu_peak(device: str) -> None:
    if cuda_ok(device):
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb(device: str) -> float:
    if not cuda_ok(device):
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


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
    out = {
        "gpu_util_percent": 0.0,
        "gpu_memory_used_mb_smi": 0.0,
        "gpu_memory_total_mb_smi": 0.0,
        "gpu_power_watts": 0.0,
        "gpu_temperature_c": 0.0,
    }
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        line = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=2,
        ).strip().splitlines()[0]
        vals = [v.strip() for v in line.split(",")]
        out.update(
            {
                "gpu_util_percent": float(vals[0]),
                "gpu_memory_used_mb_smi": float(vals[1]),
                "gpu_memory_total_mb_smi": float(vals[2]),
                "gpu_power_watts": float(vals[3]),
                "gpu_temperature_c": float(vals[4]),
            }
        )
    except Exception:
        pass
    return out


def avg(xs: Iterable[float]) -> float:
    values = list(xs)
    return float(sum(values) / max(len(values), 1))


def pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def reward_mode(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    counts: Dict[float, int] = {}
    for x in xs:
        key = round(float(x), 6)
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values())
    return float(max(k for k, v in counts.items() if v == max_count))


# ---------------------------------------------------------------------
# SUMO environment
# ---------------------------------------------------------------------

class SumoDrivingEnv(gym.Env):
    """
    Straight-road SUMO environment with one controlled ego vehicle and one deterministic leader.

    Observation:
      0: ego speed / target speed
      1: ego position / road length
      2: remaining distance / road length
      3: leader gap / 100
      4: leader speed / target speed

    Actions:
      0: decelerate
      1: maintain
      2: accelerate
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_dir: str,
        max_episode_steps: int = 500,
        target_speed: float = 13.9,
        seed: int = 42,
        gui: bool = False,
    ) -> None:
        super().__init__()
        self.scenario_dir = Path(scenario_dir).resolve()
        self.max_episode_steps = int(max_episode_steps)
        self.target_speed = float(target_speed)
        self.base_seed = int(seed)
        self.gui = bool(gui)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 2.0, 2.0, 1.0, 3.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.road_length = 1000.0
        self.step_count = 0
        self.last_position = 0.0
        self.conn = None
        self.route_file: Optional[Path] = None
        self._create_scenario()

    @staticmethod
    def _require_binary(name: str) -> str:
        path = shutil.which(name)
        if path is None:
            raise RuntimeError(
                f"Required SUMO binary '{name}' was not found.\n"
                "Install SUMO with:\n"
                "  apt update && apt install -y sumo sumo-tools"
            )
        return path

    def _create_scenario(self) -> None:
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        net_file = self.scenario_dir / "straight.net.xml"
        if net_file.exists():
            return

        nodes_file = self.scenario_dir / "straight.nod.xml"
        edges_file = self.scenario_dir / "straight.edg.xml"

        nodes_file.write_text(
            """<nodes>
    <node id="n0" x="0.0" y="0.0" type="priority"/>
    <node id="n1" x="1000.0" y="0.0" type="priority"/>
</nodes>
""",
            encoding="utf-8",
        )
        edges_file.write_text(
            """<edges>
    <edge id="road" from="n0" to="n1" numLanes="1" speed="25.0"/>
</edges>
""",
            encoding="utf-8",
        )

        subprocess.run(
            [
                self._require_binary("netconvert"),
                "--node-files", str(nodes_file),
                "--edge-files", str(edges_file),
                "--output-file", str(net_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _write_route_file(self, seed: int) -> Path:
        # Fixed scenario to match the repeated CARLA-style setup across episodes.
        # SUMO uses m/s, so target_speed=8.333333 corresponds to 30 km/h.
        leader_speed = float(self.target_speed)
        leader_depart = 0.0

        route_file = (
            self.scenario_dir
            / f"episode_{os.getpid()}_{id(self)}.rou.xml"
        )
        route_file.write_text(
            f"""<routes>
    <vType id="egoType"
           accel="3.0"
           decel="6.0"
           sigma="0.0"
           length="5.0"
           minGap="2.5"
           maxSpeed="25.0"
           guiShape="passenger"/>

    <vType id="leaderType"
           accel="2.0"
           decel="4.5"
           sigma="0.0"
           length="5.0"
           minGap="2.5"
           maxSpeed="{leader_speed:.3f}"
           guiShape="passenger"/>

    <route id="mainRoute" edges="road"/>

    <vehicle id="leader"
             type="leaderType"
             route="mainRoute"
             depart="{leader_depart:.3f}"
             departSpeed="{leader_speed:.3f}"/>

    <vehicle id="ego"
             type="egoType"
             route="mainRoute"
             depart="2.0"
             departSpeed="5.0"/>
</routes>
""",
            encoding="utf-8",
        )
        return route_file

    def _start_sumo(self, seed: int) -> None:
        try:
            import traci
        except ImportError as exc:
            raise RuntimeError(
                "Python package 'traci' is missing.\n"
                "Install with:\n"
                "  python -m pip install traci sumolib"
            ) from exc

        binary = self._require_binary("sumo-gui" if self.gui else "sumo")
        self.route_file = self._write_route_file(seed)

        cmd = [
            binary,
            "-n", str(self.scenario_dir / "straight.net.xml"),
            "-r", str(self.route_file),
            "--step-length", "0.2",
            "--collision.action", "remove",
            "--collision.check-junctions", "true",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--seed", str(seed),
        ]

        label = f"sumo_{os.getpid()}_{id(self)}"
        traci.start(cmd, label=label)
        self.conn = traci.getConnection(label)

        for _ in range(100):
            self.conn.simulationStep()
            if "ego" in self.conn.vehicle.getIDList():
                break

        if "ego" not in self.conn.vehicle.getIDList():
            self.close()
            raise RuntimeError("Ego vehicle failed to enter the SUMO simulation.")

        self.conn.vehicle.setSpeedMode("ego", 31)

        # Keep the leader deterministic instead of relying only on maxSpeed.
        if "leader" in self.conn.vehicle.getIDList():
            self.conn.vehicle.setSpeedMode("leader", 0)
            self.conn.vehicle.setSpeed("leader", float(self.target_speed))

    def _get_observation(self) -> np.ndarray:
        if self.conn is None or "ego" not in self.conn.vehicle.getIDList():
            return np.zeros(5, dtype=np.float32)

        speed = float(self.conn.vehicle.getSpeed("ego"))
        position = float(self.conn.vehicle.getLanePosition("ego"))
        remaining = max(0.0, self.road_length - position)

        leader = self.conn.vehicle.getLeader("ego", 1000.0)
        if leader is None:
            gap = 1000.0
            leader_speed = self.target_speed
        else:
            leader_id, gap = leader
            leader_speed = float(self.conn.vehicle.getSpeed(leader_id))

        return np.array(
            [
                speed / self.target_speed,
                position / self.road_length,
                remaining / self.road_length,
                min(float(gap), 100.0) / 100.0,
                leader_speed / self.target_speed,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.close()

        episode_seed = self.base_seed if seed is None else int(seed)
        self._start_sumo(episode_seed)

        self.step_count = 0
        self.last_position = float(
            self.conn.vehicle.getLanePosition("ego")
        )
        return self._get_observation(), {}

    def step(self, action: int):
        if self.conn is None:
            raise RuntimeError("Call env.reset() before env.step().")

        action = int(action)
        current_speed = float(self.conn.vehicle.getSpeed("ego"))

        if action == 0:
            requested_speed = max(0.0, current_speed - 2.0)
        elif action == 1:
            requested_speed = current_speed
        elif action == 2:
            requested_speed = min(25.0, current_speed + 2.0)
        else:
            raise ValueError(f"Invalid action: {action}")

        self.conn.vehicle.setSpeed("ego", requested_speed)
        self.conn.simulationStep()
        self.step_count += 1

        vehicle_ids = set(self.conn.vehicle.getIDList())
        ego_present = "ego" in vehicle_ids
        arrived_ids = set(self.conn.simulation.getArrivedIDList())
        collision_ids = set(
            self.conn.simulation.getCollidingVehiclesIDList()
        )

        arrived = "ego" in arrived_ids
        collision = "ego" in collision_ids

        if ego_present:
            position = float(self.conn.vehicle.getLanePosition("ego"))
            speed = float(self.conn.vehicle.getSpeed("ego"))
        else:
            position = self.road_length if arrived else self.last_position
            speed = 0.0

        progress = max(0.0, position - self.last_position)
        speed_error = abs(speed - self.target_speed) / self.target_speed

        reward = progress - 0.25 * speed_error
        terminated = False
        termination_reason = "running"

        if collision:
            reward -= 50.0
            terminated = True
            termination_reason = "collision"
        elif arrived or position >= self.road_length - 5.0:
            reward += 100.0
            terminated = True
            termination_reason = "goal"
        elif not ego_present:
            reward -= 10.0
            terminated = True
            termination_reason = "removed"

        truncated = self.step_count >= self.max_episode_steps
        if truncated and not terminated:
            termination_reason = "max_steps"

        self.last_position = position

        info = {
            "termination_reason": termination_reason,
            "ended_before_max_steps": bool(
                terminated and self.step_count < self.max_episode_steps
            ),
            "collision": collision,
            "collision_actor_type": "vehicle" if collision else "none",
            "collision_actor_id": -1,
            "collision_actor_role_name": "leader" if collision else "",
            "collision_intensity": 1.0 if collision else 0.0,
            "stuck": False,
            "stuck_step_count": 0,
            "position": position,
            "speed": speed,
            "progress": progress,
        }

        return (
            self._get_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close(False)
            except Exception:
                pass
            self.conn = None


# ---------------------------------------------------------------------
# DQN + RND + Count
# ---------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs: int, actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDNetwork(nn.Module):
    def __init__(self, obs: int, hidden: int, out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, int(batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


class RNDCountDQNAgent:
    def __init__(self, obs: int, actions: int, args, lr: float) -> None:
        self.obs = int(obs)
        self.actions = int(actions)
        self.gamma = float(args.gamma)
        self.batch_size = int(args.batch_size)
        self.target_update_interval = int(args.target_update_interval)
        self.device = device_from_arg(args.device)
        self.learn_steps = 0

        self.q_network = QNetwork(
            obs, actions, args.hidden_size
        ).to(self.device)
        self.target_network = QNetwork(
            obs, actions, args.hidden_size
        ).to(self.device)
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=float(lr),
        )
        self.replay_buffer = ReplayBuffer(args.replay_capacity)

        self.rnd_target = RNDNetwork(
            obs, args.hidden_size, args.rnd_output_size
        ).to(self.device)
        self.rnd_predictor = RNDNetwork(
            obs, args.hidden_size, args.rnd_output_size
        ).to(self.device)
        self.rnd_optimizer = optim.Adam(
            self.rnd_predictor.parameters(),
            lr=float(args.rnd_learning_rate),
        )

        self.rnd_target.eval()
        for parameter in self.rnd_target.parameters():
            parameter.requires_grad = False

    def tensor_state(self, state) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)

    def get_q_values(self, state) -> np.ndarray:
        was_training = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            q_values = (
                self.q_network(self.tensor_state(state))
                .detach()
                .cpu()
                .numpy()[0]
            )
        if was_training:
            self.q_network.train()
        return q_values.astype(float)

    def best_action(self, state) -> int:
        return int(np.argmax(self.get_q_values(state)))

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(
            state, action, reward, next_state, done
        )

    def intrinsic_reward(self, state) -> float:
        with torch.no_grad():
            target = self.rnd_target(self.tensor_state(state))
            prediction = self.rnd_predictor(
                self.tensor_state(state)
            )
            return float(
                F.mse_loss(
                    prediction,
                    target,
                    reduction="mean",
                ).detach().cpu().item()
            )

    def train_rnd_predictor(self, state) -> float:
        x = self.tensor_state(state)
        with torch.no_grad():
            target = self.rnd_target(x)

        prediction = self.rnd_predictor(x)
        loss = F.mse_loss(
            prediction,
            target,
            reduction="mean",
        )

        self.rnd_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.rnd_predictor.parameters(), 10.0
        )
        self.rnd_optimizer.step()
        return float(loss.detach().cpu().item())

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            target_next = self.target_network(
                next_states
            ).max(1, keepdim=True)[0]
            targets = (
                rewards
                + (1.0 - dones) * self.gamma * target_next
            )

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 10.0
        )
        self.optimizer.step()

        self.learn_steps += 1
        if (
            self.learn_steps
            % self.target_update_interval
            == 0
        ):
            self.update_target_network()

        return float(loss.detach().cpu().item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def freeze_for_eval(self) -> None:
        for model in [
            self.q_network,
            self.target_network,
            self.rnd_target,
            self.rnd_predictor,
        ]:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "rnd_target": self.rnd_target.state_dict(),
                "rnd_predictor": self.rnd_predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rnd_optimizer": self.rnd_optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "obs": self.obs,
                "actions": self.actions,
            },
            path,
        )


class CountBonus:
    def __init__(self, beta: float, bin_size: float) -> None:
        self.beta = float(beta)
        self.bin_size = max(float(bin_size), 1e-6)
        self.counts: Dict[Tuple[int, ...], int] = {}

    def key(self, state) -> Tuple[int, ...]:
        arr = np.asarray(state, dtype=float)
        return tuple(
            np.round(arr / self.bin_size)
            .astype(int)
            .tolist()
        )

    def bonus(self, state) -> Tuple[float, int]:
        key = self.key(state)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return float(self.beta / math.sqrt(count)), int(count)


# ---------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------

def median50_action(
    agent: RNDCountDQNAgent,
    state,
    actions: int,
) -> int:
    q_values = agent.get_q_values(state)
    median_value = float(np.median(q_values))
    lower = [
        index
        for index, value in enumerate(q_values)
        if float(value) <= median_value
    ]
    return int(
        random.choice(lower if lower else list(range(actions)))
    )


def base_action(
    experiment: str,
    agent: RNDCountDQNAgent,
    state,
    episode: int,
    args,
    actions: int,
) -> Tuple[int, str]:
    if experiment == "standard_epsilon":
        if random.random() < args.epsilon:
            return int(random.randrange(actions)), "epsilon_random"
        return agent.best_action(state), "greedy"

    if experiment == "median_50":
        if random.random() < args.epsilon:
            return (
                median50_action(agent, state, actions),
                "median50_explore",
            )
        return agent.best_action(state), "greedy"

    raise ValueError(experiment)




# ---------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------

def convergence_episode(
    train_rewards,
    target_reward: float,
    threshold_fraction: float,
    window: int,
) -> int:
    """Training-only convergence estimate.

    The target_reward argument is retained for call compatibility but is not
    used. Convergence is the first rolling window reaching the requested
    fraction of improvement from the initial window to the final window.
    """
    if not train_rewards:
        return 0

    rewards = np.asarray(train_rewards, dtype=float)
    actual_window = max(1, min(int(window), len(rewards)))
    rolling = np.convolve(
        rewards,
        np.ones(actual_window) / actual_window,
        mode="valid",
    )

    initial_reference = float(rolling[0])
    tail_count = min(5, len(rolling))
    final_reference = float(np.mean(rolling[-tail_count:]))
    threshold = initial_reference + float(threshold_fraction) * (
        final_reference - initial_reference
    )

    if final_reference >= initial_reference:
        for index, value in enumerate(rolling):
            if float(value) >= threshold:
                return int(index + actual_window)
    else:
        for index, value in enumerate(rolling):
            if float(value) <= threshold:
                return int(index + actual_window)

    return int(len(train_rewards))


def block_summary(
    rows,
    experiment: str,
    phase: str,
    learning_rate: float,
    block_index: int,
):
    rewards = [
        float(row["env_reward"]) for row in rows
    ]

    return {
        "phase": phase,
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "block_index": int(block_index),
        "start_episode": int(rows[0]["episode"]),
        "end_episode": int(rows[-1]["episode"]),
        "episodes_in_block": int(len(rows)),
        "average_env_reward": float(np.mean(rewards)),
        "median_env_reward": float(np.median(rewards)),
        "min_env_reward": float(np.min(rewards)),
        "max_env_reward": float(np.max(rewards)),
        "std_env_reward": float(np.std(rewards)),
        "q1_env_reward": pct(rewards, 25),
        "q3_env_reward": pct(rewards, 75),
        "total_wall_time_seconds": float(
            sum(
                float(row.get("wall_time_seconds", 0.0))
                for row in rows
            )
        ),
        "total_cpu_time_seconds": float(
            sum(
                float(row.get("cpu_time_seconds", 0.0))
                for row in rows
            )
        ),
        "total_gpu_time_seconds": float(
            sum(
                float(row.get("gpu_time_seconds", 0.0))
                for row in rows
            )
        ),
        "average_steps": float(
            np.mean(
                [
                    float(row.get("steps", 0))
                    for row in rows
                ]
            )
        ),
        "average_loss": float(
            np.mean(
                [
                    float(row.get("average_loss", 0.0))
                    for row in rows
                ]
            )
        ),
        "average_rnd_intrinsic": float(
            np.mean(
                [
                    float(
                        row.get(
                            "average_rnd_intrinsic",
                            0.0,
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "average_count_intrinsic": float(
            np.mean(
                [
                    float(
                        row.get(
                            "average_count_intrinsic",
                            0.0,
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "average_gpu_util_percent": float(
            np.mean(
                [
                    float(
                        row.get(
                            "gpu_util_percent",
                            0.0,
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "max_gpu_memory_mb": float(
            np.max(
                [
                    float(
                        row.get(
                            "gpu_memory_mb",
                            0.0,
                        )
                    )
                    for row in rows
                ]
            )
        ),
        "learning_rate": float(learning_rate),
        "lr_multiplier": 1.0,
    }


# ---------------------------------------------------------------------
# Training and testing
# ---------------------------------------------------------------------

def make_env(args, output_dir: Path) -> SumoDrivingEnv:
    return SumoDrivingEnv(
        scenario_dir=str(output_dir / "sumo_scenario"),
        max_episode_steps=args.max_episode_steps,
        target_speed=args.target_speed,
        seed=args.seed,
        gui=args.gui,
    )


def train_one(
    experiment: str,
    learning_rate: float,
    args,
    output_dir: Path,
    experts=None,
):
    env = make_env(args, output_dir)
    obs = int(env.observation_space.shape[0])
    actions = int(env.action_space.n)

    agent = RNDCountDQNAgent(
        obs,
        actions,
        args,
        learning_rate,
    )
    count_bonus = CountBonus(
        args.count_beta,
        args.count_state_bin_size,
    )

    rows = []
    block_rows = []
    train_rewards = []

    reset_gpu_peak(args.device)
    experiment_wall_start = time.time()
    experiment_cpu_start = time.process_time()

    print(
        f"\n===== TRAINING START: "
        f"{SHORT_LABELS[experiment]} =====",
        flush=True,
    )

    for episode in range(args.train_episodes):
        state, _ = env.reset(
            seed=args.seed
        )

        env_reward = 0.0
        training_reward = 0.0
        rnd_sum = 0.0
        count_sum = 0.0
        rnd_loss_sum = 0.0
        rnd_loss_count = 0
        steps = 0
        losses = []
        source_counts: Dict[str, int] = {}
        last_info: Dict = {}
        done = False

        wall_start = time.time()
        cpu_start = time.process_time()
        gpu_timer = make_gpu_timer(args.device)

        while not done:
            action, source = base_action(
                experiment,
                agent,
                state,
                episode,
                args,
                actions,
            )

            source_counts[source] = (
                source_counts.get(source, 0) + 1
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            last_info = dict(info or {})
            done = bool(terminated or truncated)

            rnd_raw = float(
                agent.intrinsic_reward(next_state)
            )
            rnd_loss = float(
                agent.train_rnd_predictor(next_state)
            )
            count_value, _ = count_bonus.bonus(
                next_state
            )

            full_reward = (
                float(reward)
                + float(args.rnd_beta) * rnd_raw
                + float(count_value)
            )

            agent.remember(
                state,
                action,
                full_reward,
                next_state,
                done,
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))

            env_reward += float(reward)
            training_reward += full_reward
            rnd_sum += rnd_raw
            count_sum += count_value
            rnd_loss_sum += rnd_loss
            rnd_loss_count += 1

            state = next_state
            steps += 1

        train_rewards.append(env_reward)

        row = {
            "phase": "train",
            "experiment": experiment,
            "method": SHORT_LABELS[experiment],
            "episode": episode,
            "env_reward": float(env_reward),
            "training_reward": float(training_reward),
            "steps": steps,
            "termination_reason": last_info.get(
                "termination_reason",
                "unknown",
            ),
            "ended_before_max_steps": bool(
                last_info.get(
                    "ended_before_max_steps",
                    steps < args.max_episode_steps,
                )
            ),
            "collision": bool(
                last_info.get("collision", False)
            ),
            "collision_actor_type": last_info.get(
                "collision_actor_type",
                "none",
            ),
            "collision_actor_id": last_info.get(
                "collision_actor_id",
                -1,
            ),
            "collision_actor_role_name": last_info.get(
                "collision_actor_role_name",
                "",
            ),
            "collision_intensity": float(
                last_info.get(
                    "collision_intensity",
                    0.0,
                )
                or 0.0
            ),
            "stuck": bool(
                last_info.get("stuck", False)
            ),
            "stuck_step_count": int(
                last_info.get(
                    "stuck_step_count",
                    0,
                )
                or 0
            ),
            "wall_time_seconds": float(
                time.time() - wall_start
            ),
            "cpu_time_seconds": float(
                time.process_time() - cpu_start
            ),
            "gpu_time_seconds": stop_gpu_timer(
                gpu_timer,
                args.device,
            ),
            "gpu_memory_mb": gpu_peak_mb(
                args.device
            ),
            "process_memory_mb": process_memory_mb(),
            **system_memory_metrics(),
            **smi_metrics(),
            "average_loss": avg(losses),
            "loss_updates": len(losses),
            "average_rnd_intrinsic": (
                rnd_sum / max(steps, 1)
            ),
            "average_count_intrinsic": (
                count_sum / max(steps, 1)
            ),
            "average_rnd_loss": (
                rnd_loss_sum
                / max(rnd_loss_count, 1)
            ),
            "replay_buffer_size": len(
                agent.replay_buffer
            ),
            "learn_steps": agent.learn_steps,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "learning_rate": learning_rate,
            "lr_multiplier": 1.0,
            "dqn_technology": (
                "DQN + RND + CountBased + "
                "TargetNetwork + ReplayBuffer + Adam"
            ),
            "rnd_beta": args.rnd_beta,
            "count_beta": args.count_beta,
            "count_state_bin_size": (
                args.count_state_bin_size
            ),
            "action_source_counts": json.dumps(
                source_counts,
                sort_keys=True,
            ),
        }
        rows.append(row)

        if (
            (episode + 1) % args.episode_block_size == 0
            or (episode + 1) == args.train_episodes
        ):
            block_size = min(
                args.episode_block_size,
                len(rows),
            )
            block_rows.append(
                block_summary(
                    rows[-block_size:],
                    experiment,
                    "train",
                    learning_rate,
                    len(block_rows) + 1,
                )
            )

        print(
            f"TRAIN {SHORT_LABELS[experiment]:14s} "
            f"ep={episode:03d} "
            f"reward={env_reward:.2f} "
            f"total={training_reward:.2f} "
            f"steps={steps} "
            f"term={row['termination_reason']} "
            f"coll={row['collision_actor_type']} "
            f"wall={row['wall_time_seconds']:.2f}s "
            f"cpu={row['cpu_time_seconds']:.2f}s "
            f"gpu={row['gpu_time_seconds']:.2f}s "
            f"ram={row['ram_used_mb']:.0f}/"
            f"{row['ram_total_mb']:.0f}MB "
            f"vram={row['gpu_memory_used_mb_smi']:.0f}/"
            f"{row['gpu_memory_total_mb_smi']:.0f}MB "
            f"loss={row['average_loss']:.5f}",
            flush=True,
        )

    agent.update_target_network()

    model_path = (
        output_dir
        / "models"
        / f"{experiment}_lrmult_1.pt"
    )
    agent.save(str(model_path))

    pd.DataFrame(rows).to_csv(
        output_dir
        / f"{experiment}_lrmult_1_train.csv",
        index=False,
    )

    runtime = {
        "phase": "train",
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "learning_rate": learning_rate,
        "lr_multiplier": 1.0,
        "total_wall_time_seconds": (
            time.time() - experiment_wall_start
        ),
        "total_cpu_time_seconds": (
            time.process_time()
            - experiment_cpu_start
        ),
        "total_gpu_time_seconds": sum(
            float(row["gpu_time_seconds"])
            for row in rows
        ),
        "gpu_memory_mb": gpu_peak_mb(
            args.device
        ),
        "process_memory_mb": process_memory_mb(),
        "episodes": args.train_episodes,
        "model_path": str(model_path),
    }

    env.close()
    print(
        f"===== TRAINING END: "
        f"{SHORT_LABELS[experiment]} =====",
        flush=True,
    )

    return (
        agent,
        train_rewards,
        rows,
        block_rows,
        runtime,
    )


def test_one(
    experiment: str,
    agent: RNDCountDQNAgent,
    args,
    learning_rate: float,
    output_dir: Path,
    experts=None,
):
    env = make_env(args, output_dir)

    agent.freeze_for_eval()
    if experts:
        for _, expert in experts:
            expert.freeze_for_eval()

    rows = []
    block_rows = []

    reset_gpu_peak(args.device)
    experiment_wall_start = time.time()
    experiment_cpu_start = time.process_time()

    print(
        f"\n===== TESTING START: "
        f"{SHORT_LABELS[experiment]} =====",
        flush=True,
    )

    with torch.no_grad():
        for episode in range(args.test_episodes):
            state, _ = env.reset(
                seed=args.seed
            )

            total_reward = 0.0
            steps = 0
            done = False
            source_counts: Dict[str, int] = {}
            last_info: Dict = {}

            wall_start = time.time()
            cpu_start = time.process_time()
            gpu_timer = make_gpu_timer(args.device)

            while not done:
                action = agent.best_action(state)
                source = "frozen_greedy"

                source_counts[source] = (
                    source_counts.get(source, 0) + 1
                )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                last_info = dict(info or {})
                done = bool(terminated or truncated)
                total_reward += float(reward)
                steps += 1

            row = {
                "phase": "test",
                "experiment": experiment,
                "method": SHORT_LABELS[experiment],
                "episode": episode,
                "env_reward": float(total_reward),
                "steps": steps,
                "termination_reason": last_info.get(
                    "termination_reason",
                    "unknown",
                ),
                "ended_before_max_steps": bool(
                    last_info.get(
                        "ended_before_max_steps",
                        steps < args.max_episode_steps,
                    )
                ),
                "collision": bool(
                    last_info.get(
                        "collision",
                        False,
                    )
                ),
                "collision_actor_type": last_info.get(
                    "collision_actor_type",
                    "none",
                ),
                "collision_actor_id": last_info.get(
                    "collision_actor_id",
                    -1,
                ),
                "collision_actor_role_name": last_info.get(
                    "collision_actor_role_name",
                    "",
                ),
                "collision_intensity": float(
                    last_info.get(
                        "collision_intensity",
                        0.0,
                    )
                    or 0.0
                ),
                "stuck": bool(
                    last_info.get("stuck", False)
                ),
                "stuck_step_count": int(
                    last_info.get(
                        "stuck_step_count",
                        0,
                    )
                    or 0
                ),
                "wall_time_seconds": float(
                    time.time() - wall_start
                ),
                "cpu_time_seconds": float(
                    time.process_time() - cpu_start
                ),
                "gpu_time_seconds": stop_gpu_timer(
                    gpu_timer,
                    args.device,
                ),
                "gpu_memory_mb": gpu_peak_mb(
                    args.device
                ),
                "process_memory_mb": (
                    process_memory_mb()
                ),
                **system_memory_metrics(),
                **smi_metrics(),
                "epsilon": args.epsilon,
                "gamma": args.gamma,
                "learning_rate": learning_rate,
                "lr_multiplier": 1.0,
                "network_frozen": True,
                "updates_during_test": 0,
                "action_source_counts": json.dumps(
                    source_counts,
                    sort_keys=True,
                ),
            }
            rows.append(row)

            if (
                (episode + 1)
                % args.episode_block_size
                == 0
                or (episode + 1)
                == args.test_episodes
            ):
                block_size = min(
                    args.episode_block_size,
                    len(rows),
                )
                block_rows.append(
                    block_summary(
                        rows[-block_size:],
                        experiment,
                        "test",
                        learning_rate,
                        len(block_rows) + 1,
                    )
                )

            print(
                f"TEST  {SHORT_LABELS[experiment]:14s} "
                f"ep={episode:03d} "
                f"reward={total_reward:.2f} "
                f"steps={steps} "
                f"term={row['termination_reason']} "
                f"coll={row['collision_actor_type']} "
                f"wall={row['wall_time_seconds']:.2f}s "
                f"cpu={row['cpu_time_seconds']:.2f}s "
                f"gpu={row['gpu_time_seconds']:.2f}s "
                f"ram={row['ram_used_mb']:.0f}/"
                f"{row['ram_total_mb']:.0f}MB "
                f"vram={row['gpu_memory_used_mb_smi']:.0f}/"
                f"{row['gpu_memory_total_mb_smi']:.0f}MB",
                flush=True,
            )

    runtime = {
        "phase": "test",
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "learning_rate": learning_rate,
        "lr_multiplier": 1.0,
        "total_wall_time_seconds": (
            time.time() - experiment_wall_start
        ),
        "total_cpu_time_seconds": (
            time.process_time()
            - experiment_cpu_start
        ),
        "total_gpu_time_seconds": sum(
            float(row["gpu_time_seconds"])
            for row in rows
        ),
        "gpu_memory_mb": gpu_peak_mb(
            args.device
        ),
        "process_memory_mb": process_memory_mb(),
        "episodes": args.test_episodes,
        "network_frozen": True,
        "updates_during_test": 0,
    }

    env.close()
    print(
        f"===== TESTING END: "
        f"{SHORT_LABELS[experiment]} =====",
        flush=True,
    )

    return rows, block_rows, runtime


# ---------------------------------------------------------------------
# Summaries and outputs
# ---------------------------------------------------------------------

def summarize(
    experiment,
    train_rewards,
    train_rows,
    test_rows,
    args,
    learning_rate,
):
    train_values = [
        float(value) for value in train_rewards
    ]
    test_values = [
        float(row["env_reward"])
        for row in test_rows
    ]

    average_test = float(np.mean(test_values))
    convergence_ep = convergence_episode(
        train_values,
        average_test,
        args.convergence_threshold_fraction,
        args.convergence_window,
    )

    train_wall = sum(
        float(row["wall_time_seconds"])
        for row in train_rows
    )
    train_cpu = sum(
        float(row["cpu_time_seconds"])
        for row in train_rows
    )
    train_gpu = sum(
        float(row["gpu_time_seconds"])
        for row in train_rows
    )

    fraction = (
        convergence_ep
        / max(float(args.train_episodes), 1.0)
    )

    return {
        "experiment": experiment,
        "method": SHORT_LABELS[experiment],
        "lr_multiplier": 1.0,
        "learning_rate": learning_rate,
        "final_learning_rate": learning_rate,
        "average_train_reward": float(
            np.mean(train_values)
        ),
        "median_train_reward": float(
            np.median(train_values)
        ),
        "std_train_reward": float(
            np.std(train_values)
        ),
        "min_train_reward": float(
            np.min(train_values)
        ),
        "max_train_reward": float(
            np.max(train_values)
        ),
        "q1_train_reward": pct(
            train_values, 25
        ),
        "q3_train_reward": pct(
            train_values, 75
        ),
        "average_test_reward": average_test,
        "median_test_reward": float(
            np.median(test_values)
        ),
        "mode_test_reward": reward_mode(
            test_values
        ),
        "min_test_reward": float(
            np.min(test_values)
        ),
        "max_test_reward": float(
            np.max(test_values)
        ),
        "range_test_reward": float(
            np.max(test_values)
            - np.min(test_values)
        ),
        "std_test_reward": float(
            np.std(test_values)
        ),
        "q1_test_reward": pct(
            test_values, 25
        ),
        "q3_test_reward": pct(
            test_values, 75
        ),
        "convergence_episode": int(
            convergence_ep
        ),
        "convergence_time_seconds": (
            fraction * train_wall
        ),
        "convergence_cpu_time_seconds": (
            fraction * train_cpu
        ),
        "convergence_gpu_time_seconds": (
            fraction * train_gpu
        ),
        "total_training_wall_time_seconds": (
            train_wall
        ),
        "total_training_cpu_time_seconds": (
            train_cpu
        ),
        "total_training_gpu_time_seconds": (
            train_gpu
        ),
        "total_training_episodes": (
            args.train_episodes
        ),
        "test_rewards": test_values,
        "dqn_technology": (
            "DQN + RND + CountBased + "
            "TargetNetwork + ReplayBuffer + Adam"
        ),
        "rnd_beta": args.rnd_beta,
        "count_beta": args.count_beta,
        "count_state_bin_size": (
            args.count_state_bin_size
        ),
        "network_frozen_during_testing": True,
    }


def save_csvs(
    results,
    train_rows,
    test_rows,
    block_rows,
    runtime_rows,
    output_dir: Path,
):
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "test_rewards"
            }
            for row in results
        ]
    ).to_csv(
        output_dir
        / "all_experiments_learning_rate_summary.csv",
        index=False,
    )

    pd.DataFrame(train_rows).to_csv(
        output_dir
        / "all_experiments_train_episode_rewards.csv",
        index=False,
    )

    pd.DataFrame(test_rows).to_csv(
        output_dir
        / "all_experiments_test_episode_rewards.csv",
        index=False,
    )

    pd.DataFrame(block_rows).to_csv(
        output_dir
        / "all_experiments_episode_block_logs.csv",
        index=False,
    )

    pd.DataFrame(runtime_rows).to_csv(
        output_dir
        / "all_experiments_runtime_logs.csv",
        index=False,
    )


def apply_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
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


def save_figure(
    figure,
    output_dir: Path,
    name: str,
) -> None:
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{name}.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def ordered(results):
    return [
        next(
            row
            for row in results
            if row["experiment"] == experiment
        )
        for experiment in EXPERIMENTS
    ]


def make_figures(
    results,
    train_rows,
    test_rows,
    block_rows,
    output_dir: Path,
    args,
):
    apply_ieee_style()

    figure_dir = output_dir / "figures_ieee"
    figure_dir.mkdir(exist_ok=True)

    result_rows = ordered(results)
    labels = [
        SHORT_LABELS[row["experiment"]]
        for row in result_rows
    ]
    x = np.arange(len(result_rows))

    # Average test reward
    figure, axis = plt.subplots(
        figsize=(5.2, 3.4)
    )
    values = [
        row["average_test_reward"]
        for row in result_rows
    ]
    errors = [
        row["std_test_reward"]
        for row in result_rows
    ]
    axis.bar(
        x,
        values,
        yerr=errors,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )
    axis.set_ylabel("Average test reward")
    axis.set_xlabel("Experiment")
    axis.set_title(
        "Average Test Reward Across All Test Episodes"
    )
    save_figure(
        figure,
        figure_dir,
        "ieee_average_test_reward",
    )

    # Test box plot
    figure, axis = plt.subplots(
        figsize=(5.4, 3.4)
    )
    axis.boxplot(
        [
            row["test_rewards"]
            for row in result_rows
        ],
        tick_labels=labels,
        showfliers=True,
    )
    axis.set_ylabel("Environment reward")
    axis.set_xlabel("Experiment")
    axis.set_title(
        "Test Reward Distribution Across All Test Episodes"
    )
    save_figure(
        figure,
        figure_dir,
        "ieee_test_reward_boxplot",
    )

    # Convergence time and episode
    for key, ylabel, name, title in [
        (
            "convergence_time_seconds",
            "Convergence time (s)",
            "ieee_convergence_time",
            "Training-Data Convergence Time",
        ),
        (
            "convergence_episode",
            "Convergence episode",
            "ieee_convergence_episode",
            "Training Convergence Episode",
        ),
    ]:
        figure, axis = plt.subplots(
            figsize=(5.2, 3.4)
        )
        axis.bar(
            x,
            [
                row[key]
                for row in result_rows
            ],
            edgecolor="black",
            linewidth=0.7,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            labels,
            rotation=15,
            ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Experiment")
        axis.set_title(title)
        save_figure(
            figure,
            figure_dir,
            name,
        )

    # Block reward plots
    block_frame = pd.DataFrame(block_rows)
    for phase, name, title, ylabel in [
        (
            "train",
            "ieee_training_reward_blocks",
            "Training Reward by Episode Block",
            "Average training reward",
        ),
        (
            "test",
            "ieee_testing_reward_blocks",
            "Testing Reward by Episode Block",
            "Average test reward",
        ),
    ]:
        figure, axis = plt.subplots(
            figsize=(5.8, 3.6)
        )

        phase_frame = (
            block_frame[
                block_frame["phase"] == phase
            ]
            if not block_frame.empty
            else pd.DataFrame()
        )

        for experiment in EXPERIMENTS:
            data = (
                phase_frame[
                    phase_frame["experiment"]
                    == experiment
                ].sort_values("block_index")
                if not phase_frame.empty
                else pd.DataFrame()
            )

            if not data.empty:
                block_labels = [
                    f"{int(start)+1}-{int(end)+1}"
                    for start, end in zip(
                        data["start_episode"],
                        data["end_episode"],
                    )
                ]
                axis.plot(
                    block_labels,
                    data[
                        "average_env_reward"
                    ].astype(float),
                    marker="o",
                    linewidth=1.6,
                    label=SHORT_LABELS[
                        experiment
                    ],
                )

        axis.set_ylabel(ylabel)
        axis.set_xlabel("Episode block")
        axis.set_title(title)
        axis.legend(frameon=False)
        save_figure(
            figure,
            figure_dir,
            name,
        )


def write_readme(
    output_dir: Path,
    args,
) -> None:
    (
        output_dir
        / "README_sumo_set4_v2.txt"
    ).write_text(
        f"""SUMO set4_v2 DQN outputs

Experiments in order:
1. Epsilon Greedy
2. Median 50

Neural setup:
DQN + RND + Count-Based intrinsic reward
+ Target Network + Replay Buffer + Adam.
No NoisyNet.

Testing is frozen:
- no optimizer updates
- no replay updates
- no RND updates
- no count updates
- no target-network updates
- environment reward only

Main CSVs:
- all_experiments_train_episode_rewards.csv
- all_experiments_test_episode_rewards.csv
- all_experiments_episode_block_logs.csv
- all_experiments_runtime_logs.csv
- all_experiments_learning_rate_summary.csv

Figures:
- figures_ieee/*.png

Configuration:
train_episodes={args.train_episodes}
test_episodes={args.test_episodes}
max_episode_steps={args.max_episode_steps}
epsilon={args.epsilon}
learning_rate={args.raw_learning_rate}
gamma={args.gamma}
rnd_beta={args.rnd_beta}
count_beta={args.count_beta}
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run(args) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        output_dir / "models"
    ).mkdir(exist_ok=True)

    learning_rate = float(
        args.raw_learning_rate
        if args.raw_learning_rate > 0
        else 5e-5
    )

    print("=" * 72, flush=True)
    print(
        "SUMO set4_v2: "
        "DQN + RND + Count + Target + Replay + Adam",
        flush=True,
    )
    print(
        "Experiments: "
        "Epsilon Greedy, Median 50",
        flush=True,
    )
    print(
        f"Python: {platform.python_version()}",
        flush=True,
    )
    print(
        f"PyTorch: {torch.__version__}",
        flush=True,
    )
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}",
        flush=True,
    )
    print(
        f"Selected device: "
        f"{device_from_arg(args.device)}",
        flush=True,
    )
    print(
        f"Output dir: {output_dir}",
        flush=True,
    )
    print("=" * 72, flush=True)

    trained = {}
    train_rewards_by_experiment = {}
    train_rows_by_experiment = {}

    all_train_rows = []
    all_test_rows = []
    all_block_rows = []
    all_runtime_rows = []
    results = []

    for experiment in EXPERIMENTS:
        (
            agent,
            rewards,
            rows,
            blocks,
            runtime,
        ) = train_one(
            experiment,
            learning_rate,
            args,
            output_dir,
        )

        trained[experiment] = agent
        train_rewards_by_experiment[experiment] = rewards
        train_rows_by_experiment[experiment] = rows

        all_train_rows.extend(rows)
        all_block_rows.extend(blocks)
        all_runtime_rows.append(runtime)

        trained[experiment].freeze_for_eval()

    for experiment in EXPERIMENTS:
        (
            test_rows,
            test_blocks,
            test_runtime,
        ) = test_one(
            experiment,
            trained[experiment],
            args,
            learning_rate,
            output_dir,
        )

        all_test_rows.extend(test_rows)
        all_block_rows.extend(test_blocks)
        all_runtime_rows.append(test_runtime)

        results.append(
            summarize(
                experiment,
                train_rewards_by_experiment[experiment],
                train_rows_by_experiment[experiment],
                test_rows,
                args,
                learning_rate,
            )
        )

    save_csvs(
        results,
        all_train_rows,
        all_test_rows,
        all_block_rows,
        all_runtime_rows,
        output_dir,
    )

    make_figures(
        results,
        all_train_rows,
        all_test_rows,
        all_block_rows,
        output_dir,
        args,
    )

    write_readme(
        output_dir,
        args,
    )

    (
        output_dir / "config.json"
    ).write_text(
        json.dumps(
            vars(args),
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved SUMO set4_v2 outputs to: "
        f"{output_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-episodes",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--episode-block-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--raw-learning-rate",
        "--learning-rate",
        dest="raw_learning_rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--rnd-beta",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--rnd-learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--rnd-output-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--count-beta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--count-state-bin-size",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--target-speed",
        type=float,
        default=8.333333,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--convergence-threshold-fraction",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/sumo/"
            "results_sumo_set4_v2_dqn_500_300_500"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

