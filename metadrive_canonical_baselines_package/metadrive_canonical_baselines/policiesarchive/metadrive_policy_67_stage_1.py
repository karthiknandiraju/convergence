#!/usr/bin/env python3
"""MetaDrive Policy 67 Stage 1 DQN.

Author: Sai Durga Karthik Nandiraju
Last updated: 2026-07-16 CEST (+0200)

Training policy
---------------
* No main-diagonal rule.
* No anti-diagonal rule.
* No triangle or middle-episode overrides.
* At every training action:
  - 80%: maximum-Q action.
  - 20%: choose uniformly between L20 and U80.
* Therefore, in expectation:
  - 80% maximum-Q.
  - 10% L20.
  - 10% U80.
* U80 samples uniformly from actions with Q-values strictly above the
  80th percentile; fallback is maximum-Q.
* L20 samples uniformly from actions with Q-values strictly below the
  20th percentile; fallback is minimum-Q.
* Schedule selection is O(1) time and O(1) memory.
* U80/L20 selection is O(A) time with O(1) extra candidate memory, where A is
  the number of discrete actions (nine by default).

Shared setup
------------
* Plain DQN, target network, replay buffer, and Adam optimizer.
* No RND and no count-based intrinsic reward.
* Frozen greedy testing: no optimizer, replay, or target updates.
* Disjoint training and testing scenarios.
* Defaults: 500 train episodes, 300 test episodes, 500 maximum steps.

Example:
    python policies/metadrive_policy_67_stage_1.py \
      --seed 11 --test-seed 100000 \
      --train-episodes 500 --test-episodes 300 \
      --max-episode-steps 500 --device cuda

Results are saved to:
    policy_results/seed_<seed>/policy_67_stage_1
"""

from __future__ import annotations

import argparse
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


EXPERIMENTS = ["policy_67_stage_1"]
SHORT_LABELS = {
    "policy_67_stage_1": "Policy 67 Stage 1",
}
COLORS = {
    "policy_67_stage_1": "#2ca02c",
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
    def _random_extreme_from_q(q: np.ndarray, maximum: bool) -> int:
        """Reservoir-sample tied extrema without allocating a candidate array."""
        extreme = float(np.max(q) if maximum else np.min(q))
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
        return self._random_extreme_from_q(q, maximum=True)

    def lowest_q_action(self, state: np.ndarray) -> int:
        """Choose the action with the lowest predicted long-term return."""
        q = self.q_values(state)
        return self._random_extreme_from_q(q, maximum=False)

    def percentile_tail_action(self, state: np.ndarray, upper: bool) -> int:
        """Sample U80/L20 in O(A) time with O(1) extra candidate memory.

        ``A`` is the discrete action count (nine with the default configuration).
        The Q vector is required network output; this method does not allocate a
        sorted action list or candidate mask.
        """
        q = self.q_values(state)
        percentile = 80.0 if upper else 20.0
        threshold = float(np.percentile(q, percentile))
        chosen = -1
        matches = 0
        for action, value in enumerate(q):
            eligible = float(value) > threshold if upper else float(value) < threshold
            if eligible:
                matches += 1
                if random.randrange(matches) == 0:
                    chosen = action
        if chosen >= 0:
            return int(chosen)
        return self._random_extreme_from_q(q, maximum=upper)

    def upper_80_action(self, state: np.ndarray) -> int:
        return self.percentile_tail_action(state, upper=True)

    def lower_20_action(self, state: np.ndarray) -> int:
        return self.percentile_tail_action(state, upper=False)

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



def select_training_action(
    experiment: str,
    agent: DQNAgent,
    state: np.ndarray,
    episode: int,
    step: int,
    args,
) -> Tuple[int, str]:
    """Apply Policy 67 Stage 1 at every training action.

    - 80% maximum-Q.
    - 20% exploration, split uniformly between L20 and U80.

    The branch decision is O(1) time and O(1) memory. L20/U80 scan the
    A available Q-values in O(A) time with O(1) extra candidate memory.
    """
    if experiment != "policy_67_stage_1":
        raise ValueError(f"Unknown experiment: {experiment}")

    if random.random() < 0.80:
        return agent.greedy_action(state), "policy_67_stage_1_argmax80"

    if random.random() < 0.50:
        return agent.lower_20_action(state), "policy_67_stage_1_l20"
    return agent.upper_80_action(state), "policy_67_stage_1_u80"



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
                training_reward = float(env_reward)
                agent.replay.add(state, action, training_reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                env_reward_total += float(env_reward)
                training_reward_total += training_reward
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
        description="MetaDrive Policy 67 Stage 1: 80% argmax and 20% L20/U80 exploration"
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
            "<project>/policy_results/seed_<seed>/policy_67_stage_1."
        ),
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed < 0 or args.test_seed < 0:
        raise ValueError("--seed and --test-seed must be non-negative.")
    if args.train_episodes < 2 or args.max_episode_steps < 2:
        raise ValueError(
            "The four-triangle schedule requires at least two training episodes "
            "and two maximum episode steps."
        )
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


def resolve_output_dir(args) -> Path:
    """Return the only permitted seed-specific policy output directory."""
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent if script_dir.name == "policies" else script_dir
    output_dir = (
        project_dir / "policy_results" / f"seed_{args.seed}" / "policy_67_stage_1"
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
    print("METADRIVE POLICY 67 STAGE 1 DQN POLICY")
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
    print("Policy name: policy_67_stage_1")
    print("Diagonal and anti-diagonal rules: disabled")
    print("All training actions: 80% maximum-Q, 10% L20, 10% U80")
    print("Schedule complexity: O(1) time / O(1) memory")
    print("L20/U80 complexity: O(A) time / O(1) extra memory")
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
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
