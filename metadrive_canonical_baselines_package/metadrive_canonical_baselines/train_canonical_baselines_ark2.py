#!/usr/bin/env python3
"""Train canonical MetaDrive collision baselines once per seed.

Baselines
---------
* epsilon: plain DQN, epsilon-greedy training, environment reward only.
* noisy: factorized-Gaussian NoisyNet DQN, no epsilon.
* rnd: plain DQN, epsilon-greedy training, normalized RND bonus during training.

Every method receives its own freshly seeded model, optimizer, replay buffer,
and environments. Frozen testing uses the same scenario seeds for all methods,
argmax actions only, and no optimizer/replay/RND updates.  Outputs are stored
under ``OUTPUT_ROOT/seed_<seed>/<method>`` and are protected from accidental
overwrite unless ``--force`` is supplied.

Each episode row records wall and CPU time. Each method directory also contains
``runtime_statistics.csv`` with total and average train/test phase timings for
use in policy-versus-baseline timing comparisons.
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
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

# Must be set before CUDA creates a cuBLAS context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    import metadrive
    from metadrive import MetaDriveEnv
except ImportError as exc:  # pragma: no cover - depends on runtime
    raise SystemExit(
        "MetaDrive is not installed. Run: python -m pip install metadrive-simulator"
    ) from exc


METHOD_LABELS = {
    "epsilon": "Epsilon Greedy",
    "noisy": "NoisyNet DQN",
    "rnd": "DQN + RND",
}
CRITICAL_CONFIG_KEYS = (
    "seed",
    "deterministic",
    "train_episodes",
    "test_episodes",
    "max_episode_steps",
    "epsilon",
    "learning_rate",
    "gamma",
    "batch_size",
    "replay_capacity",
    "target_update_steps",
    "hidden_size",
    "noisy_sigma_init",
    "rnd_beta",
    "rnd_learning_rate",
    "rnd_output_size",
    "rnd_bonus_clip",
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
)


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


def seed_everything(seed: int, deterministic: bool = True) -> None:
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
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_observation(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def observation_sha256(observation) -> str:
    array = np.ascontiguousarray(flatten_observation(observation))
    return hashlib.sha256(array.tobytes()).hexdigest()


def bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def restricted_mean_survival_time(
    times: Sequence[float], events: Sequence[bool], tau: float
) -> float:
    """Kaplan-Meier restricted mean collision-free survival through ``tau``."""
    if not times or tau <= 0:
        return 0.0
    original_t = np.asarray(times, dtype=float)
    t = np.minimum(original_t, float(tau))
    e = np.asarray(events, dtype=bool) & (original_t <= float(tau))
    survival = 1.0
    area = 0.0
    previous = 0.0
    for current in np.unique(t[e]):
        current = float(current)
        area += survival * max(0.0, current - previous)
        at_risk = int(np.sum(t >= current))
        failures = int(np.sum(e & np.isclose(t, current)))
        if at_risk and failures:
            survival *= 1.0 - failures / at_risk
        previous = current
    area += survival * max(0.0, float(tau) - previous)
    return float(area)


def collision_summary(rows: Sequence[Dict], tau: int) -> Dict[str, float]:
    collisions = sum(bool_value(row["collision"]) for row in rows)
    offroad = sum(bool_value(row["out_of_road"]) for row in rows)
    goals = sum(bool_value(row["goal_reached"]) for row in rows)
    steps = sum(int(row["steps"]) for row in rows)
    episodes = len(rows)
    return {
        "episodes": episodes,
        "collision_count": collisions,
        "out_of_road_count": offroad,
        "total_steps": steps,
        "collision_rmst_event_definition": "collision",
        "collision_rmst": restricted_mean_survival_time(
            [int(row["event_or_censor_time_steps"]) for row in rows],
            [bool_value(row["collision"]) for row in rows],
            tau,
        ),
        "collisions_per_1000_steps": 1000.0 * collisions / steps if steps else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "out_of_road_rate": offroad / episodes if episodes else 0.0,
        "goal_rate": goals / episodes if episodes else 0.0,
    }


def make_env(args, phase: str) -> MetaDriveEnv:
    if phase not in {"train", "test"}:
        raise ValueError("phase must be train or test")
    start_seed = args.seed if phase == "train" else args.test_seed
    scenario_count = args.train_episodes if phase == "train" else args.test_episodes
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
            "num_scenarios": int(max(1, scenario_count)),
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


def verify_discrete_action_space(env, args) -> int:
    action_space = env.action_space
    if not hasattr(action_space, "n"):
        raise RuntimeError("MetaDrive action space is not Discrete.")
    action_count = int(action_space.n)
    expected = int(args.discrete_steering_dim * args.discrete_throttle_dim)
    if action_count != expected or action_count != 9:
        raise RuntimeError(
            "Canonical comparisons require exactly nine discrete actions; "
            f"configured={expected}, exposed={action_count}."
        )
    invalid = [
        action for action in range(action_count)
        if not action_space.contains(action)
    ]
    if invalid or action_space.contains(action_count) or action_space.contains(-1):
        raise RuntimeError("Discrete action IDs must be contiguous from 0 through 8.")
    return action_count


def truthy(info: Dict, *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)


def parse_step_info(info: Dict, terminated: bool, truncated: bool) -> Dict:
    crash_vehicle = truthy(info, "crash_vehicle")
    crash_object = truthy(
        info, "crash_object", "crash_building", "crash_human", "crash_sidewalk"
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
    }


class NoisyLinear(nn.Module):
    """Factorized Gaussian layer used by NoisyNet DQN."""

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
    def scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        values = torch.randn(size, device=device)
        return values.sign() * values.abs().sqrt()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        noise_in = self.scaled_noise(self.in_features, self.weight_mu.device)
        noise_out = self.scaled_noise(self.out_features, self.weight_mu.device)
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
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.noisy = bool(noisy)

        def linear(input_size: int, output_size: int):
            if self.noisy:
                return NoisyLinear(input_size, output_size, sigma_init)
            return nn.Linear(input_size, output_size)

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
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


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
    def __init__(self, capacity: int, seed: int):
        self.data: Deque[Transition] = deque(maxlen=int(capacity))
        self.rng = random.Random(int(seed))

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
        return self.rng.sample(list(self.data), int(batch_size))

    def __len__(self) -> int:
        return len(self.data)


class RunningVariance:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def normalize_then_update(self, value: float, clip: float) -> float:
        variance = self.m2 / max(1, self.count - 1) if self.count > 1 else 1.0
        normalized = float(value) / math.sqrt(max(variance, 1e-8))
        self.count += 1
        delta = float(value) - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (float(value) - self.mean)
        return float(np.clip(normalized, 0.0, clip))


class BaselineAgent:
    def __init__(
        self,
        observation_size: int,
        action_count: int,
        method: str,
        args,
        device: torch.device,
    ):
        self.method = method
        self.observation_size = int(observation_size)
        self.action_count = int(action_count)
        self.device = device
        self.gamma = float(args.gamma)
        self.batch_size = int(args.batch_size)
        self.target_update_steps = int(args.target_update_steps)
        self.learn_steps = 0
        self.noisy = method == "noisy"
        self.policy_rng = random.Random(args.seed + 10_001)
        self.online = QNetwork(
            observation_size,
            action_count,
            args.hidden_size,
            noisy=self.noisy,
            sigma_init=args.noisy_sigma_init,
        ).to(device)
        self.target = QNetwork(
            observation_size,
            action_count,
            args.hidden_size,
            noisy=self.noisy,
            sigma_init=args.noisy_sigma_init,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.train(self.noisy)
        self.optimizer = optim.Adam(self.online.parameters(), lr=args.learning_rate)
        self.replay = ReplayBuffer(args.replay_capacity, args.seed + 20_003)

        self.rnd_target: Optional[RNDNetwork] = None
        self.rnd_predictor: Optional[RNDNetwork] = None
        self.rnd_optimizer: Optional[optim.Optimizer] = None
        self.rnd_normalizer = RunningVariance()
        if method == "rnd":
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

    @staticmethod
    def tie_action(candidates: np.ndarray, key: str) -> int:
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big")
        return int(candidates[offset % len(candidates)])

    def argmax(self, state: np.ndarray, key: str, use_noise: bool = False) -> int:
        if self.noisy and use_noise:
            self.online.train()
            self.online.reset_noise()
        else:
            self.online.eval()
        with torch.no_grad():
            q_values = self.online(self.tensor(state))[0]
        maximum = torch.max(q_values)
        candidates = torch.nonzero(q_values == maximum, as_tuple=False).flatten()
        candidates_np = candidates.detach().cpu().numpy()
        return self.tie_action(candidates_np, key)

    def training_action(
        self, state: np.ndarray, episode: int, step: int, epsilon: float
    ) -> Tuple[int, str]:
        key = f"train|{self.method}|{episode}|{step}"
        if self.noisy:
            return self.argmax(state, key, use_noise=True), "noisy_argmax"
        if self.policy_rng.random() < epsilon:
            return self.policy_rng.randrange(self.action_count), "epsilon_random"
        return self.argmax(state, key), "epsilon_argmax"

    def test_action(self, state: np.ndarray, seed: int, episode: int, step: int) -> int:
        return self.argmax(state, f"test|{seed}|{episode}|{step}", use_noise=False)

    def rnd_bonus_and_update(self, next_state: np.ndarray, args) -> Tuple[float, float]:
        if self.method != "rnd":
            return 0.0, 0.0
        assert self.rnd_target is not None
        assert self.rnd_predictor is not None
        assert self.rnd_optimizer is not None
        x = self.tensor(next_state)
        with torch.no_grad():
            target = self.rnd_target(x)
            prediction_before = self.rnd_predictor(x)
            raw_bonus = float(F.mse_loss(prediction_before, target).item())
        prediction = self.rnd_predictor(x)
        loss = F.mse_loss(prediction, target)
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.rnd_predictor.parameters(), 10.0)
        self.rnd_optimizer.step()
        normalized = self.rnd_normalizer.normalize_then_update(
            raw_bonus, args.rnd_bonus_clip
        )
        return float(args.rnd_beta * normalized), float(loss.detach().cpu().item())

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
        if self.noisy:
            self.online.reset_noise()
            self.target.train()
            self.target.reset_noise()
        predicted = self.online(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.gamma * next_q
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def freeze(self) -> None:
        modules: List[nn.Module] = [self.online, self.target]
        if self.rnd_target is not None:
            modules.append(self.rnd_target)
        if self.rnd_predictor is not None:
            modules.append(self.rnd_predictor)
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def checkpoint(self, args) -> Dict:
        state = {
            "method": self.method,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "observation_size": self.observation_size,
            "action_count": self.action_count,
            "config": json_safe(vars(args)),
        }
        if self.rnd_target is not None and self.rnd_predictor is not None:
            state["rnd_target"] = self.rnd_target.state_dict()
            state["rnd_predictor"] = self.rnd_predictor.state_dict()
            state["rnd_optimizer"] = self.rnd_optimizer.state_dict()
        return state


def episode_row(
    phase: str,
    method: str,
    episode: int,
    scenario_seed: int,
    initial_hash: str,
    env_reward: float,
    training_reward: float,
    steps: int,
    parsed: Dict,
    losses: Sequence[float],
    rnd_losses: Sequence[float],
    rnd_bonuses: Sequence[float],
    action_sources: Dict[str, int],
    agent: BaselineAgent,
    wall_time_seconds: float,
    cpu_time_seconds: float,
    args,
) -> Dict:
    return {
        "phase": phase,
        "experiment": method,
        "method": METHOD_LABELS[method],
        "seed": args.seed,
        "episode": episode,
        "scenario_seed": scenario_seed,
        "initial_observation_sha256": initial_hash,
        "env_reward": float(env_reward),
        "training_reward": float(training_reward),
        "steps": int(steps),
        **parsed,
        "rmst_event_definition": "collision",
        "rmst_event_observed": bool(parsed["collision"]),
        "event_or_censor_time_steps": int(steps),
        "wall_time_seconds": float(wall_time_seconds),
        "cpu_time_seconds": float(cpu_time_seconds),
        "average_loss": float(np.mean(losses)) if losses else 0.0,
        "average_rnd_loss": float(np.mean(rnd_losses)) if rnd_losses else 0.0,
        "average_rnd_bonus": float(np.mean(rnd_bonuses)) if rnd_bonuses else 0.0,
        "replay_buffer_size": len(agent.replay),
        "learn_steps": agent.learn_steps,
        "epsilon": args.epsilon if method in {"epsilon", "rnd"} else 0.0,
        "rnd_beta": args.rnd_beta if method == "rnd" else 0.0,
        "noisy_sigma_init": args.noisy_sigma_init if method == "noisy" else 0.0,
        "network_frozen": phase == "test",
        "updates_during_test": 0 if phase == "test" else "",
        "action_source_counts": json.dumps(action_sources, sort_keys=True),
    }


def run_method(
    method: str, args, device: torch.device, method_dir: Path
) -> Tuple[List[Dict], List[Dict]]:
    seed_everything(args.seed, args.deterministic)
    rows: List[Dict] = []
    train_env = make_env(args, "train")
    try:
        initial, _ = train_env.reset(seed=args.seed)
        observation_size = int(flatten_observation(initial).size)
        action_count = verify_discrete_action_space(train_env, args)
    except Exception:
        train_env.close()
        raise
    agent = BaselineAgent(observation_size, action_count, method, args, device)

    print(f"\n===== TRAIN {METHOD_LABELS[method]} / seed {args.seed} =====", flush=True)
    training_wall_start = time.perf_counter()
    training_cpu_start = time.process_time()
    try:
        for episode in range(args.train_episodes):
            episode_wall_start = time.perf_counter()
            episode_cpu_start = time.process_time()
            scenario_seed = args.seed + episode
            raw_state, _ = train_env.reset(seed=scenario_seed)
            state = flatten_observation(raw_state)
            initial_hash = observation_sha256(raw_state)
            env_total = 0.0
            train_total = 0.0
            losses: List[float] = []
            rnd_losses: List[float] = []
            rnd_bonuses: List[float] = []
            sources: Dict[str, int] = {}
            parsed = parse_step_info({}, False, False)
            for step in range(args.max_episode_steps):
                action, source = agent.training_action(state, episode, step, args.epsilon)
                sources[source] = sources.get(source, 0) + 1
                next_raw, env_reward, terminated, truncated, info = train_env.step(action)
                next_state = flatten_observation(next_raw)
                done = bool(terminated or truncated)
                rnd_bonus, rnd_loss = agent.rnd_bonus_and_update(next_state, args)
                training_reward = float(env_reward) + rnd_bonus
                agent.replay.add(state, action, training_reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                if method == "rnd":
                    rnd_losses.append(rnd_loss)
                    rnd_bonuses.append(rnd_bonus)
                env_total += float(env_reward)
                train_total += training_reward
                state = next_state
                parsed = parse_step_info(info, bool(terminated), bool(truncated))
                if done:
                    break
            rows.append(
                episode_row(
                    "train", method, episode, scenario_seed, initial_hash,
                    env_total, train_total, step + 1, parsed, losses,
                    rnd_losses, rnd_bonuses, sources, agent,
                    time.perf_counter() - episode_wall_start,
                    time.process_time() - episode_cpu_start,
                    args,
                )
            )
            if (episode + 1) % args.progress_every == 0:
                print(
                    f"{METHOD_LABELS[method]} train {episode + 1}/{args.train_episodes}",
                    flush=True,
                )
    finally:
        train_env.close()
    training_wall_seconds = time.perf_counter() - training_wall_start
    training_cpu_seconds = time.process_time() - training_cpu_start
    print(
        f"{METHOD_LABELS[method]} training duration: "
        f"{training_wall_seconds:.2f}s wall, {training_cpu_seconds:.2f}s CPU",
        flush=True,
    )

    model_path = method_dir / "model.pt"
    torch.save(agent.checkpoint(args), model_path)
    agent.freeze()
    test_env = make_env(args, "test")
    print(f"===== TEST {METHOD_LABELS[method]} / seed {args.seed} =====", flush=True)
    testing_wall_start = time.perf_counter()
    testing_cpu_start = time.process_time()
    try:
        with torch.no_grad():
            for episode in range(args.test_episodes):
                episode_wall_start = time.perf_counter()
                episode_cpu_start = time.process_time()
                scenario_seed = args.test_seed + episode
                raw_state, _ = test_env.reset(seed=scenario_seed)
                state = flatten_observation(raw_state)
                initial_hash = observation_sha256(raw_state)
                total = 0.0
                sources: Dict[str, int] = {}
                parsed = parse_step_info({}, False, False)
                for step in range(args.max_episode_steps):
                    action = agent.test_action(state, args.seed, episode, step)
                    sources["frozen_argmax"] = sources.get("frozen_argmax", 0) + 1
                    next_raw, reward, terminated, truncated, info = test_env.step(action)
                    state = flatten_observation(next_raw)
                    total += float(reward)
                    parsed = parse_step_info(info, bool(terminated), bool(truncated))
                    if terminated or truncated:
                        break
                rows.append(
                    episode_row(
                        "test", method, episode, scenario_seed, initial_hash,
                        total, total, step + 1, parsed, (), (), (), sources,
                        agent,
                        time.perf_counter() - episode_wall_start,
                        time.process_time() - episode_cpu_start,
                        args,
                    )
                )
                if (episode + 1) % args.progress_every == 0:
                    print(
                        f"{METHOD_LABELS[method]} test {episode + 1}/{args.test_episodes}",
                        flush=True,
                    )
    finally:
        test_env.close()
    testing_wall_seconds = time.perf_counter() - testing_wall_start
    testing_cpu_seconds = time.process_time() - testing_cpu_start
    print(
        f"{METHOD_LABELS[method]} testing duration: "
        f"{testing_wall_seconds:.2f}s wall, {testing_cpu_seconds:.2f}s CPU",
        flush=True,
    )
    runtime_rows = []
    for phase, wall_seconds, cpu_seconds in (
        ("train", training_wall_seconds, training_cpu_seconds),
        ("test", testing_wall_seconds, testing_cpu_seconds),
    ):
        phase_rows = [row for row in rows if row["phase"] == phase]
        runtime_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
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
    return rows, runtime_rows


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


def prepare_method_dir(path: Path, force: bool) -> None:
    manifest = path / "manifest.json"
    if manifest.exists() and not force:
        raise FileExistsError(
            f"Canonical baseline already exists: {path}. Reuse it or pass --force."
        )
    path.mkdir(parents=True, exist_ok=True)
    if force:
        for filename in (
            "model.pt", "all_episode_results.csv", "collision_metrics.csv",
            "runtime_statistics.csv", "config.json", "manifest.json",
        ):
            target = path / filename
            if target.exists():
                target.unlink()


def save_method(
    method: str,
    rows: List[Dict],
    runtime_rows: List[Dict],
    args,
    method_dir: Path,
) -> Dict:
    results_path = method_dir / "all_episode_results.csv"
    config_path = method_dir / "config.json"
    metrics_path = method_dir / "collision_metrics.csv"
    runtime_path = method_dir / "runtime_statistics.csv"
    write_csv(results_path, rows)
    write_csv(runtime_path, runtime_rows)
    config = json_safe(vars(args).copy())
    config.update({"method": method, "method_label": METHOD_LABELS[method]})
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    metric_rows = []
    for phase in ("train", "test"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        summary = collision_summary(phase_rows, args.rmst_tau)
        metric_rows.append(
            {"method": method, "method_label": METHOD_LABELS[method], "phase": phase, **summary}
        )
    write_csv(metrics_path, metric_rows)
    model_path = method_dir / "model.pt"
    manifest = {
        "completed": True,
        "environment": "MetaDrive",
        "metadrive_version": getattr(metadrive, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "seed": args.seed,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "observation_source": "flattened_metadrive_observation_only",
        "uses_engine_object_safety_scan": False,
        "test_policy": "frozen_dqn_argmax",
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
    manifest_path = method_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train canonical Epsilon, NoisyNet, and DQN+RND MetaDrive baselines"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_LABELS), default=list(METHOD_LABELS)
    )
    parser.add_argument("--output-root", type=Path, default=Path("canonical_baselines"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-steps", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--noisy-sigma-init", type=float, default=0.5)
    parser.add_argument("--rnd-beta", type=float, default=0.01)
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnd-output-size", type=int, default=64)
    parser.add_argument("--rnd-bonus-clip", type=float, default=5.0)
    parser.add_argument("--discrete-steering-dim", type=int, default=3)
    parser.add_argument("--discrete-throttle-dim", type=int, default=3)
    parser.add_argument("--map-blocks", type=int, default=3)
    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--accident-prob", type=float, default=0.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=50.0)
    parser.add_argument("--out-of-road-penalty", type=float, default=10.0)
    parser.add_argument("--metadrive-log-level", type=int, default=50)
    parser.add_argument("--rmst-tau", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    positive = {
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_episode_steps": args.max_episode_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "target_update_steps": args.target_update_steps,
        "hidden_size": args.hidden_size,
        "noisy_sigma_init": args.noisy_sigma_init,
        "rnd_learning_rate": args.rnd_learning_rate,
        "rnd_output_size": args.rnd_output_size,
        "rnd_bonus_clip": args.rnd_bonus_clip,
        "discrete_steering_dim": args.discrete_steering_dim,
        "discrete_throttle_dim": args.discrete_throttle_dim,
        "map_blocks": args.map_blocks,
        "rmst_tau": args.rmst_tau,
        "progress_every": args.progress_every,
    }
    invalid = [
        name for name, value in positive.items()
        if not math.isfinite(float(value)) or value <= 0
    ]
    if invalid:
        parser.error(
            "these arguments must be finite and positive: " + ", ".join(invalid)
        )
    if args.seed < 0 or args.test_seed < 0:
        parser.error("--seed and --test-seed must be non-negative")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must be between 0 and 1")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be between 0 and 1")
    if not 0.0 <= args.traffic_density <= 1.0:
        parser.error("--traffic-density must be between 0 and 1")
    if not 0.0 <= args.accident_prob <= 1.0:
        parser.error("--accident-prob must be between 0 and 1")
    if not math.isfinite(args.rnd_beta) or args.rnd_beta < 0.0:
        parser.error("--rnd-beta must be finite and non-negative")
    if not math.isfinite(args.success_reward):
        parser.error("--success-reward must be finite")
    if (
        not math.isfinite(args.collision_penalty)
        or not math.isfinite(args.out_of_road_penalty)
        or args.collision_penalty < 0.0
        or args.out_of_road_penalty < 0.0
    ):
        parser.error("environment penalty magnitudes must be finite and non-negative")
    if args.replay_capacity < args.batch_size:
        parser.error("--replay-capacity must be at least --batch-size")
    if args.discrete_steering_dim * args.discrete_throttle_dim != 9:
        parser.error("canonical comparisons require exactly nine discrete actions")
    train_end = args.seed + args.train_episodes - 1
    test_end = args.test_seed + args.test_episodes - 1
    if max(args.seed, args.test_seed) <= min(train_end, test_end):
        parser.error("training and testing scenario-seed ranges must not overlap")
    args.methods = list(dict.fromkeys(args.methods))
    return args


def main() -> None:
    args = parse_args()
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        print(
            f"WARNING: launch with PYTHONHASHSEED={args.seed} for complete process reproducibility",
            file=sys.stderr,
        )
    device = choose_device(args.device)
    args.device = str(device)
    seed_root = args.output_root.resolve() / f"seed_{args.seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for method in args.methods:
        method_dir = seed_root / method
        prepare_method_dir(method_dir, args.force)
        rows, runtime_rows = run_method(method, args, device, method_dir)
        manifests.append(
            save_method(method, rows, runtime_rows, args, method_dir)
        )
    index = {
        "seed": args.seed,
        "environment": "MetaDrive",
        "observation_source": "flattened_metadrive_observation_only",
        "uses_engine_object_safety_scan": False,
        "methods": manifests,
        "critical_config": critical_config(args),
        "critical_config_sha256": canonical_json_sha256(critical_config(args)),
    }
    (seed_root / "baseline_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\nCanonical baselines saved to: {seed_root}")


if __name__ == "__main__":
    main()
