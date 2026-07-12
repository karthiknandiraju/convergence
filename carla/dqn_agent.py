"""DQN agents for the real CARLA driving environment.

Includes:
1. Base DQN
2. Epsilon-greedy DQN
3. NoisyNet DQN
4. RND DQN
5. Noisy + RND DQN
6. Ensemble DQN support

All agents use function approximation, replay buffer, target network,
and real environment reward for Bellman loss updates.
"""

from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



_GPU_BANNER_PRINTED = False


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_torch_device(device: str | None = None) -> torch.device:
    """Resolve the PyTorch device and optionally require CUDA.

    RunPod should use GPU. Set REQUIRE_CUDA=1, or pass device="cuda",
    and the program will fail immediately if CUDA is not visible.
    """
    global _GPU_BANNER_PRINTED

    require_cuda = _env_bool("REQUIRE_CUDA", default=False)
    requested = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was requested, but torch.cuda.is_available() is False. "
            "On RunPod, make sure you selected a GPU pod and run Docker with --gpus all."
        )

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_CUDA=1, but CUDA GPU is not detected. "
            "Stop here instead of accidentally training on CPU."
        )

    if require_cuda and requested == "cpu":
        raise RuntimeError("REQUIRE_CUDA=1, but device='cpu' was requested.")

    resolved = torch.device(requested)

    if resolved.type == "cuda" and not _GPU_BANNER_PRINTED:
        props = torch.cuda.get_device_properties(resolved)
        print("=" * 72, flush=True)
        print(f"DQN PyTorch device : {resolved}", flush=True)
        print(f"GPU name           : {torch.cuda.get_device_name(resolved)}", flush=True)
        print(f"CUDA version       : {torch.version.cuda}", flush=True)
        print(f"PyTorch version    : {torch.__version__}", flush=True)
        print(f"Total VRAM         : {props.total_memory / 1024**3:.1f} GB", flush=True)
        print("=" * 72, flush=True)
        _GPU_BANNER_PRINTED = True
    elif resolved.type == "cpu" and not _GPU_BANNER_PRINTED:
        print("WARNING: DQN is running on CPU. Set REQUIRE_CUDA=1 on RunPod to prevent this.", flush=True)
        _GPU_BANNER_PRINTED = True

    return resolved

class QNetwork(nn.Module):
    """Standard DQN neural network."""

    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyLinear(nn.Module):
    """Noisy linear layer from Noisy Networks.

    Weight = mu + sigma * epsilon
    The trainable parameters are mu and sigma. Epsilon is resampled.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """DQN network with noisy layers near the output."""

    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self) -> None:
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class RNDFeatureNetwork(nn.Module):
    """Feature network used by Random Network Distillation."""

    def __init__(self, observation_size: int, feature_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, feature_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Stores past CARLA transitions for DQN learning."""

    def __init__(self, capacity: int = 50000):
        self.memory: Deque[Experience] = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        self.memory.append(Experience(state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """Base DQN with online network, target network, replay buffer."""

    network_cls: Type[nn.Module] = QNetwork

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 0.05,
        gamma: float = 0.9,
        batch_size: int = 64,
        replay_capacity: int = 50000,
        target_update_interval: int = 1000,
        inference_margin: float = 0.01,
        device: str | None = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.inference_margin = inference_margin
        self.learn_steps = 0
        self.device = resolve_torch_device(device)

        self.q_network = self.network_cls(observation_size, action_size).to(self.device)
        self.target_network = self.network_cls(observation_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.loss_fn = nn.MSELoss()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).detach().cpu().numpy()
        self.q_network.train()
        return q_values

    def best_action(self, state: np.ndarray) -> int:
        q_values = self.get_q_values(state)
        max_q = float(np.max(q_values))
        close_actions = np.where((max_q - q_values) <= self.inference_margin)[0]
        return int(close_actions[0])

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        return self.best_action(state)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(np.array([e.state for e in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([e.reward for e in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            best_next_q = self.target_network(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * best_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent_class": self.__class__.__name__,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "inference_margin": self.inference_margin,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "DQNAgent":
        checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(
            checkpoint["observation_size"],
            checkpoint["action_size"],
            learning_rate=checkpoint.get("learning_rate", 0.05),
            gamma=checkpoint.get("gamma", 0.9),
            inference_margin=checkpoint.get("inference_margin", 0.01),
            device=device,
        )
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        return agent


class NoisyDQNAgent(DQNAgent):
    """DQN where exploration comes from learned noisy weights, not epsilon."""

    network_cls = NoisyQNetwork

    def reset_noise(self) -> None:
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        self.reset_noise()
        return self.best_action(state)

    def learn(self) -> float | None:
        self.reset_noise()
        loss = super().learn()
        self.reset_noise()
        return loss


class RNDDQNAgent(DQNAgent):
    """DQN with Random Network Distillation intrinsic reward."""

    def __init__(self, *args, rnd_beta: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnd_beta = rnd_beta
        self.rnd_target = RNDFeatureNetwork(self.observation_size).to(self.device)
        self.rnd_predictor = RNDFeatureNetwork(self.observation_size).to(self.device)
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=self.learning_rate)
        self.rnd_loss_fn = nn.MSELoss()

    def intrinsic_reward(self, state: np.ndarray) -> float:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target_features = self.rnd_target(state_tensor)
            predicted_features = self.rnd_predictor(state_tensor)
            error = self.rnd_loss_fn(predicted_features, target_features)
        return float(error.item())

    def train_rnd_predictor(self, state: np.ndarray) -> float:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            target_features = self.rnd_target(state_tensor)
        predicted_features = self.rnd_predictor(state_tensor)
        loss = self.rnd_loss_fn(predicted_features, target_features)
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        return float(loss.item())

    def remember_with_intrinsic_reward(self, state, action, env_reward, next_state, done) -> float:
        intrinsic = self.intrinsic_reward(next_state)
        total_reward = float(env_reward) + self.rnd_beta * intrinsic
        self.remember(state, action, total_reward, next_state, done)
        self.train_rnd_predictor(next_state)
        return total_reward

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent_class": self.__class__.__name__,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "rnd_target": self.rnd_target.state_dict(),
                "rnd_predictor": self.rnd_predictor.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "inference_margin": self.inference_margin,
                "rnd_beta": self.rnd_beta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "RNDDQNAgent":
        checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(
            checkpoint["observation_size"],
            checkpoint["action_size"],
            learning_rate=checkpoint.get("learning_rate", 0.05),
            gamma=checkpoint.get("gamma", 0.9),
            inference_margin=checkpoint.get("inference_margin", 0.01),
            rnd_beta=checkpoint.get("rnd_beta", 0.01),
            device=device,
        )
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.rnd_target.load_state_dict(checkpoint["rnd_target"])
        agent.rnd_predictor.load_state_dict(checkpoint["rnd_predictor"])
        return agent


class NoisyRNDDQNAgent(RNDDQNAgent):
    """Latest-style DQN used by the CARLA SUMO-style experiments.

    This combines:
    - NoisyNet exploration through NoisyQNetwork;
    - RND intrinsic reward through RNDDQNAgent;
    - replay buffer, target network, and GPU-backed PyTorch training.

    Count-based intrinsic reward is added in the training script because it depends
    on each experiment's state visitation history.
    """

    network_cls = NoisyQNetwork

    def reset_noise(self) -> None:
        self.q_network.reset_noise()
        self.target_network.reset_noise()

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        # Epsilon is still honored for the SUMO-style Standard Epsilon experiment.
        if epsilon > 0.0 and random.random() < epsilon:
            return random.randrange(self.action_size)
        self.reset_noise()
        return self.best_action(state)

    def learn(self) -> float | None:
        self.reset_noise()
        loss = super().learn()
        self.reset_noise()
        return loss

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "NoisyRNDDQNAgent":
        checkpoint = torch.load(path, map_location=device or "cpu")
        agent = cls(
            checkpoint["observation_size"],
            checkpoint["action_size"],
            learning_rate=checkpoint.get("learning_rate", 0.05),
            gamma=checkpoint.get("gamma", 0.9),
            inference_margin=checkpoint.get("inference_margin", 0.01),
            rnd_beta=checkpoint.get("rnd_beta", 0.01),
            device=device,
        )
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        if "rnd_target" in checkpoint:
            agent.rnd_target.load_state_dict(checkpoint["rnd_target"])
        if "rnd_predictor" in checkpoint:
            agent.rnd_predictor.load_state_dict(checkpoint["rnd_predictor"])
        return agent
