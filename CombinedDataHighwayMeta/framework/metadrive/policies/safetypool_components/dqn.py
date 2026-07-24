"""Reusable neural network, replay buffer, and DQN learning agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .constants import POLICY_NAME
from .utils import json_safe


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
    """Fixed-capacity indexed ring buffer with O(batch_size) sampling."""

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
    def _deterministic_extreme_from_q(
        q: np.ndarray,
        maximum: bool,
        key: str,
    ) -> int:
        """Choose a tied extreme deterministically, matching the baseline style."""
        extreme = float(np.max(q) if maximum else np.min(q))
        candidates = np.flatnonzero(q == extreme)
        if candidates.size == 0:
            raise RuntimeError("No action was available in the Q-value vector.")
        digest = hashlib.sha256(key.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % int(candidates.size)
        return int(candidates[offset])

    def greedy_action(self, state: np.ndarray, key: str = "greedy") -> int:
        q = self.q_values(state)
        return self._deterministic_extreme_from_q(q, maximum=True, key=key)


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
                "method": POLICY_NAME,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
                "observation_size": self.observation_size,
                "action_count": self.action_count,
                "config": json_safe(vars(args)),
            },
            path,
        )
