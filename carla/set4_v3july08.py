#!/usr/bin/env python3
"""set4_v2.py

CARLA 0.9.14 DQN experiments for results/set4_v2.

Experiments, in strict order:
  1. Epsilon Greedy
  2. Median First
  3. Median 50
  4. Ensemble, trained only after the first three are finished

Neural setup for every experiment:
  DQN + RND + Count-Based intrinsic reward + target network + replay buffer + Adam
  No NoisyNet. Standard argmax best_action. Default learning rate = 5e-5.

Evaluation:
  Test networks are frozen. No optimizer, replay, RND, or target updates during testing.
  Reward graphs use all test episodes and environment rewards only.
  Convergence graphs use training data only.

Outputs:
  /workspace/results/set4_v2 by default
  CSV logs, model checkpoints, IEEE-style figures, and dashboard.
"""

from __future__ import annotations

import argparse, copy, math, os, random, subprocess, time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.carla_env import CarlaDrivingEnv

BASE_EXPERIMENTS = ["standard_epsilon", "median_50_first", "median_50"]
ENSEMBLE_EXPERIMENTS = ["ensemble_epsilon_medianfirst_median50"]
EXPERIMENTS = BASE_EXPERIMENTS + ENSEMBLE_EXPERIMENTS

SHORT_LABELS = {
    "standard_epsilon": "Epsilon",
    "median_50_first": "Median First",
    "median_50": "Median 50",
    "ensemble_epsilon_medianfirst_median50": "Ensemble",
}

COLORS = {
    "standard_epsilon": "#1f5eff",
    "median_50_first": "#1a8f3a",
    "median_50": "#ff5a1f",
    "ensemble_epsilon_medianfirst_median50": "#f5aa00",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name: str):
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


def system_memory_metrics() -> dict:
    if psutil is None:
        return {"ram_used_mb": 0.0, "ram_total_mb": 0.0, "ram_percent": 0.0}
    vm = psutil.virtual_memory()
    return {
        "ram_used_mb": float(vm.used / (1024 ** 2)),
        "ram_total_mb": float(vm.total / (1024 ** 2)),
        "ram_percent": float(vm.percent),
    }


def smi_metrics() -> dict:
    out = {"gpu_util_percent": 0.0, "gpu_memory_used_mb_smi": 0.0, "gpu_memory_total_mb_smi": 0.0, "gpu_power_watts": 0.0, "gpu_temperature_c": 0.0}
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu", "--format=csv,noheader,nounits"]
        s = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=2).strip().splitlines()[0]
        vals = [v.strip() for v in s.split(",")]
        out.update({"gpu_util_percent": float(vals[0]), "gpu_memory_used_mb_smi": float(vals[1]), "gpu_memory_total_mb_smi": float(vals[2]), "gpu_power_watts": float(vals[3]), "gpu_temperature_c": float(vals[4])})
    except Exception:
        pass
    return out


def avg(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / max(len(xs), 1))


def pct(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else 0.0


def reward_mode(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    counts: Dict[float, int] = {}
    for x in xs:
        k = round(float(x), 6)
        counts[k] = counts.get(k, 0) + 1
    m = max(counts.values())
    return float(max(k for k, v in counts.items() if v == m))


def make_env(args) -> CarlaDrivingEnv:
    return CarlaDrivingEnv(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        reward_mode=args.reward_mode,
        target_speed_kmh=args.target_speed,
        max_episode_steps=args.max_episode_steps,
        use_mock_when_carla_missing=False,
        num_traffic_vehicles=args.num_traffic_vehicles,
        num_pedestrians=args.num_pedestrians,
        realistic_traffic=args.realistic_traffic,
        traffic_manager_port=args.traffic_manager_port,
    )


class QNetwork(nn.Module):
    def __init__(self, obs: int, actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))
    def forward(self, x):
        return self.net(x)


class RNDNetwork(nn.Module):
    def __init__(self, obs: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
    def forward(self, x):
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=int(capacity))
    def push(self, s, a, r, ns, done):
        self.buffer.append(Transition(np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(ns, dtype=np.float32), bool(done)))
    def sample(self, batch: int):
        return random.sample(self.buffer, int(batch))
    def __len__(self):
        return len(self.buffer)


class RNDCountDQNAgent:
    def __init__(self, obs: int, actions: int, args, lr: float):
        self.obs = int(obs)
        self.actions = int(actions)
        self.gamma = float(args.gamma)
        self.batch_size = int(args.batch_size)
        self.target_update_interval = int(args.target_update_interval)
        self.inference_margin = float(args.inference_margin)
        self.device = device_from_arg(args.device)
        self.learn_steps = 0
        self.q_network = QNetwork(obs, actions, args.hidden_size).to(self.device)
        self.target_network = QNetwork(obs, actions, args.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(lr))
        self.replay_buffer = ReplayBuffer(args.replay_capacity)
        self.rnd_target = RNDNetwork(obs, args.hidden_size, args.rnd_output_size).to(self.device)
        self.rnd_predictor = RNDNetwork(obs, args.hidden_size, args.rnd_output_size).to(self.device)
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=float(args.rnd_learning_rate))
        self.rnd_target.eval()
        for p in self.rnd_target.parameters():
            p.requires_grad = False

    def ts(self, state):
        return torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).unsqueeze(0)

    def get_q_values(self, state) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            q = self.q_network(self.ts(state)).detach().cpu().numpy()[0]
        self.q_network.train()
        return q.astype(float)

    def best_action(self, state) -> int:
        # Standard DQN greedy action selection: choose the action with the highest Q-value.
        # No inference margin / near-tie randomization is used.
        q = self.get_q_values(state)
        return int(np.argmax(q))

    def select_action(self, state, epsilon: float) -> int:
        return int(random.randrange(self.actions)) if random.random() < epsilon else self.best_action(state)

    def remember(self, s, a, r, ns, done):
        self.replay_buffer.push(s, a, r, ns, done)

    def intrinsic_reward(self, state) -> float:
        with torch.no_grad():
            target = self.rnd_target(self.ts(state))
            pred = self.rnd_predictor(self.ts(state))
            return float(F.mse_loss(pred, target, reduction="mean").detach().cpu().item())

    def train_rnd_predictor(self, state) -> float:
        x = self.ts(state)
        with torch.no_grad():
            target = self.rnd_target(x)
        pred = self.rnd_predictor(x)
        loss = F.mse_loss(pred, target, reduction="mean")
        self.rnd_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_predictor.parameters(), 10.0)
        self.rnd_optimizer.step()
        return float(loss.detach().cpu().item())

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            target_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * target_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.update_target_network()
        return float(loss.detach().cpu().item())

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def freeze_for_eval(self):
        for m in [self.q_network, self.target_network, self.rnd_target, self.rnd_predictor]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "rnd_target": self.rnd_target.state_dict(),
            "rnd_predictor": self.rnd_predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict(),
            "learn_steps": self.learn_steps,
            "obs": self.obs,
            "actions": self.actions,
        }, path)


class CountBonus:
    def __init__(self, beta: float, bin_size: float):
        self.beta = float(beta)
        self.bin_size = max(float(bin_size), 1e-6)
        self.counts: Dict[tuple, int] = {}
    def key(self, state):
        arr = np.asarray(state, dtype=float)
        return tuple(np.round(arr / self.bin_size).astype(int).tolist())
    def bonus(self, state) -> Tuple[float, int]:
        k = self.key(state)
        n = self.counts.get(k, 0) + 1
        self.counts[k] = n
        return float(self.beta / math.sqrt(n)), int(n)


def median50_action(agent: RNDCountDQNAgent, state, actions: int) -> int:
    q = agent.get_q_values(state)
    med = float(np.median(q))
    lower = [i for i, v in enumerate(q) if float(v) <= med]
    return int(random.choice(lower if lower else list(range(actions))))


def base_action(exp: str, agent: RNDCountDQNAgent, state, episode: int, args, actions: int) -> Tuple[int, str]:
    if exp == "standard_epsilon":
        if random.random() < args.epsilon:
            return int(random.randrange(actions)), "epsilon_random"
        return agent.best_action(state), "greedy"
    if exp == "median_50":
        if random.random() < args.epsilon:
            return median50_action(agent, state, actions), "median50_explore"
        return agent.best_action(state), "greedy"
    if exp == "median_50_first":
        if episode < int(round(args.epsilon * args.train_episodes)):
            return median50_action(agent, state, actions), "median_first_phase"
        return agent.best_action(state), "greedy"
    raise ValueError(exp)


def expert_actions(state, experts):
    out, seen = [], set()
    for name, agent in experts:
        a = int(agent.best_action(state))
        if a not in seen:
            out.append((a, name, agent))
            seen.add(a)
    return out


def ensemble_action(state, own_agent, experts) -> Tuple[int, str]:
    cands = expert_actions(state, experts)
    own = int(own_agent.best_action(state))
    if own not in {a for a, _, _ in cands}:
        cands.append((own, "own", own_agent))
    own_q = own_agent.get_q_values(state)
    a, src, _ = max(cands, key=lambda item: float(own_q[item[0]]))
    return int(a), f"ensemble_{src}"


def convergence_episode(train_rewards, target_reward, threshold_fraction, window) -> int:
    if not train_rewards:
        return 0
    w = max(1, min(int(window), len(train_rewards)))
    threshold = float(threshold_fraction) * float(target_reward)
    rolling = np.convolve(np.asarray(train_rewards, dtype=float), np.ones(w) / w, mode="valid")
    for i, v in enumerate(rolling):
        if float(v) >= threshold:
            return int(i + w)
    return int(len(train_rewards))


def block_summary(rows, exp, phase, lr, block_index):
    rewards = [float(r["env_reward"]) for r in rows]
    return {
        "phase": phase, "experiment": exp, "method": SHORT_LABELS[exp], "block_index": int(block_index),
        "start_episode": int(rows[0]["episode"]), "end_episode": int(rows[-1]["episode"]), "episodes_in_block": int(len(rows)),
        "average_env_reward": float(np.mean(rewards)), "median_env_reward": float(np.median(rewards)), "min_env_reward": float(np.min(rewards)),
        "max_env_reward": float(np.max(rewards)), "std_env_reward": float(np.std(rewards)), "q1_env_reward": pct(rewards, 25), "q3_env_reward": pct(rewards, 75),
        "total_wall_time_seconds": float(sum(float(r.get("wall_time_seconds", 0.0)) for r in rows)),
        "total_cpu_time_seconds": float(sum(float(r.get("cpu_time_seconds", 0.0)) for r in rows)),
        "total_gpu_time_seconds": float(sum(float(r.get("gpu_time_seconds", 0.0)) for r in rows)),
        "average_steps": float(np.mean([float(r.get("steps", 0)) for r in rows])),
        "average_loss": float(np.mean([float(r.get("average_loss", 0.0)) for r in rows])),
        "average_rnd_intrinsic": float(np.mean([float(r.get("average_rnd_intrinsic", 0.0)) for r in rows])),
        "average_count_intrinsic": float(np.mean([float(r.get("average_count_intrinsic", 0.0)) for r in rows])),
        "average_gpu_util_percent": float(np.mean([float(r.get("gpu_util_percent", 0.0)) for r in rows])),
        "max_gpu_memory_mb": float(np.max([float(r.get("gpu_memory_mb", 0.0)) for r in rows])),
        "learning_rate": float(lr), "lr_multiplier": 1.0,
    }


def train_one(exp, lr, args, output_dir: Path, experts=None):
    env = make_env(args)
    obs = int(env.observation_space.shape[0])
    actions = int(env.action_space.n)
    agent = RNDCountDQNAgent(obs, actions, args, lr)
    count = CountBonus(args.count_beta, args.count_state_bin_size)
    rows, block_rows, train_rewards = [], [], []
    reset_gpu_peak(args.device)
    exp_wall, exp_cpu = time.time(), time.process_time()
    print(f"\n===== TRAINING START: {SHORT_LABELS[exp]} =====", flush=True)
    for ep in range(args.train_episodes):
        state, _ = env.reset()
        env_reward = train_reward = rnd_sum = cnt_sum = rnd_loss_sum = 0.0
        rnd_loss_n = steps = 0
        losses, src_counts = [], {}
        last_info = {}
        done = False
        wall, cpu, gt = time.time(), time.process_time(), make_gpu_timer(args.device)
        while not done:
            if exp in BASE_EXPERIMENTS:
                action, src = base_action(exp, agent, state, ep, args, actions)
            else:
                action, src = ensemble_action(state, agent, experts or [])
            src_counts[src] = src_counts.get(src, 0) + 1
            next_state, reward, terminated, truncated, _info = env.step(action)
            last_info = dict(_info or {})
            done = bool(terminated or truncated)
            rnd_raw = float(agent.intrinsic_reward(next_state))
            rnd_loss = float(agent.train_rnd_predictor(next_state))
            cnt_bonus, _cnt = count.bonus(next_state)
            full_reward = float(reward) + float(args.rnd_beta) * rnd_raw + float(cnt_bonus)
            agent.remember(state, action, full_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))
            env_reward += float(reward); train_reward += full_reward; rnd_sum += rnd_raw; cnt_sum += cnt_bonus; rnd_loss_sum += rnd_loss; rnd_loss_n += 1
            state = next_state; steps += 1
        train_rewards.append(env_reward)
        row = {
            "phase": "train", "experiment": exp, "method": SHORT_LABELS[exp], "episode": ep, "env_reward": float(env_reward), "training_reward": float(train_reward),
            "steps": steps, "termination_reason": last_info.get("termination_reason", "unknown"), "ended_before_max_steps": bool(last_info.get("ended_before_max_steps", steps < args.max_episode_steps)),
            "collision": bool(last_info.get("collision", False)), "collision_actor_type": last_info.get("collision_actor_type", "none"), "collision_actor_id": last_info.get("collision_actor_id", -1),
            "collision_actor_role_name": last_info.get("collision_actor_role_name", ""), "collision_intensity": float(last_info.get("collision_intensity", 0.0) or 0.0),
            "stuck": bool(last_info.get("stuck", False)), "stuck_step_count": int(last_info.get("stuck_step_count", 0) or 0),
            "wall_time_seconds": float(time.time() - wall), "cpu_time_seconds": float(time.process_time() - cpu), "gpu_time_seconds": stop_gpu_timer(gt, args.device),
            "gpu_memory_mb": gpu_peak_mb(args.device), "process_memory_mb": process_memory_mb(), **system_memory_metrics(), **smi_metrics(),
            "average_loss": avg(losses), "loss_updates": len(losses), "average_rnd_intrinsic": rnd_sum / max(steps, 1), "average_count_intrinsic": cnt_sum / max(steps, 1),
            "average_rnd_loss": rnd_loss_sum / max(rnd_loss_n, 1), "replay_buffer_size": len(agent.replay_buffer), "learn_steps": agent.learn_steps,
            "epsilon": args.epsilon, "gamma": args.gamma, "learning_rate": lr, "lr_multiplier": 1.0,
            "dqn_technology": "DQN + RND + CountBased + TargetNetwork + ReplayBuffer + Adam", "rnd_beta": args.rnd_beta, "count_beta": args.count_beta,
            "count_state_bin_size": args.count_state_bin_size, "action_source_counts": str(src_counts),
        }
        rows.append(row)
        if ((ep + 1) % args.episode_block_size == 0) or ((ep + 1) == args.train_episodes):
            block_rows.append(block_summary(rows[-min(args.episode_block_size, len(rows)):], exp, "train", lr, len(block_rows) + 1))
        print(f"TRAIN {SHORT_LABELS[exp]:14s} ep={ep:03d} reward={env_reward:.2f} steps={steps} term={row['termination_reason']} coll={row['collision_actor_type']} force={row['collision_intensity']:.1f} wall={row['wall_time_seconds']:.2f}s cpu={row['cpu_time_seconds']:.2f}s gpu={row['gpu_time_seconds']:.2f}s ram={row['ram_used_mb']:.0f}/{row['ram_total_mb']:.0f}MB vram={row['gpu_memory_used_mb_smi']:.0f}/{row['gpu_memory_total_mb_smi']:.0f}MB loss={row['average_loss']:.5f}", flush=True)
    agent.update_target_network()
    model_path = output_dir / "models" / f"{exp}_lrmult_1.pt"
    agent.save(str(model_path))
    pd.DataFrame(rows).to_csv(output_dir / f"{exp}_lrmult_1_train.csv", index=False)
    runtime = {"phase": "train", "experiment": exp, "method": SHORT_LABELS[exp], "learning_rate": lr, "lr_multiplier": 1.0, "total_wall_time_seconds": time.time() - exp_wall, "total_cpu_time_seconds": time.process_time() - exp_cpu, "total_gpu_time_seconds": sum(float(r["gpu_time_seconds"]) for r in rows), "gpu_memory_mb": gpu_peak_mb(args.device), "process_memory_mb": process_memory_mb(), "episodes": args.train_episodes, "model_path": str(model_path)}
    env.close()
    print(f"===== TRAINING END: {SHORT_LABELS[exp]} =====", flush=True)
    return agent, train_rewards, rows, block_rows, runtime


def test_one(exp, agent, args, lr, experts=None):
    env = make_env(args)
    agent.freeze_for_eval()
    if experts:
        for _, e in experts: e.freeze_for_eval()
    rows, block_rows = [], []
    reset_gpu_peak(args.device)
    exp_wall, exp_cpu = time.time(), time.process_time()
    print(f"\n===== TESTING START: {SHORT_LABELS[exp]} =====", flush=True)
    with torch.no_grad():
        for ep in range(args.test_episodes):
            state, _ = env.reset(); total = 0.0; steps = 0; done = False; src_counts = {}; last_info = {}
            wall, cpu, gt = time.time(), time.process_time(), make_gpu_timer(args.device)
            while not done:
                if exp == "ensemble_epsilon_medianfirst_median50":
                    action, src = ensemble_action(state, agent, experts or [])
                else:
                    action, src = agent.best_action(state), "frozen_greedy"
                src_counts[src] = src_counts.get(src, 0) + 1
                state, reward, terminated, truncated, _info = env.step(action)
                last_info = dict(_info or {})
                done = bool(terminated or truncated); total += float(reward); steps += 1
            row = {"phase": "test", "experiment": exp, "method": SHORT_LABELS[exp], "episode": ep, "env_reward": float(total), "steps": steps,
                   "termination_reason": last_info.get("termination_reason", "unknown"), "ended_before_max_steps": bool(last_info.get("ended_before_max_steps", steps < args.max_episode_steps)),
                   "collision": bool(last_info.get("collision", False)), "collision_actor_type": last_info.get("collision_actor_type", "none"), "collision_actor_id": last_info.get("collision_actor_id", -1),
                   "collision_actor_role_name": last_info.get("collision_actor_role_name", ""), "collision_intensity": float(last_info.get("collision_intensity", 0.0) or 0.0),
                   "stuck": bool(last_info.get("stuck", False)), "stuck_step_count": int(last_info.get("stuck_step_count", 0) or 0),
                   "wall_time_seconds": time.time() - wall, "cpu_time_seconds": time.process_time() - cpu, "gpu_time_seconds": stop_gpu_timer(gt, args.device),
                   "gpu_memory_mb": gpu_peak_mb(args.device), "process_memory_mb": process_memory_mb(), **system_memory_metrics(), **smi_metrics(), "epsilon": args.epsilon, "gamma": args.gamma,
                   "learning_rate": lr, "lr_multiplier": 1.0, "network_frozen": True, "updates_during_test": 0, "action_source_counts": str(src_counts)}
            rows.append(row)
            if ((ep + 1) % args.episode_block_size == 0) or ((ep + 1) == args.test_episodes):
                block_rows.append(block_summary(rows[-min(args.episode_block_size, len(rows)):], exp, "test", lr, len(block_rows) + 1))
            print(f"TEST  {SHORT_LABELS[exp]:14s} ep={ep:03d} reward={total:.2f} steps={steps} term={row['termination_reason']} coll={row['collision_actor_type']} force={row['collision_intensity']:.1f} wall={row['wall_time_seconds']:.2f}s cpu={row['cpu_time_seconds']:.2f}s gpu={row['gpu_time_seconds']:.2f}s ram={row['ram_used_mb']:.0f}/{row['ram_total_mb']:.0f}MB vram={row['gpu_memory_used_mb_smi']:.0f}/{row['gpu_memory_total_mb_smi']:.0f}MB", flush=True)
    runtime = {"phase": "test", "experiment": exp, "method": SHORT_LABELS[exp], "learning_rate": lr, "lr_multiplier": 1.0, "total_wall_time_seconds": time.time() - exp_wall, "total_cpu_time_seconds": time.process_time() - exp_cpu, "total_gpu_time_seconds": sum(float(r["gpu_time_seconds"]) for r in rows), "gpu_memory_mb": gpu_peak_mb(args.device), "process_memory_mb": process_memory_mb(), "episodes": args.test_episodes, "network_frozen": True, "updates_during_test": 0}
    env.close(); print(f"===== TESTING END: {SHORT_LABELS[exp]} =====", flush=True)
    return rows, block_rows, runtime


def summarize(exp, train_rewards, train_rows, test_rows, args, lr):
    tr, te = [float(x) for x in train_rewards], [float(r["env_reward"]) for r in test_rows]
    avg_te = float(np.mean(te)); conv_ep = convergence_episode(tr, avg_te, args.convergence_threshold_fraction, args.convergence_window)
    train_wall = sum(float(r["wall_time_seconds"]) for r in train_rows); train_cpu = sum(float(r["cpu_time_seconds"]) for r in train_rows); train_gpu = sum(float(r["gpu_time_seconds"]) for r in train_rows)
    frac = conv_ep / max(float(args.train_episodes), 1.0)
    return {"experiment": exp, "method": SHORT_LABELS[exp], "lr_multiplier": 1.0, "learning_rate": lr, "final_learning_rate": lr,
            "average_train_reward": float(np.mean(tr)), "median_train_reward": float(np.median(tr)), "std_train_reward": float(np.std(tr)), "min_train_reward": float(np.min(tr)), "max_train_reward": float(np.max(tr)), "q1_train_reward": pct(tr,25), "q3_train_reward": pct(tr,75),
            "average_test_reward": avg_te, "median_test_reward": float(np.median(te)), "mode_test_reward": reward_mode(te), "min_test_reward": float(np.min(te)), "max_test_reward": float(np.max(te)), "range_test_reward": float(np.max(te)-np.min(te)), "std_test_reward": float(np.std(te)), "q1_test_reward": pct(te,25), "q3_test_reward": pct(te,75),
            "convergence_episode": int(conv_ep), "convergence_time_seconds": frac * train_wall, "convergence_cpu_time_seconds": frac * train_cpu, "convergence_gpu_time_seconds": frac * train_gpu,
            "total_training_wall_time_seconds": train_wall, "total_training_cpu_time_seconds": train_cpu, "total_training_gpu_time_seconds": train_gpu, "total_training_episodes": args.train_episodes,
            "test_rewards": te, "dqn_technology": "DQN + RND + CountBased + TargetNetwork + ReplayBuffer + Adam", "rnd_beta": args.rnd_beta, "count_beta": args.count_beta, "count_state_bin_size": args.count_state_bin_size, "network_frozen_during_testing": True}


def save_csvs(results, train_rows, test_rows, block_rows, runtime_rows, output_dir):
    pd.DataFrame([{k:v for k,v in r.items() if k != "test_rewards"} for r in results]).to_csv(output_dir / "all_experiments_learning_rate_summary.csv", index=False)
    pd.DataFrame(train_rows).to_csv(output_dir / "all_experiments_train_episode_rewards.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "all_experiments_test_episode_rewards.csv", index=False)
    pd.DataFrame(block_rows).to_csv(output_dir / "all_experiments_episode_block_logs.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "all_experiments_runtime_logs.csv", index=False)


def apply_ieee_style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300, "savefig.dpi": 600, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def save_fig(fig, outdir, name):
    fig.tight_layout(); fig.savefig(outdir / f"{name}.png", bbox_inches="tight"); fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def ordered(results):
    return [next(r for r in results if r["experiment"] == exp) for exp in EXPERIMENTS]


def make_figures(results, train_rows, test_rows, block_rows, output_dir, args):
    apply_ieee_style(); figdir = output_dir / "figures_ieee"; figdir.mkdir(exist_ok=True)
    rows = ordered(results); labels = [SHORT_LABELS[r["experiment"]] for r in rows]; colors = [COLORS[r["experiment"]] for r in rows]
    # Average test reward
    fig, ax = plt.subplots(figsize=(5.2,3.4)); x=np.arange(len(rows)); vals=[r["average_test_reward"] for r in rows]; err=[r["std_test_reward"] for r in rows]
    ax.bar(x, vals, yerr=err, capsize=3, color=colors, edgecolor="black", linewidth=.7); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right"); ax.set_ylabel("Average test reward"); ax.set_xlabel("Experiment"); ax.set_title("Average Test Reward Across All Test Episodes"); save_fig(fig, figdir, "ieee_average_test_reward")
    # Boxplot
    fig, ax = plt.subplots(figsize=(5.4,3.4)); bp=ax.boxplot([r["test_rewards"] for r in rows], patch_artist=True, labels=labels, showfliers=True)
    for p,c in zip(bp["boxes"], colors): p.set_facecolor(c); p.set_alpha(.55); p.set_edgecolor("black")
    ax.set_ylabel("Environment reward"); ax.set_xlabel("Experiment"); ax.set_title("Test Reward Distribution Across All Test Episodes"); save_fig(fig, figdir, "ieee_test_reward_boxplot")
    # convergence time and episode
    for key, ylabel, name, title in [("convergence_time_seconds", "Convergence time (s)", "ieee_convergence_time", "Training-Data Convergence Time"), ("convergence_episode", "Convergence episode", "ieee_convergence_episode", "Training Convergence Episode")]:
        fig, ax = plt.subplots(figsize=(5.2,3.4)); vals=[r[key] for r in rows]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=.7); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right"); ax.set_ylabel(ylabel); ax.set_xlabel("Experiment"); ax.set_title(title); save_fig(fig, figdir, name)
    # block lines
    df = pd.DataFrame(block_rows)
    for phase, name, title, ylabel in [("train", "ieee_training_reward_blocks", "Training Reward by Episode Block", "Average training reward"), ("test", "ieee_testing_reward_blocks", "Testing Reward by Episode Block", "Average test reward")]:
        fig, ax = plt.subplots(figsize=(5.8,3.6)); sub=df[df["phase"]==phase] if not df.empty else pd.DataFrame()
        for exp in EXPERIMENTS:
            d=sub[sub["experiment"]==exp].sort_values("block_index") if not sub.empty else pd.DataFrame()
            if not d.empty:
                labs=[f"{int(a)+1}-{int(b)+1}" for a,b in zip(d["start_episode"], d["end_episode"])]
                ax.plot(labs, d["average_env_reward"].astype(float), marker="o", linewidth=1.6, label=SHORT_LABELS[exp], color=COLORS[exp])
        ax.set_ylabel(ylabel); ax.set_xlabel("Episode block"); ax.set_title(title); ax.legend(frameon=False); save_fig(fig, figdir, name)
    # Dashboard
    make_dashboard(results, test_rows, block_rows, output_dir, args)


def table_axis(ax, rows, prefix, title):
    ax.axis("off"); cols=["Method","Mean","Median","Std","Min","Max","Q1","Q3"]
    data=[[SHORT_LABELS[r["experiment"]], f"{r[f'average_{prefix}_reward']:.2f}", f"{r[f'median_{prefix}_reward']:.2f}", f"{r[f'std_{prefix}_reward']:.2f}", f"{r[f'min_{prefix}_reward']:.2f}", f"{r[f'max_{prefix}_reward']:.2f}", f"{r[f'q1_{prefix}_reward']:.2f}", f"{r[f'q3_{prefix}_reward']:.2f}"] for r in rows]
    tab=ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center"); tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.45)
    for (rr,cc), cell in tab.get_celld().items():
        cell.set_linewidth(.3)
        if rr==0: cell.set_text_props(weight="bold"); cell.set_facecolor("#f0f0f0")
        elif cc==0: cell.get_text().set_color(COLORS[rows[rr-1]["experiment"]]); cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10, color="#0000aa", fontweight="bold")


def make_dashboard(results, test_rows, block_rows, output_dir, args):
    rows=ordered(results); block_df=pd.DataFrame(block_rows); test_df=pd.DataFrame(test_rows)
    fig=plt.figure(figsize=(18,12), facecolor="white"); gs=fig.add_gridspec(4,3,height_ratios=[1.25,.95,1,.75],hspace=.42,wspace=.28)
    fig.suptitle(f"DQN in CARLA – Training & Testing Dashboard (LR multiplier = 1, Final LR = {args.raw_learning_rate:g})", fontsize=16, fontweight="bold", color="#0b123f", y=.985)
    fig.text(.5,.955,f"All rewards are ENVIRONMENT rewards only  •  Training Episodes: {args.train_episodes}  •  Testing Episodes: {args.test_episodes}  •  Episode Block Size: {args.episode_block_size}  •  Epsilon: {args.epsilon}",ha="center",va="center",fontsize=10,color="#0b123f")
    for idx, (phase, title, ylabel) in enumerate([("train", "Average Training Reward by Episode Block", "Average training reward"), ("test", "Average Testing Reward by Episode Block", "Average testing reward")]):
        ax=fig.add_subplot(gs[0,idx]); sub=block_df[block_df["phase"]==phase] if not block_df.empty else pd.DataFrame()
        for exp in EXPERIMENTS:
            d=sub[sub["experiment"]==exp].sort_values("block_index") if not sub.empty else pd.DataFrame()
            if not d.empty:
                labs=[f"{int(a)+1}-{int(b)+1}" for a,b in zip(d["start_episode"], d["end_episode"])]
                x=np.arange(len(labs)); y=d["average_env_reward"].astype(float).values
                ax.plot(x,y,marker="o",label=SHORT_LABELS[exp],color=COLORS[exp],linewidth=1.8); ax.set_xticks(x); ax.set_xticklabels(labs)
                for xx,yy in zip(x,y): ax.text(xx,yy,f"{yy:,.2f}",ha="center",va="bottom",fontsize=7,color=COLORS[exp],fontweight="bold")
        ax.set_title(title,fontsize=10,color="#0000aa",fontweight="bold"); ax.set_ylabel(ylabel); ax.set_xlabel("Episode block"); ax.legend(fontsize=7,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.32)); ax.grid(True,alpha=.25)
    ax=fig.add_subplot(gs[0,2]); x=np.arange(len(rows)); y=[r["convergence_episode"] for r in rows]
    ax.bar(x,y,color=[COLORS[r["experiment"]] for r in rows],edgecolor="black",alpha=.85); ax.set_xticks(x); ax.set_xticklabels([SHORT_LABELS[r["experiment"]] for r in rows],rotation=10); ax.set_ylabel("Convergence episode"); ax.set_title("Training Convergence Episode by Experiment",fontsize=10,color="#0000aa",fontweight="bold"); ax.grid(True,axis="y",alpha=.25)
    for i,v in enumerate(y): ax.text(i,v,f"{int(v)}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    table_axis(fig.add_subplot(gs[1,0]), rows, "train", f"Training Reward Summary Statistics (All {args.train_episodes} Episodes)")
    table_axis(fig.add_subplot(gs[1,1]), rows, "test", f"Testing Reward Summary Statistics (All {args.test_episodes} Episodes)")
    ax=fig.add_subplot(gs[1,2]); ax.axis("off"); data=[[SHORT_LABELS[r["experiment"]], int(r["convergence_episode"]), int(r["total_training_episodes"]), f"{r['total_training_wall_time_seconds']:,.2f}", f"{r['convergence_time_seconds']:,.2f}"] for r in rows]
    tab=ax.table(cellText=data, colLabels=["Method","Conv\nEpisode","Train\nEpisodes","Train\nTime(s)","Conv\nTime(s)"], loc="center", cellLoc="center"); tab.auto_set_font_size(False); tab.set_fontsize(7); tab.scale(1,1.45); ax.set_title("Convergence Summary",fontsize=10,color="#0000aa",fontweight="bold")
    ax=fig.add_subplot(gs[2,0]); bp=ax.boxplot([r["test_rewards"] for r in rows], patch_artist=True, labels=[SHORT_LABELS[r["experiment"]] for r in rows]);
    for p,r in zip(bp["boxes"],rows): p.set_facecolor(COLORS[r["experiment"]]); p.set_alpha(.55)
    ax.set_title(f"Test Reward Distribution (All {args.test_episodes} Episodes)",fontsize=10,color="#0000aa",fontweight="bold"); ax.set_ylabel("Environment reward"); ax.grid(True,axis="y",alpha=.25)
    ax=fig.add_subplot(gs[2,1])
    for exp in EXPERIMENTS:
        d=test_df[test_df["experiment"]==exp]["env_reward"].astype(float).values
        if len(d): ax.hist(d,bins=30,alpha=.35,label=SHORT_LABELS[exp],color=COLORS[exp],edgecolor="black",linewidth=.2)
    ax.set_title("Test Reward Distribution Histogram",fontsize=10,color="#0000aa",fontweight="bold"); ax.set_xlabel("Environment reward"); ax.set_ylabel("Count"); ax.legend(fontsize=7); ax.grid(True,alpha=.25)
    ax=fig.add_subplot(gs[2,2]); vals=[r["convergence_time_seconds"] for r in rows]
    ax.bar(x,vals,color=[COLORS[r["experiment"]] for r in rows],edgecolor="black",alpha=.85); ax.set_xticks(x); ax.set_xticklabels([SHORT_LABELS[r["experiment"]] for r in rows],rotation=10); ax.set_ylabel("Convergence time (s)"); ax.set_title("Convergence Time (s) by Experiment",fontsize=10,color="#0000aa",fontweight="bold"); ax.grid(True,axis="y",alpha=.25)
    for i,v in enumerate(vals): ax.text(i,v,f"{v:,.2f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax=fig.add_subplot(gs[3,0]); ax.axis("off"); best=max(rows,key=lambda r:r["average_test_reward"]); fast=min(rows,key=lambda r:r["convergence_time_seconds"]); stable=min(rows,key=lambda r:r["std_test_reward"])
    ax.text(.02,.95,f"Key Takeaways\n\n• {SHORT_LABELS[best['experiment']]} has the highest average test reward ({best['average_test_reward']:.2f}).\n• {SHORT_LABELS[fast['experiment']]} has the fastest convergence time ({fast['convergence_time_seconds']:.2f}s).\n• {SHORT_LABELS[stable['experiment']]} has the lowest test reward std ({stable['std_test_reward']:.2f}).\n• Test networks are frozen; no updates occur during testing.",ha="left",va="top",fontsize=9,bbox=dict(boxstyle="round,pad=.5",fc="white",ec="#c9d9ff"))
    ax=fig.add_subplot(gs[3,1]); ax.axis("off"); rr={r["experiment"]:i+1 for i,r in enumerate(sorted(rows,key=lambda r:r["average_test_reward"],reverse=True))}; cr={r["experiment"]:i+1 for i,r in enumerate(sorted(rows,key=lambda r:r["convergence_time_seconds"]))}; be=min(EXPERIMENTS,key=lambda e:rr[e]+cr[e]); br=next(r for r in rows if r["experiment"]==be)
    ax.text(.5,.55,f"Best Overall\n\n{SHORT_LABELS[be].upper()}\n\nAverage Test Reward: {br['average_test_reward']:.2f}\nConvergence Time: {br['convergence_time_seconds']:.2f}s\n\nRanking uses reward rank + convergence-time rank.",ha="center",va="center",fontsize=12,fontweight="bold",bbox=dict(boxstyle="round,pad=.5",fc="white",ec="#2ca25f"))
    ax=fig.add_subplot(gs[3,2]); ax.axis("off"); ax.text(.02,.95,f"Experiment Settings\n\nEnvironment           : CARLA 0.9.14 (Real)\nAlgorithm             : DQN\nNetwork               : RND + Count + Target + Replay + Adam\nNoisyNet              : Removed\nLearning Rate Mult.   : 1\nFinal Learning Rate   : {args.raw_learning_rate:g}\nEpsilon               : {args.epsilon}\nTraining Episodes     : {args.train_episodes}\nTesting Episodes      : {args.test_episodes}\nEpisode Block Size    : {args.episode_block_size}",ha="left",va="top",fontsize=9,bbox=dict(boxstyle="round,pad=.5",fc="white",ec="#c9d9ff"))
    fig.text(.01,.01,"Notes: Rewards are environment rewards only. RND/count rewards are training-only. Convergence uses training rewards and total training wall time.",fontsize=8,color="#333")
    fig.savefig(output_dir / "set4_v2_dashboard.png",dpi=300,bbox_inches="tight"); fig.savefig(output_dir / "set4_v2_dashboard.pdf",dpi=300,bbox_inches="tight"); plt.close(fig)


def write_readme(out,args):
    (out / "README_set4_v2.txt").write_text(f"""set4_v2 outputs\n\nExperiments in order:\n1. Epsilon Greedy\n2. Median First\n3. Median 50\n4. Ensemble\n\nNeural setup: DQN + RND + Count-Based + Target Network + Replay Buffer + Adam. No NoisyNet.\nTesting is frozen: no optimizer, replay, RND, or target updates.\nReward graphs use all test episodes and environment reward only.\nConvergence graphs use training data only.\n\nMain CSVs:\n- all_experiments_train_episode_rewards.csv\n- all_experiments_test_episode_rewards.csv\n- all_experiments_episode_block_logs.csv\n- all_experiments_runtime_logs.csv\n- all_experiments_learning_rate_summary.csv\n\nFigures:\n- set4_v2_dashboard.png / .pdf\n- figures_ieee/*.png and *.pdf\n\nConfiguration:\ntrain_episodes={args.train_episodes}\ntest_episodes={args.test_episodes}\nmax_episode_steps={args.max_episode_steps}\nepsilon={args.epsilon}\nlearning_rate={args.raw_learning_rate}\nrnd_beta={args.rnd_beta}\ncount_beta={args.count_beta}\n""")


def run(args):
    set_seed(args.seed); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); (out/"models").mkdir(exist_ok=True)
    lr=float(args.raw_learning_rate if args.raw_learning_rate>0 else 5e-5)
    print("="*72, flush=True); print("set4_v2: DQN + RND + Count + Target + Replay + Adam; No NoisyNet", flush=True); print(f"Output dir: {out}", flush=True); print(f"CUDA: {torch.cuda.is_available()}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True); print("="*72, flush=True)
    trained={}; train_rewards_by_exp={}; train_rows_by_exp={}; all_train=[]; all_test=[]; all_blocks=[]; all_runtime=[]; results=[]
    for exp in BASE_EXPERIMENTS:
        agent, rewards, rows, blocks, runtime = train_one(exp, lr, args, out)
        trained[exp]=agent; train_rewards_by_exp[exp]=rewards; train_rows_by_exp[exp]=rows; all_train.extend(rows); all_blocks.extend(blocks); all_runtime.append(runtime); trained[exp].freeze_for_eval()
    experts=[(exp, trained[exp]) for exp in BASE_EXPERIMENTS]
    agent, rewards, rows, blocks, runtime = train_one("ensemble_epsilon_medianfirst_median50", lr, args, out, experts=experts)
    trained["ensemble_epsilon_medianfirst_median50"]=agent; train_rewards_by_exp["ensemble_epsilon_medianfirst_median50"]=rewards; train_rows_by_exp["ensemble_epsilon_medianfirst_median50"]=rows; all_train.extend(rows); all_blocks.extend(blocks); all_runtime.append(runtime)
    for exp in EXPERIMENTS:
        test_rows, test_blocks, test_runtime = test_one(exp, trained[exp], args, lr, experts=experts if exp in ENSEMBLE_EXPERIMENTS else None)
        all_test.extend(test_rows); all_blocks.extend(test_blocks); all_runtime.append(test_runtime)
        results.append(summarize(exp, train_rewards_by_exp[exp], train_rows_by_exp[exp], test_rows, args, lr))
    save_csvs(results, all_train, all_test, all_blocks, all_runtime, out); make_figures(results, all_train, all_test, all_blocks, out, args); write_readme(out,args)
    print(f"\nSaved set4_v2 outputs to: {out}", flush=True)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=2000); p.add_argument("--timeout-seconds", type=float, default=10.0)
    p.add_argument("--reward-mode", default="ontology_combined"); p.add_argument("--target-speed", type=float, default=30.0); p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--train-episodes", type=int, default=500); p.add_argument("--test-episodes", type=int, default=300); p.add_argument("--episode-block-size", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.2); p.add_argument("--gamma", type=float, default=0.99); p.add_argument("--raw-learning-rate", type=float, default=5e-5); p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--replay-capacity", type=int, default=50000); p.add_argument("--target-update-interval", type=int, default=1000); p.add_argument("--inference-margin", type=float, default=0.01)
    p.add_argument("--rnd-beta", type=float, default=0.01); p.add_argument("--rnd-learning-rate", type=float, default=1e-4); p.add_argument("--rnd-output-size", type=int, default=64)
    p.add_argument("--count-beta", type=float, default=0.05); p.add_argument("--count-state-bin-size", type=float, default=1.0)
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=42); p.add_argument("--convergence-threshold-fraction", type=float, default=0.95); p.add_argument("--convergence-window", type=int, default=10)
    p.add_argument("--num-traffic-vehicles", type=int, default=0)
    p.add_argument("--num-pedestrians", type=int, default=0)
    p.add_argument("--realistic-traffic", action="store_true")
    p.add_argument("--traffic-manager-port", type=int, default=8000)
    p.add_argument("--output-dir", default="/workspace/results/set4_v2")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

