#!/usr/bin/env python3
"""set3_v2_ensemble_only.py

CARLA 0.9.14 DQN experiments for results/set3_v2.

Workflow:
  1. Load existing Noisy + Count checkpoint and training CSV
  2. Load existing RND + Count checkpoint and training CSV
  3. Train only the Ensemble using the frozen expert action candidates
  4. Test all three frozen policies and regenerate combined CSVs/graphs

Neural setup:
  Uses the Set 3 agents from src.dqn_agent.
  Noisy + Count: NoisyDQNAgent + count-based intrinsic reward.
  RND + Count: RNDDQNAgent + RND + count-based intrinsic reward.
  Ensemble: NoisyRNDDQNAgent own network + trained Noisy/RND expert action candidates.

Evaluation:
  Test networks are frozen in behavior: no optimizer, replay, RND, or target updates during testing.
  Reward graphs use all test episodes and environment rewards only.
  Convergence graphs use training data only.

Outputs:
  /workspace/results/set3_v2 by default
  CSV logs, model checkpoints, IEEE-style figures, and dashboard.
"""

from __future__ import annotations

import argparse, math, os, random, subprocess, time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

from src.carla_env import CarlaDrivingEnv
from src.dqn_agent import DQNAgent, NoisyDQNAgent, RNDDQNAgent, NoisyRNDDQNAgent

BASE_EXPERIMENTS = ["noisy_count", "rnd_count"]
ENSEMBLE_EXPERIMENTS = ["ensemble_own_noisy_rnd_count"]
EXPERIMENTS = BASE_EXPERIMENTS + ENSEMBLE_EXPERIMENTS

SHORT_LABELS = {
    "noisy_count": "Noisy + Count",
    "rnd_count": "RND + Count",
    "ensemble_own_noisy_rnd_count": "Ensemble",
}

COLORS = {
    "noisy_count": "#1f5eff",
    "rnd_count": "#ff5a1f",
    "ensemble_own_noisy_rnd_count": "#1a8f3a",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def smi_metrics() -> dict:
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
        s = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=2).strip().splitlines()[0]
        vals = [v.strip() for v in s.split(",")]
        out.update({
            "gpu_util_percent": float(vals[0]),
            "gpu_memory_used_mb_smi": float(vals[1]),
            "gpu_memory_total_mb_smi": float(vals[2]),
            "gpu_power_watts": float(vals[3]),
            "gpu_temperature_c": float(vals[4]),
        })
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
    )


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


def build_agent(exp: str, obs: int, actions: int, args, lr: float) -> DQNAgent:
    common = dict(
        observation_size=obs,
        action_size=actions,
        learning_rate=lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_interval=args.target_update_interval,
        inference_margin=args.inference_margin,
        device=args.device,
    )
    if exp == "noisy_count":
        return NoisyDQNAgent(**common)
    if exp == "rnd_count":
        return RNDDQNAgent(**common, rnd_beta=args.rnd_beta)
    if exp == "ensemble_own_noisy_rnd_count":
        return NoisyRNDDQNAgent(**common, rnd_beta=args.rnd_beta)
    raise ValueError(exp)


def select_ensemble_action(state: np.ndarray, own: DQNAgent, experts: Sequence[Tuple[str, DQNAgent]]) -> Tuple[int, str]:
    candidates = []
    seen = set()
    own_action = int(own.best_action(state))
    candidates.append((own_action, "own"))
    seen.add(own_action)
    for name, agent in experts:
        action = int(agent.best_action(state))
        if action not in seen:
            candidates.append((action, name))
            seen.add(action)
    q = np.asarray(own.get_q_values(state), dtype=float)
    action, source = max(candidates, key=lambda item: float(q[item[0]]))
    return int(action), f"ensemble_{source}"


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
        "phase": phase,
        "experiment": exp,
        "method": SHORT_LABELS[exp],
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
        "total_wall_time_seconds": float(sum(float(r.get("wall_time_seconds", 0.0)) for r in rows)),
        "total_cpu_time_seconds": float(sum(float(r.get("cpu_time_seconds", 0.0)) for r in rows)),
        "total_gpu_time_seconds": float(sum(float(r.get("gpu_time_seconds", 0.0)) for r in rows)),
        "average_steps": float(np.mean([float(r.get("steps", 0)) for r in rows])),
        "average_loss": float(np.mean([float(r.get("average_loss", 0.0)) for r in rows])),
        "average_rnd_intrinsic": float(np.mean([float(r.get("average_rnd_intrinsic", 0.0)) for r in rows])),
        "average_count_intrinsic": float(np.mean([float(r.get("average_count_intrinsic", 0.0)) for r in rows])),
        "average_gpu_util_percent": float(np.mean([float(r.get("gpu_util_percent", 0.0)) for r in rows])),
        "max_gpu_memory_mb": float(np.max([float(r.get("gpu_memory_mb", 0.0)) for r in rows])),
        "learning_rate": float(lr),
        "lr_multiplier": 1.0,
    }


def maybe_update_target(agent) -> None:
    if hasattr(agent, "update_target_network"):
        agent.update_target_network()


def maybe_save_agent(agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(agent, "save"):
        agent.save(str(path))
    else:
        torch.save(agent, str(path))




def load_agent_checkpoint(agent, path: Path, device: str):
    """Load checkpoints saved either by agent.save(), torch.save(state_dict), or torch.save(agent)."""
    if not path.exists():
        raise FileNotFoundError(f"Required expert checkpoint not found: {path}")

    # Prefer a class-provided load method because it knows the checkpoint format.
    if hasattr(agent, "load"):
        try:
            agent.load(str(path))
            return agent
        except TypeError:
            try:
                agent.load(str(path), map_location=device)
                return agent
            except Exception:
                pass
        except Exception:
            pass

    checkpoint = torch.load(str(path), map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    candidate_dicts = []
    if isinstance(checkpoint, dict):
        candidate_dicts.append(checkpoint)
        for key in ("state_dict", "model_state_dict", "policy_state_dict", "q_network_state_dict", "online_network_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                candidate_dicts.append(value)

    modules = []
    for name in ("model", "network", "q_network", "online_network", "policy_net", "policy_network"):
        obj = getattr(agent, name, None)
        if isinstance(obj, torch.nn.Module):
            modules.append(obj)
    if isinstance(agent, torch.nn.Module):
        modules.insert(0, agent)

    errors = []
    for module in modules:
        for state_dict in candidate_dicts:
            try:
                module.load_state_dict(state_dict)
                return agent
            except Exception as exc:
                errors.append(str(exc))

    raise RuntimeError(
        f"Could not load checkpoint {path}. The agent has no compatible load method/state dict. "
        f"Last errors: {errors[-3:]}"
    )


def load_existing_training_rows(output_dir: Path):
    rows_by_exp = {}
    rewards_by_exp = {}
    all_rows = []
    all_blocks = []
    for exp in BASE_EXPERIMENTS:
        csv_path = output_dir / f"{exp}_lrmult_1_train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Required training CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "experiment" not in df.columns:
            df["experiment"] = exp
        if "method" not in df.columns:
            df["method"] = SHORT_LABELS[exp]
        rows = df.to_dict("records")
        rows_by_exp[exp] = rows
        rewards_by_exp[exp] = [float(v) for v in df["env_reward"].tolist()]
        all_rows.extend(rows)
        block_index = 0
        for start in range(0, len(rows), 100):
            chunk = rows[start:start + 100]
            if chunk:
                block_index += 1
                all_blocks.append(block_summary(chunk, exp, "train", float(chunk[0].get("learning_rate", 1e-4)), block_index))
    return rows_by_exp, rewards_by_exp, all_rows, all_blocks


def train_one(exp, lr, args, output_dir: Path, experts=None):
    env = make_env(args)
    obs = int(env.observation_space.shape[0])
    actions = int(env.action_space.n)
    agent = build_agent(exp, obs, actions, args, lr)
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
        done = False
        wall, cpu, gt = time.time(), time.process_time(), make_gpu_timer(args.device)

        while not done:
            if exp == "ensemble_own_noisy_rnd_count":
                action, src = select_ensemble_action(state, agent, experts or [])
            elif exp == "rnd_count":
                action, src = int(agent.select_action(state, epsilon=args.epsilon)), "rnd_epsilon"
            else:
                action, src = int(agent.select_action(state, epsilon=0.0)), "noisy_policy"
            src_counts[src] = src_counts.get(src, 0) + 1

            next_state, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            rnd_raw = 0.0
            rnd_loss = 0.0
            if hasattr(agent, "intrinsic_reward"):
                rnd_raw = float(agent.intrinsic_reward(next_state))
            if hasattr(agent, "train_rnd_predictor"):
                rnd_loss = float(agent.train_rnd_predictor(next_state))

            cnt_bonus, _cnt = count.bonus(next_state)
            full_reward = float(reward) + float(args.rnd_beta) * rnd_raw + float(cnt_bonus)
            agent.remember(state, action, full_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))

            env_reward += float(reward)
            train_reward += full_reward
            rnd_sum += rnd_raw
            cnt_sum += cnt_bonus
            rnd_loss_sum += rnd_loss
            rnd_loss_n += 1
            state = next_state
            steps += 1

        train_rewards.append(env_reward)
        row = {
            "phase": "train",
            "experiment": exp,
            "method": SHORT_LABELS[exp],
            "episode": ep,
            "env_reward": float(env_reward),
            "training_reward": float(train_reward),
            "steps": steps,
            "wall_time_seconds": float(time.time() - wall),
            "cpu_time_seconds": float(time.process_time() - cpu),
            "gpu_time_seconds": stop_gpu_timer(gt, args.device),
            "gpu_memory_mb": gpu_peak_mb(args.device),
            "process_memory_mb": process_memory_mb(),
            **smi_metrics(),
            "average_loss": avg(losses),
            "loss_updates": len(losses),
            "average_rnd_intrinsic": rnd_sum / max(steps, 1),
            "average_count_intrinsic": cnt_sum / max(steps, 1),
            "average_rnd_loss": rnd_loss_sum / max(rnd_loss_n, 1),
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "learning_rate": lr,
            "lr_multiplier": 1.0,
            "dqn_technology": "Set3: Noisy/RND/Count candidate ensemble",
            "rnd_beta": args.rnd_beta,
            "count_beta": args.count_beta,
            "count_state_bin_size": args.count_state_bin_size,
            "action_source_counts": str(src_counts),
        }
        rows.append(row)
        if ((ep + 1) % args.episode_block_size == 0) or ((ep + 1) == args.train_episodes):
            block_rows.append(block_summary(rows[-min(args.episode_block_size, len(rows)):], exp, "train", lr, len(block_rows) + 1))
        print(
            f"TRAIN {SHORT_LABELS[exp]:14s} ep={ep:03d} reward={env_reward:.2f} steps={steps} "
            f"wall={row['wall_time_seconds']:.2f}s cpu={row['cpu_time_seconds']:.2f}s "
            f"gpu={row['gpu_time_seconds']:.2f}s gpu_util={row['gpu_util_percent']:.1f}% "
            f"vram={row['gpu_memory_used_mb_smi']:.0f}/{row['gpu_memory_total_mb_smi']:.0f}MB "
            f"loss={row['average_loss']:.5f}",
            flush=True,
        )

    maybe_update_target(agent)
    model_path = output_dir / "models" / f"{exp}_lrmult_1.pt"
    maybe_save_agent(agent, model_path)
    pd.DataFrame(rows).to_csv(output_dir / f"{exp}_lrmult_1_train.csv", index=False)
    runtime = {
        "phase": "train",
        "experiment": exp,
        "method": SHORT_LABELS[exp],
        "learning_rate": lr,
        "lr_multiplier": 1.0,
        "total_wall_time_seconds": time.time() - exp_wall,
        "total_cpu_time_seconds": time.process_time() - exp_cpu,
        "total_gpu_time_seconds": sum(float(r["gpu_time_seconds"]) for r in rows),
        "gpu_memory_mb": gpu_peak_mb(args.device),
        "process_memory_mb": process_memory_mb(),
        "episodes": args.train_episodes,
        "model_path": str(model_path),
    }
    env.close()
    print(f"===== TRAINING END: {SHORT_LABELS[exp]} =====", flush=True)
    return agent, train_rewards, rows, block_rows, runtime


def test_one(exp, agent, args, lr, experts=None):
    env = make_env(args)
    rows, block_rows = [], []
    reset_gpu_peak(args.device)
    exp_wall, exp_cpu = time.time(), time.process_time()
    print(f"\n===== TESTING START: {SHORT_LABELS[exp]} =====", flush=True)

    with torch.no_grad():
        for ep in range(args.test_episodes):
            state, _ = env.reset()
            total = 0.0
            steps = 0
            done = False
            src_counts = {}
            wall, cpu, gt = time.time(), time.process_time(), make_gpu_timer(args.device)

            while not done:
                if exp == "ensemble_own_noisy_rnd_count":
                    action, src = select_ensemble_action(state, agent, experts or [])
                else:
                    action, src = int(agent.best_action(state)), "frozen_greedy"
                src_counts[src] = src_counts.get(src, 0) + 1
                state, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)
                total += float(reward)
                steps += 1

            row = {
                "phase": "test",
                "experiment": exp,
                "method": SHORT_LABELS[exp],
                "episode": ep,
                "env_reward": float(total),
                "steps": steps,
                "wall_time_seconds": time.time() - wall,
                "cpu_time_seconds": time.process_time() - cpu,
                "gpu_time_seconds": stop_gpu_timer(gt, args.device),
                "gpu_memory_mb": gpu_peak_mb(args.device),
                "process_memory_mb": process_memory_mb(),
                **smi_metrics(),
                "epsilon": args.epsilon,
                "gamma": args.gamma,
                "learning_rate": lr,
                "lr_multiplier": 1.0,
                "network_frozen": True,
                "updates_during_test": 0,
                "action_source_counts": str(src_counts),
            }
            rows.append(row)
            if ((ep + 1) % args.episode_block_size == 0) or ((ep + 1) == args.test_episodes):
                block_rows.append(block_summary(rows[-min(args.episode_block_size, len(rows)):], exp, "test", lr, len(block_rows) + 1))
            print(
                f"TEST  {SHORT_LABELS[exp]:14s} ep={ep:03d} reward={total:.2f} steps={steps} "
                f"wall={row['wall_time_seconds']:.2f}s cpu={row['cpu_time_seconds']:.2f}s "
                f"gpu={row['gpu_time_seconds']:.2f}s gpu_util={row['gpu_util_percent']:.1f}% "
                f"vram={row['gpu_memory_used_mb_smi']:.0f}/{row['gpu_memory_total_mb_smi']:.0f}MB",
                flush=True,
            )

    runtime = {
        "phase": "test",
        "experiment": exp,
        "method": SHORT_LABELS[exp],
        "learning_rate": lr,
        "lr_multiplier": 1.0,
        "total_wall_time_seconds": time.time() - exp_wall,
        "total_cpu_time_seconds": time.process_time() - exp_cpu,
        "total_gpu_time_seconds": sum(float(r["gpu_time_seconds"]) for r in rows),
        "gpu_memory_mb": gpu_peak_mb(args.device),
        "process_memory_mb": process_memory_mb(),
        "episodes": args.test_episodes,
        "network_frozen": True,
        "updates_during_test": 0,
    }
    env.close()
    print(f"===== TESTING END: {SHORT_LABELS[exp]} =====", flush=True)
    return rows, block_rows, runtime


def summarize(exp, train_rewards, train_rows, test_rows, args, lr):
    tr = [float(x) for x in train_rewards]
    te = [float(r["env_reward"]) for r in test_rows]
    avg_te = float(np.mean(te))
    conv_ep = convergence_episode(tr, avg_te, args.convergence_threshold_fraction, args.convergence_window)
    train_wall = sum(float(r["wall_time_seconds"]) for r in train_rows)
    train_cpu = sum(float(r["cpu_time_seconds"]) for r in train_rows)
    train_gpu = sum(float(r["gpu_time_seconds"]) for r in train_rows)
    frac = conv_ep / max(float(args.train_episodes), 1.0)
    return {
        "experiment": exp,
        "method": SHORT_LABELS[exp],
        "lr_multiplier": 1.0,
        "learning_rate": lr,
        "final_learning_rate": lr,
        "average_train_reward": float(np.mean(tr)),
        "median_train_reward": float(np.median(tr)),
        "std_train_reward": float(np.std(tr)),
        "min_train_reward": float(np.min(tr)),
        "max_train_reward": float(np.max(tr)),
        "q1_train_reward": pct(tr, 25),
        "q3_train_reward": pct(tr, 75),
        "average_test_reward": avg_te,
        "median_test_reward": float(np.median(te)),
        "mode_test_reward": reward_mode(te),
        "min_test_reward": float(np.min(te)),
        "max_test_reward": float(np.max(te)),
        "range_test_reward": float(np.max(te) - np.min(te)),
        "std_test_reward": float(np.std(te)),
        "q1_test_reward": pct(te, 25),
        "q3_test_reward": pct(te, 75),
        "convergence_episode": int(conv_ep),
        "convergence_time_seconds": frac * train_wall,
        "convergence_cpu_time_seconds": frac * train_cpu,
        "convergence_gpu_time_seconds": frac * train_gpu,
        "total_training_wall_time_seconds": train_wall,
        "total_training_cpu_time_seconds": train_cpu,
        "total_training_gpu_time_seconds": train_gpu,
        "total_training_episodes": args.train_episodes,
        "test_rewards": te,
        "dqn_technology": "Set3: Noisy/RND/Count candidate ensemble",
        "rnd_beta": args.rnd_beta,
        "count_beta": args.count_beta,
        "count_state_bin_size": args.count_state_bin_size,
        "network_frozen_during_testing": True,
    }


def save_csvs(results, train_rows, test_rows, block_rows, runtime_rows, output_dir):
    pd.DataFrame([{k: v for k, v in r.items() if k != "test_rewards"} for r in results]).to_csv(output_dir / "all_experiments_learning_rate_summary.csv", index=False)
    pd.DataFrame(train_rows).to_csv(output_dir / "all_experiments_train_episode_rewards.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "all_experiments_test_episode_rewards.csv", index=False)
    pd.DataFrame(block_rows).to_csv(output_dir / "all_experiments_episode_block_logs.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(output_dir / "all_experiments_runtime_logs.csv", index=False)


def apply_ieee_style():
    plt.rcParams.update({
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
    })


def save_fig(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ordered(results):
    return [next(r for r in results if r["experiment"] == exp) for exp in EXPERIMENTS]


def make_figures(results, train_rows, test_rows, block_rows, output_dir, args):
    apply_ieee_style()
    figdir = output_dir / "figures_ieee"
    figdir.mkdir(exist_ok=True)
    rows = ordered(results)
    labels = [SHORT_LABELS[r["experiment"]] for r in rows]
    colors = [COLORS[r["experiment"]] for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    vals = [r["average_test_reward"] for r in rows]
    err = [r["std_test_reward"] for r in rows]
    ax.bar(x, vals, yerr=err, capsize=3, color=colors, edgecolor="black", linewidth=.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Average test reward")
    ax.set_xlabel("Experiment")
    ax.set_title("Average Test Reward Across All Test Episodes")
    save_fig(fig, figdir, "ieee_average_test_reward")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bp = ax.boxplot([r["test_rewards"] for r in rows], patch_artist=True, labels=labels, showfliers=True, showmeans=True)
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c)
        p.set_alpha(.55)
        p.set_edgecolor("black")
    ax.set_ylabel("Environment reward")
    ax.set_xlabel("Experiment")
    ax.set_title("Test Reward Distribution Across All Test Episodes")
    save_fig(fig, figdir, "ieee_test_reward_boxplot")

    train_df = pd.DataFrame(train_rows)
    train_data = [train_df[train_df["experiment"] == exp]["env_reward"].astype(float).values for exp in EXPERIMENTS]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bp = ax.boxplot(train_data, patch_artist=True, labels=labels, showfliers=True, showmeans=True)
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c)
        p.set_alpha(.55)
        p.set_edgecolor("black")
    ax.set_ylabel("Training environment reward")
    ax.set_xlabel("Experiment")
    ax.set_title("Training Reward Distribution Across All Episodes")
    save_fig(fig, figdir, "ieee_training_reward_boxplot")

    for key, ylabel, name, title in [
        ("convergence_time_seconds", "Convergence time (s)", "ieee_convergence_time", "Training-Data Convergence Time"),
        ("convergence_episode", "Convergence episode", "ieee_convergence_episode", "Training Convergence Episode"),
    ]:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        vals = [r[key] for r in rows]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Experiment")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=8)
        save_fig(fig, figdir, name)

    df = pd.DataFrame(block_rows)
    for phase, name, title, ylabel in [
        ("train", "ieee_training_reward_blocks", "Training Reward by Episode Block", "Average training reward"),
        ("test", "ieee_testing_reward_blocks", "Testing Reward by Episode Block", "Average test reward"),
    ]:
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        sub = df[df["phase"] == phase] if not df.empty else pd.DataFrame()
        for exp in EXPERIMENTS:
            d = sub[sub["experiment"] == exp].sort_values("block_index") if not sub.empty else pd.DataFrame()
            if not d.empty:
                labs = [f"{int(a)+1}-{int(b)+1}" for a, b in zip(d["start_episode"], d["end_episode"])]
                ax.plot(labs, d["average_env_reward"].astype(float), marker="o", linewidth=1.6, label=SHORT_LABELS[exp], color=COLORS[exp])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode block")
        ax.set_title(title)
        ax.legend(frameon=False)
        save_fig(fig, figdir, name)

    make_dashboard(results, test_rows, block_rows, output_dir, args)


def table_axis(ax, rows, prefix, title):
    ax.axis("off")
    cols = ["Method", "Mean", "Median", "Std", "Min", "Max", "Q1", "Q3"]
    data = [[
        SHORT_LABELS[r["experiment"]],
        f"{r[f'average_{prefix}_reward']:.2f}",
        f"{r[f'median_{prefix}_reward']:.2f}",
        f"{r[f'std_{prefix}_reward']:.2f}",
        f"{r[f'min_{prefix}_reward']:.2f}",
        f"{r[f'max_{prefix}_reward']:.2f}",
        f"{r[f'q1_{prefix}_reward']:.2f}",
        f"{r[f'q3_{prefix}_reward']:.2f}",
    ] for r in rows]
    tab = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.45)
    for (rr, cc), cell in tab.get_celld().items():
        cell.set_linewidth(.3)
        if rr == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        elif cc == 0:
            cell.get_text().set_color(COLORS[rows[rr - 1]["experiment"]])
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10, color="#0000aa", fontweight="bold")


def make_dashboard(results, test_rows, block_rows, output_dir, args):
    rows = ordered(results)
    block_df = pd.DataFrame(block_rows)
    test_df = pd.DataFrame(test_rows)
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs = fig.add_gridspec(4, 3, height_ratios=[1.25, .95, 1, .75], hspace=.42, wspace=.28)
    fig.suptitle(f"Set3_v2 DQN in CARLA – Training & Testing Dashboard (Final LR = {args.raw_learning_rate:g})", fontsize=16, fontweight="bold", color="#0b123f", y=.985)
    fig.text(.5, .955, f"Environment rewards only  •  Train Episodes: {args.train_episodes}  •  Test Episodes: {args.test_episodes}  •  Block Size: {args.episode_block_size}  •  Epsilon: {args.epsilon}", ha="center", va="center", fontsize=10, color="#0b123f")

    for idx, (phase, title, ylabel) in enumerate([("train", "Average Training Reward by Episode Block", "Average training reward"), ("test", "Average Testing Reward by Episode Block", "Average testing reward")]):
        ax = fig.add_subplot(gs[0, idx])
        sub = block_df[block_df["phase"] == phase] if not block_df.empty else pd.DataFrame()
        for exp in EXPERIMENTS:
            d = sub[sub["experiment"] == exp].sort_values("block_index") if not sub.empty else pd.DataFrame()
            if not d.empty:
                labs = [f"{int(a)+1}-{int(b)+1}" for a, b in zip(d["start_episode"], d["end_episode"])]
                xx = np.arange(len(labs))
                yy = d["average_env_reward"].astype(float).values
                ax.plot(xx, yy, marker="o", label=SHORT_LABELS[exp], color=COLORS[exp], linewidth=1.8)
                ax.set_xticks(xx)
                ax.set_xticklabels(labs)
        ax.set_title(title, fontsize=10, color="#0000aa", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode block")
        ax.legend(fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.32))
        ax.grid(True, alpha=.25)

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(rows))
    y = [r["convergence_episode"] for r in rows]
    ax.bar(x, y, color=[COLORS[r["experiment"]] for r in rows], edgecolor="black", alpha=.85)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[r["experiment"]] for r in rows], rotation=10)
    ax.set_ylabel("Convergence episode")
    ax.set_title("Training Convergence Episode", fontsize=10, color="#0000aa", fontweight="bold")
    for i, v in enumerate(y):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    table_axis(fig.add_subplot(gs[1, 0]), rows, "train", f"Training Reward Summary (All {args.train_episodes} Episodes)")
    table_axis(fig.add_subplot(gs[1, 1]), rows, "test", f"Testing Reward Summary (All {args.test_episodes} Episodes)")

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    data = [[SHORT_LABELS[r["experiment"]], int(r["convergence_episode"]), int(r["total_training_episodes"]), f"{r['total_training_wall_time_seconds']:,.2f}", f"{r['convergence_time_seconds']:,.2f}"] for r in rows]
    tab = ax.table(cellText=data, colLabels=["Method", "Conv\nEpisode", "Train\nEpisodes", "Train\nTime(s)", "Conv\nTime(s)"], loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    tab.scale(1, 1.45)
    ax.set_title("Convergence Summary", fontsize=10, color="#0000aa", fontweight="bold")

    ax = fig.add_subplot(gs[2, 0])
    bp = ax.boxplot([r["test_rewards"] for r in rows], patch_artist=True, labels=[SHORT_LABELS[r["experiment"]] for r in rows], showmeans=True)
    for p, r in zip(bp["boxes"], rows):
        p.set_facecolor(COLORS[r["experiment"]])
        p.set_alpha(.55)
    ax.set_title(f"Test Reward Distribution (All {args.test_episodes} Episodes)", fontsize=10, color="#0000aa", fontweight="bold")
    ax.set_ylabel("Environment reward")

    ax = fig.add_subplot(gs[2, 1])
    for exp in EXPERIMENTS:
        d = test_df[test_df["experiment"] == exp]["env_reward"].astype(float).values
        if len(d):
            ax.hist(d, bins=30, alpha=.35, label=SHORT_LABELS[exp], color=COLORS[exp], edgecolor="black", linewidth=.2)
    ax.set_title("Test Reward Distribution Histogram", fontsize=10, color="#0000aa", fontweight="bold")
    ax.set_xlabel("Environment reward")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    vals = [r["convergence_time_seconds"] for r in rows]
    ax.bar(x, vals, color=[COLORS[r["experiment"]] for r in rows], edgecolor="black", alpha=.85)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[r["experiment"]] for r in rows], rotation=10)
    ax.set_ylabel("Convergence time (s)")
    ax.set_title("Convergence Time by Experiment", fontsize=10, color="#0000aa", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax = fig.add_subplot(gs[3, 0])
    ax.axis("off")
    best = max(rows, key=lambda r: r["average_test_reward"])
    fast = min(rows, key=lambda r: r["convergence_time_seconds"])
    stable = min(rows, key=lambda r: r["std_test_reward"])
    ax.text(.02, .95, f"Key Takeaways\n\n• {SHORT_LABELS[best['experiment']]} has the highest average test reward ({best['average_test_reward']:.2f}).\n• {SHORT_LABELS[fast['experiment']]} has the fastest convergence time ({fast['convergence_time_seconds']:.2f}s).\n• {SHORT_LABELS[stable['experiment']]} has the lowest test reward std ({stable['std_test_reward']:.2f}).\n• Test networks are frozen; no updates occur during testing.", ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#c9d9ff"))

    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    rr = {r["experiment"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: r["average_test_reward"], reverse=True))}
    cr = {r["experiment"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: r["convergence_time_seconds"]))}
    be = min(EXPERIMENTS, key=lambda e: rr[e] + cr[e])
    br = next(r for r in rows if r["experiment"] == be)
    ax.text(.5, .55, f"Best Overall\n\n{SHORT_LABELS[be].upper()}\n\nAverage Test Reward: {br['average_test_reward']:.2f}\nConvergence Time: {br['convergence_time_seconds']:.2f}s\n\nRanking uses reward rank + convergence-time rank.", ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#2ca25f"))

    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    ax.text(.02, .95, f"Experiment Settings\n\nEnvironment           : CARLA 0.9.14 (Real)\nAlgorithm             : DQN\nSet                   : Set3_v2\nExperiments           : Noisy+Count, RND+Count, Ensemble\nLearning Rate Mult.   : 1\nFinal Learning Rate   : {args.raw_learning_rate:g}\nEpsilon               : {args.epsilon}\nTraining Episodes     : {args.train_episodes}\nTesting Episodes      : {args.test_episodes}\nEpisode Block Size    : {args.episode_block_size}\nConvergence Threshold : {args.convergence_threshold_fraction:g}", ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#c9d9ff"))

    fig.text(.01, .01, "Notes: Rewards are environment rewards only. RND/count rewards are training-only. Convergence uses training rewards and total training wall time.", fontsize=8, color="#333")
    fig.savefig(output_dir / "set3_v2_dashboard.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "set3_v2_dashboard.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(out, args):
    (out / "README_set3_v2.txt").write_text(f"""set3_v2 outputs

Experiments in order:
1. Noisy + Count
2. RND + Count
3. Ensemble Own + Noisy/RND expert candidates

Neural setup: Set 3 agents from src.dqn_agent with count-based intrinsic reward and RND where applicable.
Testing is frozen: no optimizer, replay, RND, or target updates.
Reward graphs use all test episodes and environment reward only.
Convergence graphs use training data only.

Main CSVs:
- all_experiments_train_episode_rewards.csv
- all_experiments_test_episode_rewards.csv
- all_experiments_episode_block_logs.csv
- all_experiments_runtime_logs.csv
- all_experiments_learning_rate_summary.csv

Figures:
- set3_v2_dashboard.png / .pdf
- figures_ieee/*.png and *.pdf

Configuration:
train_episodes={args.train_episodes}
test_episodes={args.test_episodes}
max_episode_steps={args.max_episode_steps}
epsilon={args.epsilon}
learning_rate={args.raw_learning_rate}
rnd_beta={args.rnd_beta}
count_beta={args.count_beta}
convergence_threshold_fraction={args.convergence_threshold_fraction}
convergence_window={args.convergence_window}
""")


def run(args):
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    lr = float(args.raw_learning_rate if args.raw_learning_rate > 0 else 1e-4)
    print("=" * 72, flush=True)
    print("set3_v2 ensemble-only: load Noisy/RND experts, train Ensemble, test all", flush=True)
    print(f"Output dir: {out}", flush=True)
    print(f"CUDA: {torch.cuda.is_available()}  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
    print("=" * 72, flush=True)

    # Determine dimensions from the real CARLA environment once.
    env = make_env(args)
    obs = int(env.observation_space.shape[0])
    actions = int(env.action_space.n)
    env.close()

    trained = {}
    train_rows_by_exp, train_rewards_by_exp, all_train, all_blocks = load_existing_training_rows(out)
    all_test, all_runtime, results = [], [], []

    # Load the two already-trained expert models. They are not retrained.
    for exp in BASE_EXPERIMENTS:
        checkpoint = out / "models" / f"{exp}_lrmult_1.pt"
        print(f"Loading {SHORT_LABELS[exp]} expert: {checkpoint}", flush=True)
        agent = build_agent(exp, obs, actions, args, lr)
        trained[exp] = load_agent_checkpoint(agent, checkpoint, args.device)

    experts = [(exp, trained[exp]) for exp in BASE_EXPERIMENTS]

    # Train only the Ensemble. Existing ensemble outputs are replaced intentionally.
    ens_exp = "ensemble_own_noisy_rnd_count"
    agent, rewards, rows, blocks, runtime = train_one(ens_exp, lr, args, out, experts=experts)
    trained[ens_exp] = agent
    train_rewards_by_exp[ens_exp] = rewards
    train_rows_by_exp[ens_exp] = rows
    all_train.extend(rows)
    all_blocks.extend(blocks)
    all_runtime.append(runtime)

    # Test all three models using frozen behavior and environment reward only.
    for exp in EXPERIMENTS:
        test_rows, test_blocks, test_runtime = test_one(
            exp, trained[exp], args, lr,
            experts=experts if exp in ENSEMBLE_EXPERIMENTS else None,
        )
        all_test.extend(test_rows)
        all_blocks.extend(test_blocks)
        all_runtime.append(test_runtime)
        results.append(summarize(
            exp, train_rewards_by_exp[exp], train_rows_by_exp[exp],
            test_rows, args, lr,
        ))

    save_csvs(results, all_train, all_test, all_blocks, all_runtime, out)
    make_figures(results, all_train, all_test, all_blocks, out, args)
    write_readme(out, args)
    print(f"\nSaved ensemble-only Set3_v2 outputs to: {out}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout-seconds", type=float, default=10.0)
    p.add_argument("--reward-mode", default="ontology_combined")
    p.add_argument("--target-speed", type=float, default=30.0)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--train-episodes", type=int, default=500)
    p.add_argument("--test-episodes", type=int, default=300)
    p.add_argument("--episode-block-size", type=int, default=100)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--raw-learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=50000)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--inference-margin", type=float, default=0.01)
    p.add_argument("--rnd-beta", type=float, default=0.01)
    p.add_argument("--count-beta", type=float, default=0.05)
    p.add_argument("--count-state-bin-size", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    p.add_argument("--convergence-window", type=int, default=10)
    p.add_argument("--output-dir", default="/workspace/results/set3_v2_july09")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

