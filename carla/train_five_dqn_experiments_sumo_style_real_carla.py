"""Train/test SUMO-style five DQN experiments in REAL CARLA.

This file mirrors the SUMO experiment structure/labels:
  1. median_50                  -> Median 50
  2. standard_epsilon           -> Standard Epsilon / Epsilon Greedy
  3. median_50_first            -> Median 50 First
  4. exp4_own_table_candidates  -> Exp 4: Own Table + Expert Candidates
  5. exp5_precomputed_best      -> Exp 5: Precomputed Best Table

Creates exactly four output graphs:
  average_reward_vs_learning_rate.png
  best_learning_rate_by_experiment.png
  convergence_time_vs_experiment.png
  average_reward_vs_epsilon.png

Important CARLA notes:
- This script uses real CARLA only. CARLA must already be running and reachable.
- It keeps the exact same five SUMO experiment labels shown in your screenshot.
- Every experiment uses a neural DQN, not a Q-table.
- Every DQN is the same advanced architecture: NoisyNet + RND + count-based bonus.
- Exp 4 and Exp 5 are DQN equivalents of the SUMO table logic:
  Exp 4 uses own DQN values to choose among expert candidates + own candidate.
  Exp 5 initializes from the best expert and chooses using precomputed/expert values.
- Graphs and CSV summaries use environment reward only. RND/count bonuses are training-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.carla_env import CarlaDrivingEnv
from src.dqn_agent import DQNAgent, NoisyRNDDQNAgent

BASE_EXPERIMENTS = ["median_50", "standard_epsilon", "median_50_first"]
ENSEMBLE_EXPERIMENTS = ["exp4_own_table_candidates", "exp5_precomputed_best"]
EXPERIMENTS = BASE_EXPERIMENTS + ENSEMBLE_EXPERIMENTS

EXPERIMENT_LABELS = {
    "median_50": "Median 50",
    "standard_epsilon": "Standard Epsilon",
    "median_50_first": "Median 50 First",
    "exp4_own_table_candidates": "Exp 4: Own Table + Expert Candidates",
    "exp5_precomputed_best": "Exp 5: Precomputed Best Table",
}

EXPERIMENT_COLORS = {
    "median_50": "tab:orange",
    "standard_epsilon": "tab:blue",
    "median_50_first": "tab:green",
    "exp4_own_table_candidates": "tab:purple",
    "exp5_precomputed_best": "tab:red",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_real_carla_env(args) -> CarlaDrivingEnv:
    return CarlaDrivingEnv(
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout_seconds,
        reward_mode=args.reward_mode,
        target_speed_kmh=args.target_speed,
        max_episode_steps=args.max_episode_steps,
        use_mock_when_carla_missing=False,
    )


def average(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def reward_mode(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    rounded = [round(float(v), 6) for v in values]
    counts: Dict[float, int] = {}
    for v in rounded:
        counts[v] = counts.get(v, 0) + 1
    best_count = max(counts.values())
    return float(max(v for v, c in counts.items() if c == best_count))


def calculate_score(test_rewards: Sequence[float]) -> float:
    avg = float(np.mean(test_rewards))
    med = float(np.median(test_rewards))
    reward_range = float(np.max(test_rewards) - np.min(test_rewards))
    return float(0.20 * avg + 0.50 * med - 0.30 * reward_range)


def calculate_convergence_episode(train_rewards: Sequence[float], target_reward: float, threshold_fraction: float, window: int) -> int:
    if not train_rewards:
        return 0
    window = max(1, min(int(window), len(train_rewards)))
    threshold = float(threshold_fraction) * float(target_reward)
    rewards = np.asarray(train_rewards, dtype=float)
    rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
    for index, value in enumerate(rolling):
        if float(value) >= threshold:
            return int(index + window)
    return int(len(train_rewards))


class CountBasedBonus:
    """Simple state-count intrinsic reward used by every CARLA SUMO-style DQN.

    State vector values are divided by bin_size and rounded to form a discrete key.
    Bonus = beta / sqrt(N(state_key)). This is used only for training updates;
    reported rewards and plots remain pure environment rewards.
    """

    def __init__(self, beta: float = 0.05, bin_size: float = 1.0):
        self.beta = float(beta)
        self.bin_size = max(float(bin_size), 1e-6)
        self.counts: Dict[tuple[int, ...], int] = {}

    def key(self, state: np.ndarray) -> tuple[int, ...]:
        arr = np.asarray(state, dtype=float)
        return tuple(np.round(arr / self.bin_size).astype(int).tolist())

    def bonus(self, state: np.ndarray) -> tuple[float, int]:
        key = self.key(state)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return float(self.beta / math.sqrt(count)), int(count)


def build_agent(obs_size: int, action_size: int, args, learning_rate: float) -> DQNAgent:
    # All five experiments use the same advanced DQN technology:
    # NoisyNet + RND + target network + replay buffer + CUDA support.
    return NoisyRNDDQNAgent(
        observation_size=obs_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_interval=args.target_update_interval,
        inference_margin=args.inference_margin,
        rnd_beta=args.rnd_beta,
        device=args.device,
    )


def select_median50_action(agent: DQNAgent, state: np.ndarray, action_size: int) -> int:
    q = np.asarray(agent.get_q_values(state), dtype=float)
    median = float(np.median(q))
    lower_half = [i for i, value in enumerate(q) if float(value) <= median]
    if not lower_half:
        lower_half = list(range(action_size))
    return int(random.choice(lower_half))


def select_base_action(experiment: str, agent: DQNAgent, state: np.ndarray, episode: int, args, action_size: int) -> int:
    if experiment == "standard_epsilon":
        return int(agent.select_action(state, epsilon=args.epsilon))
    if experiment == "median_50":
        if random.random() < args.epsilon:
            return select_median50_action(agent, state, action_size)
        return int(agent.best_action(state))
    if experiment == "median_50_first":
        median_episode_count = int(round(args.epsilon * args.train_episodes))
        if episode < median_episode_count:
            return select_median50_action(agent, state, action_size)
        return int(agent.best_action(state))
    raise ValueError(f"Unknown base experiment: {experiment}")


def unique_expert_actions(state: np.ndarray, experts: Sequence[Tuple[str, DQNAgent]]) -> list[Tuple[int, str, DQNAgent]]:
    out = []
    seen = set()
    for name, expert in experts:
        action = int(expert.best_action(state))
        if action not in seen:
            out.append((action, name, expert))
            seen.add(action)
    return out


def select_exp4_action(state: np.ndarray, own_agent: DQNAgent, experts: Sequence[Tuple[str, DQNAgent]]) -> int:
    candidates = unique_expert_actions(state, experts)
    own_action = int(own_agent.best_action(state))
    if own_action not in {a for a, _, _ in candidates}:
        candidates.append((own_action, "own", own_agent))
    own_q = np.asarray(own_agent.get_q_values(state), dtype=float)
    return int(max(candidates, key=lambda item: float(own_q[item[0]]))[0])


def select_exp5_action(state: np.ndarray, own_agent: DQNAgent, experts: Sequence[Tuple[str, DQNAgent]]) -> int:
    candidates = unique_expert_actions(state, experts)
    own_action = int(own_agent.best_action(state))
    if own_action not in {a for a, _, _ in candidates}:
        candidates.append((own_action, "own", own_agent))
    scored = []
    for action, name, agent in candidates:
        q = np.asarray(agent.get_q_values(state), dtype=float)
        scored.append((float(q[action]), int(action), name))
    scored.sort(reverse=True, key=lambda x: x[0])
    return int(scored[0][1])


def save_model(agent: DQNAgent, output_dir: Path, experiment: str, lr_mult: float, suffix: str = "") -> Path:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{experiment}_lrmult_{lr_mult:g}{suffix}.pt"
    agent.save(str(path))
    return path


def train_one_experiment(experiment: str, learning_rate: float, lr_mult: float, args, output_dir: Path,
                         experts: Sequence[Tuple[str, DQNAgent]] | None = None,
                         init_from: DQNAgent | None = None) -> Tuple[DQNAgent, List[float], Path]:
    env = make_real_carla_env(args)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = build_agent(obs_size, action_size, args, learning_rate)
    if init_from is not None:
        agent.q_network.load_state_dict(copy.deepcopy(init_from.q_network.state_dict()))
        agent.target_network.load_state_dict(copy.deepcopy(init_from.target_network.state_dict()))
        if hasattr(agent, "rnd_target") and hasattr(init_from, "rnd_target"):
            agent.rnd_target.load_state_dict(copy.deepcopy(init_from.rnd_target.state_dict()))
        if hasattr(agent, "rnd_predictor") and hasattr(init_from, "rnd_predictor"):
            agent.rnd_predictor.load_state_dict(copy.deepcopy(init_from.rnd_predictor.state_dict()))

    count_bonus = CountBasedBonus(args.count_beta, args.count_state_bin_size)
    rows = []
    train_rewards: List[float] = []

    for episode in range(args.train_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        losses = []
        steps = 0
        done = False
        start = time.time()
        while not done:
            if experiment in BASE_EXPERIMENTS:
                action = select_base_action(experiment, agent, state, episode, args, action_size)
            elif experiment == "exp4_own_table_candidates":
                action = select_exp4_action(state, agent, experts or [])
            elif experiment == "exp5_precomputed_best":
                action = select_exp5_action(state, agent, experts or [])
            else:
                raise ValueError(experiment)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            # All five experiments use the same advanced DQN training signal:
            # environment reward + RND intrinsic novelty + count-based novelty.
            rnd_intrinsic = 0.0
            if hasattr(agent, "intrinsic_reward") and hasattr(agent, "train_rnd_predictor"):
                rnd_intrinsic = float(agent.intrinsic_reward(next_state))
                agent.train_rnd_predictor(next_state)
            count_intrinsic, state_count = count_bonus.bonus(next_state)
            training_reward = float(reward) + float(args.rnd_beta) * rnd_intrinsic + count_intrinsic

            agent.remember(state, action, training_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))
            total_reward += float(reward)
            state = next_state
            steps += 1
        train_rewards.append(total_reward)
        rows.append({
            "phase": "train", "experiment": experiment, "episode": episode,
            "env_reward": total_reward, "steps": steps,
            "convergence_time_seconds": time.time() - start,
            "average_loss": average(losses), "epsilon": args.epsilon,
            "gamma": args.gamma, "learning_rate": learning_rate, "lr_multiplier": lr_mult,
            "dqn_technology": "NoisyNet + RND + CountBased",
            "rnd_beta": args.rnd_beta, "count_beta": args.count_beta,
            "count_state_bin_size": args.count_state_bin_size,
        })
        print(f"TRAIN {EXPERIMENT_LABELS[experiment]:42s} ep={episode:03d} reward={total_reward:.2f} steps={steps}")

    pd.DataFrame(rows).to_csv(output_dir / f"{experiment}_lrmult_{lr_mult:g}_train.csv", index=False)
    path = save_model(agent, output_dir, experiment, lr_mult)
    env.close()
    return agent, train_rewards, path


def test_one_experiment(experiment: str, agent: DQNAgent, args, learning_rate: float, lr_mult: float,
                        experts: Sequence[Tuple[str, DQNAgent]] | None = None) -> list[dict]:
    env = make_real_carla_env(args)
    rows = []
    for episode in range(args.test_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        start = time.time()
        while not done:
            if experiment == "exp4_own_table_candidates":
                action = select_exp4_action(state, agent, experts or [])
            elif experiment == "exp5_precomputed_best":
                action = select_exp5_action(state, agent, experts or [])
            else:
                action = int(agent.best_action(state))
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            steps += 1
        rows.append({
            "phase": "test", "experiment": experiment, "episode": episode,
            "env_reward": total_reward, "steps": steps,
            "convergence_time_seconds": time.time() - start,
            "epsilon": args.epsilon, "gamma": args.gamma,
            "learning_rate": learning_rate, "lr_multiplier": lr_mult,
        })
        print(f"TEST  {EXPERIMENT_LABELS[experiment]:42s} ep={episode:03d} reward={total_reward:.2f} steps={steps}")
    env.close()
    return rows


def summarize_result(experiment: str, lr_mult: float, lr: float, train_rewards: Sequence[float], test_rows: Sequence[dict], args) -> dict:
    rewards = [float(r["env_reward"]) for r in test_rows]
    avg = float(np.mean(rewards))
    med = float(np.median(rewards))
    rng = float(np.max(rewards) - np.min(rewards))
    return {
        "experiment": experiment,
        "lr_multiplier": float(lr_mult),
        "learning_rate": float(lr),
        "final_learning_rate": float(lr),
        "average_train_reward": float(np.mean(train_rewards)),
        "average_test_reward": avg,
        "median_test_reward": med,
        "mode_test_reward": reward_mode(rewards),
        "min_test_reward": float(np.min(rewards)),
        "max_test_reward": float(np.max(rewards)),
        "range_test_reward": rng,
        "std_test_reward": float(np.std(rewards)),
        "best_score": calculate_score(rewards),
        "convergence_episode": calculate_convergence_episode(train_rewards, avg, args.convergence_threshold_fraction, args.convergence_window),
        "test_rewards": rewards,
        "dqn_technology": "NoisyNet + RND + CountBased",
        "rnd_beta": args.rnd_beta,
        "count_beta": args.count_beta,
        "count_state_bin_size": args.count_state_bin_size,
    }


def text_panel(fig, x: float, y: float, text: str, color: str) -> None:
    fig.text(x, y, text, ha="left", va="top", fontsize=7, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.94), zorder=20)


def save_average_reward_vs_learning_rate(results: list[dict], output_dir: Path, args) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for idx, exp in enumerate(EXPERIMENTS):
        rows = sorted([r for r in results if r["experiment"] == exp], key=lambda r: r["learning_rate"])
        xs = [r["learning_rate"] for r in rows]
        ys = [r["average_test_reward"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=EXPERIMENT_COLORS[exp], label=EXPERIMENT_LABELS[exp])
        best = max(rows, key=lambda r: (r["average_test_reward"], r["best_score"]))
        ax.scatter([best["learning_rate"]], [best["average_test_reward"]], marker="*", s=260,
                   color=EXPERIMENT_COLORS[exp], edgecolor="black", zorder=8)
        text_panel(fig, 0.70, 0.82 - idx * 0.115,
                   f"Best {EXPERIMENT_LABELS[exp]}\nLR={best['learning_rate']:.3g}\nAvg={best['average_test_reward']:.1f}\nS={best['best_score']:.1f}",
                   EXPERIMENT_COLORS[exp])
    ax.set_xscale("log")
    ticks = sorted({float(r["learning_rate"]) for r in results})
    ax.set_xticks(ticks); ax.set_xticklabels([f"{v:.6g}" for v in ticks], rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Learning rate"); ax.set_ylabel("Average test reward")
    ax.set_title("CARLA Average Reward vs Learning Rate - SUMO-style Experiments", pad=14)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", bbox_to_anchor=(1.03, 0.02), fontsize=8, frameon=True)
    fig.text(0.70, 0.92, "Best starred configurations", ha="left", va="top", fontsize=9, weight="bold")
    fig.text(0.5, 0.985, f"epsilon={args.epsilon:g}, gamma={args.gamma:g}; S=0.20×Average + 0.50×Median - 0.30×Range", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.67, 0.92])
    fig.savefig(output_dir / "average_reward_vs_learning_rate.png", dpi=300)
    plt.close(fig)


def save_best_learning_rate_graph(results: list[dict], output_dir: Path) -> None:
    best_rows = [max([r for r in results if r["experiment"] == exp], key=lambda r: (r["average_test_reward"], r["best_score"])) for exp in EXPERIMENTS]
    data = [r["test_rewards"] for r in best_rows]
    positions = list(range(1, len(best_rows)+1))
    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    box = ax1.boxplot(data, positions=positions, patch_artist=True, widths=0.55)
    for patch, row in zip(box["boxes"], best_rows):
        patch.set_facecolor(EXPERIMENT_COLORS[row["experiment"]]); patch.set_alpha(0.45); patch.set_edgecolor("black")
    for idx, (x, row) in enumerate(zip(positions, best_rows)):
        exp = row["experiment"]
        ax1.scatter(x, row["average_test_reward"], marker="*", s=250, color=EXPERIMENT_COLORS[exp], edgecolor="black", zorder=8)
        text_panel(fig, 0.70, 0.82 - idx * 0.115,
                   f"{EXPERIMENT_LABELS[exp]}\nAvg={row['average_test_reward']:.1f}\nLR={row['learning_rate']:.3g}\nS={row['best_score']:.1f}", EXPERIMENT_COLORS[exp])
    ax1.set_xticks(positions); ax1.set_xticklabels([f"{r['learning_rate']:.3g}" for r in best_rows], fontsize=8)
    ax1.set_xlabel("Learning rate"); ax1.set_ylabel("Average test reward / distribution")
    ax1.set_title("CARLA Best Learning Rate by Experiment", pad=16); ax1.grid(True, axis="y", alpha=0.35)
    ax2 = ax1.twinx(); ax2.plot(positions, [r["best_score"] for r in best_rows], marker="D", linestyle="--", linewidth=1.4, color="black", label="Score S"); ax2.set_ylabel("Score S")
    handles = [Patch(facecolor=EXPERIMENT_COLORS[e], alpha=0.45, label=EXPERIMENT_LABELS[e]) for e in EXPERIMENTS]
    handles.append(Patch(facecolor="white", edgecolor="black", label="Star = maximum average reward"))
    # Keep experiment legend on the right-side panel, outside plot axes,
    # matching the SUMO screenshot request.
    ax1.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.03, 0.02), fontsize=8, frameon=True)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.03, 0.98), fontsize=8, frameon=True)
    fig.text(0.70, 0.92, "Best box-plot configurations", ha="left", va="top", fontsize=9, weight="bold")
    fig.text(0.5, 0.985, "Labels are in the right-side panel, outside plot axes.", ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.67, 0.92])
    fig.savefig(output_dir / "best_learning_rate_by_experiment.png", dpi=300)
    plt.close(fig)


def save_convergence_graph(results: list[dict], output_dir: Path) -> None:
    best_rows = [max([r for r in results if r["experiment"] == exp], key=lambda r: (r["average_test_reward"], r["best_score"])) for exp in EXPERIMENTS]
    xs = list(range(1, len(best_rows)+1)); vals = [int(r["convergence_episode"]) for r in best_rows]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    bars = ax.bar(xs, vals, width=0.6, edgecolor="black")
    for idx, (bar, row) in enumerate(zip(bars, best_rows)):
        exp = row["experiment"]; bar.set_color(EXPERIMENT_COLORS[exp]); bar.set_alpha(0.65)
        ax.scatter(idx+1, row["convergence_episode"], marker="*", s=230, color=EXPERIMENT_COLORS[exp], edgecolor="black", zorder=8)
        ax.text(idx+1+0.12, row["convergence_episode"], f"{int(row['convergence_episode'])}", ha="left", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        text_panel(fig, 0.70, 0.82 - idx * 0.115,
                   f"{EXPERIMENT_LABELS[exp]}\nConv={int(row['convergence_episode'])} ep\nLR={row['learning_rate']:.3g}\nAvg={row['average_test_reward']:.1f}", EXPERIMENT_COLORS[exp])
    ax.set_xticks(xs); ax.set_xticklabels([EXPERIMENT_LABELS[r["experiment"]] for r in best_rows], fontsize=8)
    ax.set_xlabel("Experiment"); ax.set_ylabel("Convergence episode")
    ax.set_title("CARLA Convergence Time vs Experiments", pad=16); ax.grid(True, axis="y", alpha=0.35)
    if vals: ax.set_ylim(0, max(vals) * 1.18 + 1)
    fig.text(0.70, 0.92, "Convergence summaries", ha="left", va="top", fontsize=9, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.67, 0.92])
    fig.savefig(output_dir / "convergence_time_vs_experiment.png", dpi=300)
    plt.close(fig)


def save_average_reward_vs_epsilon(eps_results: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    eps_values = sorted({float(r["epsilon"]) for r in eps_results})
    for idx, exp in enumerate(EXPERIMENTS):
        rows = sorted([r for r in eps_results if r["experiment"] == exp], key=lambda r: r["epsilon"])
        xs = [r["epsilon"] for r in rows]; ys = [r["average_test_reward"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.7, color=EXPERIMENT_COLORS[exp], label=EXPERIMENT_LABELS[exp])
        best = max(rows, key=lambda r: (r["average_test_reward"], r["best_score"]))
        ax.scatter([best["epsilon"]], [best["average_test_reward"]], marker="*", s=260, color=EXPERIMENT_COLORS[exp], edgecolor="black", zorder=8)
        text_panel(fig, 0.70, 0.82 - idx * 0.105, f"Best {EXPERIMENT_LABELS[exp]}\nε={best['epsilon']:.3g}\nAvg={best['average_test_reward']:.1f}\nS={best['best_score']:.1f}", EXPERIMENT_COLORS[exp])
    ax.set_xticks(eps_values); ax.set_xticklabels([f"{v:.2f}".rstrip("0").rstrip(".") if v else "0" for v in eps_values], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Epsilon"); ax.set_ylabel("Average test reward")
    ax.set_title("CARLA Average Reward vs Epsilon - SUMO-style Experiments", pad=14); ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", bbox_to_anchor=(1.03, 0.02), fontsize=8, frameon=True)
    fig.text(0.70, 0.92, "Best epsilon configurations", ha="left", va="top", fontsize=9, weight="bold")
    fig.tight_layout(rect=[0, 0, 0.67, 0.92])
    fig.savefig(output_dir / "average_reward_vs_epsilon.png", dpi=300)
    plt.close(fig)


def write_csvs(results: list[dict], test_rows: list[dict], output_dir: Path) -> None:
    clean = []
    for r in results:
        d = {k: v for k, v in r.items() if k != "test_rewards"}
        clean.append(d)
    pd.DataFrame(clean).to_csv(output_dir / "all_experiments_learning_rate_summary.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output_dir / "all_experiments_test_episode_rewards.csv", index=False)


def parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(',') if x.strip()]
    if not vals: raise ValueError("List cannot be empty")
    return vals


def run_sweep(args) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    lr_mults = parse_float_list(args.lr_multipliers)
    raw_lr = args.raw_learning_rate if args.raw_learning_rate > 0 else 1.0 / max(args.train_episodes * args.max_episode_steps, 1)
    results: list[dict] = []
    all_test_rows: list[dict] = []

    for lr_mult in lr_mults:
        lr = raw_lr * lr_mult
        print(f"\n=== LR multiplier {lr_mult:g}; final LR={lr:.8g} ===")
        trained: Dict[str, DQNAgent] = {}
        train_rewards_by_exp: Dict[str, List[float]] = {}
        for exp in BASE_EXPERIMENTS:
            agent, train_rewards, _ = train_one_experiment(exp, lr, lr_mult, args, output_dir)
            trained[exp] = agent; train_rewards_by_exp[exp] = train_rewards
        experts = [(exp, trained[exp]) for exp in BASE_EXPERIMENTS]
        agent4, train4, _ = train_one_experiment("exp4_own_table_candidates", lr, lr_mult, args, output_dir, experts=experts)
        trained["exp4_own_table_candidates"] = agent4; train_rewards_by_exp["exp4_own_table_candidates"] = train4
        best_base_exp = max(BASE_EXPERIMENTS, key=lambda e: average(train_rewards_by_exp[e]))
        agent5, train5, _ = train_one_experiment("exp5_precomputed_best", lr, lr_mult, args, output_dir, experts=experts, init_from=trained[best_base_exp])
        trained["exp5_precomputed_best"] = agent5; train_rewards_by_exp["exp5_precomputed_best"] = train5

        for exp in EXPERIMENTS:
            rows = test_one_experiment(exp, trained[exp], args, lr, lr_mult, experts=experts if exp in ENSEMBLE_EXPERIMENTS else None)
            all_test_rows.extend(rows)
            results.append(summarize_result(exp, lr_mult, lr, train_rewards_by_exp[exp], rows, args))

    write_csvs(results, all_test_rows, output_dir)
    save_average_reward_vs_learning_rate(results, output_dir, args)
    save_best_learning_rate_graph(results, output_dir)
    save_convergence_graph(results, output_dir)

    if args.run_epsilon_sweep:
        best = max(results, key=lambda r: (r["average_test_reward"], r["best_score"]))
        fixed_lr_mult = args.epsilon_sweep_lr_multiplier if args.epsilon_sweep_lr_multiplier > 0 else float(best["lr_multiplier"])
        eps_results = []
        for eps in parse_float_list(args.epsilon_values):
            local = argparse.Namespace(**vars(args)); local.epsilon = float(eps); local.lr_multipliers = str(fixed_lr_mult); local.run_epsilon_sweep = False
            tmp_dir = output_dir / f"epsilon_{eps:g}"; tmp_dir.mkdir(parents=True, exist_ok=True); local.output_dir = str(tmp_dir)
            # lightweight nested run for one LR, but collect summaries only
            before = []
            # Duplicate the one-LR training/testing code to avoid recursive plotting.
            lr = raw_lr * fixed_lr_mult
            trained = {}; train_rewards_by_exp = {}
            for exp in BASE_EXPERIMENTS:
                agent, tr, _ = train_one_experiment(exp, lr, fixed_lr_mult, local, tmp_dir); trained[exp]=agent; train_rewards_by_exp[exp]=tr
            experts = [(exp, trained[exp]) for exp in BASE_EXPERIMENTS]
            agent4, tr4, _ = train_one_experiment("exp4_own_table_candidates", lr, fixed_lr_mult, local, tmp_dir, experts=experts); trained["exp4_own_table_candidates"]=agent4; train_rewards_by_exp["exp4_own_table_candidates"]=tr4
            best_base_exp = max(BASE_EXPERIMENTS, key=lambda e: average(train_rewards_by_exp[e]))
            agent5, tr5, _ = train_one_experiment("exp5_precomputed_best", lr, fixed_lr_mult, local, tmp_dir, experts=experts, init_from=trained[best_base_exp]); trained["exp5_precomputed_best"]=agent5; train_rewards_by_exp["exp5_precomputed_best"]=tr5
            for exp in EXPERIMENTS:
                rows = test_one_experiment(exp, trained[exp], local, lr, fixed_lr_mult, experts=experts if exp in ENSEMBLE_EXPERIMENTS else None)
                s = summarize_result(exp, fixed_lr_mult, lr, train_rewards_by_exp[exp], rows, local); s["epsilon"] = float(eps); eps_results.append(s)
        pd.DataFrame([{k:v for k,v in r.items() if k != "test_rewards"} for r in eps_results]).to_csv(output_dir / "epsilon_sweep_average_reward_summary.csv", index=False)
        save_average_reward_vs_epsilon(eps_results, output_dir)
    else:
        pd.DataFrame(columns=["experiment", "epsilon", "average_test_reward"]).to_csv(output_dir / "epsilon_sweep_average_reward_summary.csv", index=False)

    print(f"\nSaved SUMO-style CARLA outputs to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout-seconds", type=float, default=10.0)
    p.add_argument("--reward-mode", default="ontology_combined")
    p.add_argument("--target-speed", type=float, default=30.0)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--train-episodes", type=int, default=20)
    p.add_argument("--test-episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--raw-learning-rate", type=float, default=0.0, help="0 means raw_lr=1/(train_episodes*max_episode_steps)")
    p.add_argument("--lr-multipliers", default="1,1.25,0.25,0.5,0.75,1.5,1.75,2,2.5,3,4,5")
    p.add_argument("--inference-margin", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=50000)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--rnd-beta", type=float, default=0.01, help="RND intrinsic reward scale used by every experiment during training.")
    p.add_argument("--count-beta", type=float, default=0.05, help="Count-based intrinsic reward scale used by every experiment during training.")
    p.add_argument("--count-state-bin-size", type=float, default=1.0, help="State bin size for count-based exploration.")
    p.add_argument("--device", default="cuda", help="Use cuda on RunPod. Use --device cpu only for local debugging.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convergence-threshold-fraction", type=float, default=0.95)
    p.add_argument("--convergence-window", type=int, default=10)
    p.add_argument("--run-epsilon-sweep", action="store_true")
    p.add_argument("--epsilon-values", default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.30,0.40,0.50")
    p.add_argument("--epsilon-sweep-lr-multiplier", type=float, default=0.0)
    p.add_argument("--output-dir", default="results/five_dqn_sumo_style_real_carla")
    return p.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
