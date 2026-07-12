"""Train/test Graph Set 2 epsilon/median experiments in REAL CARLA 0.9.14.

Experiment set 2 contains exactly:
  1. standard_epsilon                       -> Epsilon Greedy
  2. median_50_first                        -> Median First 50
  3. median_50                              -> Median 50
  4. ensemble_epsilon_medianfirst_median50  -> Ensemble of Epsilon, Median First 50, Median 50

All four experiments use the same advanced neural network:
  NoisyNet + RND + Count-based DQN, optimized with Adam.

Creates exactly three output graphs with clean labels and no stars/scores:
  average_reward_vs_experiment.png
  reward_boxplot_by_experiment.png
  convergence_time_boxplot_by_experiment.png

Important CARLA notes:
- This script uses real CARLA only. CARLA must already be running and reachable.
- It keeps the exact same five SUMO experiment labels shown in your screenshot.
- Every experiment uses a neural DQN, not a Q-table.
- Every DQN is the same advanced architecture: NoisyNet + RND + count-based bonus.
- Every DQN uses target networks and replay buffer through NoisyRNDDQNAgent.
- Epsilon is fixed to 0.2 only; epsilon sweep is disabled.
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

BASE_EXPERIMENTS = ["standard_epsilon", "median_50_first", "median_50"]
ENSEMBLE_EXPERIMENTS = ["ensemble_epsilon_medianfirst_median50"]
EXPERIMENTS = BASE_EXPERIMENTS + ENSEMBLE_EXPERIMENTS

EXPERIMENT_LABELS = {
    "standard_epsilon": "Epsilon",
    "median_50_first": "Med First",
    "median_50": "Med 50",
    "ensemble_epsilon_medianfirst_median50": "Ensemble",
}

EXPERIMENT_COLORS = {
    "standard_epsilon": "tab:blue",
    "median_50_first": "tab:green",
    "median_50": "tab:orange",
    "ensemble_epsilon_medianfirst_median50": "tab:purple",
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


def select_ensemble_action(state: np.ndarray, own_agent: DQNAgent, experts: Sequence[Tuple[str, DQNAgent]]) -> int:
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
            elif experiment == "ensemble_epsilon_medianfirst_median50":
                action = select_ensemble_action(state, agent, experts or [])
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

            # Always update this experiment's own neural network.
            agent.remember(state, action, training_reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(float(loss))

            # Ensemble rule: epsilon, median-first-50, and median-50 expert
            # networks are read-only candidate action providers. Only the
            # ensemble's own Noisy+RND+Count DQN is updated above.

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
            if experiment == "ensemble_epsilon_medianfirst_median50":
                action = select_ensemble_action(state, agent, experts or [])
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
        "convergence_episode": calculate_convergence_episode(train_rewards, avg, args.convergence_threshold_fraction, args.convergence_window),
        "test_rewards": rewards,
        "dqn_technology": "NoisyNet + RND + CountBased",
        "rnd_beta": args.rnd_beta,
        "count_beta": args.count_beta,
        "count_state_bin_size": args.count_state_bin_size,
    }


def text_panel(fig, x: float, y: float, text: str, color: str) -> None:
    fig.text(x, y, text, ha="left", va="top", fontsize=6.5, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.94), zorder=20)


def _fixed_lr_rows(results: list[dict]) -> list[dict]:
    """Return one row per experiment. With one fixed LR this is just the only row."""
    rows = []
    for exp in EXPERIMENTS:
        exp_rows = [r for r in results if r["experiment"] == exp]
        if exp_rows:
            rows.append(max(exp_rows, key=lambda r: r["average_test_reward"]))
    return rows


def _right_panel(fig, title: str, lines: list[str]) -> None:
    """Write all explanations outside the plot area on the right side."""
    fig.text(0.62, 0.91, title, ha="left", va="top", fontsize=10, weight="bold")
    y = 0.86
    for line in lines:
        fig.text(
            0.62,
            y,
            line,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.96),
        )
        y -= 0.14


def save_average_reward_vs_experiment(results: list[dict], output_dir: Path, args) -> None:
    """Graph 1: one fixed learning rate; compare average test reward by experiment."""
    rows = _fixed_lr_rows(results)
    xs = list(range(1, len(rows) + 1))
    ys = [r["average_test_reward"] for r in rows]

    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(xs, ys, width=0.62, edgecolor="black")
    for bar, row in zip(bars, rows):
        exp = row["experiment"]
        bar.set_color(EXPERIMENT_COLORS[exp])
        bar.set_alpha(0.70)

    ax.set_xticks(xs)
    ax.set_xticklabels([EXPERIMENT_LABELS[r["experiment"]] for r in rows], rotation=15, ha="right")
    ax.set_xlabel("Experiment", labelpad=14)
    ax.set_ylabel("Average test reward (reward units)", labelpad=10)
    ax.set_title("CARLA Average Reward vs Experiment", pad=14)
    ax.grid(True, axis="y", alpha=0.35)

    panel_lines = [
        f"{EXPERIMENT_LABELS[r['experiment']]}: avg reward={r['average_test_reward']:.2f} reward units\nLR={r['learning_rate']:.1e}, epsilon=0.2"
        for r in rows
    ]
    _right_panel(fig, "Graph indicators", panel_lines)
    fig.text(
        0.62,
        0.20,
        "Network: NoisyNet + RND + count-based DQN\nTarget network: enabled\nReplay buffer: enabled\nScores: not used",
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.96),
    )
    fig.subplots_adjust(left=0.08, right=0.56, bottom=0.20, top=0.88)
    fig.savefig(output_dir / "average_reward_vs_experiment.png", dpi=300)
    plt.close(fig)


def save_reward_boxplot_by_experiment(results: list[dict], output_dir: Path, args) -> None:
    """Graph 2: test reward distribution for each experiment at the fixed LR."""
    rows = _fixed_lr_rows(results)
    data = [r["test_rewards"] for r in rows]
    positions = list(range(1, len(rows) + 1))

    fig, ax = plt.subplots(figsize=(16, 7))
    box = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.55, showmeans=False)
    for patch, row in zip(box["boxes"], rows):
        patch.set_facecolor(EXPERIMENT_COLORS[row["experiment"]])
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels([EXPERIMENT_LABELS[r["experiment"]] for r in rows], rotation=15, ha="right")
    ax.set_xlabel("Experiment", labelpad=14)
    ax.set_ylabel("Test episode reward distribution (reward units)", labelpad=10)
    ax.set_title("CARLA Reward Box Plot by Experiment", pad=14)
    ax.grid(True, axis="y", alpha=0.35)

    panel_lines = [
        f"{EXPERIMENT_LABELS[r['experiment']]}: avg={r['average_test_reward']:.2f}, median={r['median_test_reward']:.2f}\nmin={r['min_test_reward']:.2f}, max={r['max_test_reward']:.2f} reward units"
        for r in rows
    ]
    _right_panel(fig, "Box-plot indicators", panel_lines)
    fig.text(
        0.62,
        0.20,
        f"Fixed learning rate={rows[0]['learning_rate']:.1e}\nFixed epsilon=0.2\nNo symbols/stars/scores are used",
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.96),
    )
    fig.subplots_adjust(left=0.08, right=0.56, bottom=0.20, top=0.88)
    fig.savefig(output_dir / "reward_boxplot_by_experiment.png", dpi=300)
    plt.close(fig)


def save_convergence_time_boxplot_by_experiment(test_rows: list[dict], output_dir: Path, args) -> None:
    """Graph 3: wall-clock episode runtime distribution by experiment.

    The value comes from time.time() at the start and end of each test episode:
    convergence_time_seconds = episode_end_time - episode_start_time.
    Units are seconds.
    """
    data = []
    labels = []
    colors = []
    means = []
    for exp in EXPERIMENTS:
        vals = [float(r["convergence_time_seconds"]) for r in test_rows if r["experiment"] == exp]
        if vals:
            data.append(vals)
            labels.append(EXPERIMENT_LABELS[exp])
            colors.append(EXPERIMENT_COLORS[exp])
            means.append(float(np.mean(vals)))

    fig, ax = plt.subplots(figsize=(16, 7))
    positions = list(range(1, len(data) + 1))
    box = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.55, showmeans=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("Experiment", labelpad=14)
    ax.set_ylabel("Convergence time per test episode (seconds)", labelpad=10)
    ax.set_title("CARLA Convergence Time Box Plot by Experiment", pad=14)
    ax.grid(True, axis="y", alpha=0.35)

    panel_lines = [
        f"{label}: mean={mean:.3f} seconds\nvalues are per-test-episode wall-clock runtimes"
        for label, mean in zip(labels, means)
    ]
    _right_panel(fig, "Convergence-time indicators", panel_lines)
    fig.text(
        0.62,
        0.20,
        "Calculation:\nstart = time.time() before episode loop\nend = time.time() after done=True\nconvergence time = end - start\nUnit: seconds, not milliseconds",
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.96),
    )
    fig.subplots_adjust(left=0.08, right=0.56, bottom=0.20, top=0.88)
    fig.savefig(output_dir / "convergence_time_boxplot_by_experiment.png", dpi=300)
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
    # Fixed setup requested: epsilon only 0.2 and one industry-standard DQN learning rate.
    args.epsilon = 0.2
    args.run_epsilon_sweep = False
    lr_mults = [1.0]
    raw_lr = args.raw_learning_rate if args.raw_learning_rate > 0 else 1e-4
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
        agent_ens, train_ens, _ = train_one_experiment("ensemble_epsilon_medianfirst_median50", lr, lr_mult, args, output_dir, experts=experts)
        trained["ensemble_epsilon_medianfirst_median50"] = agent_ens; train_rewards_by_exp["ensemble_epsilon_medianfirst_median50"] = train_ens

        for exp in EXPERIMENTS:
            rows = test_one_experiment(exp, trained[exp], args, lr, lr_mult, experts=experts if exp in ENSEMBLE_EXPERIMENTS else None)
            all_test_rows.extend(rows)
            results.append(summarize_result(exp, lr_mult, lr, train_rewards_by_exp[exp], rows, args))

    write_csvs(results, all_test_rows, output_dir)
    save_average_reward_vs_experiment(results, output_dir, args)
    save_reward_boxplot_by_experiment(results, output_dir, args)
    save_convergence_time_boxplot_by_experiment(all_test_rows, output_dir, args)

    print(f"\nSaved Graph Set 2 CARLA outputs to: {output_dir}")


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
    p.add_argument("--raw-learning-rate", type=float, default=1e-4, help="Industry-standard default for Adam DQN. Use 0 only if you want raw_lr=1e-4 fallback.")
    p.add_argument("--lr-multipliers", default="1", help="Fixed to one LR multiplier for this version.")
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
    p.add_argument("--run-epsilon-sweep", action="store_true", help="Ignored in this version; epsilon is forced to 0.2.")
    p.add_argument("--epsilon-values", default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.30,0.40,0.50")
    p.add_argument("--epsilon-sweep-lr-multiplier", type=float, default=0.0)
    p.add_argument("--output-dir", default="results/graph_set_2_fixed_lr_eps02_three_graphs")
    return p.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
