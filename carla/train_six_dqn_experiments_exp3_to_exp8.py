"""Train and test 6 DQN exploration experiments in the REAL CARLA environment only.

Experiments kept/designed from the original file:
3.  noisy                    - NoisyNet DQN
4.  rnd                      - RND / prediction-based exploration DQN
5.  ensemble_expert_q        - Exp3/Exp4 experts propose best actions; choose by expert Q/reward estimate
6.  ensemble_own_q           - Exp3/Exp4 experts propose best actions; choose by this experiment's own DQN Q
7.  ensemble_expert_q_count  - Exp5 + count-based intrinsic reward during training
8.  ensemble_own_q_count     - Exp6 + count-based intrinsic reward during training

Important:
- Only 6 experiments are run and plotted.
- Exp1 Base, Exp2 Epsilon, and random_dqn are removed.
- Every experiment has its own DQN / target network / replay buffer / optimizer / saved model.
- Exp3 and Exp4 train independently.
- Exp5, Exp6, Exp7, and Exp8 use already-trained Exp3 Noisy and Exp4 RND models as frozen action proposers.
- Count-based reward is used only for Exp7 and Exp8 during training.
- RND intrinsic reward is used only for Exp4 during training.
- Testing and all graphs use environment reward only.
- No mock environment is used. CARLA must be running.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.carla_env import CarlaDrivingEnv
from src.dqn_agent import DQNAgent, NoisyDQNAgent, RNDDQNAgent


EXPERIMENTS = [
    "noisy",
    "rnd",
    "ensemble_expert_q",
    "ensemble_own_q",
    "ensemble_expert_q_count",
    "ensemble_own_q_count",
]

PLOT_LABELS = {
    "noisy": "Exp3 Noisy",
    "rnd": "Exp4 RND",
    "ensemble_expert_q": "Exp5 Exp3+Exp4 Expert-Q",
    "ensemble_own_q": "Exp6 Exp3+Exp4 Own-Q",
    "ensemble_expert_q_count": "Exp7 Exp5 + Count",
    "ensemble_own_q_count": "Exp8 Exp6 + Count",
}

EXPERT_Q_EXPERIMENTS = {"ensemble_expert_q", "ensemble_expert_q_count"}
OWN_Q_EXPERIMENTS = {"ensemble_own_q", "ensemble_own_q_count"}
COUNT_EXPERIMENTS = {"ensemble_expert_q_count", "ensemble_own_q_count"}
ENSEMBLE_EXPERIMENTS = EXPERT_Q_EXPERIMENTS | OWN_Q_EXPERIMENTS


class CountBasedBonus:
    """Count-based intrinsic reward for continuous CARLA observations.

    The continuous state vector is discretized into bins. The bonus decreases as
    the same bin is visited more often:

        bonus = count_beta / sqrt(N(state_bin))

    This is used only during training for Exp10 and Exp11.
    """

    def __init__(self, beta: float, bin_size: float):
        self.beta = float(beta)
        self.bin_size = max(float(bin_size), 1e-8)
        self.counts: dict[tuple[int, ...], int] = {}

    def _key(self, state) -> tuple[int, ...]:
        return tuple(int(round(float(value) / self.bin_size)) for value in state)

    def bonus(self, state) -> tuple[float, int]:
        key = self._key(state)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return self.beta / math.sqrt(count), count


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
    return sum(values) / max(len(values), 1)


def build_agent(name: str, obs_size: int, action_size: int, args):
    common = dict(
        observation_size=obs_size,
        action_size=action_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_interval=args.target_update_interval,
        inference_margin=args.inference_margin,
        device=args.device,
    )
    if name == "noisy":
        return NoisyDQNAgent(**common)
    if name == "rnd":
        return RNDDQNAgent(**common, rnd_beta=args.rnd_beta)
    return DQNAgent(**common)


def choose_train_action(agent, experiment: str, state, args, action_size: int) -> int:
    if experiment == "noisy":
        return agent.select_action(state, epsilon=0.0)
    if experiment == "rnd":
        return agent.select_action(state, epsilon=args.epsilon)
    raise ValueError(f"Unknown independent experiment: {experiment}")


def train_independent_experiment(experiment: str, args, output_dir: Path) -> Path:
    """Train Exp3 Noisy or Exp4 RND independently."""
    env = make_real_carla_env(args)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = build_agent(experiment, obs_size, action_size, args)
    rows = []

    for episode in range(args.train_episodes):
        state, _ = env.reset()
        total_env_reward = 0.0
        total_training_reward = 0.0
        total_count_bonus = 0.0
        losses = []
        steps = 0
        start_time = time.time()
        done = False

        while not done:
            action = choose_train_action(agent, experiment, state, args, action_size)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if experiment == "rnd":
                # Exp4 RND: environment reward + RND intrinsic reward for training only.
                training_reward = agent.remember_with_intrinsic_reward(state, action, env_reward, next_state, done)
            else:
                training_reward = float(env_reward)
                agent.remember(state, action, training_reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            total_env_reward += float(env_reward)
            total_training_reward += float(training_reward)
            state = next_state
            steps += 1

        row = {
            "phase": "train",
            "experiment": experiment,
            "episode": episode,
            "env_reward": total_env_reward,
            "training_reward": total_training_reward,
            "count_bonus_total": total_count_bonus,
            "steps": steps,
            "convergence_time_seconds": time.time() - start_time,
            "average_loss": average(losses),
            "epsilon": args.epsilon if experiment == "rnd" else 0.0,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "inference_margin": args.inference_margin,
        }
        rows.append(row)
        print(f"TRAIN {PLOT_LABELS[experiment]:24s} episode={episode:03d} env_reward={total_env_reward:.2f} steps={steps}")

    model_path = output_dir / "models" / f"{experiment}.pt"
    agent.save(str(model_path))
    pd.DataFrame(rows).to_csv(output_dir / f"{experiment}_train.csv", index=False)
    env.close()
    return model_path


def load_agent_by_name(name: str, args, output_dir: Path):
    path = output_dir / "models" / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing model for {name}: {path}")
    if name == "noisy":
        return NoisyDQNAgent.load(str(path), device=args.device)
    if name == "rnd":
        return RNDDQNAgent.load(str(path), device=args.device)
    return DQNAgent.load(str(path), device=args.device)


def load_noisy_rnd_experts(args, output_dir: Path):
    """Exp3 Noisy and Exp4 RND experts."""
    return [
        ("noisy", load_agent_by_name("noisy", args, output_dir)),
        ("rnd", load_agent_by_name("rnd", args, output_dir)),
    ]


def unique_expert_best_actions(state, experts: Sequence[tuple[str, object]]):
    candidates = []
    seen_actions = set()
    for expert_name, expert in experts:
        action = int(expert.best_action(state))
        if action not in seen_actions:
            candidates.append((action, expert_name, expert))
            seen_actions.add(action)
    return candidates


def select_by_expert_q(state, experts: Sequence[tuple[str, object]], margin: float):
    """Exp5/Exp8/Exp10 rule: compare each expert's own Q estimate for its best action."""
    candidates = unique_expert_best_actions(state, experts)
    scored = []
    for action, expert_name, expert in candidates:
        q_values = expert.get_q_values(state)
        scored.append((float(q_values[action]), int(action), expert_name))
    scored.sort(key=lambda item: item[0], reverse=True)

    best_q = scored[0][0]
    close_candidates = [item for item in scored if (best_q - item[0]) <= margin]
    chosen_q, chosen_action, chosen_expert = close_candidates[0]
    return int(chosen_action), chosen_expert, float(chosen_q), scored


def select_by_own_q(state, chooser_agent, experts: Sequence[tuple[str, object]], margin: float):
    """Exp6/Exp9/Exp11 rule: experts propose actions, chooser's own DQN evaluates them."""
    candidates = unique_expert_best_actions(state, experts)
    own_q_values = chooser_agent.get_q_values(state)
    scored = [(float(own_q_values[action]), int(action), expert_name) for action, expert_name, _ in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_q = scored[0][0]
    close_candidates = [item for item in scored if (best_q - item[0]) <= margin]
    chosen_q, chosen_action, chosen_expert = close_candidates[0]
    return int(chosen_action), chosen_expert, float(chosen_q), scored


def load_experts_for_ensemble(experiment: str, args, output_dir: Path):
    if experiment in ENSEMBLE_EXPERIMENTS:
        return load_noisy_rnd_experts(args, output_dir)
    raise ValueError(f"No expert set defined for {experiment}")


def select_ensemble_action(experiment: str, state, chooser_agent, experts, margin: float):
    if experiment in EXPERT_Q_EXPERIMENTS:
        return select_by_expert_q(state, experts, margin)
    if experiment in OWN_Q_EXPERIMENTS:
        return select_by_own_q(state, chooser_agent, experts, margin)
    raise ValueError(f"Unknown ensemble experiment: {experiment}")


def train_ensemble_experiment(experiment: str, args, output_dir: Path) -> Path:
    """Train Exp5, Exp6, Exp7, or Exp8."""
    env = make_real_carla_env(args)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    chooser_agent = build_agent(experiment, obs_size, action_size, args)
    experts = load_experts_for_ensemble(experiment, args, output_dir)
    count_bonus = CountBasedBonus(args.count_beta, args.count_state_bin_size) if experiment in COUNT_EXPERIMENTS else None
    rows = []

    for episode in range(args.train_episodes):
        state, _ = env.reset()
        total_env_reward = 0.0
        total_training_reward = 0.0
        total_count_bonus = 0.0
        losses = []
        steps = 0
        start_time = time.time()
        done = False

        while not done:
            action, selected_expert, selected_q, scored = select_ensemble_action(
                experiment, state, chooser_agent, experts, args.inference_margin
            )
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if count_bonus is not None:
                bonus, state_count = count_bonus.bonus(next_state)
            else:
                bonus, state_count = 0.0, 0

            # Exp7/Exp8 use env + count bonus for training only.
            # Exp5/Exp6 use env reward only for training.
            training_reward = float(env_reward) + float(bonus)

            # Only this experiment's DQN is updated. Experts are frozen.
            chooser_agent.remember(state, action, training_reward, next_state, done)
            loss = chooser_agent.learn()
            if loss is not None:
                losses.append(loss)

            total_env_reward += float(env_reward)
            total_training_reward += float(training_reward)
            total_count_bonus += float(bonus)
            state = next_state
            steps += 1

        row = {
            "phase": "train",
            "experiment": experiment,
            "episode": episode,
            "env_reward": total_env_reward,
            "training_reward": total_training_reward,
            "count_bonus_total": total_count_bonus,
            "count_beta": args.count_beta if experiment in COUNT_EXPERIMENTS else 0.0,
            "count_state_bin_size": args.count_state_bin_size if experiment in COUNT_EXPERIMENTS else 0.0,
            "steps": steps,
            "convergence_time_seconds": time.time() - start_time,
            "average_loss": average(losses),
            "epsilon": 0.0,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "inference_margin": args.inference_margin,
        }
        rows.append(row)
        print(
            f"TRAIN {PLOT_LABELS[experiment]:24s} episode={episode:03d} "
            f"env_reward={total_env_reward:.2f} count_bonus={total_count_bonus:.2f} steps={steps}"
        )

    model_path = output_dir / "models" / f"{experiment}.pt"
    chooser_agent.save(str(model_path))
    pd.DataFrame(rows).to_csv(output_dir / f"{experiment}_train.csv", index=False)
    env.close()
    return model_path


def test_experiment(experiment: str, args, output_dir: Path) -> list[Dict]:
    """Test using trained models only. No learning and no intrinsic/count rewards."""
    env = make_real_carla_env(args)
    rows = []

    agent = load_agent_by_name(experiment, args, output_dir)
    experts = None
    if experiment in ENSEMBLE_EXPERIMENTS:
        experts = load_experts_for_ensemble(experiment, args, output_dir)

    for episode in range(args.test_episodes):
        state, _ = env.reset()
        total_env_reward = 0.0
        steps = 0
        start_time = time.time()
        done = False

        while not done:
            if experts is not None:
                action, _, _, _ = select_ensemble_action(experiment, state, agent, experts, args.inference_margin)
            else:
                action = agent.best_action(state)

            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_env_reward += float(env_reward)
            state = next_state
            steps += 1

        row = {
            "phase": "test",
            "experiment": experiment,
            "episode": episode,
            "env_reward": total_env_reward,
            "steps": steps,
            "convergence_time_seconds": time.time() - start_time,
        }
        rows.append(row)
        print(f"TEST  {PLOT_LABELS[experiment]:24s} episode={episode:03d} env_reward={total_env_reward:.2f} steps={steps}")

    pd.DataFrame(rows).to_csv(output_dir / f"{experiment}_test.csv", index=False)
    env.close()
    return rows


def plot_boxplot(df: pd.DataFrame, experiments: list[str], column: str, title: str, ylabel: str, output_path: Path) -> None:
    data = [df.loc[df["experiment"] == exp, column].values for exp in experiments]
    labels = [PLOT_LABELS.get(exp, exp) for exp in experiments]

    plt.figure(figsize=(max(20, len(experiments) * 2.2), 9))
    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        widths=0.55,
        patch_artist=True,
    )
    means = [float(pd.Series(values).mean()) if len(values) else 0.0 for values in data]
    for index, mean_value in enumerate(means, start=1):
        plt.text(index, mean_value, f"mean={mean_value:.1f}", ha="center", va="bottom", fontsize=9, rotation=90)

    plt.title(title, fontsize=18)
    plt.xlabel("Experiment", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240)
    plt.close()


def plot_single_experiment(df: pd.DataFrame, experiment: str, output_dir: Path) -> None:
    sub = df[df["experiment"] == experiment].copy()
    if sub.empty:
        return

    label = PLOT_LABELS.get(experiment, experiment)
    x = sub["episode"].astype(int)

    # Reward per test episode.
    plt.figure(figsize=(9, 5))
    plt.bar(x, sub["env_reward"])
    mean_reward = float(sub["env_reward"].mean())
    plt.axhline(mean_reward, linestyle="--", linewidth=1.5, label=f"Mean = {mean_reward:.1f}")
    plt.title(f"{label}: Environment Reward per Test Episode", fontsize=14)
    plt.xlabel("Test Episode")
    plt.ylabel("Episode Environment Reward")
    plt.xticks(x)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_reward_by_episode.png", dpi=220)
    plt.close()

    # Convergence/time per test episode.
    plt.figure(figsize=(9, 5))
    plt.bar(x, sub["convergence_time_seconds"])
    mean_time = float(sub["convergence_time_seconds"].mean())
    plt.axhline(mean_time, linestyle="--", linewidth=1.5, label=f"Mean = {mean_time:.1f}s")
    plt.title(f"{label}: Convergence Time per Test Episode", fontsize=14)
    plt.xlabel("Test Episode")
    plt.ylabel("Episode Time / Convergence Time (seconds)")
    plt.xticks(x)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment}_convergence_time_by_episode.png", dpi=220)
    plt.close()


def plot_all_results(test_rows: list[Dict], output_dir: Path) -> None:
    df = pd.DataFrame(test_rows)
    df.to_csv(output_dir / "all_test_results.csv", index=False)

    ordered = [name for name in EXPERIMENTS if name in set(df["experiment"])]

    plot_boxplot(
        df,
        ordered,
        "env_reward",
        "Environment Reward vs Experiment - All 6 Experiments",
        "Episode Environment Reward",
        output_dir / "reward_boxplot_all_6.png",
    )
    plot_boxplot(
        df,
        ordered,
        "convergence_time_seconds",
        "Convergence Time vs Experiment - All 6 Experiments",
        "Episode Time / Convergence Time (seconds)",
        output_dir / "convergence_time_boxplot_all_6.png",
    )

    # Backward-compatible names.
    plot_boxplot(
        df,
        ordered,
        "env_reward",
        "Environment Reward vs Experiment - All 6 Experiments",
        "Episode Environment Reward",
        output_dir / "reward_boxplot.png",
    )
    plot_boxplot(
        df,
        ordered,
        "convergence_time_seconds",
        "Convergence Time vs Experiment - All 6 Experiments",
        "Episode Time / Convergence Time (seconds)",
        output_dir / "convergence_time_boxplot.png",
    )

    per_dir = output_dir / "per_experiment_graphs"
    per_dir.mkdir(parents=True, exist_ok=True)
    for experiment in ordered:
        plot_single_experiment(df, experiment, per_dir)


def run_all(args) -> None:
    output_dir = Path(args.output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

    if args.train:
        # First train independent source models: Exp3 Noisy and Exp4 RND.
        for experiment in ["noisy", "rnd"]:
            train_independent_experiment(experiment, args, output_dir)

        # Exp5 and Exp6 use only Exp3 Noisy + Exp4 RND as frozen experts.
        train_ensemble_experiment("ensemble_expert_q", args, output_dir)
        train_ensemble_experiment("ensemble_own_q", args, output_dir)

        # Exp7 = Exp5 + count-based methods.
        train_ensemble_experiment("ensemble_expert_q_count", args, output_dir)

        # Exp8 = Exp6 + count-based methods.
        train_ensemble_experiment("ensemble_own_q_count", args, output_dir)

    if args.test:
        all_rows = []
        for experiment in EXPERIMENTS:
            all_rows.extend(test_experiment(experiment, args, output_dir))
        plot_all_results(all_rows, output_dir)
        print(f"Saved models, CSV files, combined box plots, and per-experiment graphs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--reward-mode", default="ontology_combined")
    parser.add_argument("--target-speed", type=float, default=30.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--inference-margin", type=float, default=0.01)
    parser.add_argument("--rnd-beta", type=float, default=0.01)

    parser.add_argument("--count-beta", type=float, default=0.05)
    parser.add_argument("--count-state-bin-size", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/six_dqn_experiments_exp3_to_exp8_real_carla")

    parser.add_argument("--train", action="store_true", help="Train all six experiments.")
    parser.add_argument("--test", action="store_true", help="Test all six trained experiments and plot graphs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.train and not args.test:
        args.train = True
        args.test = True
    run_all(args)
