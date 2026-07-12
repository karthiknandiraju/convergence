#!/usr/bin/env python3
"""
SUMO tabular Q-learning experiment with two exploration strategies.

Experiments
-----------
1. epsilon_greedy
   - At every training step: with probability epsilon choose any action randomly;
     otherwise choose argmax(Q).

2. median_50
   - At every training step: with probability epsilon choose randomly from actions
     whose Q-value is <= the current median Q-value; otherwise choose argmax(Q).

Notes
-----
- This is pure TABULAR Q-learning: no neural network, no replay buffer,
  no target network, and no optimizer.
- Continuous observations are discretized into bins.
- Each experiment uses its own independent Q-table.
- Testing is frozen and greedy only (argmax); no Q-table updates.
- Default run: 500 training episodes, 300 test episodes, 500 max steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


# -----------------------------------------------------------------------------
# SUMO environment
# -----------------------------------------------------------------------------

class SumoDrivingEnv(gym.Env):
    """
    Single-lane SUMO environment with one controlled ego vehicle and one leader.

    Observation:
        0: ego speed / target speed
        1: ego position / road length
        2: remaining distance / road length
        3: leader gap / 100
        4: leader speed / target speed

    Actions:
        0: decelerate
        1: maintain speed
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
            high=np.array([3.0, 2.0, 2.0, 10.0, 3.0], dtype=np.float32),
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
                "Install SUMO with:\n  apt update && apt install -y sumo sumo-tools"
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
        rng = np.random.default_rng(seed)
        leader_speed = float(rng.uniform(8.0, 16.0))
        leader_depart = float(rng.uniform(0.0, 0.8))

        route_file = self.scenario_dir / f"episode_{os.getpid()}_{id(self)}.rou.xml"
        route_file.write_text(
            f"""<routes>
    <vType id="egoType" accel="3.0" decel="6.0" sigma="0.0"
           length="5.0" minGap="2.5" maxSpeed="25.0" guiShape="passenger"/>
    <vType id="leaderType" accel="2.0" decel="4.5" sigma="0.0"
           length="5.0" minGap="2.5" maxSpeed="{leader_speed:.3f}" guiShape="passenger"/>
    <route id="mainRoute" edges="road"/>
    <vehicle id="leader" type="leaderType" route="mainRoute"
             depart="{leader_depart:.3f}" departSpeed="{leader_speed:.3f}"/>
    <vehicle id="ego" type="egoType" route="mainRoute"
             depart="2.0" departSpeed="5.0"/>
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
                "Install it with:\n  python -m pip install traci sumolib"
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
                min(float(gap), 1000.0) / 100.0,
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
        self.last_position = float(self.conn.vehicle.getLanePosition("ego"))
        return self._get_observation(), {}

    def step(self, action: int):
        if self.conn is None:
            raise RuntimeError("Call env.reset() before env.step().")

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
        arrived = "ego" in set(self.conn.simulation.getArrivedIDList())
        collision = "ego" in set(self.conn.simulation.getCollidingVehiclesIDList())

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
        term_reason = "running"
        if collision:
            reward -= 50.0
            terminated = True
            term_reason = "collision"
        elif arrived or position >= self.road_length - 5.0:
            reward += 100.0
            terminated = True
            term_reason = "goal"
        elif not ego_present:
            reward -= 10.0
            terminated = True
            term_reason = "removed"

        truncated = self.step_count >= self.max_episode_steps
        if truncated and not terminated:
            term_reason = "max_steps"

        self.last_position = position
        info = {
            "term_reason": term_reason,
            "position": position,
            "speed": speed,
            "progress": progress,
            "collision": collision,
        }
        return self._get_observation(), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close(False)
            except Exception:
                pass
            self.conn = None


# -----------------------------------------------------------------------------
# Tabular Q-learning
# -----------------------------------------------------------------------------

class StateDiscretizer:
    """Convert the 5D continuous observation into a finite tabular state."""

    def __init__(self, bins: Tuple[int, int, int, int, int]) -> None:
        self.bins = tuple(int(x) for x in bins)
        if any(x < 2 for x in self.bins):
            raise ValueError("Every discretization dimension must have at least 2 bins.")

        # Observation ranges match SumoDrivingEnv.observation_space.
        lows = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        highs = np.array([3.0, 2.0, 2.0, 10.0, 3.0], dtype=np.float64)
        self.edges = [
            np.linspace(lows[i], highs[i], self.bins[i] + 1)[1:-1]
            for i in range(5)
        ]

    @property
    def shape(self) -> Tuple[int, int, int, int, int]:
        return self.bins

    def encode(self, observation: np.ndarray) -> Tuple[int, int, int, int, int]:
        obs = np.asarray(observation, dtype=np.float64)
        return tuple(
            int(np.digitize(obs[i], self.edges[i], right=False))
            for i in range(5)
        )


class TabularQLearningAgent:
    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_count: int,
        alpha: float,
        gamma: float,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.action_count = int(action_count)
        self.q_table = np.zeros((*state_shape, action_count), dtype=np.float32)

    def q_values(self, state_index: Tuple[int, ...]) -> np.ndarray:
        return self.q_table[state_index]

    def greedy_action(self, state_index: Tuple[int, ...]) -> int:
        # Random tie-breaking avoids always favoring action 0 when values are equal.
        q = self.q_values(state_index)
        best = np.flatnonzero(q == np.max(q))
        return int(random.choice(best.tolist()))

    def median_lower_half_action(self, state_index: Tuple[int, ...]) -> int:
        q = self.q_values(state_index)
        median_value = float(np.median(q))
        candidates = np.flatnonzero(q <= median_value)
        if len(candidates) == 0:
            return self.greedy_action(state_index)
        return int(random.choice(candidates.tolist()))

    def update(
        self,
        state_index: Tuple[int, ...],
        action: int,
        reward: float,
        next_state_index: Tuple[int, ...],
        done: bool,
    ) -> None:
        old_q = float(self.q_table[state_index + (action,)])
        next_max = 0.0 if done else float(np.max(self.q_table[next_state_index]))
        target = float(reward) + self.gamma * next_max
        self.q_table[state_index + (action,)] = old_q + self.alpha * (target - old_q)


def select_training_action(
    agent: TabularQLearningAgent,
    state_index: Tuple[int, ...],
    policy: str,
    epsilon: float,
    episode: int,
    train_episodes: int,
) -> Tuple[int, str]:
    if policy == "epsilon_greedy":
        if random.random() < epsilon:
            return random.randrange(agent.action_count), "epsilon_random"
        return agent.greedy_action(state_index), "greedy"

    if policy == "median_50":
        if random.random() < epsilon:
            return agent.median_lower_half_action(state_index), "median_lower_half"
        return agent.greedy_action(state_index), "greedy"

    raise ValueError(f"Unknown policy: {policy}")


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

POLICIES = ["epsilon_greedy", "median_50"]
DISPLAY_NAMES = {
    "epsilon_greedy": "Epsilon Greedy",
    "median_50": "Median 50",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_experiment(policy: str, args, output_dir: Path) -> List[Dict]:
    set_seed(args.seed)

    env = SumoDrivingEnv(
        scenario_dir=str(output_dir / "sumo_scenario"),
        max_episode_steps=args.max_episode_steps,
        target_speed=args.target_speed,
        seed=args.seed,
        gui=args.gui,
    )
    discretizer = StateDiscretizer(tuple(args.state_bins))
    agent = TabularQLearningAgent(
        state_shape=discretizer.shape,
        action_count=int(env.action_space.n),
        alpha=args.alpha,
        gamma=args.gamma,
    )

    rows: List[Dict] = []
    print(f"\n===== TRAINING START: {DISPLAY_NAMES[policy]} =====", flush=True)
    train_start = time.time()

    for episode in range(args.train_episodes):
        state, _ = env.reset(seed=args.seed + episode)
        state_idx = discretizer.encode(state)
        total_reward = 0.0
        term_reason = "max_steps"
        source_counts: Dict[str, int] = {}
        episode_start = time.time()

        for step in range(args.max_episode_steps):
            action, source = select_training_action(
                agent,
                state_idx,
                policy,
                args.epsilon,
                episode,
                args.train_episodes,
            )
            source_counts[source] = source_counts.get(source, 0) + 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            next_state_idx = discretizer.encode(next_state)

            agent.update(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            total_reward += float(reward)

            if done:
                term_reason = str(info.get("term_reason", "done"))
                break

        row = {
            "phase": "train",
            "policy": policy,
            "method": DISPLAY_NAMES[policy],
            "episode": episode,
            "reward": total_reward,
            "steps": step + 1,
            "term_reason": term_reason,
            "wall_seconds": time.time() - episode_start,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "action_source_counts": json.dumps(source_counts, sort_keys=True),
            "q_table_nonzero": int(np.count_nonzero(agent.q_table)),
        }
        rows.append(row)
        print(
            f"TRAIN {DISPLAY_NAMES[policy]:16s} ep={episode:04d} "
            f"reward={total_reward:9.2f} steps={step + 1:3d} "
            f"term={term_reason:10s} wall={row['wall_seconds']:.2f}s",
            flush=True,
        )

    print(
        f"===== TRAINING END: {DISPLAY_NAMES[policy]} "
        f"({time.time() - train_start:.2f}s) =====",
        flush=True,
    )

    model_dir = output_dir / "tables"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / f"{policy}_q_table.npz",
        q_table=agent.q_table,
        state_bins=np.asarray(args.state_bins, dtype=np.int32),
        alpha=np.asarray(args.alpha),
        gamma=np.asarray(args.gamma),
        epsilon=np.asarray(args.epsilon),
    )

    print(f"\n===== TESTING START: {DISPLAY_NAMES[policy]} =====", flush=True)
    test_start = time.time()

    # Frozen test: greedy only and no updates.
    for episode in range(args.test_episodes):
        state, _ = env.reset(seed=args.seed + 100000 + episode)
        state_idx = discretizer.encode(state)
        total_reward = 0.0
        term_reason = "max_steps"
        episode_start = time.time()

        for step in range(args.max_episode_steps):
            action = agent.greedy_action(state_idx)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            state_idx = discretizer.encode(next_state)

            if terminated or truncated:
                term_reason = str(info.get("term_reason", "done"))
                break

        row = {
            "phase": "test",
            "policy": policy,
            "method": DISPLAY_NAMES[policy],
            "episode": episode,
            "reward": total_reward,
            "steps": step + 1,
            "term_reason": term_reason,
            "wall_seconds": time.time() - episode_start,
            "network_frozen": True,
            "updates_during_test": 0,
            "q_table_nonzero": int(np.count_nonzero(agent.q_table)),
        }
        rows.append(row)
        print(
            f"TEST  {DISPLAY_NAMES[policy]:16s} ep={episode:04d} "
            f"reward={total_reward:9.2f} steps={step + 1:3d} "
            f"term={term_reason:10s} wall={row['wall_seconds']:.2f}s",
            flush=True,
        )

    print(
        f"===== TESTING END: {DISPLAY_NAMES[policy]} "
        f"({time.time() - test_start:.2f}s) =====",
        flush=True,
    )
    env.close()
    return rows


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    window = max(1, min(window, len(arr)))
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def save_outputs(rows: List[Dict], args, output_dir: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "all_episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        train_rewards = [float(r["reward"]) for r in rows if r["phase"] == "train" and r["policy"] == policy]
        test_rewards = [float(r["reward"]) for r in rows if r["phase"] == "test" and r["policy"] == policy]
        summary[policy] = {
            "train_average_reward": float(np.mean(train_rewards)),
            "train_median_reward": float(np.median(train_rewards)),
            "train_std_reward": float(np.std(train_rewards)),
            "test_average_reward": float(np.mean(test_rewards)),
            "test_median_reward": float(np.median(test_rewards)),
            "test_std_reward": float(np.std(test_rewards)),
            "test_min_reward": float(np.min(test_rewards)),
            "test_max_reward": float(np.max(test_rewards)),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Training and test reward plots.
    for phase in ("train", "test"):
        plt.figure(figsize=(7.2, 4.2))
        for policy in POLICIES:
            subset = [r for r in rows if r["phase"] == phase and r["policy"] == policy]
            x = [int(r["episode"]) for r in subset]
            y = [float(r["reward"]) for r in subset]
            if phase == "train" and len(y) >= 20:
                ma = moving_average(y, 20)
                plt.plot(x[len(x) - len(ma):], ma, label=DISPLAY_NAMES[policy], linewidth=1.4)
            else:
                plt.plot(x, y, label=DISPLAY_NAMES[policy], linewidth=1.1)
        plt.xlabel("Episode")
        plt.ylabel("Environment reward")
        plt.title(f"SUMO Tabular Q-Learning {phase.capitalize()} Reward")
        plt.legend(frameon=False)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{phase}_reward_vs_episode.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Test boxplot.
    test_groups = [
        [float(r["reward"]) for r in rows if r["phase"] == "test" and r["policy"] == policy]
        for policy in POLICIES
    ]
    plt.figure(figsize=(7.0, 4.2))
    plt.boxplot(test_groups, tick_labels=[DISPLAY_NAMES[p] for p in POLICIES], showmeans=True)
    plt.xlabel("Experiment")
    plt.ylabel("Environment reward")
    plt.title("SUMO Tabular Q-Learning Test Reward Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "test_reward_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Average test reward.
    means = [float(np.mean(group)) for group in test_groups]
    plt.figure(figsize=(7.0, 4.2))
    bars = plt.bar([DISPLAY_NAMES[p] for p in POLICIES], means, edgecolor="black", linewidth=0.7)
    plt.xlabel("Experiment")
    plt.ylabel("Average test reward")
    plt.title("SUMO Tabular Q-Learning Average Test Reward")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "average_test_reward.png", dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.1, help="Tabular Q-learning learning rate")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--state-bins",
        type=int,
        nargs=5,
        default=[8, 12, 12, 10, 8],
        metavar=("SPEED", "POSITION", "REMAINING", "GAP", "LEADER_SPEED"),
        help="Discretization bins for the five observation dimensions",
    )

    parser.add_argument("--target-speed", type=float, default=13.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_sumo_tabular_two_experiments")
    parser.add_argument("--gui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SUMO TABULAR Q-LEARNING EXPERIMENT")
    print("=" * 72)
    print("Python:", platform.python_version())
    print("Algorithm: pure tabular Q-learning")
    print("Experiments:", ", ".join(POLICIES))
    print("Separate Q-table per experiment: yes")
    print("Frozen greedy testing: yes")
    print("Train/Test/Max steps:", args.train_episodes, args.test_episodes, args.max_episode_steps)
    print("Epsilon:", args.epsilon)
    print("State bins:", tuple(args.state_bins))
    print("Q-table states:", int(np.prod(args.state_bins)))
    print("Q-table entries per experiment:", int(np.prod(args.state_bins)) * 3)
    print("=" * 72)

    all_rows: List[Dict] = []
    for policy in POLICIES:
        all_rows.extend(run_experiment(policy, args, output_dir))

    save_outputs(all_rows, args, output_dir)
    print("\nExperiment completed successfully.")
    print("Results saved to:", output_dir)


if __name__ == "__main__":
    main()
