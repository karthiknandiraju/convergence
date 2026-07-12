"""Train a DQN agent in CARLA using ontology based reward shaping."""

import argparse
import csv
import os

from src.carla_env import CarlaDrivingEnv
from src.dqn_agent import DQNAgent


def train(
    reward_mode: str,
    episodes: int,
    target_speed_kmh: float,
    max_episode_steps: int,
    use_mock_when_carla_missing: bool,
) -> None:
    env = CarlaDrivingEnv(
        reward_mode=reward_mode,
        target_speed_kmh=target_speed_kmh,
        max_episode_steps=max_episode_steps,
        use_mock_when_carla_missing=use_mock_when_carla_missing,
    )

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(observation_size=observation_size, action_size=action_size)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    rows = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        losses = []
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        average_loss = sum(losses) / max(len(losses), 1)
        rows.append(
            {
                "episode": episode,
                "reward_mode": reward_mode,
                "total_reward": total_reward,
                "steps": steps,
                "epsilon": epsilon,
                "average_loss": average_loss,
            }
        )
        print(f"episode {episode} reward {total_reward:.2f} steps {steps} epsilon {epsilon:.3f}")

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    agent.save(f"results/models/dqn_{reward_mode}.pt")

    output_file = f"results/training_dqn_{reward_mode}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"Saved model to results/models/dqn_{reward_mode}.pt")
    print(f"Saved training log to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", default="ontology_combined")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--target-speed", type=float, default=30.0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    train(
        reward_mode=args.reward_mode,
        episodes=args.episodes,
        target_speed_kmh=args.target_speed,
        max_episode_steps=args.max_episode_steps,
        use_mock_when_carla_missing=args.mock,
    )
