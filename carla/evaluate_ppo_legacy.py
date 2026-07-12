"""Evaluate a trained PPO agent and save simple metrics."""

import argparse
import csv
import os

from stable_baselines3 import PPO

from src.carla_env import CarlaDrivingEnv


def evaluate(model_path: str, reward_mode: str, episodes: int = 5) -> None:
    env = CarlaDrivingEnv(reward_mode=reward_mode)
    model = PPO.load(model_path)

    rows = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        collisions = 0
        lane_invasions = 0
        speed_sum = 0.0
        lane_offset_sum = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            collisions += int(info["collision"])
            lane_invasions += int(info["lane_invasion"])
            speed_sum += info["speed_kmh"]
            lane_offset_sum += abs(info["lane_offset_m"])

        rows.append({
            "episode": episode,
            "reward_mode": reward_mode,
            "total_reward": total_reward,
            "steps": steps,
            "average_speed_kmh": speed_sum / max(steps, 1),
            "average_lane_offset_m": lane_offset_sum / max(steps, 1),
            "collisions": collisions,
            "lane_invasions": lane_invasions,
        })

    os.makedirs("results", exist_ok=True)
    output_file = f"results/evaluation_{reward_mode}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"Saved evaluation results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward-mode", default="ontology_combined")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.model, args.reward_mode, args.episodes)
