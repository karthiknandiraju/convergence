"""
Train PPO in CARLA using Stable-Baselines3.

PPO is used because CARLA driving actions are continuous:
steer, throttle, and brake.
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.carla_env import CarlaDrivingEnv


def train(reward_mode: str, total_timesteps: int, target_speed_kmh: float) -> None:
    """Train a PPO model for one reward mode."""
    env = CarlaDrivingEnv(
        reward_mode=reward_mode,
        target_speed_kmh=target_speed_kmh,
        max_episode_steps=500,
    )
    env = Monitor(env, filename=f"results/monitor_{reward_mode}.csv")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="results/tensorboard",
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("results/models", exist_ok=True)
    model.save(f"results/models/ppo_{reward_mode}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-mode", default="ontology_combined")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--target-speed", type=float, default=30.0)
    args = parser.parse_args()

    train(args.reward_mode, args.timesteps, args.target_speed)
