"""Run a simple reward-function comparison without long PPO training."""

from src.ontology import make_default_state, SensorState
from src.reward_functions import compute_reward


REWARD_MODES = [
    "baseline",
    "lane_centering",
    "heading_alignment",
    "goal_progression",
    "speed_regulation",
    "ontology_combined",
]


def main() -> None:
    state = make_default_state()
    print("Reward comparison for a safe centered driving state")
    for mode in REWARD_MODES:
        reward = compute_reward(state, mode)
        print(f"{mode}: {reward:.3f}")

    collision_state = make_default_state()
    collision_state.sensors = SensorState(collision=True, lane_invasion=False)
    print("Reward comparison for a collision state")
    for mode in REWARD_MODES:
        reward = compute_reward(collision_state, mode)
        print(f"{mode}: {reward:.3f}")


if __name__ == "__main__":
    main()
