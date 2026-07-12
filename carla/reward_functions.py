"""
Reward functions for thesis experiments.

The learning algorithm stays fixed. We only change the reward function.
This lets us study the effect of reward shaping clearly.
"""

from math import exp
from src.ontology import DrivingState


def baseline_reward(state: DrivingState) -> float:
    """
    Baseline reward.

    This reward gives basic progress reward and large safety penalties.
    It is intentionally simple, so it can be compared with shaped rewards.
    """
    reward = 0.0
    reward += state.road.progress_m

    if state.sensors.collision:
        reward -= 100.0
    if state.sensors.lane_invasion:
        reward -= 10.0

    return reward


def lane_centering_reward(state: DrivingState) -> float:
    """
    Reward for staying close to the lane center.

    lane_offset_m is zero when the car is in the center.
    Larger absolute offset means worse lane keeping.
    """
    offset = abs(state.road.lane_offset_m)
    return exp(-offset)


def heading_alignment_reward(state: DrivingState) -> float:
    """
    Reward for facing the same direction as the road.

    heading_error_deg is zero when the vehicle direction matches the lane direction.
    """
    error = abs(state.road.heading_error_deg)
    return exp(-error / 30.0)


def goal_progression_reward(state: DrivingState) -> float:
    """
    Reward for moving forward along the route.

    Positive progress means the vehicle moved toward the goal.
    Negative progress means it moved backward or away from the route.
    """
    return state.road.progress_m


def speed_regulation_reward(state: DrivingState, target_speed_kmh: float = 30.0) -> float:
    """
    Reward for driving near a target speed.

    Driving too slowly is inefficient.
    Driving too fast is unsafe.
    """
    speed_error = abs(state.vehicle.speed_kmh - target_speed_kmh)
    return exp(-speed_error / target_speed_kmh)


def safety_reward(state: DrivingState) -> float:
    """
    Safety reward based on ontology concepts.

    Collisions, lane invasions, pedestrians, obstacles, and red lights are safety concepts.
    """
    reward = 0.0

    if state.sensors.collision:
        reward -= 100.0
    if state.sensors.lane_invasion:
        reward -= 10.0

    if state.pedestrian.distance_m < 5.0:
        reward -= 20.0
    elif state.pedestrian.distance_m < 10.0:
        reward -= 5.0

    if state.obstacle.distance_m < 5.0:
        reward -= 20.0
    elif state.obstacle.distance_m < 10.0:
        reward -= 5.0

    if state.traffic_light.is_red and state.traffic_light.distance_m < 8.0:
        reward -= 15.0

    return reward


def ontology_combined_reward(state: DrivingState, target_speed_kmh: float = 30.0) -> float:
    """
    Combined ontology-based reward shaping.

    This is the main reward for the thesis.
    It maps ontology concepts to reward terms.

    Vehicle speed maps to speed regulation.
    Road lane offset maps to lane centering.
    Road heading error maps to heading alignment.
    Road progress maps to goal progression.
    Pedestrian, obstacle, traffic light, collision, and lane invasion map to safety.
    """
    reward = 0.0

    reward += 1.0 * goal_progression_reward(state)
    reward += 1.0 * lane_centering_reward(state)
    reward += 1.0 * heading_alignment_reward(state)
    reward += 0.5 * speed_regulation_reward(state, target_speed_kmh)
    reward += safety_reward(state)

    return reward


def compute_reward(state: DrivingState, reward_mode: str, target_speed_kmh: float = 30.0) -> float:
    """Select a reward function by name."""
    if reward_mode == "baseline":
        return baseline_reward(state)
    if reward_mode == "lane_centering":
        return baseline_reward(state) + lane_centering_reward(state)
    if reward_mode == "heading_alignment":
        return baseline_reward(state) + heading_alignment_reward(state)
    if reward_mode == "goal_progression":
        return baseline_reward(state) + goal_progression_reward(state)
    if reward_mode == "speed_regulation":
        return baseline_reward(state) + speed_regulation_reward(state, target_speed_kmh)
    if reward_mode == "ontology_combined":
        return ontology_combined_reward(state, target_speed_kmh)

    raise ValueError(f"Unknown reward mode: {reward_mode}")
