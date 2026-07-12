"""
Rule-based baseline agent.

This agent does not learn.
It follows fixed rules and is used to check the full experiment pipeline.
"""

import numpy as np
from src.ontology import DrivingState


class RuleBasedAgent:
    """
    Simple hand-written driving policy.

    It is useful for debugging before PPO training because it can run without learning.
    """

    def __init__(self, target_speed_kmh: float = 30.0):
        self.target_speed_kmh = target_speed_kmh

    def act(self, state: DrivingState) -> np.ndarray:
        """
        Convert ontology state into CARLA control values.

        Action format:
        action[0] is steer, from -1.0 left to 1.0 right
        action[1] is throttle, from 0.0 to 1.0
        action[2] is brake, from 0.0 to 1.0
        """
        steer = 0.0
        throttle = 0.4
        brake = 0.0

        # Rule 1: steer back toward the lane center.
        steer -= 0.5 * state.road.lane_offset_m

        # Rule 2: correct vehicle direction using heading error.
        steer -= 0.02 * state.road.heading_error_deg

        # Rule 3: keep speed near the target speed.
        if state.vehicle.speed_kmh < self.target_speed_kmh - 5.0:
            throttle = 0.6
            brake = 0.0
        elif state.vehicle.speed_kmh > self.target_speed_kmh + 5.0:
            throttle = 0.0
            brake = 0.3

        # Rule 4: brake for close pedestrian or obstacle.
        if state.pedestrian.distance_m < 8.0 or state.obstacle.distance_m < 8.0:
            throttle = 0.0
            brake = 0.8

        # Rule 5: stop at a nearby red traffic light.
        if state.traffic_light.is_red and state.traffic_light.distance_m < 10.0:
            throttle = 0.0
            brake = 1.0

        steer = float(np.clip(steer, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        return np.array([steer, throttle, brake], dtype=np.float32)
