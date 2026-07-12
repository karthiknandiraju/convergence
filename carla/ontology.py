from typing import List, Tuple, Dict, Set
from typing import List
"""
Ontology module for the CARLA reward-shaping thesis project.

The goal of this file is to make the ontology explicit in code.
Your handwritten ontology contains concepts such as vehicle, driver/agent,
pedestrian, obstacle, road, speed, position, traffic light, sensors, and safety.

In reinforcement learning, these concepts become part of the state.
The reward function then uses this structured state to compute reward.
"""

from dataclasses import dataclass


@dataclass
class VehicleState:
    """Information about the ego vehicle controlled by the agent."""

    speed_kmh: float
    acceleration: float = 0.0


@dataclass
class RoadState:
    """Information about the vehicle position relative to the road."""

    lane_offset_m: float
    heading_error_deg: float
    progress_m: float


@dataclass
class TrafficLightState:
    """Information about the nearest traffic light."""

    is_red: bool = False
    distance_m: float = 100.0


@dataclass
class PedestrianState:
    """Information about the nearest pedestrian."""

    distance_m: float = 100.0


@dataclass
class ObstacleState:
    """Information about the nearest obstacle or vehicle."""

    distance_m: float = 100.0


@dataclass
class WeatherState:
    """Simple weather information. This can be expanded later."""

    rain: float = 0.0
    fog: float = 0.0


@dataclass
class SensorState:
    """Sensor events from CARLA sensors."""

    collision: bool = False
    lane_invasion: bool = False


@dataclass
class DrivingState:
    """Full ontology-based state used by reward functions."""

    vehicle: VehicleState
    road: RoadState
    traffic_light: TrafficLightState
    pedestrian: PedestrianState
    obstacle: ObstacleState
    weather: WeatherState
    sensors: SensorState

    def to_vector(self) -> List[float]:
        """
        Convert the ontology state into numeric values.

        PPO cannot directly read Python objects such as VehicleState or RoadState.
        It needs a numeric vector, so we convert ontology concepts into numbers.
        """
        return [
            self.vehicle.speed_kmh,
            self.road.lane_offset_m,
            self.road.heading_error_deg,
            self.road.progress_m,
            1.0 if self.traffic_light.is_red else 0.0,
            self.traffic_light.distance_m,
            self.pedestrian.distance_m,
            self.obstacle.distance_m,
            self.weather.rain,
            self.weather.fog,
            1.0 if self.sensors.collision else 0.0,
            1.0 if self.sensors.lane_invasion else 0.0,
        ]


def make_default_state() -> DrivingState:
    """Create a safe default state for unit tests."""
    return DrivingState(
        vehicle=VehicleState(speed_kmh=30.0),
        road=RoadState(lane_offset_m=0.0, heading_error_deg=0.0, progress_m=1.0),
        traffic_light=TrafficLightState(is_red=False, distance_m=100.0),
        pedestrian=PedestrianState(distance_m=100.0),
        obstacle=ObstacleState(distance_m=100.0),
        weather=WeatherState(rain=0.0, fog=0.0),
        sensors=SensorState(collision=False, lane_invasion=False),
    )
