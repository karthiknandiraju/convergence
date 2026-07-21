"""Compatibility adapter from the existing framework contract to HighwayEnv.

The training and policy files in this package intentionally keep their original
command-line arguments and nine-action assumptions.  This adapter translates
those settings to ``highway-v0`` while exposing the small set of vehicle/lane
attributes used by the advanced policies' safety-vector extractor.

It does *not* change a policy's DQN, replay, pool, mask, or selection logic.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, Optional

import gymnasium as gym
import highway_env  # noqa: F401  # Import registers the HighwayEnv IDs.
import numpy as np


ENVIRONMENT_ID = os.environ.get("HIGHWAY_ENV_ID", "highway-v0")
ACTION_COUNT = 9


def _as_bool(value: Any) -> bool:
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


class _HighwayLaneProxy:
    """Give a HighwayEnv lane the lane-method names expected by the policies."""

    def __init__(self, lane: Any):
        self._lane = lane

    def local_coordinates(self, position):
        return self._lane.local_coordinates(position)

    def width_at(self, longitudinal: float) -> float:
        method = getattr(self._lane, "width_at", None)
        if callable(method):
            return float(method(longitudinal))
        return float(getattr(self._lane, "width", 4.0))

    def heading_theta_at(self, longitudinal: float) -> float:
        method = getattr(self._lane, "heading_theta_at", None)
        if callable(method):
            return float(method(longitudinal))
        return float(self._lane.heading_at(longitudinal))

    @property
    def width(self) -> float:
        value = getattr(self._lane, "width", 4.0)
        return float(value() if callable(value) else value)

    def __getattr__(self, name: str):
        return getattr(self._lane, name)


class _HighwayVehicleProxy:
    """Present one HighwayEnv road vehicle through the prior safety interface."""

    def __init__(self, vehicle: Any):
        self._vehicle = vehicle

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self._vehicle.position, dtype=np.float32)

    @property
    def speed(self) -> float:
        return float(getattr(self._vehicle, "speed", 0.0))

    @property
    def speed_km_h(self) -> float:
        return self.speed * 3.6

    @property
    def heading_theta(self) -> float:
        return float(getattr(self._vehicle, "heading", 0.0))

    @property
    def velocity(self) -> np.ndarray:
        velocity = getattr(self._vehicle, "velocity", None)
        if velocity is not None:
            return np.asarray(velocity, dtype=np.float32)
        return np.asarray(
            [
                self.speed * math.cos(self.heading_theta),
                self.speed * math.sin(self.heading_theta),
            ],
            dtype=np.float32,
        )

    @property
    def lane(self) -> Optional[_HighwayLaneProxy]:
        lane = getattr(self._vehicle, "lane", None)
        return _HighwayLaneProxy(lane) if lane is not None else None

    @property
    def width(self) -> float:
        return float(getattr(self._vehicle, "WIDTH", 2.0))

    @property
    def length(self) -> float:
        return float(getattr(self._vehicle, "LENGTH", 5.0))

    @property
    def top_down_width(self) -> float:
        return self.width

    @property
    def top_down_length(self) -> float:
        return self.length

    def __getattr__(self, name: str):
        return getattr(self._vehicle, name)


class _HighwayEngineProxy:
    """Expose HighwayEnv road vehicles as the policy's hazard-object collection."""

    def __init__(self, owner: "HighwayEnvAdapter"):
        self._owner = owner

    def get_objects(self, predicate=None) -> Dict[str, _HighwayVehicleProxy]:
        objects = self._owner._road_vehicle_proxies()
        if callable(predicate):
            objects = [item for item in objects if predicate(item)]
        return {f"vehicle_{index}": item for index, item in enumerate(objects)}

    @property
    def objects(self) -> Dict[str, _HighwayVehicleProxy]:
        return self.get_objects()


class HighwayEnvAdapter:
    """Legacy-policy facade backed exclusively by the official ``highway-v0`` env.

    The facade exists so all active policies retain the same CLI and algorithm.
    The observation is HighwayEnv's fixed-size Kinematics observation, flattened
    by each existing policy exactly as before.  The nine discrete actions are
    HighwayEnv's built-in 3 longitudinal x 3 lateral low-level action grid.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        legacy = dict(config or {})
        steering_dim = int(legacy.get("discrete_steering_dim", 3))
        throttle_dim = int(legacy.get("discrete_throttle_dim", 3))
        if steering_dim != 3 or throttle_dim != 3:
            raise ValueError(
                "This HighwayEnv framework preserves the required nine-action "
                "contract, so --discrete-steering-dim and "
                "--discrete-throttle-dim must both be 3."
            )

        lanes_count = max(2, int(legacy.get("map", 3)))
        density = max(0.0, float(legacy.get("traffic_density", 0.2)))
        vehicles_count = max(0, int(round(250.0 * density)))
        horizon = max(1, int(legacy.get("horizon", 1000)))
        # Keep the network input shape stable when traffic density changes.
        observation_vehicles = 15

        highway_config: Dict[str, Any] = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": observation_vehicles,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "absolute": False,
                "normalize": True,
                "order": "sorted",
                "see_behind": True,
                "observe_intentions": False,
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 3,
                "dynamical": False,
                "clip": True,
            },
            "lanes_count": lanes_count,
            "vehicles_count": vehicles_count,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": horizon,
            "simulation_frequency": 15,
            "policy_frequency": 1,
            "collision_reward": 0.0,
            "right_lane_reward": 0.1,
            "high_speed_reward": 1.0,
            "lane_change_reward": 0.0,
            "reward_speed_range": [20, 30],
            "normalize_reward": False,
            "offroad_terminal": True,
            "show_trajectories": False,
            "render_agent": True,
        }
        render_mode = "human" if bool(legacy.get("use_render", False)) else None
        self._env = gym.make(
            ENVIRONMENT_ID,
            config=highway_config,
            render_mode=render_mode,
        )
        self._legacy_config = legacy
        self.highway_config = highway_config
        self.config = dict(legacy)
        self.config.update(
            {
                "environment_id": ENVIRONMENT_ID,
                "highway_env_config": highway_config,
                "discrete_action": True,
                "use_multi_discrete": False,
                "discrete_steering_dim": 3,
                "discrete_throttle_dim": 3,
            }
        )
        self._collision_penalty = float(legacy.get("crash_vehicle_penalty", 50.0))
        self._out_of_road_penalty = float(legacy.get("out_of_road_penalty", 50.0))
        self._vehicle_cache: Dict[int, _HighwayVehicleProxy] = {}
        self._episode_steps = 0
        self.engine = _HighwayEngineProxy(self)

        action_count = getattr(self.action_space, "n", None)
        if action_count is None or int(action_count) != ACTION_COUNT:
            self._env.close()
            raise RuntimeError(
                f"HighwayEnv must expose exactly {ACTION_COUNT} actions; "
                f"received {action_count!r}."
            )

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def vehicle(self) -> Optional[_HighwayVehicleProxy]:
        raw = getattr(self._env.unwrapped, "vehicle", None)
        return self._proxy_for(raw) if raw is not None else None

    def _proxy_for(self, raw: Any) -> _HighwayVehicleProxy:
        key = id(raw)
        proxy = self._vehicle_cache.get(key)
        if proxy is None or proxy._vehicle is not raw:
            proxy = _HighwayVehicleProxy(raw)
            self._vehicle_cache[key] = proxy
        return proxy

    def _road_vehicle_proxies(self) -> Iterable[_HighwayVehicleProxy]:
        road = getattr(self._env.unwrapped, "road", None)
        vehicles = list(getattr(road, "vehicles", ()) or ())
        return [self._proxy_for(raw) for raw in vehicles]

    def _out_of_road(self) -> bool:
        raw = getattr(self._env.unwrapped, "vehicle", None)
        if raw is None:
            return False
        return not _as_bool(getattr(raw, "on_road", True))

    def _enrich_info(
        self,
        info: Optional[Dict[str, Any]],
        *,
        terminated: bool = False,
        truncated: bool = False,
    ) -> Dict[str, Any]:
        result = dict(info or {})
        raw = getattr(self._env.unwrapped, "vehicle", None)
        crashed = _as_bool(getattr(raw, "crashed", False)) if raw is not None else False
        out_of_road = self._out_of_road()
        result.update(
            {
                "crash_vehicle": crashed,
                "crash_object": False,
                "crash": crashed,
                "crashed": crashed,
                "out_of_road": out_of_road,
                "arrive_dest": False,
                "arrived": False,
                "success": False,
                "max_step": bool(truncated),
                "environment_id": ENVIRONMENT_ID,
                "episode_steps": self._episode_steps,
            }
        )
        if crashed:
            result["termination_reason"] = "collision"
        elif out_of_road:
            result["termination_reason"] = "out_of_road"
        elif truncated:
            result["termination_reason"] = "max_steps"
        elif terminated:
            result["termination_reason"] = "terminated"
        else:
            result["termination_reason"] = "running"
        return result

    def reset(self, *, seed: Optional[int] = None, options=None):
        self._vehicle_cache.clear()
        self._episode_steps = 0
        observation, info = self._env.reset(seed=seed, options=options)
        return observation, self._enrich_info(info)

    def step(self, action):
        action_id = int(action)
        if not 0 <= action_id < ACTION_COUNT:
            raise ValueError(f"Action must be in [0, {ACTION_COUNT - 1}], got {action!r}.")
        observation, reward, terminated, truncated, info = self._env.step(action_id)
        self._episode_steps += 1
        enriched = self._enrich_info(
            info, terminated=bool(terminated), truncated=bool(truncated)
        )
        crashed = bool(enriched["crash_vehicle"])
        out_of_road = bool(enriched["out_of_road"])
        if crashed:
            reward = float(reward) - self._collision_penalty
        elif out_of_road:
            reward = float(reward) - self._out_of_road_penalty
        terminated = bool(terminated or crashed or out_of_road)
        return observation, float(reward), terminated, bool(truncated), enriched

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)


__all__ = ["ACTION_COUNT", "ENVIRONMENT_ID", "HighwayEnvAdapter"]
