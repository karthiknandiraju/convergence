"""
Gymnasium wrapper for CARLA.

This file shows the structure needed by Stable-Baselines3.
It is written to be understandable for a beginner.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.ontology import (
    DrivingState,
    VehicleState,
    RoadState,
    TrafficLightState,
    PedestrianState,
    ObstacleState,
    WeatherState,
    SensorState,
)
from src.reward_functions import compute_reward


class CarlaDrivingEnv(gym.Env):
    """
    CARLA environment for DQN.

    Observation is a numeric vector produced from the ontology state.
    DQN needs a discrete action space, so this wrapper maps action ids to CARLA controls.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_seconds: float = 10.0,
        reward_mode: str = "ontology_combined",
        target_speed_kmh: float = 30.0,
        max_episode_steps: int = 500,
        use_mock_when_carla_missing: bool = False,
        stuck_speed_threshold_kmh: float = 1.0,
        stuck_patience_steps: int = 50,
        num_traffic_vehicles: int = 0,
        num_pedestrians: int = 0,
        realistic_traffic: bool = False,
        traffic_manager_port: int = 8000,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.reward_mode = reward_mode
        self.target_speed_kmh = target_speed_kmh
        self.max_episode_steps = max_episode_steps
        self.use_mock_when_carla_missing = use_mock_when_carla_missing
        self.stuck_speed_threshold_kmh = stuck_speed_threshold_kmh
        self.stuck_patience_steps = stuck_patience_steps
        self.num_traffic_vehicles = int(num_traffic_vehicles)
        self.num_pedestrians = int(num_pedestrians)
        self.realistic_traffic = bool(realistic_traffic)
        self.traffic_manager_port = int(traffic_manager_port)

        self.client = None
        self.world = None
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.background_vehicles = []
        self.background_walkers = []
        self.walker_controllers = []
        self.traffic_manager = None
        self.last_collision_actor_type = "none"
        self.last_collision_actor_id = -1
        self.last_collision_actor_role_name = ""
        self.last_collision_intensity = 0.0
        self.collision_happened = False
        self.lane_invasion_happened = False
        self.previous_location = None
        self.step_count = 0
        self.stuck_step_count = 0

        # Observation vector is created by DrivingState.to_vector.
        self.observation_space = spaces.Box(
            low=np.array([0, -5, -180, -20, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 5, 180, 20, 1, 200, 200, 200, 100, 100, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # DQN works with discrete actions.
        # Each integer is converted to one CARLA control command in _action_to_control.
        self.discrete_actions = [
            (-0.40, 0.35, 0.00),
            (-0.20, 0.45, 0.00),
            (0.00, 0.55, 0.00),
            (0.20, 0.45, 0.00),
            (0.40, 0.35, 0.00),
            (0.00, 0.00, 0.60),
            (0.00, 0.20, 0.00),
        ]
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        self._connect_to_carla()

    def _connect_to_carla(self) -> None:
        """Connect to CARLA if the carla package is available."""
        try:
            import carla

            self.carla = carla
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout_seconds)
            self.world = self.client.get_world()
            try:
                self.traffic_manager = self.client.get_trafficmanager(self.traffic_manager_port)
                self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
                self.traffic_manager.set_synchronous_mode(False)
            except Exception:
                self.traffic_manager = None
        except Exception as exc:
            if self.use_mock_when_carla_missing:
                self.carla = None
                self.client = None
                self.world = None
            else:
                raise RuntimeError(
                    "Could not connect to CARLA. Start CarlaUnreal.exe first and check Python API compatibility."
                ) from exc

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start a new episode."""
        super().reset(seed=seed)
        self.step_count = 0
        self.collision_happened = False
        self.lane_invasion_happened = False
        self.last_collision_actor_type = "none"
        self.last_collision_actor_id = -1
        self.last_collision_actor_role_name = ""
        self.last_collision_intensity = 0.0
        self.previous_location = None
        self.stuck_step_count = 0

        if self.world is not None:
            self._spawn_vehicle()
            if self.realistic_traffic:
                self._spawn_background_traffic()
            time.sleep(0.2)

        state = self._get_state()
        return np.array(state.to_vector(), dtype=np.float32), {}

    def step(self, action: int):
        """Apply one discrete DQN action and return next observation, reward, done flags, and info."""
        self.step_count += 1
        action_id = int(action)
        steer, throttle, brake = self._action_to_control(action_id)

        if self.vehicle is not None:
            control = self.carla.VehicleControl(
                steer=float(steer),
                throttle=float(throttle),
                brake=float(brake),
            )
            self.vehicle.apply_control(control)
            try:
                self.world.tick()
            except Exception:
                # CARLA 0.9.14 can be asynchronous depending on server settings.
                # If tick is not allowed, wait for the next server frame instead.
                self.world.wait_for_tick()

        state = self._get_state()
        reward = compute_reward(state, self.reward_mode, self.target_speed_kmh)

        # Stronger collision penalty for DQN training stability.
        # reward_functions.py already applies a -100 collision penalty in ontology_combined.
        # This additional -100 makes the effective collision penalty approximately -200.
        if state.sensors.collision:
            reward -= 100.0

        # Stuck detection: if the vehicle is almost not moving for many
        # consecutive steps, end the episode early. This avoids wasting
        # hundreds of steps when the car is blocked or has failed to start.
        if state.vehicle.speed_kmh < self.stuck_speed_threshold_kmh:
            self.stuck_step_count += 1
        else:
            self.stuck_step_count = 0

        stuck = self.stuck_step_count >= self.stuck_patience_steps

        terminated = bool(state.sensors.collision)
        truncated = self.step_count >= self.max_episode_steps or stuck

        if state.sensors.collision:
            termination_reason = "collision"
        elif stuck:
            termination_reason = "stuck"
        elif self.step_count >= self.max_episode_steps:
            termination_reason = "max_steps"
        else:
            termination_reason = "running"

        info = {
            "reward_mode": self.reward_mode,
            "speed_kmh": state.vehicle.speed_kmh,
            "lane_offset_m": state.road.lane_offset_m,
            "heading_error_deg": state.road.heading_error_deg,
            "collision": state.sensors.collision,
            "collision_actor_type": self.last_collision_actor_type,
            "collision_actor_id": self.last_collision_actor_id,
            "collision_actor_role_name": self.last_collision_actor_role_name,
            "collision_intensity": self.last_collision_intensity,
            "lane_invasion": state.sensors.lane_invasion,
            "stuck": stuck,
            "stuck_step_count": self.stuck_step_count,
            "termination_reason": termination_reason,
            "ended_before_max_steps": bool(termination_reason in ("collision", "stuck")),
            "action_id": action_id,
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
        }

        return np.array(state.to_vector(), dtype=np.float32), float(reward), terminated, truncated, info

    def _action_to_control(self, action_id: int) -> tuple[float, float, float]:
        """Convert a DQN action number into steer, throttle, brake."""
        if action_id < 0 or action_id >= len(self.discrete_actions):
            raise ValueError(f"Invalid action id {action_id}")
        return self.discrete_actions[action_id]

    def _spawn_vehicle(self) -> None:
        
        self._destroy_actors()

        blueprint_library = self.world.get_blueprint_library()

        candidates = [
            "vehicle.*model3*",
            "vehicle.tesla.model3",
            "vehicle.mercedes.sprinter",
            "vehicle.audi.tt",
            "vehicle.lincoln.mkz_2020",
            "vehicle.*",
        ]

        vehicle_bp = None
        for pattern in candidates:
            matches = blueprint_library.filter(pattern)
            if len(matches) > 0:
                vehicle_bp = matches[0]
                print(f"Using vehicle blueprint: {vehicle_bp.id}")
                break

        if vehicle_bp is None:
            raise RuntimeError("No vehicle blueprint found in CARLA.")

        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            raise RuntimeError("No spawn points found in CARLA map.")

        spawn_point = spawn_points[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.previous_location = self.vehicle.get_location()

        # Attach safety sensors so ontology rewards can detect real CARLA events.
        self._spawn_safety_sensors()

    def _spawn_background_traffic(self) -> None:
        """Spawn background vehicles and pedestrians for realistic CARLA episodes.

        The ego vehicle still uses the fixed spawn point for controlled experiments.
        Background actors are respawned every episode and destroyed by _destroy_actors().
        """
        if self.world is None:
            return

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = list(self.world.get_map().get_spawn_points())

        # Avoid using the ego vehicle spawn point so NPCs do not overlap the ego car.
        npc_spawn_points = spawn_points[1:] if len(spawn_points) > 1 else spawn_points[:]
        random.shuffle(npc_spawn_points)

        # -------------------------
        # Spawn background vehicles
        # -------------------------
        vehicle_bps = list(blueprint_library.filter("vehicle.*"))
        vehicle_bps = [
            bp for bp in vehicle_bps
            if not bp.id.lower().endswith("isetta")
            and "carlamotors" not in bp.id.lower()
            and "microlino" not in bp.id.lower()
        ]

        vehicles_spawned = 0
        for spawn_point in npc_spawn_points:
            if vehicles_spawned >= self.num_traffic_vehicles:
                break
            if not vehicle_bps:
                break

            bp = random.choice(vehicle_bps)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                if colors:
                    bp.set_attribute("color", random.choice(colors))

            try:
                actor = self.world.try_spawn_actor(bp, spawn_point)
            except Exception:
                actor = None

            if actor is None:
                continue

            self.background_vehicles.append(actor)
            vehicles_spawned += 1

            try:
                if self.traffic_manager is not None:
                    actor.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_manager.ignore_lights_percentage(actor, 0.0)
                    self.traffic_manager.ignore_signs_percentage(actor, 0.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(-10.0, 20.0))
                else:
                    actor.set_autopilot(True)
            except Exception:
                pass

        # -------------------------
        # Spawn pedestrians/walkers
        # -------------------------
        walker_bps = list(blueprint_library.filter("walker.pedestrian.*"))
        controller_bp = None
        try:
            controller_bp = blueprint_library.find("controller.ai.walker")
        except Exception:
            controller_bp = None

        walkers_spawned = 0
        for _ in range(max(0, self.num_pedestrians) * 3):
            if walkers_spawned >= self.num_pedestrians:
                break
            if not walker_bps or controller_bp is None:
                break

            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue

            transform = self.carla.Transform(location)
            bp = random.choice(walker_bps)

            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            if bp.has_attribute("speed"):
                speeds = bp.get_attribute("speed").recommended_values
                # CARLA usually stores walking speed at index 1 and running at index 2.
                if len(speeds) > 1:
                    bp.set_attribute("speed", speeds[1])

            try:
                walker = self.world.try_spawn_actor(bp, transform)
            except Exception:
                walker = None

            if walker is None:
                continue

            self.background_walkers.append(walker)
            walkers_spawned += 1

            try:
                controller = self.world.spawn_actor(controller_bp, self.carla.Transform(), attach_to=walker)
                self.walker_controllers.append(controller)
                controller.start()
                dest = self.world.get_random_location_from_navigation()
                if dest is not None:
                    controller.go_to_location(dest)
                controller.set_max_speed(random.uniform(1.0, 1.6))
            except Exception:
                pass

        print(
            f"Spawned realistic traffic: vehicles={len(self.background_vehicles)}/{self.num_traffic_vehicles}, "
            f"pedestrians={len(self.background_walkers)}/{self.num_pedestrians}",
            flush=True,
        )

    def _nearest_dynamic_actor_distances(self, ego_location) -> tuple[float, float]:
        """Return nearest pedestrian and vehicle/obstacle distances to the ego car."""
        nearest_pedestrian = 100.0
        nearest_obstacle = 100.0

        for walker in list(getattr(self, "background_walkers", [])):
            try:
                nearest_pedestrian = min(nearest_pedestrian, float(ego_location.distance(walker.get_location())))
            except Exception:
                pass

        for actor in list(getattr(self, "background_vehicles", [])):
            try:
                nearest_obstacle = min(nearest_obstacle, float(ego_location.distance(actor.get_location())))
            except Exception:
                pass

        return nearest_pedestrian, nearest_obstacle

    def _get_state(self) -> DrivingState:
        """Read CARLA and create an ontology-based DrivingState."""
        if self.vehicle is None:
            return DrivingState(
                vehicle=VehicleState(speed_kmh=0.0),
                road=RoadState(lane_offset_m=0.0, heading_error_deg=0.0, progress_m=0.0),
                traffic_light=TrafficLightState(False, 100.0),
                pedestrian=PedestrianState(100.0),
                obstacle=ObstacleState(100.0),
                weather=WeatherState(0.0, 0.0),
                sensors=SensorState(False, False),
            )

        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        transform = self.vehicle.get_transform()
        location = transform.location
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True)

        lane_offset_m = location.distance(waypoint.transform.location)
        heading_error_deg = self._angle_difference(transform.rotation.yaw, waypoint.transform.rotation.yaw)

        progress_m = 0.0
        if self.previous_location is not None:
            # Signed forward progress. Positive means movement along the lane direction;
            # negative means moving backward relative to the current waypoint direction.
            forward = waypoint.transform.get_forward_vector()
            dx = location.x - self.previous_location.x
            dy = location.y - self.previous_location.y
            dz = location.z - self.previous_location.z
            progress_m = dx * forward.x + dy * forward.y + dz * forward.z
        self.previous_location = location

        traffic_light = self.vehicle.get_traffic_light()
        is_red = False
        traffic_distance = 100.0
        if traffic_light is not None:
            is_red = str(traffic_light.get_state()).lower().endswith("red")
            traffic_distance = location.distance(traffic_light.get_location())

        # Dynamic actor distances are used by the ontology reward when traffic is enabled.
        pedestrian_distance, obstacle_distance = self._nearest_dynamic_actor_distances(location)

        # Collision is terminal, so it remains true until reset.
        # Lane invasion is treated as a one-step event so the agent is not
        # punished forever after one lane crossing.
        lane_invasion_now = self.lane_invasion_happened
        self.lane_invasion_happened = False

        return DrivingState(
            vehicle=VehicleState(speed_kmh=speed_kmh),
            road=RoadState(lane_offset_m=lane_offset_m, heading_error_deg=heading_error_deg, progress_m=progress_m),
            traffic_light=TrafficLightState(is_red=is_red, distance_m=traffic_distance),
            pedestrian=PedestrianState(distance_m=pedestrian_distance),
            obstacle=ObstacleState(distance_m=obstacle_distance),
            weather=WeatherState(rain=0.0, fog=0.0),
            sensors=SensorState(collision=self.collision_happened, lane_invasion=lane_invasion_now),
        )


    def _spawn_safety_sensors(self) -> None:
        """Attach collision and lane-invasion sensors to the ego vehicle."""
        if self.world is None or self.vehicle is None:
            return

        blueprint_library = self.world.get_blueprint_library()

        collision_bp = blueprint_library.find("sensor.other.collision")
        lane_bp = blueprint_library.find("sensor.other.lane_invasion")

        sensor_transform = self.carla.Transform(self.carla.Location(x=0.0, z=0.0))

        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            sensor_transform,
            attach_to=self.vehicle,
        )
        self.collision_sensor.listen(self._on_collision)

        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_bp,
            sensor_transform,
            attach_to=self.vehicle,
        )
        self.lane_invasion_sensor.listen(self._on_lane_invasion)

    def _on_collision(self, event) -> None:
        """CARLA collision callback with actor type and impact intensity."""
        self.collision_happened = True
        try:
            other = event.other_actor
            self.last_collision_actor_type = getattr(other, "type_id", "unknown")
            self.last_collision_actor_id = int(getattr(other, "id", -1))
            try:
                self.last_collision_actor_role_name = other.attributes.get("role_name", "")
            except Exception:
                self.last_collision_actor_role_name = ""
            impulse = event.normal_impulse
            self.last_collision_intensity = float(math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2))
        except Exception:
            self.last_collision_actor_type = "unknown"
            self.last_collision_actor_id = -1
            self.last_collision_actor_role_name = ""
            self.last_collision_intensity = 0.0

    def _on_lane_invasion(self, event) -> None:
        """CARLA lane-invasion callback."""
        self.lane_invasion_happened = True

    @staticmethod
    def _angle_difference(a: float, b: float) -> float:
        """Return smallest difference between two angles in degrees."""
        diff = (a - b + 180.0) % 360.0 - 180.0
        return diff

    @staticmethod
    def _actor_id(actor):
        """Return a CARLA actor id without calling simulator-side actor functions."""
        if actor is None:
            return None
        try:
            actor_id = getattr(actor, "id", None)
            if actor_id is None:
                return None
            return int(actor_id)
        except Exception:
            return None

    def _batch_destroy_actor_ids(self, actor_ids) -> None:
        """Destroy actors by id using CARLA batch commands.

        Important: do NOT call actor.is_alive, actor.stop(), or actor.destroy() here.
        In CARLA 0.9.14, those Python actor methods can abort the whole process with
        a C++ runtime error if the simulator has already destroyed the actor. Batch
        DestroyActor by id is much safer for realistic traffic/walker cleanup.
        """
        if self.client is None or self.carla is None:
            return

        clean_ids = []
        seen = set()
        for actor_id in actor_ids:
            try:
                actor_id = int(actor_id)
            except Exception:
                continue
            if actor_id <= 0 or actor_id in seen:
                continue
            clean_ids.append(actor_id)
            seen.add(actor_id)

        if not clean_ids:
            return

        try:
            commands = [self.carla.command.DestroyActor(actor_id) for actor_id in clean_ids]
            self.client.apply_batch(commands)
        except Exception:
            # Last-resort safety: cleanup must never kill the training script.
            pass

        # Let CARLA process actor destruction before the next reset/spawn.
        try:
            if self.world is not None:
                self.world.wait_for_tick()
        except Exception:
            try:
                time.sleep(0.05)
            except Exception:
                pass

    def _destroy_actors(self) -> None:
        """Destroy CARLA actors created by this environment without unsafe actor calls."""
        # Capture ids first. Accessing actor.id is local metadata and avoids calling
        # simulator-side methods such as is_alive/stop/destroy, which caused:
        # "trying to operate on a destroyed actor".
        sensor_ids = []
        for sensor_name in ("collision_sensor", "lane_invasion_sensor"):
            sensor_ids.append(self._actor_id(getattr(self, sensor_name, None)))
            setattr(self, sensor_name, None)

        controller_ids = [self._actor_id(a) for a in list(getattr(self, "walker_controllers", []))]
        walker_ids = [self._actor_id(a) for a in list(getattr(self, "background_walkers", []))]
        vehicle_ids = [self._actor_id(a) for a in list(getattr(self, "background_vehicles", []))]
        ego_id = self._actor_id(self.vehicle)

        # Clear references before sending destroy commands so callbacks/close cannot
        # accidentally reuse stale actor objects.
        self.walker_controllers = []
        self.background_walkers = []
        self.background_vehicles = []
        self.vehicle = None

        # Destroy attached sensors first, then walker controllers, then walkers/NPCs,
        # and ego vehicle last. Everything is done by id through CARLA batch commands.
        self._batch_destroy_actor_ids(sensor_ids)
        self._batch_destroy_actor_ids(controller_ids)
        self._batch_destroy_actor_ids(walker_ids + vehicle_ids + [ego_id])

    def close(self) -> None:
        self._destroy_actors()

