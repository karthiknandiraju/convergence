"""Fast installation and adapter check; it does not train a model."""

from __future__ import annotations

import itertools

import numpy as np

from highway_env_adapter import ACTION_COUNT, ENVIRONMENT_ID, HighwayEnvAdapter


def main() -> None:
    config = {
        "discrete_action": True,
        "use_multi_discrete": False,
        "discrete_steering_dim": 3,
        "discrete_throttle_dim": 3,
        "horizon": 20,
        "map": 3,
        "traffic_density": 0.2,
        "crash_vehicle_penalty": 50.0,
        "out_of_road_penalty": 50.0,
    }
    env = HighwayEnvAdapter(config)
    try:
        initial_observation, info = env.reset(seed=27)
        assert env.action_space.n == ACTION_COUNT
        assert initial_observation.shape == (15, 6)
        assert np.isfinite(initial_observation).all()
        assert info["termination_reason"] == "running"

        hazards = env.engine.get_objects()
        assert env.vehicle is not None
        assert sum(item is env.vehicle for item in hazards.values()) == 1
        lane = env.vehicle.lane
        assert lane is not None
        longitudinal, _ = lane.local_coordinates(env.vehicle.position)
        assert lane.width_at(longitudinal) > 0.0
        assert np.isfinite(lane.heading_theta_at(longitudinal))

        for action in range(ACTION_COUNT):
            env.reset(seed=100 + action)
            observation, reward, _, _, step_info = env.step(action)
            assert observation.shape == initial_observation.shape
            assert np.isfinite(observation).all()
            assert isinstance(reward, float)
            for key in ("crash_vehicle", "crash_object", "out_of_road", "max_step"):
                assert key in step_info
    finally:
        env.close()

    first = HighwayEnvAdapter(config)
    second = HighwayEnvAdapter(config)
    try:
        first_observation, _ = first.reset(seed=1234)
        second_observation, _ = second.reset(seed=1234)
        assert np.array_equal(first_observation, second_observation)
    finally:
        first.close()
        second.close()

    vectors = list(itertools.product((-1, 0, 1), repeat=2))
    print(f"PASS: {ENVIRONMENT_ID}, observation=(15, 6), actions={vectors}")


if __name__ == "__main__":
    main()
