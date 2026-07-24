"""MetaDrive creation and canonical episode-termination parsing."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import Any, Dict

# ``Any`` keeps annotations importable when MetaDrive is absent.
MetaDriveEnv = Any


def _load_metadrive_env_class():
    """Import MetaDrive only when an experiment actually creates an environment."""
    try:
        from metadrive import MetaDriveEnv
    except ImportError as exc:
        raise RuntimeError(
            "MetaDrive is not installed. Run: "
            "python -m pip install metadrive-simulator"
        ) from exc
    return MetaDriveEnv


def metadrive_version() -> str:
    """Return the installed MetaDrive version without importing its engine."""
    try:
        return distribution_version("metadrive-simulator")
    except PackageNotFoundError:
        return "not-installed"


def make_env(args, phase: str) -> MetaDriveEnv:
    if phase not in {"train", "test"}:
        raise ValueError("phase must be train or test")
    if phase == "train":
        start_seed, num_scenarios = args.seed, args.train_episodes
    else:
        start_seed, num_scenarios = args.test_seed, args.test_episodes
    config = {
        "use_render": bool(args.render),
        "image_observation": False,
        "log_level": int(args.metadrive_log_level),
        "discrete_action": True,
        "use_multi_discrete": False,
        "discrete_steering_dim": int(args.discrete_steering_dim),
        "discrete_throttle_dim": int(args.discrete_throttle_dim),
        "horizon": int(args.max_episode_steps),
        "truncate_as_terminate": False,
        "start_seed": int(start_seed),
        "num_scenarios": int(max(1, num_scenarios)),
        "map": int(args.map_blocks),
        "traffic_density": float(args.traffic_density),
        "random_traffic": False,
        "accident_prob": float(args.accident_prob),
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "out_of_road_done": True,
        "success_reward": float(args.success_reward),
        "crash_vehicle_penalty": float(args.collision_penalty),
        "crash_object_penalty": float(args.collision_penalty),
        "out_of_road_penalty": float(args.out_of_road_penalty),
    }
    return _load_metadrive_env_class()(config)

def truthy(info: Dict, *keys: str) -> bool:
    return any(bool(info.get(key, False)) for key in keys)

def parse_step_info(info: Dict, terminated: bool, truncated: bool) -> Dict:
    crash_vehicle = truthy(info, "crash_vehicle")
    crash_object = truthy(
        info,
        "crash_object",
        "crash_building",
        "crash_human",
        "crash_sidewalk",
    )
    collision = crash_vehicle or crash_object or truthy(info, "crash", "crashed")
    out_of_road = truthy(info, "out_of_road")
    goal_reached = truthy(info, "arrive_dest", "arrived", "success")
    max_steps = truthy(info, "max_step") or bool(truncated)

    if collision:
        reason = "collision"
    elif out_of_road:
        reason = "out_of_road"
    elif goal_reached:
        reason = "goal"
    elif max_steps:
        reason = "max_steps"
    elif terminated:
        reason = "terminated"
    else:
        reason = "running"

    return {
        "termination_reason": reason,
        "policy_safety_stop": False,
        "collision": collision,
        "crash_vehicle": crash_vehicle,
        "crash_object": crash_object,
        "out_of_road": out_of_road,
        "goal_reached": goal_reached,
        "max_steps_reached": max_steps,
        "step_cost": float(info.get("cost", 0.0) or 0.0),
    }

class AllActionsBlockedEpisode(RuntimeError):
    """Signal a local safety stop without aborting the complete run."""

def policy_safety_stop_info() -> Dict:
    """Return an auditable non-crash episode-stop result."""
    parsed = parse_step_info({}, False, False)
    parsed["termination_reason"] = "policy_safety_stop"
    parsed["policy_safety_stop"] = True
    return parsed

def selected_rmst_event(parsed: Dict, event_definition: str) -> bool:
    if event_definition == "collision":
        return bool(parsed["collision"])
    if event_definition == "safety":
        return bool(
            parsed["collision"]
            or parsed["out_of_road"]
            or parsed.get("policy_safety_stop", False)
        )
    raise ValueError(f"Unknown RMST event definition: {event_definition}")
