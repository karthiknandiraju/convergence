"""Safety-feature extraction, normalization, and threshold calibration."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .constants import DIRECTIONAL_SAFETY_RELATIVE_FLOORS


def directional_safety_relative_improvements(
    current_safety: np.ndarray,
    centroid_safeties: np.ndarray,
) -> np.ndarray:
    """Return stable safer-direction changes for the four directional fields."""
    current = np.asarray(current_safety, dtype=np.float32)[:4]
    centroids = np.asarray(centroid_safeties, dtype=np.float32)[:, :4]
    safer_directions = np.asarray(
        [1.0, 1.0, -1.0, -1.0], dtype=np.float32
    )
    denominators = np.maximum(
        np.abs(centroids), DIRECTIONAL_SAFETY_RELATIVE_FLOORS
    )
    return (current - centroids) * safer_directions / denominators

def capacity_fallback_valid_mask(
    relaxed_general_cosine_ok: np.ndarray,
    relaxed_general_rms_ok: np.ndarray,
    safety_cosine_ok: np.ndarray,
    safety_rms_ok: np.ndarray,
    directional_safety_ok: np.ndarray,
    strict_general_ok: np.ndarray,
) -> np.ndarray:
    """Compose the fallback gates; either failed safety gate rejects a match."""
    return (
        relaxed_general_cosine_ok
        & relaxed_general_rms_ok
        & safety_cosine_ok
        & safety_rms_ok
        & directional_safety_ok
        & ~strict_general_ok
    )

def _finite_float(value, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)

def _finite_float_or_none(value) -> Optional[float]:
    """Return a finite float, or None when a safety reading is unavailable."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def _angle_difference_radians(a: float, b: float) -> float:
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)

def _vehicle_lane(vehicle):
    navigation = getattr(vehicle, "navigation", None)
    lane = getattr(navigation, "current_lane", None)
    return lane if lane is not None else getattr(vehicle, "lane", None)

def _is_collision_hazard(obj) -> bool:
    """Best-effort identification of physical collision hazards.

    This intentionally avoids depending on one MetaDrive release's exact class
    names. It first rejects non-physical runtime helpers, then accepts known
    physical hazards, and finally rejects non-collidable map infrastructure.
    """
    class_name = obj.__class__.__name__.lower()
    module_name = obj.__class__.__module__.lower()
    text = f"{module_name}.{class_name}"

    helper_tokens = (
        "navigation", "camera", "sensor", "engine", "manager",
        "policy", "renderer", "nodepath",
    )
    if any(token in text for token in helper_tokens):
        return False

    included_tokens = (
        "vehicle", "pedestrian", "human", "cyclist", "bicycle",
        "cone", "barrier", "obstacle", "trafficobject", "traffic_object",
        "building", "sidewalk",
    )
    if any(token in text for token in included_tokens):
        return True

    infrastructure_tokens = (
        "lane", "road", "map", "light", "marking", "terrain",
    )
    if any(token in text for token in infrastructure_tokens):
        return False

    # Version-tolerant duck typing. A positioned object with collision geometry
    # or vehicle-like kinematics is treated as a hazard. Static map helpers are
    # filtered above.
    has_position = getattr(obj, "position", None) is not None
    has_collision_geometry = any(
        hasattr(obj, attribute)
        for attribute in (
            "collision_node", "collision_nodes", "body", "chassis",
            "top_down_width", "top_down_length", "WIDTH", "LENGTH",
        )
    )
    has_kinematics = any(
        hasattr(obj, attribute)
        for attribute in ("velocity", "speed", "speed_km_h", "heading_theta")
    )
    return bool(has_position and (has_collision_geometry or has_kinematics))

def _nearest_collision_hazard_distance(
    env, ego_position: np.ndarray, cap: float
) -> Optional[float]:
    """Return a verified hazard distance, or None when scanning is unavailable."""
    engine = getattr(env, "engine", None)
    if engine is None:
        return None

    objects = None
    getter = getattr(engine, "get_objects", None)
    if callable(getter):
        try:
            objects = getter()
        except TypeError:
            try:
                objects = getter(lambda _: True)
            except Exception:
                objects = None
        except Exception:
            objects = None
    if objects is None:
        objects = getattr(engine, "objects", None)

    if isinstance(objects, dict):
        iterable = objects.values()
    elif objects is None:
        return None
    else:
        try:
            iterable = iter(objects)
        except TypeError:
            return None

    ego = getattr(env, "vehicle", None)
    best = float(cap)
    for obj in iterable:
        if obj is ego or not _is_collision_hazard(obj):
            continue
        position = getattr(obj, "position", None)
        if position is None:
            return None
        try:
            point = np.asarray(position, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if point.size < 2 or not np.all(np.isfinite(point[:2])):
            return None
        distance = float(np.linalg.norm(point[:2] - ego_position[:2]))
        if not math.isfinite(distance):
            return None
        if 1e-6 < distance < best:
            best = distance
    return float(min(best, cap))

def extract_safety_vector(env, args) -> Optional[np.ndarray]:
    """Extract five verified pre-action safety variables.

    Missing or non-finite engine data returns ``None``.  The caller then skips
    pool matching and evidence updates for that state instead of fabricating a
    safety vector whose large clearances could look safer than reality.
    """
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        return None

    try:
        position = np.asarray(vehicle.position, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if position.size < 2 or not np.all(np.isfinite(position[:2])):
        return None

    speed_km_h_value = getattr(vehicle, "speed_km_h", None)
    if speed_km_h_value is not None:
        speed_value = _finite_float_or_none(speed_km_h_value)
    else:
        fallback_speed = _finite_float_or_none(
            getattr(vehicle, "speed", None)
        )
        if fallback_speed is None:
            return None
        if args.safety_speed_fallback_unit == "mps":
            speed_value = fallback_speed * 3.6
        else:
            speed_value = fallback_speed
    if speed_value is None:
        return None
    speed = float(np.clip(abs(speed_value), 0.0, args.safety_speed_cap))

    lane = _vehicle_lane(vehicle)
    if lane is None:
        return None

    local_coordinates = getattr(lane, "local_coordinates", None)
    if not callable(local_coordinates):
        return None
    try:
        longitudinal_value, lateral_value = local_coordinates(position[:2])
    except Exception:
        return None
    longitudinal = _finite_float_or_none(longitudinal_value)
    lateral = _finite_float_or_none(lateral_value)
    if longitudinal is None or lateral is None:
        return None
    lane_offset = abs(lateral)

    lane_width: Optional[float] = None
    width_at = getattr(lane, "width_at", None)
    if callable(width_at):
        try:
            lane_width = _finite_float_or_none(width_at(longitudinal))
        except Exception:
            lane_width = None
    if lane_width is None or lane_width <= 0.0:
        width_value = getattr(lane, "width", None)
        if callable(width_value):
            try:
                width_value = width_value(longitudinal)
            except TypeError:
                try:
                    width_value = width_value()
                except Exception:
                    width_value = None
            except Exception:
                width_value = None
        lane_width = _finite_float_or_none(width_value)
    if lane_width is None or lane_width <= 0.0:
        return None
    lane_boundary_clearance = min(
        max(0.0, lane_width / 2.0 - lane_offset),
        float(args.safety_lane_boundary_cap),
    )

    vehicle_heading = _finite_float_or_none(
        getattr(vehicle, "heading_theta", None)
    )
    heading_at = getattr(lane, "heading_theta_at", None)
    if vehicle_heading is None or not callable(heading_at):
        return None
    try:
        lane_heading = _finite_float_or_none(heading_at(longitudinal))
    except Exception:
        return None
    if lane_heading is None:
        return None
    heading_error = _angle_difference_radians(vehicle_heading, lane_heading)

    nearest_hazard = _nearest_collision_hazard_distance(
        env, position, float(args.safety_nearest_object_cap)
    )
    if nearest_hazard is None:
        return None
    safety = np.asarray(
        [
            lane_boundary_clearance,
            nearest_hazard,
            lane_offset,
            heading_error,
            speed,
        ],
        dtype=np.float32,
    )
    return safety if np.all(np.isfinite(safety)) else None

class RunningObservationNormalizer:
    """Welford normalizer fitted on training data and then frozen."""

    def __init__(self, dimension: int, epsilon: float = 1e-6):
        self.dimension = int(dimension)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)
        self.frozen = False

    def update(self, state: np.ndarray) -> None:
        if self.frozen:
            return
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("Observation dimension changed during normalization.")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / float(self.count)
        delta2 = vector - self.mean
        self.m2 += delta * delta2

    def standard_deviation(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.dimension, dtype=np.float64)
        variance = self.m2 / float(self.count - 1)
        return np.sqrt(np.maximum(variance, self.epsilon))

    def transform(self, state: np.ndarray) -> np.ndarray:
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self.dimension:
            raise ValueError("Observation dimension changed during normalization.")
        normalized = (vector - self.mean) / (
            self.standard_deviation() + self.epsilon
        )
        return normalized.astype(np.float32)

    def freeze(self) -> None:
        self.frozen = True

    def statistics(self) -> Dict:
        std = self.standard_deviation()
        return {
            "normalizer_observation_count": int(self.count),
            "normalizer_frozen": bool(self.frozen),
            "normalizer_mean_absolute_mean": float(np.mean(np.abs(self.mean))),
            "normalizer_mean_standard_deviation": float(np.mean(std)),
            "normalizer_min_standard_deviation": float(np.min(std)),
            "normalizer_max_standard_deviation": float(np.max(std)),
        }

class SimilarityThresholdCalibrator:
    """Calibrate matching thresholds in one fixed normalized coordinate space."""

    def __init__(
        self,
        max_pairs: int,
        fallback_similarity: float,
        fallback_distance: float,
        fallback_candidate_distance: float,
        seed: int,
    ):
        self.max_pairs = int(max_pairs)
        self.fallback_similarity = float(fallback_similarity)
        self.fallback_distance = float(fallback_distance)
        self.fallback_candidate_distance = float(fallback_candidate_distance)
        self.rng = random.Random(int(seed))
        self.positive_cosines: Deque[float] = deque(maxlen=self.max_pairs)
        self.positive_distances: Deque[float] = deque(maxlen=self.max_pairs)
        self.negative_cosines: Deque[float] = deque(maxlen=self.max_pairs)
        self.negative_distances: Deque[float] = deque(maxlen=self.max_pairs)
        self.previous: Optional[np.ndarray] = None
        self.reference_buffer: Deque[np.ndarray] = deque(maxlen=512)
        self.frozen = False

    @staticmethod
    def pair_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        an = float(np.linalg.norm(a))
        bn = float(np.linalg.norm(b))
        if an == 0.0 or bn == 0.0:
            cosine = 1.0 if np.array_equal(a, b) else 0.0
        else:
            cosine = float(np.dot(a, b) / (an * bn))
        # RMS distance is stable in a z-scored space and does not divide by
        # a potentially near-zero representative norm.
        rms_distance = float(np.sqrt(np.mean(np.square(a - b))))
        return cosine, rms_distance

    def reset_episode(self) -> None:
        self.previous = None

    def observe(self, normalized_state: np.ndarray) -> None:
        if self.frozen:
            return
        vector = np.asarray(normalized_state, dtype=np.float32).copy()

        # Adjacent states are treated as likely-positive pairs.
        if self.previous is not None:
            cosine, distance = self.pair_metrics(vector, self.previous)
            if np.isfinite(cosine) and np.isfinite(distance):
                self.positive_cosines.append(cosine)
                self.positive_distances.append(distance)

        # Random older states provide likely-negative contrast pairs.
        if len(self.reference_buffer) >= 8:
            reference = self.reference_buffer[
                self.rng.randrange(len(self.reference_buffer))
            ]
            cosine, distance = self.pair_metrics(vector, reference)
            if np.isfinite(cosine) and np.isfinite(distance):
                self.negative_cosines.append(cosine)
                self.negative_distances.append(distance)

        self.reference_buffer.append(vector)
        self.previous = vector

    def derive(self) -> Tuple[float, float, float]:
        if (
            len(self.positive_cosines) < 20
            or len(self.positive_distances) < 20
        ):
            return (
                self.fallback_similarity,
                self.fallback_distance,
                self.fallback_candidate_distance,
            )

        positive_cosine_q25 = float(
            np.percentile(self.positive_cosines, 25)
        )
        positive_distance_q75 = float(
            np.percentile(self.positive_distances, 75)
        )

        if self.negative_cosines and self.negative_distances:
            negative_cosine_q90 = float(
                np.percentile(self.negative_cosines, 90)
            )
            negative_distance_q10 = float(
                np.percentile(self.negative_distances, 10)
            )
            cosine = max(
                positive_cosine_q25,
                negative_cosine_q90,
            )
            distance = min(
                positive_distance_q75,
                max(positive_distance_q75 * 0.5, negative_distance_q10),
            )
        else:
            cosine = positive_cosine_q25
            distance = positive_distance_q75

        cosine = max(0.85, min(0.995, cosine))
        distance = max(0.05, min(2.0, distance))
        # Calibration is allowed to tighten a candidate gate, never expand it
        # beyond the configured fallback/cap (0.25 with the policy defaults).
        candidate_distance = min(
            self.fallback_candidate_distance,
            max(distance, distance * 1.20),
        )
        return cosine, distance, candidate_distance

    def freeze(self) -> None:
        self.frozen = True

    def statistics(self) -> Dict:
        return {
            "calibration_positive_pair_count": len(self.positive_cosines),
            "calibration_negative_pair_count": len(self.negative_cosines),
            "threshold_calibrator_frozen": bool(self.frozen),
            "calibration_positive_cosine_q25": (
                float(np.percentile(self.positive_cosines, 25))
                if self.positive_cosines
                else math.nan
            ),
            "calibration_positive_distance_q75": (
                float(np.percentile(self.positive_distances, 75))
                if self.positive_distances
                else math.nan
            ),
            "calibration_negative_cosine_q90": (
                float(np.percentile(self.negative_cosines, 90))
                if self.negative_cosines
                else math.nan
            ),
            "calibration_negative_distance_q10": (
                float(np.percentile(self.negative_distances, 10))
                if self.negative_distances
                else math.nan
            ),
        }
