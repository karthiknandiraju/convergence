"""Structural and compatibility tests for the modular SafetyPool policy.

The tests use Python and NumPy only. They can therefore validate source parity,
CLI defaults, and memory composition without MetaDrive, PyTorch, or a GPU.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
POLICIES_ROOT = SOURCE_ROOT / "policies"
ORIGINAL = POLICIES_ROOT / "safetypool.py"
MODULAR = POLICIES_ROOT / "safetypool_components"
EXPECTED_ORIGINAL_SHA256 = (
    "e0a3e38f4a277c4b58e8eccdd2f746db72bf557e9783d2f522c1b5e5ef12e890"
)


def parsed(path: Path) -> ast.Module:
    """Parse a Python file using its declared UTF-8 source text."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def named_definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Index top-level functions and classes by name."""
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def canonical(node: ast.AST) -> str:
    """Return a location-independent syntax representation."""
    return ast.dump(node, include_attributes=False)


class ModularPolicyTests(unittest.TestCase):
    """Protect the archived source and the modular compatibility contract."""

    def test_original_policy_is_unchanged(self) -> None:
        digest = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        self.assertEqual(digest, EXPECTED_ORIGINAL_SHA256)

    def test_every_new_python_file_has_a_module_docstring(self) -> None:
        files = [
            POLICIES_ROOT / "__init__.py",
            POLICIES_ROOT / "safetypool_modular.py",
            *sorted(MODULAR.rglob("*.py")),
        ]
        for path in files:
            with self.subTest(path=path.relative_to(SOURCE_ROOT)):
                self.assertTrue(ast.get_docstring(parsed(path)))

    def test_all_pool_methods_are_preserved_exactly_once(self) -> None:
        original_tree = parsed(ORIGINAL)
        original_pool = next(
            node
            for node in original_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SimilarStateActionPools"
        )
        original_methods = {
            node.name: canonical(node)
            for node in original_pool.body
            if isinstance(node, ast.FunctionDef)
        }

        modular_methods: dict[str, str] = {}
        for path in sorted((MODULAR / "memory").glob("*.py")):
            for class_node in (
                node for node in parsed(path).body if isinstance(node, ast.ClassDef)
            ):
                if not class_node.name.endswith("Mixin"):
                    continue
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef):
                        self.assertNotIn(method.name, modular_methods)
                        modular_methods[method.name] = canonical(method)

        self.assertEqual(len(original_methods), 97)
        self.assertEqual(modular_methods, original_methods)

    def test_non_adapter_definitions_match_the_monolith(self) -> None:
        mapping = {
            "utils.py": [
                "set_seed",
                "choose_device",
                "flatten_observation",
                "sha256_file",
                "canonical_json_sha256",
                "json_safe",
                "observation_sha256",
                "avg",
                "restricted_mean_survival_time",
            ],
            "safety.py": [
                "directional_safety_relative_improvements",
                "capacity_fallback_valid_mask",
                "_finite_float",
                "_finite_float_or_none",
                "_angle_difference_radians",
                "_vehicle_lane",
                "_is_collision_hazard",
                "_nearest_collision_hazard_distance",
                "extract_safety_vector",
                "RunningObservationNormalizer",
                "SimilarityThresholdCalibrator",
            ],
            "dqn.py": ["QNetwork", "Transition", "ReplayBuffer", "DQNAgent"],
            "action_selection.py": [
                "best_available_action",
                "select_training_action",
            ],
            "experiment.py": [
                "episode_row",
                "verify_discrete_action_space",
                "run_frozen_test_phase",
                "run_experiment",
            ],
            "metrics.py": [
                "make_summary",
                "bool_value",
                "collision_summary",
                "policy_safety_stop_summary",
            ],
            "plotting.py": ["apply_ieee_style", "save_figure", "make_figures"],
            "pool_reporting.py": [
                "save_pool_statistics",
                "make_pool_figures",
            ],
            "persistence.py": [
                "write_csv",
                "critical_config",
                "baseline_shared_config",
                "canonical_episode_rows",
                "save_framework_compatibility_outputs",
                "update_baseline_index",
                "save_outputs",
            ],
        }
        original = named_definitions(parsed(ORIGINAL))
        for filename, names in mapping.items():
            modular = named_definitions(parsed(MODULAR / filename))
            for name in names:
                with self.subTest(module=filename, definition=name):
                    self.assertEqual(canonical(modular[name]), canonical(original[name]))

    def test_parser_defaults_preserve_the_experiment_contract(self) -> None:
        # Import only the light configuration module; no engine or DQN is loaded.
        sys.path.insert(0, str(SOURCE_ROOT))
        try:
            from policies.safetypool_components.configuration import build_parser

            args = build_parser().parse_args(["--seed", "17"])
        finally:
            sys.path.remove(str(SOURCE_ROOT))

        self.assertEqual(args.train_episodes, 500)
        self.assertEqual(args.test_episodes, 300)
        self.assertEqual(args.max_episode_steps, 500)
        self.assertEqual(args.test_seed, 100000)
        self.assertEqual(args.max_state_pools, 500)
        self.assertEqual(args.max_state_candidates, 125)
        self.assertEqual(args.candidate_hard_limit, 150)
        self.assertEqual(args.candidate_batch_evict_count, 25)

    def test_memory_responsibilities_are_separate_mixins(self) -> None:
        expected = {
            "storage.py": "PoolStorageMixin",
            "matching.py": "PoolMatchingMixin",
            "capacity.py": "PoolCapacityMixin",
            "outcomes.py": "PoolOutcomeMixin",
            "candidates.py": "PoolCandidateManagementMixin",
            "state_processing.py": "PoolStateProcessingMixin",
            "selection.py": "PoolActionSelectionMixin",
            "diagnostics.py": "PoolDiagnosticsMixin",
        }
        for filename, class_name in expected.items():
            with self.subTest(module=filename):
                names = named_definitions(parsed(MODULAR / "memory" / filename))
                self.assertIn(class_name, names)

    def test_composed_memory_can_create_and_validate_a_candidate(self) -> None:
        sys.path.insert(0, str(SOURCE_ROOT))
        try:
            from policies.safetypool_components.memory import (
                SimilarStateActionPools,
            )

            pools = SimilarStateActionPools(
                max_pools=4,
                maximum_pool_capacity=4,
                max_candidates=3,
                candidate_hard_limit=4,
                candidate_batch_evict_count=1,
                candidate_promotion_visits=4,
                hazard_memory_capacity=1,
                safe_confirmation_visits=2,
                safety_horizon_steps=5,
                minimum_progress_reward=0.01,
                warning_block_threshold=2,
                candidate_recent_protection_episodes=3,
                capacity_review_interval=25,
                action_count=9,
                observation_size=25,
                safety_size=5,
                similarity_threshold=0.90,
                distance_threshold=0.20,
                candidate_similarity_threshold=0.90,
                candidate_distance_threshold=0.25,
                safety_similarity_threshold=0.98,
                safety_distance_threshold=0.10,
                candidate_safety_distance_threshold=0.125,
                close_enough_fallback=True,
                general_cosine_relaxation=0.02,
                general_rms_relaxation=0.02,
                capacity_fallback_general_variation=0.10,
                capacity_fallback_safety_improvement=0.10,
                auto_calibrate_thresholds=False,
                calibration_state_count=10,
                calibration_max_pairs=20,
                seed=17,
                candidate_centroid_shift_threshold=0.01,
                candidate_stable_updates=2,
                max_candidate_centroid_updates=4,
                centroid_shift_threshold=0.01,
                centroid_stable_updates=3,
                max_centroid_updates=10,
                centroid_stability_distance_threshold=0.10,
                pool_storage_dtype="float32",
            )
            result = pools.process_state(
                np.ones(25, dtype=np.float32),
                np.asarray([2.0, 10.0, 0.1, 0.1, 30.0], dtype=np.float32),
                episode=1,
            )
            pools.validate_invariants()
        finally:
            sys.path.remove(str(SOURCE_ROOT))

        self.assertEqual(result, (None, 0, "candidate_created_argmax"))
        self.assertEqual(len(pools.records), 1)
        self.assertEqual(len(pools.status_ids[pools.CANDIDATE]), 1)


if __name__ == "__main__":
    unittest.main()
