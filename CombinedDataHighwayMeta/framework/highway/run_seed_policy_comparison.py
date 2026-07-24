#!/usr/bin/env python3
"""Run one or more HighwayEnv policies for a seed and compare with baselines.

Place this file beside ``train_canonical_baselines.py`` and
``compare_collision_policies.py``. Policy scripts must accept the common CLI
arguments used by the chapter policies:

    --seed --test-seed --train-episodes --test-episodes
    --max-episode-steps --device --output-dir --rmst-tau

Examples
--------
Run one policy for seed 11 and compare it with existing seed-11 baselines:

    python run_seed_policy_comparison.py --seed 11 \
      --policy-files policies/Karthikeya27adv23.py --device cuda

Create missing baselines before running the policy:

    python run_seed_policy_comparison.py --seed 67 \
      --policy-files policies/Karthikeya27adv23.py --device cuda \
      --train-baselines-if-missing

Compare several policies together:

    python run_seed_policy_comparison.py --seed 11 \
      --policy-files policies/Karthikeya27adv8956.py \
      policies/Karthikeya27adv23.py --device cuda

Reuse existing policy outputs without retraining:

    python run_seed_policy_comparison.py --seed 11 \
      --policy-files policies/Karthikeya27adv23.py --skip-policy-training
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


BASELINE_METHODS = ("epsilon", "noisy", "rnd")
REQUIRED_COLUMNS = {
    "phase",
    "experiment",
    "method",
    "episode",
    "scenario_seed",
    "steps",
    "collision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HighwayEnv policies for any seed and generate collision-only "
            "comparisons against matching canonical baselines."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--policy-files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more policy Python files.",
    )
    parser.add_argument(
        "--policy-names",
        nargs="+",
        default=None,
        help="Optional output names matching --policy-files in the same order.",
    )
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=300)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--rmst-tau", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument(
        "--train-baselines-if-missing",
        action="store_true",
        help="Train only missing epsilon/noisy/RND baselines for this seed.",
    )
    parser.add_argument(
        "--skip-policy-training",
        action="store_true",
        help="Reuse policy_results/seed_<seed>/<policy-name> outputs.",
    )
    parser.add_argument(
        "--force-policy",
        action="store_true",
        help="Allow a policy to overwrite an existing result CSV.",
    )
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Forward this exception flag to the comparison script.",
    )
    parser.add_argument(
        "--allow-scenario-mismatch",
        action="store_true",
        help="Forward this exception flag to the comparison script.",
    )
    parser.add_argument(
        "--comparison-name",
        default=None,
        help="Optional final folder name under comparisons/seed_<seed>/.",
    )
    args = parser.parse_args()

    if args.policy_names is not None and len(args.policy_names) != len(args.policy_files):
        parser.error("--policy-names must contain one name per --policy-files entry")
    if min(
        args.train_episodes,
        args.test_episodes,
        args.max_episode_steps,
        args.rmst_tau,
        args.block_size,
    ) <= 0:
        parser.error("episode counts, step counts, RMST tau, and block size must be positive")
    if args.rmst_tau > args.max_episode_steps:
        parser.error("--rmst-tau cannot exceed --max-episode-steps")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples cannot be negative")
    train_end = args.seed + args.train_episodes - 1
    test_end = args.test_seed + args.test_episodes - 1
    if max(args.seed, args.test_seed) <= min(train_end, test_end):
        parser.error("training and testing scenario-seed ranges overlap")
    return args


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not name:
        raise ValueError(f"Invalid empty policy/comparison name derived from {value!r}")
    return name


def resolve_from_project(path: Path, project: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def run(command: list[str], env: dict[str, str], cwd: Path) -> None:
    print("\n$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def missing_baselines(baseline_root: Path) -> list[str]:
    return [
        method
        for method in BASELINE_METHODS
        if not (baseline_root / method / "all_episode_results.csv").is_file()
    ]


def validate_policy_output(output_dir: Path, args: argparse.Namespace) -> Path:
    csv_path = output_dir / "all_episode_results.csv"
    config_path = output_dir / "config.json"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Policy completed without creating the required file: {csv_path}"
        )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Policy output lacks config.json, required for fair comparison: {config_path}"
        )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {missing}")
        phases = {str(row.get("phase", "")).strip().lower() for row in reader}
    if not {"train", "test"}.issubset(phases):
        raise ValueError(f"{csv_path} must contain both train and test rows")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "seed": args.seed,
        "test_seed": args.test_seed,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "max_episode_steps": args.max_episode_steps,
        "rmst_tau": args.rmst_tau,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, expected={expected_value!r}"
            for key, (saved, expected_value) in mismatches.items()
        )
        raise ValueError(f"Policy configuration mismatch in {config_path}: {details}")
    return csv_path


def unique_names(values: Iterable[str]) -> None:
    values = list(values)
    if len(set(values)) != len(values):
        raise ValueError(f"Policy names must be unique: {values}")


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    baseline_trainer = project / "train_canonical_baselines.py"
    comparator = project / "compare_collision_policies.py"
    for required in (baseline_trainer, comparator):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing {required.name}. Place this runner in the canonical-baseline project root."
            )

    policy_files = [resolve_from_project(path, project) for path in args.policy_files]
    for policy_file in policy_files:
        if not policy_file.is_file():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

    policy_names = (
        [safe_name(name) for name in args.policy_names]
        if args.policy_names
        else [safe_name(path.stem) for path in policy_files]
    )
    unique_names(policy_names)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    baseline_root = project / "canonical_baselines" / f"seed_{args.seed}"
    missing = missing_baselines(baseline_root)
    if missing:
        if not args.train_baselines_if_missing:
            raise FileNotFoundError(
                f"Missing canonical baselines for seed {args.seed}: {', '.join(missing)}. "
                "Rerun with --train-baselines-if-missing."
            )
        baseline_command = [
            sys.executable,
            "-u",
            str(baseline_trainer),
            "--seed",
            str(args.seed),
            "--methods",
            *missing,
            "--device",
            args.device,
            "--train-episodes",
            str(args.train_episodes),
            "--test-episodes",
            str(args.test_episodes),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--test-seed",
            str(args.test_seed),
            "--rmst-tau",
            str(args.rmst_tau),
            "--output-root",
            str(project / "canonical_baselines"),
        ]
        run(baseline_command, env, project)

    policy_dirs: list[Path] = []
    for policy_file, policy_name in zip(policy_files, policy_names):
        output_dir = project / "policy_results" / f"seed_{args.seed}" / policy_name
        result_csv = output_dir / "all_episode_results.csv"
        if not args.skip_policy_training:
            if result_csv.exists() and not args.force_policy:
                raise FileExistsError(
                    f"Policy output already exists: {result_csv}. Use "
                    "--skip-policy-training to reuse it or --force-policy to overwrite it."
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            policy_command = [
                sys.executable,
                "-u",
                str(policy_file),
                "--seed",
                str(args.seed),
                "--test-seed",
                str(args.test_seed),
                "--train-episodes",
                str(args.train_episodes),
                "--test-episodes",
                str(args.test_episodes),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--device",
                args.device,
                "--rmst-tau",
                str(args.rmst_tau),
                "--output-dir",
                str(output_dir),
            ]
            if args.force_policy:
                policy_command.append("--force")
            run(policy_command, env, project)
        validate_policy_output(output_dir, args)
        policy_dirs.append(output_dir)

    comparison_name = safe_name(
        args.comparison_name
        or (policy_names[0] if len(policy_names) == 1 else "all_policies")
    )
    comparison_dir = project / "comparisons" / f"seed_{args.seed}" / comparison_name
    compare_command = [
        sys.executable,
        "-u",
        str(comparator),
        "--baseline-root",
        str(baseline_root),
        "--policy-dir",
        *[str(path) for path in policy_dirs],
        "--rmst-tau",
        str(args.rmst_tau),
        "--block-size",
        str(args.block_size),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--output-dir",
        str(comparison_dir),
    ]
    if args.allow_config_mismatch:
        compare_command.append("--allow-config-mismatch")
    if args.allow_scenario_mismatch:
        compare_command.append("--allow-scenario-mismatch")
    run(compare_command, env, project)

    print("\nCompleted successfully.")
    print("Seed:", args.seed)
    print("Policy results:")
    for path in policy_dirs:
        print(" ", path)
    print("Comparison results:", comparison_dir)
    print("Collision metrics:", comparison_dir / "collision_metrics.csv")
    print("Comparison plots:", comparison_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Command failed with exit status {error.returncode}") from error
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
