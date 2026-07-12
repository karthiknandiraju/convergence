"""Run both CARLA 0.9.14 experiment graph sets.

Set 1 output: results/graph_set_1_noisy_rnd_count
Set 2 output: results/graph_set_2_epsilon_median_noisy_rnd_count

CARLA 0.9.14 must already be running before this script starts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["1", "2", "both"], default="both")
    parser.add_argument("--extra", default="", help="Extra arguments passed to each training script, for example: '--train-episodes 100 --test-episodes 20 --run-epsilon-sweep'")
    args = parser.parse_args()
    extra = args.extra.split() if args.extra else []
    jobs = []
    if args.set in {"1", "both"}:
        jobs.append([sys.executable, "-m", "src.train_set1_noisy_rnd_count_experiments", *extra])
    if args.set in {"2", "both"}:
        jobs.append([sys.executable, "-m", "src.train_set2_epsilon_median_noisy_rnd_count", *extra])
    for cmd in jobs:
        print("\n=== Running:", " ".join(cmd), "===\n", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
