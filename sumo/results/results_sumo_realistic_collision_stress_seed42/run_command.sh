#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DQNMedian50/convergence/sumo

source sumo_collision_env/bin/activate

export SUMO_HOME=/usr/share/sumo

cd src

python -u sumo_tabular_median50.py \
  --train-episodes 500 \
  --test-episodes 300 \
  --max-episode-steps 500 \
  --epsilon 0.2 \
  --alpha 0.1 \
  --gamma 0.99 \
  --target-speed 8.333333 \
  --collision-penalty -50 \
  --ego-speed-mode 30 \
  --collision-mingap-factor 0.0 \
  --road-length 800 \
  --progress-reward-scale 100 \
  --traffic-vehicles 2 \
  --leader-brake-probability 0.5 \
  --leader-brake-start-min 80 \
  --leader-brake-start-max 220 \
  --leader-brake-duration 25 \
  --leader-brake-speed 2.0 \
  --leader-brake-decel-seconds 2.0 \
  --seed 42 \
  --output-dir /workspace/DQNMedian50/convergence/sumo/results/results_sumo_realistic_collision_stress_seed42
