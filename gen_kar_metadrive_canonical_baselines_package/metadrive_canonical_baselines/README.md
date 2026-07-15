# Canonical MetaDrive Baselines and Collision Comparisons

This package trains each baseline **once per seed** and reuses its frozen raw
data for every later policy comparison. It prevents Epsilon, NoisyNet, or RND
values from changing merely because a different policy script retrained them.

## Baselines

All three methods use an online DQN, target network, replay buffer, Adam,
Huber loss, gradient clipping, and periodic target synchronization.

| Method | Training exploration | DQN training reward | Frozen test |
|---|---|---|---|
| Epsilon Greedy | Random action with probability `epsilon`, otherwise argmax | Environment reward | Argmax |
| NoisyNet DQN | Factorized-Gaussian NoisyLinear layers; no epsilon | Environment reward | Noise disabled, argmax |
| DQN + RND | Same epsilon-greedy action rule as Epsilon | Environment reward + normalized/clipped RND bonus | RND disabled, argmax |

The default hyperparameters match the supplied MetaDrive experiments:

- 500 training episodes, 300 frozen-test episodes, 500 maximum steps
- learning rate `5e-5`, gamma `0.99`, batch size `64`
- replay capacity `50,000`, target update every `1,000` learning steps
- two hidden layers of 128 units
- epsilon `0.2`, NoisyNet sigma `0.5`
- RND beta `0.01`, RND learning rate `1e-4`

## Dependencies

Use the same environment in which the existing MetaDrive experiments run:

```bash
python -m pip install metadrive-simulator torch numpy pandas matplotlib
```

## 1. Train canonical baselines

Set process-level determinism before Python starts:

```bash
SEED=11
PYTHONHASHSEED=$SEED \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python train_canonical_baselines.py \
  --seed "$SEED" \
  --methods epsilon noisy rnd \
  --device cuda \
  --output-root canonical_baselines
```

This creates:

```text
canonical_baselines/
└── seed_11/
    ├── baseline_index.json
    ├── epsilon/
    │   ├── model.pt
    │   ├── all_episode_results.csv
    │   ├── collision_metrics.csv
    │   ├── config.json
    │   └── manifest.json
    ├── noisy/
    └── rnd/
```

The trainer refuses to overwrite a completed baseline. Use `--force` only
when intentionally replacing the canonical run.

For multiple seeds:

```bash
for SEED in 11 48 67
do
  PYTHONHASHSEED=$SEED \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python train_canonical_baselines.py \
    --seed "$SEED" \
    --methods epsilon noisy rnd \
    --device cuda \
    --output-root canonical_baselines
done
```

Never retrain seed-11 Epsilon inside Chapter 6B, 6E, 12B, or 12C. Point every
seed-11 comparison at `canonical_baselines/seed_11/epsilon`.

## 2. Run a policy

The policy may use its existing training script. Its result folder must contain
`all_episode_results.csv` with at least:

```text
phase, experiment, method, episode, scenario_seed, steps, collision
```

For paired frozen testing, every method must use testing scenario seeds
`100000` through `100299`.

## 3. Compare baselines with policies

```bash
python compare_collision_policies.py \
  --baseline-root canonical_baselines/seed_11 \
  --policy-dir \
    /path/to/ch6bresults_11_new \
    /path/to/ch6br_results_11 \
    /path/to/ch6e_results_11 \
    /path/to/ch12bresults_11 \
  --rmst-tau 500 \
  --block-size 25 \
  --output-dir comparisons/seed_11
```

The comparator validates the available configurations and frozen-test scenario
sets before producing results. Do not use the mismatch override flags for a
formal experiment.

Baseline experiments embedded inside policy folders—such as another
`standard_epsilon` or `noisy_dqn` run—are excluded automatically. Therefore,
the comparison always uses the canonical baseline bank. Pass
`--include-embedded-baselines` only for diagnostic work.

## Collision metrics

Only three primary metrics are used:

1. **Collision RMST** through 500 steps — higher is better.
2. **Collisions per 1,000 steps** — lower is better.
3. **Collision rate per episode** — lower is better.

The three test bar charts include episode-bootstrap 95% confidence intervals.

## Training and testing collision box plots

A collision flag is binary, so a raw episode-level box plot would contain only
zeros and ones. Instead, the comparator divides each phase into consecutive
episode blocks and calculates collisions per 1,000 steps within each block.

The default block size is 25 episodes:

- 500 training episodes produce 20 block observations per method.
- 300 testing episodes produce 12 block observations per method.

Change it with `--block-size`.

## Comparison outputs

```text
comparisons/seed_11/
├── collision_metrics.csv
├── collision_block_values.csv
├── combined_episode_results.csv
├── comparison_manifest.json
├── test_collision_rmst.png/.pdf
├── test_collisions_per_1000_steps.png/.pdf
├── test_collision_rate.png/.pdf
├── train_collision_boxplot.png/.pdf
├── test_collision_boxplot.png/.pdf
└── train_test_collision_boxplots.png/.pdf
```

## Experimental rule

A canonical baseline is keyed by both seed and environment configuration. If
the environment configuration changes, create a new baseline bank. For a fixed
configuration and seed, reuse the existing model and CSVs instead of retraining.



## Comparsion Command 

python run_seed_policy_comparison.py --seed 11 --policy-files policies/metadrivech67.py --policy-names ch67 --device cuda --skip-policy-training

python run_seed_policy_comparison.py --seed 11 --policy-files policies/metadrivech6b.py --policy-names ch6b --device cuda --skip-policy-training

python compare_collision_policies.py --seed 67 --rmst-tau 500 --block-size 25 --output-dir comparisons/seed_67/all_policies_timed

## policy Run Command 

python -u policies/metadrivech67.py --seed 11 --test-seed 100000 --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda

python -u policies/metadrivech6b.py --seed 11 --test-seed 100000 --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda

PYTHONHASHSEED=67 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u policies/metadrivech6b.py --seed 67 --test-seed 100000 --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda & PYTHONHASHSEED=67 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u policies/metadrivech67.py --seed 67 --test-seed 100000 --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda & wait

## Command to train base line 

PYTHONHASHSEED=67 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u train_canonical_baselines.py --seed 67 --methods epsilon noisy rnd --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda --output-root canonical_baselines_timed

PYTHONHASHSEED=27 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u train_canonical_baselines.py --seed 27 --methods epsilon noisy rnd --train-episodes 500 --test-episodes 300 --max-episode-steps 500 --device cuda --output-root canonical_baselines_timed