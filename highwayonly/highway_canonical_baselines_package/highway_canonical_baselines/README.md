# HighwayEnv canonical baselines and safety-pool policies

This package is the HighwayEnv version of the supplied framework. It keeps the
active DQN, replay, training/testing, comparison, safety-pool, mask, retirement,
and output logic intact. Only the environment integration and environment
metadata were changed.

Both included policies are wired directly to `HighwayEnvAdapter`; neither has
a MetaDrive fallback or a custom-environment override.

## What maps to HighwayEnv

| Existing framework contract | HighwayEnv implementation |
|---|---|
| Nine discrete actions | Built-in `DiscreteAction` with 3 acceleration choices × 3 steering choices |
| Flattened observation | Fixed `(15, 6)` normalized Kinematics observation, flattened by the existing code |
| `--map-blocks` | Number of highway lanes, with a minimum of 2 |
| `--traffic-density` | Road vehicle count: `round(250 × density)` |
| `--max-episode-steps` | HighwayEnv episode duration |
| Collision/out-of-road termination | Native HighwayEnv state, normalized to the existing result keys |
| Collision/out-of-road penalties | Existing configured penalties are applied once by the adapter |
| Direct engine safety scan | HighwayEnv road vehicles exposed through a compatibility view |

Highway driving has no destination-arrival event, so `success_reward` has no
direct equivalent. `accident_prob` is retained in saved configurations for
cross-method compatibility but is not applied by HighwayEnv.

## Nine-action ordering

The action ID is a pair `(acceleration, steering)` where each value is low,
neutral, or high:

| ID | Acceleration | Steering |
|---:|---|---|
| 0 | brake | left |
| 1 | brake | straight |
| 2 | brake | right |
| 3 | neutral | left |
| 4 | neutral | straight |
| 5 | neutral | right |
| 6 | accelerate | left |
| 7 | accelerate | straight |
| 8 | accelerate | right |

The policies still see action IDs `0..8`; therefore action masks and per-action
pool evidence retain the same shape and complexity.

## Install and verify

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python smoke_test_highway.py
```

The smoke test creates real `highway-v0` environments, checks deterministic
seeded resets, exercises every action, and verifies the lane/vehicle safety view.

## Train canonical baselines

```bash
python train_canonical_baselines.py \
  --seed 5 \
  --output-root canonical_baselines
```

This trains the same Epsilon-Greedy, NoisyNet, and DQN+RND methods and writes the
same train/test result contract beneath `canonical_baselines/seed_5/`.

The `_ark1`, `_ark2`, and `_ark3` trainer variants are also included and use the
same adapter.

## Run one advanced policy

```bash
PYTHONHASHSEED=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -u policies/Karthikeya27adv23.py \
  --seed 5 \
  --test-seed 100000 \
  --train-episodes 500 \
  --test-episodes 300 \
  --max-episode-steps 500 \
  --device cuda \
  --output-root policy_results
```

The other included Highway policy is `policies/Karthikeya27adv8956.py`. Its
internal policy and output name are also `Karthikeya27adv8956`.

## Train and compare in one command

```bash
python run_seed_policy_comparison.py \
  --seed 5 \
  --policy-files policies/Karthikeya27adv23.py \
  --train-baselines-if-missing \
  --train-episodes 500 \
  --test-episodes 300 \
  --max-episode-steps 500 \
  --test-seed 100000 \
  --device cuda
```

To run and compare both included policies sequentially, supply both paths:

```bash
python -u run_seed_policy_comparison.py \
  --seed 5 \
  --policy-files policies/Karthikeya27adv8956.py policies/Karthikeya27adv23.py \
  --train-baselines-if-missing \
  --device cuda \
  --comparison-name all_highway_policies
```

Outputs use `canonical_baselines/`, `policy_results/`, and `comparisons/`.

## Fair-comparison boundary

Within this package, all policies use the same HighwayEnv adapter, observation
shape, action grid, seeds, and episode limits. The canonical baselines use only
the flattened Kinematics observation. Advanced policies additionally scan the
environment's road-vehicle objects for their safety vector, and their manifests
truthfully retain `uses_engine_object_safety_scan: true`.

Compare methods only against the HighwayEnv canonical baselines produced by
this package; results from other simulators are not on the same reward or road
distribution.

## Adapter location

`highway_env_adapter.py` is the only environment compatibility layer. An
identical copy is present in `policies/` so a policy can be launched directly as
`python policies/<name>.py` without modifying `PYTHONPATH`.
