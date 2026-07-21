# Validation report

Conversion validation completed on 2026-07-22.

## Passed static checks

- Every included Python file compiles successfully.
- The baseline trainer, `Karthikeya27adv8956`, and `Karthikeya27adv23` import
  `HighwayEnvAdapter` and contain no MetaDrive import or constructor.
- Both policies save `environment: HighwayEnv` and the installed HighwayEnv
  version in their completion manifests.
- `Karthikeya27adv8956.py` uses `Karthikeya27adv8956` consistently as its
  internal method, model, safety-memory, and output-folder name.
- The root and direct-policy copies of `highway_env_adapter.py` are
  byte-identical.
- Comparison defaults now resolve the same `canonical_baselines/seed_<seed>`
  tree written by the trainer.

## Runtime check before a long campaign

After installing `requirements.txt`, run:

```bash
python smoke_test_highway.py
```

This creates real `highway-v0` environments, verifies deterministic seeded
resets, checks the fixed `(15, 6)` Kinematics observation, exercises all nine
actions, and checks the vehicle/lane safety view. A full 500-episode campaign
is intentionally not bundled as a package validation step.
