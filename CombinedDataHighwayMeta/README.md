# SafetyPool collision-RMST paper and reproducibility package

This folder accompanies the revised paper. The paper reports **Collision
Restricted Mean Survival Time (RMST)** only. Highway maximum-step completion
and Highway collisions per 1,000 steps are intentionally not reported.

## Contents

- `paper/`: revised PDF, Word document, and LaTeX source.
- `policy/`: the exact SafetyPool policy implementations used for MetaDrive
  and HighwayEnv.
- `framework/`: simulator adapters, baseline trainers, comparison scripts,
  analysis code, requirements, and environment records.
- `data/highway/`: all 300 Collision-RMST test episodes for every method in
  every selected Highway seed, plus seed-level and aggregate analyses.
- `data/metadrive/`: the collision-RMST-only seed-level and aggregate data
  underlying the MetaDrive result.
- `figures/`: paper figures in PDF and PNG.
- `recomputed/`: verification output regenerated from the packaged analysis
  script and collision-only episode data.

## Highway Collision RMST result

The matched 23-seed subset is:

`3, 5, 7, 9, 13, 17, 27, 33, 38, 42, 45, 48, 54, 63, 65, 67, 69, 73, 76, 77, 93, 108, 111`

Collision RMST IQM values at the 500-step censoring horizon:

- SafetyPool: 122.15
- Epsilon-Greedy DQN: 5.79
- NoisyNet DQN: 5.68
- DQN + RND: 6.44

The analysis uses `scipy.stats.trim_mean(values, 0.25)` for the seed-level IQM,
50,000 deterministic seed-bootstrap resamples for 95% intervals, and paired
two-sided Wilcoxon signed-rank tests with Holm correction.

## Reproduce the analysis

From `framework/analysis/`:

```bash
python compute_collision_rmst.py
```

The supplied script reads the collision-only episode file and regenerates the
Highway seed-level RMST table, aggregate IQM table, paired tests, and figure.

## Scope note

The Highway result is explicitly labeled as the matched 23-seed subset
analysis. Broader all-seed sensitivity analysis, alternative subset rules, and
additional driving outcomes are future work.
