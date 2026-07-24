# SafetyPool policy implementations

This directory intentionally contains both policy forms:

```text
policies/
├── safetypool.py                  # unchanged 7,927-line archival source
├── safetypool_modular.py          # same-CLI modular launcher
└── safetypool_components/         # reusable implementation package
```

`safetypool.py` is retained unchanged for exact experimental provenance. Its
SHA-256 is:

```text
e0a3e38f4a277c4b58e8eccdd2f746db72bf557e9783d2f522c1b5e5ef12e890
```

For new work, use:

```bash
python -u policies/safetypool_modular.py \
    --seed 17 \
    --test-seed 100000 \
    --train-episodes 500 \
    --test-episodes 300 \
    --max-episode-steps 500 \
    --device cuda
```

The arguments, default policy name, output folder, metrics, manifests, and raw
result schemas remain compatible with the monolithic implementation.

See `01_documentation/MODULAR_POLICY_ARCHITECTURE.md` for the responsibility
map, dependency flow, extension guidance, and verification details.

