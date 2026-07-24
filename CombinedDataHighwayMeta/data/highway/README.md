# Highway Collision RMST data

This directory contains the complete collision-event test data used by the
paper's selected 23-seed Highway analysis: 23 seeds × 4 methods × 300 test
episodes = 27,600 episode rows.

The episode CSVs contain only fields needed to audit Collision RMST: method,
training seed, episode, matched scenario seed, initial-observation checksum,
collision event indicator, event-or-censor time, censoring horizon, and frozen
test-policy checks.

The authoritative user-supplied archive is `highway(9).zip`. Its SHA-256 and
the seed/data checks are recorded in `DATA_VALIDATION.json`.
