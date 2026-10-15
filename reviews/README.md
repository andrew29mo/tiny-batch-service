# Review of the original heavy experiment

[JSON](heavy-633eec0.json) and [HTML](heavy-633eec0.html) audit the original experiment from immutable commit [633eec0](https://github.com/andrew29mo/tiny-batch-service/tree/633eec023aaf1600807433df7d5a20e3eaac439e). They are separate from the newly measured schema-4 reports.

The approximately 1.40× statement described only two arrival-check-passing pairs. Passing is a post-execution outcome that may depend on performance; selecting on it can introduce bias. It is withdrawn as a confirmed effect estimate. The audit retains all four pairs at every rate and recomputes every original successful prediction, outcome, timing/RSS statistic and compressed-record hash. The all-pair median is descriptive; it cannot repair distorted arrivals either.

An actual same-environment refit of the formal training-only path reproduced all eight frozen arrays bit for bit. Its elapsed time and exact-array observations are retained in the JSON. Routine checks do not rerun fitting or assert cross-platform training identity.

Verification requires the original Git commit's objects (a full clone; CI fetches history):

```sh
python tools/audit_heavy_review.py --verify
```

The checker reopens the original committed index, fixture and every raw trial through Git, instead of treating the updated working tree as historical evidence. The historical experiment lacked loaded-worker/NPZ snapshot attestations; the audit states that limitation explicitly. Current reports add those attestations, full memory bracketing and all-pair assessment; historical records are not backfilled with invented fields.
