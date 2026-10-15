# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Success counts include all trials. Throughput and latency cells are medians of per-trial statistics; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Timeout | Success/s | p50 ms | p95 ms | p99 ms | Paired RSS MiB | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100.0 | single | 1920/1920 | 0 | 0 | 100.2 | 1.106 | 1.800 | 2.127 | 103.95 | pass |
| 100.0 | dynamic | 1920/1920 | 0 | 0 | 100.1 | 3.384 | 4.070 | 4.437 | 103.94 | pass |
| 1000.0 | single | 1920/1920 | 0 | 0 | 1000.3 | 0.488 | 0.577 | 0.730 | 103.69 | pass |
| 1000.0 | dynamic | 1920/1920 | 0 | 0 | 997.9 | 1.618 | 2.726 | 2.844 | 103.70 | pass |
| 3000.0 | single | 1920/1920 | 0 | 0 | 2996.6 | 0.346 | 0.463 | 1.209 | 103.91 | pass |
| 3000.0 | dynamic | 1920/1920 | 0 | 0 | 2980.9 | 1.697 | 2.621 | 2.905 | 103.70 | pass |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and server-arrival lag is retained. Arrival warnings mean delivered traffic was not close enough to the planned schedule for a clean comparison.

The loaded model is a tiny frozen Digits MLP. This experiment does not establish throughput or latency for larger networks, accelerators or distributed deployments. Paired RSS includes HTTP, runtime, IPC and audit overhead; it excludes the generator and can double-count shared pages.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
