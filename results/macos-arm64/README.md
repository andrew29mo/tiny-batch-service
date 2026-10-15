# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Success counts include all trials. Throughput and latency cells are medians of per-trial statistics; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Header lag p99 ms | Admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100.0 | single | 1920/1920 | 0 | 0 | 100.2 | 1.185 | 1.806 | 2.301 | 2.327 | 1.029 | 1.562 | 103.86 | 17.692 | pass |
| 100.0 | dynamic | 1920/1920 | 0 | 0 | 100.1 | 3.416 | 3.911 | 4.353 | 4.446 | 1.025 | 1.506 | 104.05 | 25.357 | pass |
| 1000.0 | single | 1920/1920 | 0 | 0 | 1000.6 | 0.438 | 0.540 | 0.828 | 0.654 | 0.313 | 0.470 | 103.95 | 5.729 | pass |
| 1000.0 | dynamic | 1920/1920 | 0 | 0 | 995.6 | 1.591 | 2.708 | 2.879 | 2.797 | 0.390 | 0.593 | 103.58 | 5.739 | pass |
| 3000.0 | single | 1920/1920 | 0 | 0 | 2996.0 | 0.367 | 0.641 | 1.440 | 0.715 | 0.780 | 1.101 | 103.72 | 5.735 | pass |
| 3000.0 | dynamic | 1920/1920 | 0 | 0 | 2969.4 | 1.733 | 2.726 | 3.151 | 2.838 | 0.754 | 0.976 | 103.92 | 5.726 | pass |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.

The loaded model is a tiny frozen Digits MLP. This experiment does not establish throughput or latency for larger networks, accelerators or distributed deployments. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
