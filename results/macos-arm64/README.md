# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Counts include all trials; rejection percentage is total rejected / total offered. Throughput and latency cells are medians of per-trial statistics; arrival-lag p99 is the worst trial; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Rejected % | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Worst header lag p99 ms | Worst admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Backend+IPC busy | Mean batch rows | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100.0 | single | 1920/1920 | 0 | 0.00% | 0 | 100.2 | 1.211 | 1.888 | 2.195 | 2.430 | 1.023 | 1.529 | 104.20 | 21.636 | 3.8% | 1.00 | 4/4 pass |
| 100.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 100.1 | 3.388 | 4.148 | 4.585 | 4.707 | 1.646 | 2.504 | 103.94 | 33.212 | 3.9% | 1.00 | 4/4 pass |
| 1000.0 | single | 1920/1920 | 0 | 0.00% | 0 | 1000.4 | 0.477 | 0.558 | 0.650 | 0.663 | 0.262 | 0.398 | 103.70 | 5.734 | 13.4% | 1.00 | 4/4 pass |
| 1000.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 995.2 | 1.588 | 2.683 | 2.828 | 2.759 | 0.477 | 0.754 | 103.56 | 5.897 | 5.8% | 2.97 | 4/4 pass |
| 3000.0 | single | 1920/1920 | 0 | 0.00% | 0 | 2996.9 | 0.351 | 0.486 | 1.351 | 0.579 | 0.670 | 0.906 | 104.22 | 5.737 | 27.7% | 1.00 | 4/4 pass |
| 3000.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 2971.0 | 1.691 | 2.609 | 2.900 | 2.693 | 0.663 | 0.880 | 103.83 | 5.843 | 6.3% | 7.01 | 4/4 pass |

Paired throughput ratio is dynamic / single: median [min, max]. Deltas are medians of paired differences, not differences of pooled rates. Positive deltas mean dynamic has more rejections or worse successful-request p95. Every trial, including arrival warnings, is included; passing pairs count both modes passing the coarse arrival check. Survivor bias applies under overload; do not compare latency without outcome rates.

| Offered req/s | Throughput ratio | Rejection delta pp | Scheduled success p95 delta ms | Passing arrival pairs |
|---:|---:|---:|---:|---:|
| 100.0 | 0.999 [0.999, 1.000] | +0.00 | +2.333 | 4/4 |
| 1000.0 | 0.995 [0.991, 0.998] | +0.00 | +2.087 | 4/4 |
| 3000.0 | 0.992 [0.984, 0.997] | +0.00 | +2.108 | 4/4 |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.

Model: digits-tiny-mlp, 2,410 parameters, held-out reference accuracy 436/450. This is a deliberately chosen systems workload, not an efficient-classifier recommendation or evidence for all networks/accelerators. Backend+IPC busy is dispatch-to-result wall time divided by the observation interval; it is not CPU utilization. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
