# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Counts include all trials; rejection percentage is total rejected / total offered. Throughput and latency cells are medians of per-trial statistics; arrival-lag p99 is the worst trial; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Rejected % | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Worst header lag p99 ms | Worst admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Backend+IPC busy | Mean batch rows | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500.0 | single | 24000/24000 | 0 | 0.00% | 0 | 500.0 | 0.813 | 1.167 | 1.843 | 1.323 | 0.626 | 1.006 | 139.91 | 41.190 | 21.1% | 1.00 | 4/4 pass |
| 500.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 499.9 | 2.746 | 3.585 | 3.902 | 3.701 | 3.178 | 4.253 | 136.94 | 37.519 | 16.7% | 1.96 | 4/4 pass |
| 3500.0 | single | 24000/24000 | 0 | 0.00% | 0 | 3496.7 | 0.494 | 1.609 | 6.565 | 1.782 | 5.026 | 5.199 | 145.53 | 19.129 | 76.4% | 1.00 | 4/4 pass |
| 3500.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 3493.6 | 1.927 | 2.720 | 3.326 | 2.819 | 17.693 | 18.004 | 137.25 | 20.670 | 16.0% | 8.05 | 4/4 pass |
| 6000.0 | single | 14881/24000 | 9119 | 38.00% | 0 | 3874.7 | 15.232 | 24.223 | 37.757 | 34.645 | 71.413 | 71.643 | 142.73 | 42.990 | 96.1% | 1.00 | 2/4 pass |
| 6000.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 5975.4 | 1.854 | 2.734 | 3.077 | 3.359 | 11.613 | 11.755 | 139.73 | 19.397 | 18.4% | 13.04 | 4/4 pass |
| 10000.0 | single | 8449/24000 | 15551 | 64.80% | 0 | 3061.7 | 18.441 | 41.183 | 52.548 | 96.274 | 554.300 | 554.553 | 141.53 | 32.705 | 95.5% | 1.00 | 1/4 pass |
| 10000.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 9881.8 | 2.157 | 2.636 | 3.450 | 18.708 | 719.249 | 719.404 | 137.81 | 32.519 | 25.9% | 15.92 | 3/4 pass |

Paired throughput ratio is dynamic / single: median [min, max]. Deltas are medians of paired differences, not differences of pooled rates. Positive deltas mean dynamic has more rejections or worse successful-request p95. Every trial, including arrival warnings, is included; passing pairs count both modes passing the coarse arrival check. Survivor bias applies under overload; do not compare latency without outcome rates.

| Offered req/s | Throughput ratio | Rejection delta pp | Scheduled success p95 delta ms | Passing arrival pairs |
|---:|---:|---:|---:|---:|
| 500.0 | 1.000 [1.000, 1.000] | +0.00 | +2.346 | 4/4 |
| 3500.0 | 0.999 [0.998, 1.000] | +0.00 | +1.300 | 4/4 |
| 6000.0 | 1.557 [1.400, 2.407] | -34.56 | -31.330 | 2/4 |
| 10000.0 | 2.858 [2.654, 3.428] | -64.43 | -42.910 | 1/4 |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.

Model: digits-wide-mlp, 4,274,186 parameters, held-out reference accuracy 444/450. This is a deliberately chosen systems workload, not an efficient-classifier recommendation or evidence for all networks/accelerators. Backend+IPC busy is dispatch-to-result wall time divided by the observation interval; it is not CPU utilization. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
