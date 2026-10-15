# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Counts include all trials; rejection percentage is total rejected / total offered. Throughput and latency cells are medians of per-trial statistics; arrival-lag p99 is the worst trial; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Rejected % | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Worst header lag p99 ms | Worst admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Backend+IPC busy | Mean batch rows | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100.0 | single | 1920/1920 | 0 | 0.00% | 0 | 100.2 | 1.286 | 2.047 | 2.372 | 2.605 | 2.358 | 2.763 | 104.09 | 15.892 | 4.0% | 1.00 | 4/4 pass |
| 100.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 100.1 | 3.555 | 4.281 | 4.680 | 4.836 | 1.193 | 1.728 | 104.08 | 33.455 | 4.0% | 1.00 | 4/4 pass |
| 1000.0 | single | 1920/1920 | 0 | 0.00% | 0 | 1000.4 | 0.489 | 0.578 | 0.686 | 0.678 | 1.709 | 3.192 | 104.06 | 7.422 | 13.9% | 1.00 | 4/4 pass |
| 1000.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 997.2 | 1.644 | 2.728 | 3.290 | 2.808 | 0.593 | 0.861 | 103.86 | 5.727 | 6.4% | 2.96 | 4/4 pass |
| 3000.0 | single | 1920/1920 | 0 | 0.00% | 0 | 2996.6 | 0.344 | 0.470 | 1.319 | 0.545 | 1.654 | 1.855 | 103.92 | 5.733 | 27.2% | 1.00 | 4/4 pass |
| 3000.0 | dynamic | 1920/1920 | 0 | 0.00% | 0 | 2985.7 | 1.670 | 2.590 | 2.935 | 2.659 | 0.670 | 0.858 | 103.88 | 5.717 | 5.7% | 7.06 | 4/4 pass |

All-pair descriptive throughput ratio is dynamic / single: median [min, max]. Deltas are medians of paired differences, not differences of pooled rates. Every prespecified pair is included. Any arrival-warning pair makes the case inconclusive for a matched-arrival benefit claim. Passing is a post-run outcome that may depend on mode/performance: selecting only passing pairs can introduce selection bias; no pass-only effect estimate is endorsed. All pairs passing the coarse check still does not establish causality or statistical significance. Successful-only latency has survivor bias; consider all rejection/timeout counts.

| Offered req/s | All-pair descriptive ratio | Rejection delta pp | Scheduled success p95 delta ms | Passing arrival pairs | Assessment |
|---:|---:|---:|---:|---:|---|
| 100.0 | 1.000 [0.999, 1.000] | +0.00 | +2.231 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |
| 1000.0 | 0.997 [0.994, 0.998] | +0.00 | +2.147 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |
| 3000.0 | 0.996 [0.995, 0.997] | +0.00 | +2.114 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.

Model: digits-tiny-mlp, 2,410 parameters, held-out reference accuracy 436/450. This is a deliberately chosen systems workload, not an efficient-classifier recommendation or evidence for all networks/accelerators. Backend+IPC busy is dispatch-to-result wall time divided by the observation interval; it is not CPU utilization. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
