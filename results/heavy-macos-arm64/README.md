# Recorded HTTP batching experiment

These are actual local HTTP/ORT measurements, not simulated timings. No general speedup claim is made.

Counts include all trials; rejection percentage is total rejected / total offered. Throughput and latency cells are medians of per-trial statistics; arrival-lag p99 is the worst trial; RSS is the maximum paired sampled process sum. Rejection/timeout counts must be considered with successful-only latency.

| Offered req/s | Mode | Success/offered | Rejected | Rejected % | Timeout | Success/s | Dispatch p50 ms | p95 ms | p99 ms | Scheduled p95 ms | Worst header lag p99 ms | Worst admission lag p99 ms | HTTP+worker RSS MiB | Max RSS gap ms | Backend+IPC busy | Mean batch rows | Arrival check |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500.0 | single | 24000/24000 | 0 | 0.00% | 0 | 499.9 | 0.816 | 1.355 | 1.993 | 1.524 | 0.630 | 1.005 | 173.44 | 27.164 | 21.2% | 1.00 | 4/4 pass |
| 500.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 499.8 | 2.890 | 3.800 | 4.559 | 4.001 | 0.718 | 1.173 | 164.81 | 38.552 | 17.6% | 1.97 | 4/4 pass |
| 3500.0 | single | 23999/24000 | 1 | 0.00% | 0 | 3496.8 | 0.482 | 1.973 | 7.244 | 2.427 | 7.980 | 8.103 | 174.36 | 21.683 | 75.7% | 1.00 | 4/4 pass |
| 3500.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 3493.4 | 1.918 | 2.699 | 2.928 | 2.776 | 4.820 | 4.891 | 168.58 | 19.752 | 15.7% | 8.04 | 4/4 pass |
| 6000.0 | single | 16638/24000 | 7362 | 30.68% | 0 | 4115.1 | 15.534 | 16.739 | 19.660 | 17.047 | 14.984 | 15.093 | 171.45 | 22.910 | 96.0% | 1.00 | 4/4 pass |
| 6000.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 5975.6 | 1.854 | 2.758 | 3.180 | 3.385 | 17.604 | 17.833 | 168.72 | 22.108 | 18.8% | 13.04 | 4/4 pass |
| 10000.0 | single | 8872/24000 | 15128 | 63.03% | 0 | 3584.6 | 17.361 | 22.696 | 30.560 | 36.836 | 24.848 | 24.981 | 170.30 | 25.076 | 96.0% | 1.00 | 4/4 pass |
| 10000.0 | dynamic | 24000/24000 | 0 | 0.00% | 0 | 9801.3 | 2.163 | 2.847 | 3.872 | 19.067 | 160.886 | 161.062 | 168.61 | 23.373 | 26.7% | 15.85 | 3/4 pass |

All-pair descriptive throughput ratio is dynamic / single: median [min, max]. Deltas are medians of paired differences, not differences of pooled rates. Every prespecified pair is included. Any arrival-warning pair makes the case inconclusive for a matched-arrival benefit claim. Passing is a post-run outcome that may depend on mode/performance: selecting only passing pairs can introduce selection bias; no pass-only effect estimate is endorsed. All pairs passing the coarse check still does not establish causality or statistical significance. Successful-only latency has survivor bias; consider all rejection/timeout counts.

| Offered req/s | All-pair descriptive ratio | Rejection delta pp | Scheduled success p95 delta ms | Passing arrival pairs | Assessment |
|---:|---:|---:|---:|---:|---|
| 500.0 | 1.000 [1.000, 1.000] | +0.00 | +2.503 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |
| 3500.0 | 0.999 [0.999, 1.000] | +0.00 | +0.384 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |
| 6000.0 | 1.452 [1.449, 1.472] | -30.48 | -13.661 | 4/4 | descriptive_only_all_pairs_pass_coarse_check |
| 10000.0 | 2.719 [2.325, 2.763] | -62.96 | -9.898 | 3/4 | inconclusive_arrival_fidelity |

Fixed policy: batch 1 / 0 ms versus batch 16 / 2 ms; equal capacity 64 counts waiting and physically running requests. Request deadline 250 ms. Four alternating fresh-process trial pairs are recommended; inspect the recorded repeat count.

The same seeded arrival/input trace is replayed for both modes. Actual header-send and admission lag is shown. Passing the fixed 25 ms p99 arrival check does not prove identical actual arrivals or burstiness at sub-millisecond spacing. Scheduled latency includes generator delay; dispatch latency does not. Arrival warnings preclude a clean comparison.

Model: digits-wide-mlp, 4,274,186 parameters, held-out reference accuracy 444/450. This is a deliberately chosen systems workload, not an efficient-classifier recommendation or evidence for all networks/accelerators. Backend+IPC busy is dispatch-to-result wall time divided by the observation interval; it is not CPU utilization. Paired RSS includes HTTP, runtime, IPC, diagnostic serialization and audit overhead; it excludes the generator and multiprocessing resource tracker and can double-count shared pages. Small RSS differences do not establish a memory-efficiency ranking. Actual sampling gaps are shown; short peaks can be missed.

- [Raw JSON](report.json)
- [Offline HTML](report.html)
- [Methodology](../../docs/METHODOLOGY.md)
