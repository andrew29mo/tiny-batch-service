# Tiny Batch Service

A local dynamic-batching inference service with bounded admission, explicit request lifecycles, and reproducible HTTP measurements.

The experiment asks: **When do batching gains justify the extra waiting time under the same scheduled arrival traffic?** It wraps ONNX Runtime CPU; it does not implement a new inference engine or claim a universal speedup.

## MVP

- Loopback HTTP process plus one spawned ORT worker, one batch in flight, FIFO scheduling.
- Dispatch at maximum batch size or the oldest request's batching-wait deadline. Partial final batches are never padded or discarded.
- Equal admission capacity across modes counts queued and physically running requests, including expired/cancelled work that is still computing.
- Queued cancellations/timeouts release capacity without computing. Running ones resolve the request but retain capacity until computation returns; native computation is not interrupted.
- Explicit overload rejection, duplicate-ID detection within bounded audit retention, disconnect cancellation, shutdown resolution and fail-closed worker errors.
- Independent sklearn/NumPy reference checks, raw client/server records, per-batch IDs, memory samples, JSON and offline HTML reports.

## Run

Python 3.12 on macOS/Linux:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/make_model.py
tiny-batch-service examples/generated/digits/model.onnx \
  --batch-size 16 --wait-ms 2 --capacity 64 --port 8080
```

Only the fixed `[N,64] float32 → [N,10] probabilities` contract is supported. Two frozen Digits models share the same 1,347/450 training/test split: the original 64→32→10 MLP (2,410 parameters, reference accuracy 436/450), and a 64→1024→2048→1024→10 MLP (4,274,186 parameters, reference accuracy 444/450). The frozen wider model uses a prespecified 60-epoch protocol with only training IDs; no held-out fitting, early stopping, architecture search or accuracy-based model selection occurs. Review separately refit that formal path and reproduced all eight frozen parameter arrays bit for bit on the same environment; this is not a cross-platform training guarantee. Frozen weights are included; normal preparation and CI do not retrain. Runtime success means agreement with the reference, not that every digit is classified correctly. The local split does not establish writer-independent accuracy. The wider network is a deliberate systems workload, not a recommended efficient Digits classifier or a proxy for every production model.

To serve the heavier model:

```sh
python tools/make_heavy_model.py
tiny-batch-service examples/generated/heavy/model.onnx \
  --batch-size 16 --wait-ms 2 --capacity 64 --port 8080
```

## HTTP contract

`POST /v1/infer` accepts exactly `id`, `input` (64 finite numbers, absolute value ≤1000) and `timeout_ms` (0–60000, exclusive of zero). IDs use 1–64 ASCII letters/digits/underscore/hyphen. The timeout begins at validated server admission; it is not a guaranteed client end-to-end deadline.

| Outcome | HTTP | Behavior |
|---|---:|---|
| Success | 200 | Matching ID, ten probabilities and predicted class |
| Invalid input | 400 | Rejected before admission |
| Body too large | 413 | Request body exceeds 8 KiB |
| Duplicate recent/active ID | 409 | Original request is preserved |
| Cancelled | 409 | No successful prediction returned |
| Capacity full | 429 | Immediate rejection after input validation |
| Request deadline | 504 | Queued work removed; running computation drains |
| Shutdown/worker failure | 503 | Pending requests resolve; failed worker is not silently restarted |

`DELETE /v1/requests/{id}` explicitly cancels an outstanding request. It returns 200 when cancellation changes its state, 409 when already terminal and still recorded, or 404 when unknown/evicted. Client disconnection also cancels an outstanding waiter. `GET /health` exposes readiness; `GET /v1/stats` exposes a bounded diagnostic audit. IDs may be reused after their terminal audit record is evicted; this is not durable idempotency storage.

The service has no authentication or tenant isolation and binds loopback only. Other local users/processes can access its endpoints; use trusted models and inputs. Admission bounds inference work, not all possible TCP connections or HTTP transport allocations. Request bodies are limited to 8 KiB. See [security scope](SECURITY.md).

## Measurements

[Recorded experiment and raw reports](results/macos-arm64/README.md) compare batch 1 / 0 ms with batch 16 / 2 ms at equal total capacity 64. Four fresh-process pairs alternate mode order at each offered rate. Each mode replays the same seeded schedule and inputs. Actual header-send/server-arrival lag is retained; excessive lag produces an arrival-fidelity warning.

On the recorded tiny-model workload, all requests succeeded at the tested offered rates and dynamic batching increased latency without a throughput gain. This measures behavior at those offered loads, not maximum sustainable capacity; it does not rule out benefits for heavier models or other traffic patterns.

[Heavier-model saturation/overload experiment](results/heavy-macos-arm64/README.md) uses 500, 3,500, 6,000 and 10,000 scheduled requests/s, 6,000 requests per trial, and four alternating trial pairs. Rates were frozen after a retained exploratory sweep, before the repeated experiment. These cover low load, a candidate near single-request saturation, single-request overload, and higher load. Observed rejection and backend/IPC occupancy determine the interpretation; a rate name alone does not prove saturation. The exploratory 15k/20k probes exceeded the fixed arrival-lag budget and cannot establish the dynamic service's maximum capacity. Their raw results remain available. No sleep, duplicated model operations or gates are used to manufacture compute cost.

**The earlier approximately 1.40× benefit claim is withdrawn as a confirmed effect estimate.** It described only the 2/4 passing pairs at 6k in commit `633eec0`. Arrival-check outcome is measured after execution and may depend on mode/performance; conditioning on that outcome can introduce selection bias. The original all-pair median was 1.557×, but mixing warning trials does not produce a trustworthy replacement effect estimate either. The [immutable original-data audit](reviews/heavy-633eec0.html) rechecks all 192,000 requests, retains every pair and documents the same-environment training reproduction. Historical numbers are not relabeled as measurements of the updated implementation.

Current JSON/HTML reports assess **every prespecified pair**. If any pair has an arrival warning, that rate is `inconclusive_arrival_fidelity`; no passing-only aggregate is endorsed. Even all pairs passing the coarse 25 ms check supports only descriptive observations, not causal, statistical-significance or universal crossover claims. Rejection, timeout, latency and memory remain visible for all trials. A higher descriptive ratio under distorted traffic does not establish a reliable batching speedup or the dynamic service's maximum capacity.

```sh
python tools/benchmark.py --output results/local/experiment
python tools/check_project.py --directory results/local/experiment
python tools/benchmark.py --profile heavy --output results/local/heavy-experiment
python tools/check_project.py --directory results/local/heavy-experiment
```

The report presents success throughput, successful-request dispatch p50/p95/p99, scheduled-arrival p95, all terminal outcomes and latencies, queue times, batch distributions and near-contemporaneous HTTP-plus-worker RSS. Scheduled latency includes generator delay; dispatch latency does not. Header-send/admission lag is shown alongside latency. Passing the fixed 25 ms p99 arrival check is a coarse sanity check, not proof of identical delivered traffic at sub-millisecond spacing. Rejections/timeouts remain visible: dropping work is not presented as a speedup. Reference validation is outside the latency timer but affects generator scheduling. Measurements are descriptive, warm, single-machine observations; startup, caches and thermal state are not controlled.

Memory reports include the actual maximum sampling gap and paired-read duration. HTTP-plus-worker RSS includes audit/statistics serialization overhead but excludes the load generator and multiprocessing resource-tracker helper; it is not total deployment memory. Shared pages can be counted twice and short peaks can be missed. Small RSS differences do not justify an efficiency ranking.

Each current worker loads ONNX from one byte snapshot and reports its digest; the benchmark reads its NPZ input/reference archive from one byte snapshot and records its digest. Both must match the report's frozen artifacts in every trial. Memory samples must bracket planned time zero through physical compute drain, with a forced final observation. These checks strengthen new measurements; the original experiment and exploratory calibration do not contain retroactively invented attestations.

The byte-snapshot worker supports the bundled self-contained ONNX models; ONNX external-data sidecar models are outside this contract.

Installed ORT retains serialized bytes in its session even after the worker drops its local reference. That real memory cost is included in both modes' RSS. The loader changed in this review, so earlier path-loader RSS and new RSS are different baselines, not evidence of a batching memory gain or regression.

The HTML contains per-trial details and paired throughput ratios alongside rejection rates and latency. Ratios include the range across all pairs, not a confidence interval. Under overload, successful-only latency has survivor bias and must be read with rejection/timeout counts; lower latency obtained by dropping work is not a benefit. Backend+IPC busy fraction is dispatch-to-result wall time, not CPU utilization. These finite traces do not establish a production steady-state capacity/SLO.

The tiny report embeds raw records. The heavier `report.json` indexes hash-linked `raw/*.json.gz` files containing every request, server terminal, batch and RSS observation; `calibration.json.gz` retains every exploratory probe. Keep the complete folder for offline verification. Gzip reduces storage only; observations are not sampled or discarded. GitHub does not host HTML as a running service.

## Verify

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests
python tools/run_mutations.py
python tools/check_project.py
python tools/check_project.py --directory results/heavy-macos-arm64
python tools/audit_heavy_review.py --verify
```

Deterministic scheduler tests use a controllable clock, including delayed deadline callbacks and a tail waiting behind a busy worker. HTTP tests run the actual service and ORT worker; controlled in-flight tests gate the real backend, then execute actual inference after cancellation/timeout. A mixed three-row batch checks one success, one timeout and one cancellation, followed by a one-row tail, with exact outcome/computed/discarded counts. Such gates are test controls and are never used in performance measurements. Seven deliberate source faults must fail behavioral assertions, not merely import or source-hash checks. The evidence checker also rejects contradictory request/batch times, forged occupancy, and RSS samples outside their recorded observation window. CI runs Linux/macOS correctness and real HTTP timing smoke; smoke timings are not performance gates.

The heavier tests check training-only inputs, exact exported frozen arrays, all 450 held-out HTTP probabilities/classes and tail coverage, plus compressed-record corruption and index mismatch. The verifier reconstructs occupancy and rejects overload records that did not occur at full capacity. Audit retention is 4,096 for tiny trials and 16,384 for longer heavy trials; both modes always use the same limit within an experiment. Admission capacity remains 64.

Optional retraining writes a new local artifact, leaving frozen evidence intact: `python tools/train_heavy_model.py --output examples/generated/retrained-heavy`. Reproducing training on another numerical stack need not produce byte-identical weights; ordinary verification uses the committed frozen arrays.

[Architecture and methodology](docs/METHODOLOGY.md) · [Related work and practical lessons](docs/RELATED_WORK.md) · [Third-party attribution](THIRD_PARTY_NOTICES.md)

## License

Original project contributions, including service code, tools, tests, documentation and newly trained heavy-model weights, are available under the [MIT License](LICENSE), Copyright (c) 2026 Andrew Yuan.

The reused tiny model retains its [upstream MIT notice](LICENSES/inference-doctor-MIT.txt). The Digits dataset and applicable dataset-derived fixture/report values retain their CC BY 4.0 attribution and terms; the project MIT license does not replace those terms. See [third-party materials and data attribution](THIRD_PARTY_NOTICES.md) for sources, modifications and scope. Installed dependencies retain their own licenses.
