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

Only the fixed `[N,64] float32 → [N,10] probabilities` MVP contract is supported. The supplied frozen Digits model has a 1,347/450 training/test split and reference accuracy 436/450. There is no new training or test-based model selection. Runtime success means agreement with the reference, not that every digit is classified correctly. The local split does not establish writer-independent accuracy.

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

```sh
python tools/benchmark.py --output results/local/experiment
python tools/check_project.py --directory results/local/experiment
```

The report presents success throughput, successful-request dispatch p50/p95/p99, scheduled-arrival p95, all terminal outcomes and latencies, queue times, batch distributions and near-contemporaneous HTTP-plus-worker RSS. Scheduled latency includes generator delay; dispatch latency does not. Header-send/admission lag is shown alongside latency. Passing the fixed 25 ms p99 arrival check is a coarse sanity check, not proof of identical delivered traffic at sub-millisecond spacing. Rejections/timeouts remain visible: dropping work is not presented as a speedup. Reference validation is outside the latency timer but affects generator scheduling. Measurements are descriptive, warm, single-machine observations; startup, caches and thermal state are not controlled.

Memory reports include the actual maximum sampling gap and paired-read duration. HTTP-plus-worker RSS includes audit/statistics serialization overhead but excludes the load generator and multiprocessing resource-tracker helper; it is not total deployment memory. Shared pages can be counted twice and short peaks can be missed. Small RSS differences do not justify an efficiency ranking.

The HTML is a compact summary with per-trial details and links to the complete raw JSON and retained fixture in the same folder. Keep these files together for offline inspection. Raw JSON is compact to avoid duplicating tens of megabytes of whitespace; use a JSON viewer or `python -m json.tool` to inspect it. GitHub does not host the HTML as a running service.

## Verify

```sh
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests
python tools/run_mutations.py
python tools/check_project.py
```

Deterministic scheduler tests use a controllable clock, including delayed deadline callbacks and a tail waiting behind a busy worker. HTTP tests run the actual service and ORT worker; controlled in-flight tests gate the real backend, then execute actual inference after cancellation/timeout. A mixed three-row batch checks one success, one timeout and one cancellation, followed by a one-row tail, with exact outcome/computed/discarded counts. Such gates are test controls and are never used in performance measurements. Seven deliberate source faults must fail behavioral assertions, not merely import or source-hash checks. The evidence checker also rejects contradictory request/batch times, forged occupancy, and RSS samples outside their recorded observation window. CI runs Linux/macOS correctness and real HTTP timing smoke; smoke timings are not performance gates.

[Architecture and methodology](docs/METHODOLOGY.md) · [Third-party attribution](THIRD_PARTY_NOTICES.md)

No project license has been selected for the newly written service code. The reused frozen model retains its original MIT notice; dataset-derived values retain applicable CC BY 4.0 terms. No license from a sibling project is implicitly assigned to the new code.
