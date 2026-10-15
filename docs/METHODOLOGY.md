# Architecture and measurement contract

## Components

The HTTP process validates requests and owns a FIFO ordered queue, a recent-ID/audit map, timer callbacks and one dispatcher task. Its single backend bridge sends work only when the ORT process is idle. IPC is a private multiprocessing pipe; serialization is internal trusted data, not an external pickle API. An asynchronous bridge thread keeps synchronous pipe IO off the HTTP event loop. There is only one exchange at a time, so no hidden executor work queue grows with incoming requests.

The spawned worker initializes the same CPUExecutionProvider session in both modes: one intra-op thread, one inter-op thread, sequential execution and full graph optimization. Runtime fallback retry is disabled. Numerical-library thread environment variables are fixed before its NumPy/ORT imports. It warms model shapes 1 and 32 identically. Every trial then sends eight independently checked HTTP warmup examples before the measured trace.

Both modes share admission, queue, IPC, runtime, response and audit code. Only maximum batch size and batching-wait duration differ. Capacity 64 counts all entries queued or physically running. Thus a cancelled/expired running batch cannot free slots before its computation completes. The latest request to fill the queue cannot reset the oldest wait deadline. Dispatch rechecks request deadlines even if a timer callback was delayed.

## Lifecycle and limits

Queued entries become running, successful, cancelled, timed out, shutdown or failed. Terminal resolution occurs once. Queued terminal entries leave admission immediately; running terminal entries leave only after backend completion. Cancelled/timed-out output is discarded, and computed/discarded row counters remain observable. Shutdown resolves queued entries with 503 and waits for the current worker exchange. Backend timeout/failure marks the service unhealthy and resolves affected work; automatic restart is outside the MVP.

Audit retention defaults to 4,096 request records and 4,096 batches. Active entries are never evicted; terminal history can be. Measurement runs reject evicted evidence and generator-side unsent/transport-error outcomes instead of silently dropping them. The finite HTTP front end is a local experiment, not hardened protection against hostile connection floods, slow clients or arbitrary ONNX execution. Worker response timeout is ten seconds; a stuck worker is terminated rather than hanging shutdown indefinitely.

## Offered traffic and fairness

For each rate, the controller generates a fixed seed-42 list of IDs, indices into 450 held-out inputs and arrival offsets `i/rate`. Every trial of either mode receives the identical list. The generator does not wait for a preceding response before scheduling the next request. It has a separate 512-active-request cap; unsent work is recorded and invalidates the comparison.

Runs alternate single/dynamic then dynamic/single across four rounds. Each run uses fresh HTTP and ORT processes on the same host. Case/rate order is fixed. Host OS scheduling, numerical/runtime caches, thermals and generator/server resource contention remain uncontrolled. No outliers are removed, and no cross-machine or statistical ranking is claimed.

Planned arrivals cannot be perfectly reproduced by a real OS/network. Raw records preserve client dispatch, actual headers-sent, server admission and response-decoded times, relative to a shared host monotonic origin. Per-trial p99 headers-sent and server-admission lag must each be ≤25 ms for the arrival-fidelity check to pass. This tolerance is fixed in the acceptance policy. Warnings mean a clean traffic-matched performance conclusion is not supported, even though the scheduled trace was the same. Raw timing differences remain available rather than being normalized away.

This fixed threshold is only a coarse check: 25 ms is much longer than the interarrival spacing at the higher offered rates. A pass does not establish identical actual arrival timing, interarrival distributions or burstiness. Tables expose actual p99 header/admission lag and scheduled-arrival latency so generator delay is visible. Interpret sub-millisecond rankings cautiously even when both modes pass; this experiment does not demonstrate a causal production performance ranking.

## Metrics

- Client latency: both actual dispatch to fully decoded HTTP response and planned arrival to fully decoded response are retained, before independent output validation. The latter includes generator scheduling delay. Successful-only percentiles and all-terminal percentiles are separate. Output checks still execute for every successful response and can affect subsequent generator scheduling.
- Success throughput: reference-checked successful responses divided by the observation interval from scheduled time zero through the latest of final response decoding, final planned arrival or final physical batch completion. This includes draining computation after request timeout/cancellation; offered/requested rate is reported separately.
- Queuing: validated admission until batch dispatch, when computation occurs. Timeouts that never start are retained as such, not invented as completed queue observations.
- Runtime call time: the worker's synchronous `session.run`, excluding IPC, conversion, Python-list output conversion and validation. Per-batch IPC/dispatch wall time and IDs are also retained; this is distinct from client end-to-end latency.
- Memory: controller samples HTTP and worker RSS in sequence at a requested 5 ms interval, retaining read start/end timestamps and both values. The observation window starts after warmup, before planned time zero, and ends after all responses, physical compute drain and diagnostic stats collection. The verifier checks ordered sample times within that window. Actual sample count, maximum gap (including boundary gaps), window duration and maximum paired-read duration are retained; the table exposes the worst gap. The paired peak is the maximum sum from one sampling observation, not the sum of independent process maxima. Scheduling can be slower, short spikes may be missed, and shared pages may be counted twice. Per-process OS lifetime high-water values are separate evidence and are not added into a supposed simultaneous peak. Sampling includes warm resident allocations, runtime, IPC, audit buffers and diagnostic stats serialization; it excludes the generator/controller, its reference arrays and the multiprocessing resource-tracker helper. It is not total deployment memory; small RSS differences do not support a memory-efficiency ranking.

Tables display medians of per-trial throughput/percentiles and the maximum paired RSS across trials. Counts sum all trials. No speedup headline is automatically generated. Few finite trial samples are descriptive, not confidence intervals or evidence about production service tail behavior.

## Correctness and provenance

The frozen 64→32 ReLU→10 Softmax model and split originate in inference-doctor, with MIT attribution retained. Preparation checks installed public dataset hashes and disjoint training/test IDs, exports the original weights as float32 ONNX, and checks a NumPy float32 computation against frozen sklearn float64 `predict_proba` cast to float32. No model fitting or calibration occurs here. Each HTTP success must match every probability at fixed `rtol=1e-4`, `atol=1e-5` and have the same class and request ID. Agreement is distinct from ground-truth accuracy: the reference model correctly labels 436/450 held-out examples.

The measured fixture is retained losslessly as JSON. Reconstructing its NPZ must reproduce the recorded hash exactly; regeneration on another architecture must match inputs/labels/IDs exactly and references within fixed tolerance. The ONNX model/state hashes must agree. Package source hashes cover every package Python file, including timing, scheduling and report code. CLI scripts and other files remain identified by Git history. JSON records cannot alone prove that past physical hardware events occurred.

The evidence checker revalidates output values, all request IDs, full batch coverage, counts, timings, capacity, process cleanup, repeated-trial order, summary arithmetic and generated HTML. Source mutation tests intentionally avoid recorded-source checks so a changed hash alone does not count as a detected behavioral bug.

Schema 2 additionally reconciles every request's compute completion with its batch, enforces client/header/admission/terminal chronology and non-overlapping batch execution, reconstructs physical occupancy from per-request admission/release times, and checks memory-window chronology. These checks make contradictory evidence detectable; they are not cryptographic proof that a benchmark happened. The full benchmark is rerun when package source changes rather than replacing the source hash in old measurements.
