# Related work and practical lessons

Primary sources checked on 2026-10-15. These are design and measurement references, not benchmarks against the systems below. No Triton, Ray Serve or wrk2 comparison was executed. Their current documentation may describe versions newer than this project's pinned dependencies.

## Batching trades waiting time for execution efficiency

[NVIDIA Triton's dynamic batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) documents dispatch on a full batch or a bounded queue delay, with a partial batch sent when the delay expires. Its tuning procedure measures latency/throughput and increases size or delay within a latency budget. Queue size and queue timeout are separate policy settings.

Here, the oldest admission fixes the batching-wait deadline; later requests do not restart it. Tests exercise partial tails, busy-worker tails and delayed deadline callbacks against the production scheduler. Equal capacity counts both queued and physically running rows, a stricter boundary than a queue-only limit. The batching wait is not a guaranteed end-to-end latency bound when the worker is busy. Our fixed 16-row/2 ms policy is one experiment, not an optimal configuration or a production recommendation.

[Ray Serve's dynamic batching guide](https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html) similarly separates maximum size, wait timeout and concurrent batches. It requires output cardinality to match the input list so responses can be routed back to their callers. This project uses one in-flight batch and verifies every returned row against an independent reference, including mixed success/timeout/cancellation batches. Variable-cost inputs and concurrent workers would need a different capacity model; they are outside this MVP.

## Cancellation is not automatic interruption of native computation

[Triton's cancellation documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/request_cancellation.html) describes cancellation checks in scheduling and backend cooperation after dispatch. [Issue #8030](https://github.com/triton-inference-server/server/issues/8030) is a public example of users asking about that support; the official guide is the reference for its scope.

Our ORT call is not interrupted on a client deadline or cancellation. The HTTP outcome resolves, but physically running rows retain capacity until completion and their outputs are discarded. Tests hold then release the real backend and verify actual computation, capacity rejection and exact discarded-row accounting. Logical cancellation alone must not be counted as freed compute or completed successful work.

## A fixed planned rate does not guarantee faithful delivered traffic

[wrk2's measurement explanation](https://github.com/giltene/wrk2#readme) discusses coordinated omission: measuring only from actual send time can hide load-generator delay. It measures latency from intended send time and also documents OS timing granularity and calibration limitations.

Our open-loop controller retains planned arrivals, dispatch, headers sent, server admission and response decoding. Both scheduled-arrival and dispatch latency are reported. Unsent requests invalidate a complete comparison. The 25 ms p99 arrival check is only a coarse warning mechanism; it is much longer than interarrival spacing at the higher rates. Every prespecified pair remains in the report, including warnings. Selecting only post-run passing pairs does not repair a distorted experiment.

A useful next measurement improvement would be an independently provisioned load generator, longer trials and explicit delivered-interarrival/burstiness analysis, while retaining all outcomes. Rejections, timeouts and physical compute drain must remain in the accounting. This has not been implemented or measured here, and moving the generator to another host would require a new clock/latency measurement design.

## What these references support

They support the relevance of the lifecycle and measurement problems being tested. They do not validate this implementation or establish competitive performance. Correctness evidence comes from this repository's production-path tests and retained raw records. The current tiny-model observations show added waiting without throughput benefit; heavier-model ratios are descriptive within the documented arrival and host limitations. RSS sampling is process accounting, not a unique physical-memory measurement or an efficiency ranking.
