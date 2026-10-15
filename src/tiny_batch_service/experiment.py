"""Open-loop HTTP trace replay in a process distinct from both server processes."""

import asyncio
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time

import aiohttp
import numpy as np
import psutil


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hash():
    return fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path(__file__).parent.glob("*.py"))})


def distribution(values):
    values = sorted(values)
    def quantile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)
    return {"count": len(values), **{name: quantile(q) if values else None
                                   for name, q in (("min", 0), ("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))}}


class ManagedServer:
    def __init__(self, model, *, batch_size=16, wait_ms=2., capacity=64, audit_limit=4096):
        self.model = str(Path(model).resolve())
        self.settings = dict(batch_size=batch_size, wait_ms=wait_ms, capacity=capacity, audit_limit=audit_limit)
        self.process = self.worker = None
        self.metadata = None
        self.lifecycle = {}

    async def __aenter__(self):
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", 0))
        except BaseException:
            sock.close()
            raise
        sock.listen(128)
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        command = [sys.executable, *(["-O"] if sys.flags.optimize else []), "-m", "tiny_batch_service", self.model,
                   "--socket-fd", str(sock.fileno())]
        for key, value in self.settings.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1")
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, pass_fds=(sock.fileno(),), stdout=self.log, stderr=self.log, env=env)
        sock.close()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=.3)) as session:
                limit = time.monotonic() + 20
                while time.monotonic() < limit:
                    if self.process.poll() is not None:
                        raise RuntimeError("Server exited during initialization")
                    try:
                        async with session.get(self.url + "/health") as response:
                            body = await response.json()
                            if response.status == 200 and body["ready"]:
                                self.worker = psutil.Process(body["worker"]["pid"])
                                self.worker_created = self.worker.create_time()
                                self.metadata = {"worker": {k: v for k, v in body["worker"].items() if k not in ("pid", "high_water_rss_bytes")},
                                                 "http": body["http_environment"],
                                                 "generator": {"python": platform.python_version(), "numpy": np.__version__,
                                                               "aiohttp": aiohttp.__version__, "psutil": psutil.__version__}}
                                return self
                    except (aiohttp.ClientError, TimeoutError):
                        pass
                    await asyncio.sleep(.03)
                raise RuntimeError("Server readiness timed out")
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *args):
        forced = False
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 12)
            except subprocess.TimeoutExpired:
                forced = True
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 3)
        worker_alive = False
        if self.worker is not None:
            try:
                worker_alive = self.worker.is_running() and self.worker.create_time() == self.worker_created
                if worker_alive:
                    self.worker.terminate()
                    await asyncio.to_thread(self.worker.wait, 3)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        self.lifecycle = {"forced_http_kill": forced, "worker_alive_after_http_exit": worker_alive,
                          "http_exit_code": self.process.returncode if self.process else None}
        self.log.close()


def make_trace(count, rate, seed=42):
    rng = np.random.default_rng(seed)
    return [{"id": f"r{i:06d}", "at_s": i / rate, "index": int(rng.integers(450))} for i in range(count)]


def validate_response(body, reference):
    output = np.asarray(body.get("probabilities"))
    if (output.shape != (10,) or not np.isfinite(output).all()
            or not np.allclose(output, reference, rtol=1e-4, atol=1e-5)
            or body.get("class_id") != int(reference.argmax())):
        raise ValueError("HTTP prediction differs from independent reference")


async def trial(model, fixture, trace, settings):
    fixture_bytes = Path(fixture).read_bytes()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    with np.load(io.BytesIO(fixture_bytes), allow_pickle=False) as archive:
        inputs, reference = archive["inputs"], archive["reference"]
    del fixture_bytes
    client_records, memory_samples = [], []
    server = ManagedServer(model, **settings)
    async with server:
        loop = asyncio.get_running_loop()
        # Connections may overlap; a separate explicit generator cap accounts for unsent work.
        connector = aiohttp.TCPConnector(limit=0)
        tracing = aiohttp.TraceConfig()
        async def headers_sent(session, context, params):
            context.trace_request_ctx["headers_sent_s"] = loop.time()
        tracing.on_request_headers_sent.append(headers_sent)
        async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=10),
                                         trace_configs=[tracing]) as client:
            # Same eight HTTP warmup examples in both modes; retained separately, excluded from trace metrics.
            for i in range(8):
                async with client.post(server.url + "/v1/infer", json={"id": f"warm{i}", "input": inputs[i].tolist(), "timeout_ms": 5000},
                                       trace_request_ctx={}) as response:
                    body = await response.json()
                    if response.status != 200:
                        raise ValueError("Warmup failed")
                    validate_response(body, reference[i])
            t0 = loop.time() + .1
            memory_window_start = loop.time() - t0
            monitoring = True
            http = psutil.Process(server.process.pid)
            def sample_memory():
                sample = {"at_s": loop.time() - t0, "http_rss_bytes": http.memory_info().rss,
                          "worker_rss_bytes": server.worker.memory_info().rss}
                sample["finished_s"] = loop.time() - t0
                memory_samples.append(sample)
            async def monitor():
                while monitoring:
                    sample_memory()
                    await asyncio.sleep(.005)
            monitoring_task = asyncio.create_task(monitor())
            tasks = set()
            all_tasks = []
            async def request(item):
                sent = loop.time()
                record = {**item, "dispatch_s": sent - t0, "headers_sent_s": None,
                          "finished_s": None, "status": None, "http_status": None, "output_checked": False}
                context = {}
                received_at = None
                try:
                    async with client.post(server.url + "/v1/infer", json={"id": item["id"],
                                           "input": inputs[item["index"]].tolist(), "timeout_ms": 250},
                                           trace_request_ctx=context) as response:
                        body = await response.json()
                        received_at = loop.time()
                        record.update(status=body["status"], http_status=response.status)
                        if body.get("id") != item["id"]:
                            raise ValueError("Response ID differs")
                        if response.status == 200:
                            validate_response(body, reference[item["index"]])
                            record.update(output_checked=True, probabilities=body["probabilities"], class_id=body["class_id"])
                except (aiohttp.ClientError, TimeoutError):
                    record["status"] = "transport_error"
                finally:
                    record["finished_s"] = (received_at if received_at is not None else loop.time()) - t0
                    record["headers_sent_s"] = context.get("headers_sent_s", t0) - t0 if context else None
                    client_records.append(record)
            try:
                for item in trace:
                    await asyncio.sleep(max(0., t0 + item["at_s"] - loop.time()))
                    if len(tasks) >= 512:
                        client_records.append({**item, "dispatch_s": None, "headers_sent_s": None,
                                               "finished_s": loop.time() - t0, "status": "generator_unsent",
                                               "http_status": None, "output_checked": False})
                        continue
                    task = asyncio.create_task(request(item))
                    tasks.add(task)
                    all_tasks.append(task)
                    task.add_done_callback(tasks.discard)
                # Retain completed tasks too: task exceptions must never disappear after removal from the active set.
                outcomes = await asyncio.gather(*all_tasks, return_exceptions=True)
                for _ in range(500):
                    async with client.get(server.url + "/v1/stats", trace_request_ctx={}) as response:
                        audit = await response.json()
                    if audit["occupancy"] == 0:
                        break
                    await asyncio.sleep(.01)
                else:
                    raise ValueError("Server failed to drain")
            finally:
                monitoring = False
                await monitoring_task
                sample_memory()  # Bracket the entire active/drain interval, even for short trials.
            memory_window_end = loop.time() - t0
            if any(isinstance(value, BaseException) for value in outcomes):
                raise ValueError("A request failed output or protocol validation")
            trace_ids = {r["id"] for r in trace}
            # Server timestamps use the same host monotonic clock as the generator.
            records = [{k: (v - t0 if k.endswith("_s") and v is not None else v) for k, v in r.items()}
                       for r in audit["records"] if r["id"] in trace_ids]
            batches = [{**b, "started_s": b["started_s"] - t0, "finished_s": b["finished_s"] - t0}
                       for b in audit["batches"] if any(i in trace_ids for i in b["ids"])]
            result = {"settings": settings, "environment": server.metadata, "trace_sha256": fingerprint(trace),
                      "workload": {"loaded_model_sha256": server.metadata["worker"]["loaded_model_sha256"],
                                   "fixture_sha256": fixture_sha256},
                      "client": sorted(client_records, key=lambda r: r["id"]), "server_records": records,
                      "batches": batches, "audit": {k: v for k, v in audit.items() if k not in ("records", "batches")},
                      "rss_samples": memory_samples,
                      "memory_window": {"start_s": memory_window_start, "end_s": memory_window_end}}
    result["lifecycle"] = server.lifecycle
    return result


def summarize(run, trace, *, require_memory_bracketing=True):
    def finite(*values):
        return all(type(v) in (int, float) and np.isfinite(v) for v in values)

    records = run["client"]
    if len(records) != len(trace) or {r["id"] for r in records} != {r["id"] for r in trace}:
        raise ValueError("Incomplete or duplicate client coverage")
    ledger = {r["id"]: r for r in run["server_records"]}
    if len(ledger) != len(run["server_records"]):
        raise ValueError("Duplicate server coverage")
    counts = Counter(r["status"] for r in records)
    if counts["transport_error"] or counts["generator_unsent"]:
        raise ValueError("Generator could not deliver the offered workload")
    if set(ledger) != {r["id"] for r in records}:
        raise ValueError("Missing server terminal evidence")
    successful, scheduled_success, terminal_latency, scheduled_terminal = [], [], [], []
    lag, receive_lag, queues, admission_events = [], [], [], []
    for item, actual in zip(sorted(trace, key=lambda x: x["id"]), records):
        if any(actual[k] != item[k] for k in ("id", "at_s", "index")):
            raise ValueError("Trace identity changed")
        server = ledger[item["id"]]
        if server["status"] != actual["status"] or server["terminal_s"] is None:
            raise ValueError("Client/server terminal state differs")
        if (not finite(*(server[k] for k in ("received_s", "deadline_s", "terminal_s")),
                       *(actual[k] for k in ("dispatch_s", "headers_sent_s", "finished_s")))
                or not item["at_s"] <= actual["dispatch_s"] <= actual["headers_sent_s"] <= server["received_s"]
                or not server["received_s"] <= server["terminal_s"] <= actual["finished_s"]
                or not np.isclose(server["deadline_s"] - server["received_s"], .25, rtol=0, atol=1e-8)):
            raise ValueError("Contradictory server timing")
        if actual["status"] not in ("success", "timeout", "overloaded"):
            raise ValueError("Unexpected runtime or protocol outcome")
        expected_status = {"success": 200, "timeout": 504, "overloaded": 429}[actual["status"]]
        if actual["http_status"] != expected_status:
            raise ValueError("HTTP status differs")
        elapsed = (actual["finished_s"] - actual["dispatch_s"]) * 1000
        if elapsed <= 0 or actual["headers_sent_s"] is None:
            raise ValueError("Invalid client timing")
        terminal_latency.append(elapsed)
        scheduled_elapsed = (actual["finished_s"] - item["at_s"]) * 1000
        scheduled_terminal.append(scheduled_elapsed)
        lag.append((actual["headers_sent_s"] - item["at_s"]) * 1000)
        receive_lag.append((server["received_s"] - item["at_s"]) * 1000)
        if actual["status"] == "success":
            if not actual["output_checked"] or server["compute_finished_s"] is None:
                raise ValueError("Unchecked success")
            successful.append(elapsed)
            scheduled_success.append(scheduled_elapsed)
            if (server["terminal_s"] >= server["deadline_s"] or server["phase_at_terminal"] != "running"
                    or server["reason"] is not None):
                raise ValueError("Late or uncomputed success")
        elif actual["status"] == "timeout":
            if server["terminal_s"] < server["deadline_s"] or server["reason"] != "request_deadline":
                raise ValueError("Premature or unexplained timeout")
        if server["started_s"] is not None:
            if (not finite(server["started_s"], server["compute_finished_s"])
                    or not server["received_s"] <= server["started_s"] <= server["compute_finished_s"]
                    or server["phase_at_terminal"] != "running" or actual["status"] == "overloaded"
                    or (actual["status"] == "success" and server["compute_finished_s"] > server["terminal_s"])):
                raise ValueError("Contradictory computation timing")
            queues.append((server["started_s"] - server["received_s"]) * 1000)
        elif (server["compute_finished_s"] is not None or actual["status"] == "success"
              or server["phase_at_terminal"] != "queued"):
            raise ValueError("Unstarted computation has output or wrong phase")
        if actual["status"] != "overloaded":
            released = server["compute_finished_s"] if server["started_s"] is not None else server["terminal_s"]
            admission_events += [(server["received_s"], 1), (released, -1)]
        else:
            admission_events.append((server["received_s"], 0))
    computed = [i for b in run["batches"] for i in b["ids"]]
    expected_computed = [i for i, r in ledger.items() if r["started_s"] is not None]
    if sorted(computed) != sorted(expected_computed) or len(set(computed)) != len(computed):
        raise ValueError("Missing, repeated or unexpected computed requests")
    prior_finish = 0.
    for b in run["batches"]:
        if (not finite(b["started_s"], b["finished_s"], b["inference_ms"])
                or not prior_finish <= b["started_s"] < b["finished_s"]
                or b["rows"] != len(b["ids"]) or not 1 <= b["rows"] <= run["settings"]["batch_size"]
                or not 0 < b["inference_ms"] <= (b["finished_s"] - b["started_s"]) * 1000):
            raise ValueError("Invalid batch cardinality")
        for identifier in b["ids"]:
            row = ledger[identifier]
            if row["started_s"] > b["started_s"] or row["compute_finished_s"] != b["finished_s"]:
                raise ValueError("Batch and per-request computation timing differ")
        prior_finish = b["finished_s"]
    occupancy, reconstructed_peak = 0, 1  # Eight sequential warmups each occupy one slot.
    for _, delta in sorted(admission_events):
        if delta == 0 and occupancy != run["settings"]["capacity"]:
            raise ValueError("Rejection occurred without physical capacity being full")
        occupancy += delta
        reconstructed_peak = max(reconstructed_peak, occupancy)
        if not 0 <= occupancy <= run["settings"]["capacity"]:
            raise ValueError("Per-request evidence violates physical capacity")
    if (run["audit"]["occupancy"] != 0 or run["audit"]["audit_evicted"] or run["audit"]["batch_evicted"]
            or run["audit"]["running"] != 0 or run["audit"]["queued"] != 0
            or run["audit"]["peak_occupancy"] != reconstructed_peak
            or run["audit"]["capacity"] != run["settings"]["capacity"]
            or run["audit"]["batch_size"] != run["settings"]["batch_size"]
            or run["audit"]["wait_ms"] != run["settings"]["wait_ms"]
            or not run["audit"]["healthy"] or run["audit"]["closing"]
            or run["lifecycle"] != {"forced_http_kill": False, "worker_alive_after_http_exit": False, "http_exit_code": 0}):
        raise ValueError("Incomplete audit, capacity violation, or unclean process exit")
    expected_counts = {"offered": len(trace) + 8, "admitted": len(trace) - counts["overloaded"] + 8,
                       "success": counts["success"] + 8, "overloaded": counts["overloaded"],
                       "timeout": counts["timeout"], "computed_rows": len(computed) + 8,
                       "discarded_computed_rows": sum(r["status"] != "success" for r in ledger.values() if r["started_s"] is not None)}
    if ({k: v for k, v in run["audit"]["counts"].items() if v}
            != {k: v for k, v in expected_counts.items() if v}):
        raise ValueError("Server counters differ from per-request evidence")
    duration = max(max(r["finished_s"] for r in records), trace[-1]["at_s"],
                   max((b["finished_s"] for b in run["batches"]), default=0))
    samples = run["rss_samples"]
    if not samples or any(type(s[k]) is not int or s[k] <= 0 for s in samples for k in ("http_rss_bytes", "worker_rss_bytes")):
        raise ValueError("Invalid memory samples")
    if require_memory_bracketing and (len(samples) < 2 or samples[0]["finished_s"] > 0 or samples[-1]["at_s"] < duration):
        raise ValueError("RSS samples do not bracket active traffic and physical compute drain")
    window = run["memory_window"]
    if not finite(window["start_s"], window["end_s"]) or not window["start_s"] <= 0 <= duration <= window["end_s"]:
        raise ValueError("Memory observation window does not cover the trial")
    previous = window["start_s"]
    for sample in samples:
        if (not finite(sample["at_s"], sample["finished_s"])
                or not previous <= sample["at_s"] <= sample["finished_s"] <= window["end_s"]):
            raise ValueError("Memory samples outside observation window or out of order")
        previous = sample["finished_s"]
    gaps = ([samples[0]["at_s"] - window["start_s"]]
            + [b["at_s"] - a["at_s"] for a, b in zip(samples, samples[1:])]
            + [window["end_s"] - samples[-1]["at_s"]])
    return {"counts": dict(counts), "observation_s": duration, "success_per_second": len(successful) / duration,
            "rejection_rate": counts["overloaded"] / len(trace), "timeout_rate": counts["timeout"] / len(trace),
            "success_rate": counts["success"] / len(trace),
            "backend_busy_fraction": sum(b["finished_s"] - b["started_s"] for b in run["batches"]) / duration,
            "mean_batch_rows": len(computed) / len(run["batches"]) if run["batches"] else 0.,
            "success_latency_ms": distribution(successful), "all_terminal_latency_ms": distribution(terminal_latency),
            "scheduled_success_latency_ms": distribution(scheduled_success),
            "scheduled_all_terminal_latency_ms": distribution(scheduled_terminal),
            "header_send_lag_ms": distribution(lag), "server_arrival_lag_ms": distribution(receive_lag),
            "queue_ms": distribution(queues), "batch_sizes": {str(k): v for k, v in Counter(b["rows"] for b in run["batches"]).items()},
            "peak_http_rss_bytes": max(s["http_rss_bytes"] for s in samples),
            "peak_worker_rss_bytes": max(s["worker_rss_bytes"] for s in samples),
            "peak_paired_rss_bytes": max(s["http_rss_bytes"] + s["worker_rss_bytes"] for s in samples),
            "memory_observation_s": window["end_s"] - window["start_s"],
            "rss_sample_count": len(samples), "max_rss_gap_ms": max(gaps) * 1000,
            "max_paired_read_ms": max(s["finished_s"] - s["at_s"] for s in samples) * 1000,
            "arrival_fidelity_passed": distribution(lag)["p99"] <= 25 and distribution(receive_lag)["p99"] <= 25}


def host_metadata():
    cpu = None
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return {"os": platform.system(), "os_release": platform.release(), "machine": platform.machine(),
            "cpu_model": cpu, "logical_cpus": psutil.cpu_count(), "python": platform.python_version()}
