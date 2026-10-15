"""One FIFO dispatcher; logical terminal states never free running capacity early."""

import asyncio
from collections import Counter, OrderedDict
from dataclasses import dataclass
import math


class LoopClock:
    def now(self):
        return asyncio.get_running_loop().time()

    def call_at(self, when, callback):
        return asyncio.get_running_loop().call_at(when, callback)


@dataclass
class Entry:
    identifier: str
    values: list
    future: asyncio.Future
    record: dict
    expiry: object = None
    phase: str = "queued"


class Scheduler:
    def __init__(self, backend, *, batch_size=16, wait_ms=2., capacity=128, audit_limit=4096, clock=None):
        if (batch_size < 1 or capacity < batch_size or audit_limit < capacity
                or not math.isfinite(wait_ms) or wait_ms < 0):
            raise ValueError("Require 1 <= batch_size <= capacity <= audit_limit and finite nonnegative wait")
        self.backend = backend
        self.batch_size, self.wait_s = batch_size, wait_ms / 1000
        self.capacity, self.audit_limit = capacity, audit_limit
        self.clock = clock or LoopClock()
        self.entries, self.pending, self.records = {}, OrderedDict(), OrderedDict()
        self.counts = Counter()
        self.batches = []
        self.audit_evicted = self.batch_evicted = 0
        self.wake = asyncio.Event()
        self.closing, self.healthy = False, True
        self.batch_timer = None
        self.peak_occupancy = 0
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self._run())

    def submit(self, identifier, values, timeout_ms):
        if identifier in self.records or identifier in self.entries:
            raise ValueError("Duplicate active or recently recorded ID")
        now = self.clock.now()
        record = {"id": identifier, "received_s": now, "deadline_s": now + timeout_ms / 1000,
                  "started_s": None, "terminal_s": None, "status": None,
                  "phase_at_terminal": None, "reason": None, "compute_finished_s": None}
        entry = Entry(identifier, values, asyncio.get_running_loop().create_future(), record)
        self.records[identifier] = record
        self.counts["offered"] += 1
        if self.closing or not self.healthy:
            self._terminal(entry, "unavailable", 503)
        elif len(self.entries) >= self.capacity:
            self._terminal(entry, "overloaded", 429)
        else:
            self.entries[identifier] = self.pending[identifier] = entry
            self.counts["admitted"] += 1
            self.peak_occupancy = max(self.peak_occupancy, len(self.entries))
            entry.expiry = self.clock.call_at(record["deadline_s"], lambda: self._expire(entry))
            self.wake.set()
        self._trim()
        return entry.future

    def _trim(self):
        while len(self.records) > self.audit_limit:
            removable = next((i for i in self.records if i not in self.entries), None)
            if removable is None:
                break
            self.records.pop(removable)
            self.audit_evicted += 1

    def _terminal(self, entry, status, http_status, *, reason=None, output=None):
        if entry.future.done():
            return False
        entry.record.update(status=status, terminal_s=self.clock.now(), phase_at_terminal=entry.phase, reason=reason)
        if entry.expiry is not None:
            entry.expiry.cancel()
        self.counts[status] += 1
        result = {"id": entry.identifier, "status": status, "http_status": http_status}
        if output is not None:
            result.update(probabilities=output, class_id=max(range(len(output)), key=output.__getitem__))
        entry.future.set_result(result)
        if entry.phase == "queued":
            self.pending.pop(entry.identifier, None)
            self.entries.pop(entry.identifier, None)
        self.wake.set()
        return True

    def _expire(self, entry):
        self._terminal(entry, "timeout", 504, reason="request_deadline")

    def cancel(self, identifier, reason="explicit_cancel"):
        entry = self.entries.get(identifier)
        if entry is None or entry.future.done():
            return False
        return self._terminal(entry, "cancelled", 409, reason=reason)

    async def _run(self):
        while True:
            await self.wake.wait()
            self.wake.clear()
            if self.batch_timer is not None:
                self.batch_timer.cancel()
                self.batch_timer = None
            if not self.pending:
                if self.closing:
                    return
                continue
            now = self.clock.now()
            # Expire again at dispatch: timer callbacks can be delayed by event-loop scheduling.
            for entry in list(self.pending.values()):
                if now >= entry.record["deadline_s"]:
                    self._expire(entry)
            if not self.pending:
                self.wake.set()
                continue
            first = next(iter(self.pending.values()))
            flush_at = first.record["received_s"] + self.wait_s
            if len(self.pending) < self.batch_size and now < flush_at and not self.closing:
                self.batch_timer = self.clock.call_at(flush_at, self.wake.set)
                continue
            batch = []
            while self.pending and len(batch) < self.batch_size:
                _, entry = self.pending.popitem(last=False)
                entry.phase = "running"
                entry.record["started_s"] = self.clock.now()
                batch.append(entry)
            started = self.clock.now()
            try:
                result = await self.backend.infer([e.values for e in batch])
                outputs = result["outputs"]
                if (len(outputs) != len(batch) or any(len(o) != 10 or
                        any(not math.isfinite(v) for v in o) for o in outputs)
                        or not math.isfinite(result["inference_ms"]) or result["inference_ms"] <= 0):
                    raise ValueError("Malformed worker output")
                finished = self.clock.now()
                self.batches.append({"ids": [e.identifier for e in batch], "rows": len(batch),
                                     "started_s": started, "finished_s": finished,
                                     "inference_ms": result["inference_ms"]})
                if len(self.batches) > self.audit_limit:
                    self.batches.pop(0)
                    self.batch_evicted += 1
                self.counts["computed_rows"] += len(batch)
                for entry, output in zip(batch, outputs):
                    entry.record["compute_finished_s"] = finished
                    if not entry.future.done():
                        if finished >= entry.record["deadline_s"]:
                            self._expire(entry)
                        else:
                            self._terminal(entry, "success", 200, output=output)
                    if entry.record["status"] != "success":
                        self.counts["discarded_computed_rows"] += 1
            except Exception:
                self.healthy = False
                for entry in [*batch, *list(self.pending.values())]:
                    self._terminal(entry, "failed", 503, reason="worker_failure")
            finally:
                for entry in batch:
                    self.entries.pop(entry.identifier, None)
                self._trim()
                self.wake.set()

    def snapshot(self):
        return {"counts": dict(self.counts), "capacity": self.capacity, "batch_size": self.batch_size,
                "wait_ms": self.wait_s * 1000, "healthy": self.healthy, "closing": self.closing,
                "occupancy": len(self.entries), "queued": len(self.pending),
                "running": len(self.entries) - len(self.pending), "peak_occupancy": self.peak_occupancy,
                "audit_evicted": self.audit_evicted, "batch_evicted": self.batch_evicted,
                "records": list(self.records.values()), "batches": list(self.batches)}

    async def close(self):
        self.closing = True
        for entry in list(self.pending.values()):
            self._terminal(entry, "shutdown", 503, reason="service_shutdown")
        self.wake.set()
        if self.task is not None:
            await self.task
        if self.batch_timer is not None:
            self.batch_timer.cancel()
