import asyncio
import heapq
import unittest

from tiny_batch_service.scheduler import Scheduler


class Handle:
    def __init__(self, callback):
        self.callback, self.cancelled = callback, False

    def cancel(self):
        self.cancelled = True


class ManualClock:
    def __init__(self):
        self.time = 100.
        self.events = []
        self.serial = 0

    def now(self):
        return self.time

    def call_at(self, when, callback):
        handle = Handle(callback)
        self.serial += 1
        heapq.heappush(self.events, (when, self.serial, handle))
        return handle

    def advance(self, seconds):
        self.time += seconds
        while self.events and self.events[0][0] <= self.time:
            _, _, handle = heapq.heappop(self.events)
            if not handle.cancelled:
                handle.callback()


class ControlledBackend:
    def __init__(self):
        self.calls = []

    async def infer(self, values):
        future = asyncio.get_running_loop().create_future()
        self.calls.append((values, future))
        return await future

    def finish(self, index=-1):
        values, future = self.calls[index]
        future.set_result({"outputs": [[row[0] + j for j in range(10)] for row in values], "inference_ms": .01})


async def turns():
    for _ in range(8):
        await asyncio.sleep(0)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = ManualClock()
        self.backend = ControlledBackend()
        self.scheduler = Scheduler(self.backend, batch_size=4, wait_ms=10, capacity=8, clock=self.clock)
        self.scheduler.start()

    async def asyncTearDown(self):
        # Release a held backend even when an assertion fails, so failure is not hidden by a hang.
        self.scheduler.closing = True
        for values, future in self.backend.calls:
            if not future.done():
                future.set_result({"outputs": [[v[0] + j for j in range(10)] for v in values], "inference_ms": .01})
        await self.scheduler.close()

    def submit(self, name, value, timeout=1000):
        return self.scheduler.submit(name, [value] * 64, timeout)

    async def test_tail_deadline_and_exact_response_mapping(self):
        futures = [self.submit(str(i), i) for i in range(3)]
        await turns()
        self.assertEqual(len(self.backend.calls), 0)
        self.clock.advance(.009)
        await turns()
        self.assertEqual(len(self.backend.calls), 0)
        self.clock.advance(.002)
        await turns()
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(len(self.backend.calls[0][0]), 3)
        self.backend.finish()
        await turns()
        for i, future in enumerate(futures):
            self.assertTrue(future.done())
            self.assertEqual(future.result()["id"], str(i))
            self.assertEqual(future.result()["probabilities"], [i + j for j in range(10)])
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)

    async def test_full_batch_dispatches_without_wait(self):
        futures = [self.submit(str(i), i) for i in range(4)]
        await turns()
        self.assertEqual(len(self.backend.calls), 1)
        self.backend.finish()
        await turns()
        self.assertTrue(all(f.result()["status"] == "success" for f in futures))

    async def test_queued_timeout_and_cancel_never_reach_backend(self):
        timed = self.submit("timeout", 1, timeout=5)
        cancelled = self.submit("cancel", 2)
        self.assertTrue(self.scheduler.cancel("cancel"))
        self.clock.advance(.006)
        await turns()
        self.assertTrue(timed.done())
        self.assertEqual(timed.result()["status"], "timeout")
        self.assertEqual(cancelled.result()["status"], "cancelled")
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)

    async def test_running_cancel_retains_capacity_until_compute_finishes(self):
        self.scheduler.batch_size = self.scheduler.capacity = 1
        active = self.submit("active", 1)
        await turns()
        self.assertTrue(self.scheduler.cancel("active"))
        self.assertEqual(active.result()["status"], "cancelled")
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 1)
        rejected = self.submit("excess", 2)
        self.assertTrue(rejected.done())
        self.assertEqual(rejected.result()["http_status"], 429)
        self.backend.finish()
        await turns()
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)
        self.assertEqual(self.scheduler.counts["discarded_computed_rows"], 1)
        self.assertFalse(self.scheduler.cancel("active"))

    async def test_running_timeout_does_not_cancel_worker_or_leak_capacity(self):
        self.scheduler.batch_size = 1
        active = self.submit("timeout", 3, timeout=5)
        await turns()
        self.clock.advance(.006)
        await turns()
        self.assertTrue(active.done())
        self.assertEqual(active.result()["http_status"], 504)
        self.assertEqual(self.scheduler.snapshot()["running"], 1)
        self.assertFalse(self.backend.calls[0][1].cancelled())
        self.backend.finish()
        await turns()
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)

    async def test_capacity_includes_computing_and_queued_work(self):
        self.scheduler.batch_size = 1
        active = self.submit("active", 0)
        await turns()
        waiting = [self.submit(str(i), i) for i in range(7)]
        rejected = self.submit("overflow", 10)
        self.assertTrue(rejected.done())
        self.assertEqual(rejected.result()["status"], "overloaded")
        self.assertEqual(self.scheduler.snapshot()["peak_occupancy"], 8)
        for i in range(7):
            self.scheduler.cancel(str(i))
        self.backend.finish()
        await turns()
        self.assertEqual(active.result()["status"], "success")

    async def test_shutdown_resolves_waiters_and_worker_error_fails_closed(self):
        queued = self.submit("queued", 1)
        await self.scheduler.close()
        self.assertEqual(queued.result()["status"], "shutdown")
        rejected = self.submit("late", 2)
        self.assertEqual(rejected.result()["http_status"], 503)

    async def test_failure_resolves_active_and_queued_and_rejects_new(self):
        self.scheduler.batch_size = 1
        active = self.submit("active", 0)
        await turns()
        queued = self.submit("queued", 1)
        self.backend.calls[0][1].set_exception(RuntimeError("deliberate worker fault"))
        await turns()
        self.assertEqual(active.result()["status"], "failed")
        self.assertEqual(queued.result()["status"], "failed")
        self.assertFalse(self.scheduler.healthy)
        self.assertEqual(self.submit("new", 2).result()["http_status"], 503)

    async def test_duplicate_ids_and_bounded_audit_retention(self):
        future = self.submit("same", 1)
        with self.assertRaises(ValueError):
            self.submit("same", 2)
        self.scheduler.cancel("same")
        self.scheduler.audit_limit = 8
        for i in range(20):
            self.submit(str(i), 1)
            self.scheduler.cancel(str(i))
        self.assertEqual(len(self.scheduler.records), 8)
        self.assertGreater(self.scheduler.audit_evicted, 0)

    async def test_completion_cancel_race_resolves_exactly_once(self):
        self.scheduler.batch_size = 1
        cancelled = self.submit("cancel_wins", 1)
        await turns()
        self.backend.finish()
        self.assertTrue(self.scheduler.cancel("cancel_wins"))
        await turns()
        self.assertEqual(cancelled.result()["status"], "cancelled")
        self.assertEqual(self.scheduler.counts["cancelled"], 1)
        self.assertEqual(self.scheduler.counts["success"], 0)
        completed = self.submit("completion_wins", 2)
        await turns()
        self.backend.finish()
        await turns()
        self.assertFalse(self.scheduler.cancel("completion_wins"))
        self.assertEqual(completed.result()["status"], "success")
        self.assertEqual(self.scheduler.counts["success"], 1)
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)

    async def test_busy_worker_does_not_restart_tail_wait(self):
        first = [self.submit(f"first{i}", i) for i in range(4)]
        await turns()
        tail = [self.submit(f"tail{i}", 10 + i) for i in range(3)]
        self.clock.advance(.020)
        await turns()
        self.assertEqual(len(self.backend.calls), 1)
        self.backend.finish(0)
        await turns()
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(len(self.backend.calls[1][0]), 3)
        self.backend.finish(1)
        await turns()
        self.assertTrue(all(f.result()["status"] == "success" for f in first + tail))
        self.assertEqual(self.scheduler.counts["computed_rows"], 7)

    async def test_deadline_rechecked_when_timer_callbacks_are_delayed(self):
        queued = self.submit("expired_before_dispatch", 1, timeout=5)
        self.clock.time += .020  # Deliberately leave scheduled timer callbacks unexecuted.
        await turns()
        self.assertTrue(queued.done())
        self.assertEqual(queued.result()["status"], "timeout")
        self.assertEqual(self.backend.calls, [])
        self.scheduler.batch_size = 1
        running = self.submit("expired_before_completion", 2, timeout=5)
        await turns()
        self.clock.time += .020
        self.backend.finish()
        await turns()
        self.assertTrue(running.done())
        self.assertEqual(running.result()["status"], "timeout")
        self.assertEqual(self.scheduler.counts["computed_rows"], 1)
        self.assertEqual(self.scheduler.counts["discarded_computed_rows"], 1)
        self.assertEqual(self.scheduler.snapshot()["occupancy"], 0)


if __name__ == "__main__":
    unittest.main()
