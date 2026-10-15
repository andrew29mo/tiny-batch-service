import asyncio
import importlib.util
from pathlib import Path
import tempfile
import unittest

import aiohttp
from aiohttp.test_utils import TestClient, TestServer
import numpy as np

from tiny_batch_service.backend import ProcessBackend
from tiny_batch_service.experiment import ManagedServer, validate_response, trial, summarize, make_trace
from tiny_batch_service.server import create_app

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_model", ROOT / "tools/make_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temp.name)
        cls.manifest = model.prepare(cls.fixture)
        with np.load(cls.fixture / "fixture.npz") as archive:
            cls.inputs, cls.reference = archive["inputs"], archive["reference"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def body(self, identifier, index=0, timeout=1000):
        return {"id": identifier, "input": self.inputs[index].tolist(), "timeout_ms": timeout}

    async def test_real_separate_process_http_outputs_and_tail(self):
        server = ManagedServer(self.fixture / "model.onnx", batch_size=8, wait_ms=30, capacity=512)
        async with server:
            async with aiohttp.ClientSession() as client:
                async def call(i):
                    async with client.post(server.url + "/v1/infer", json=self.body(str(i), i, timeout=5000)) as response:
                        body = await response.json()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(body["id"], str(i))
                        validate_response(body, self.reference[i])
                await asyncio.gather(*(call(i) for i in range(450)))
                async with client.get(server.url + "/v1/stats") as response:
                    stats = await response.json()
                self.assertEqual(stats["counts"]["success"], 450)
                self.assertEqual(stats["counts"]["computed_rows"], 450)
                self.assertEqual(sum(b["rows"] for b in stats["batches"]), 450)
                self.assertTrue(any(0 < b["rows"] < 8 for b in stats["batches"]))
                self.assertEqual(stats["occupancy"], 0)
        self.assertEqual(server.lifecycle, {"http_exit_code": 0, "forced_http_kill": False, "worker_alive_after_http_exit": False})

    async def test_real_http_queued_timeout_cancel_and_invalid_duplicate(self):
        async with ManagedServer(self.fixture / "model.onnx", batch_size=8, wait_ms=200, capacity=8) as server:
            async with aiohttp.ClientSession() as client:
                active = asyncio.create_task(client.post(server.url + "/v1/infer", json=self.body("cancel")))
                for _ in range(100):
                    async with client.get(server.url + "/v1/stats") as response:
                        stats = await response.json()
                    if stats["queued"]:
                        break
                    await asyncio.sleep(.001)
                async with client.delete(server.url + "/v1/requests/cancel") as response:
                    self.assertEqual(response.status, 200)
                response = await active
                self.assertEqual(response.status, 409)
                self.assertEqual((await response.json())["status"], "cancelled")
                response.release()
                async with client.post(server.url + "/v1/infer", json=self.body("timeout", timeout=10)) as response:
                    self.assertEqual(response.status, 504)
                async with client.post(server.url + "/v1/infer", json=self.body("cancel")) as response:
                    self.assertEqual(response.status, 409)
                invalid = self.body("bad")
                invalid["input"][-1] = float("nan")
                async with client.post(server.url + "/v1/infer", json=invalid) as response:
                    self.assertEqual(response.status, 400)
                invalid_cases = [self.body("bad", timeout=0), {**self.body("bad"), "input": [0] * 63},
                                 {**self.body("bad"), "input": [True] * 64},
                                 {**self.body("bad"), "id": "invalid/id"},
                                 {**self.body("bad"), "extra": 1}]
                for body in invalid_cases:
                    async with client.post(server.url + "/v1/infer", json=body) as response:
                        self.assertEqual(response.status, 400)
                async with client.post(server.url + "/v1/infer", data="{", headers={"Content-Type": "application/json"}) as response:
                    self.assertEqual(response.status, 400)
                async with client.post(server.url + "/v1/infer", data="x" * 9000, headers={"Content-Type": "application/json"}) as response:
                    self.assertEqual(response.status, 413)
                async with client.get(server.url + "/v1/stats") as response:
                    stats = await response.json()
                self.assertEqual(stats["counts"].get("computed_rows", 0), 0)

    async def test_deep_json_is_invalid_without_admission_and_service_recovers(self):
        async with ManagedServer(self.fixture / "model.onnx", batch_size=1, wait_ms=0, capacity=8) as server:
            async with aiohttp.ClientSession() as client:
                # Deeply nested JSON below the body limit is still not an inference object.
                payload = "[" * 2000 + "0" + "]" * 2000
                self.assertLess(len(payload), 8192)
                async with client.post(server.url + "/v1/infer", data=payload,
                                       headers={"Content-Type": "application/json"}) as response:
                    self.assertEqual(response.status, 400)
                    self.assertEqual(await response.json(), {"status": "invalid_request"})
                async with client.get(server.url + "/v1/stats") as response:
                    audit = await response.json()
                self.assertEqual(audit["counts"].get("offered", 0), 0)
                self.assertEqual(audit["occupancy"], 0)
                async with client.post(server.url + "/v1/infer", json=self.body("after_deep_json")) as response:
                    self.assertEqual(response.status, 200)
                    validate_response(await response.json(), self.reference[0])

    async def test_real_http_running_cancel_timeout_overload_and_compute_drain(self):
        # Hold the real backend before executing ORT, so state transitions are deterministic.
        # On release, the actual model still computes cancelled/expired running requests.
        outer = self
        class GateBackend(ProcessBackend):
            def __init__(self, path):
                super().__init__(path)
                self.gate = asyncio.Event()
                self.entered = asyncio.Event()
                self.executed = 0
                outer.backend = self

            async def infer(self, values):
                self.entered.set()
                await self.gate.wait()
                result = await super().infer(values)
                self.executed += len(values)
                return result
        app = create_app(self.fixture / "model.onnx", batch_size=1, wait_ms=0, capacity=1, backend_factory=GateBackend)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for identifier, timeout, status in (("cancel", 1000, 409), ("timeout", 20, 504)):
                self.backend.gate.clear()
                self.backend.entered.clear()
                pending = asyncio.create_task(client.post("/v1/infer", json=self.body(identifier, timeout=timeout)))
                await asyncio.wait_for(self.backend.entered.wait(), 2)
                if identifier == "cancel":
                    response = await client.delete("/v1/requests/" + identifier)
                    self.assertEqual(response.status, 200)
                    response.release()
                response = await asyncio.wait_for(pending, 2)
                self.assertEqual(response.status, status)
                response.release()
                response = await client.post("/v1/infer", json=self.body("excess_" + identifier))
                self.assertEqual(response.status, 429)
                response.release()
                self.backend.gate.set()
                for _ in range(100):
                    response = await client.get("/v1/stats")
                    stats = await response.json()
                    response.release()
                    if stats["occupancy"] == 0:
                        break
                    await asyncio.sleep(.001)
                self.assertEqual(stats["occupancy"], 0)
            self.assertEqual(self.backend.executed, 2)
            self.assertEqual(stats["counts"]["discarded_computed_rows"], 2)
        finally:
            self.backend.gate.set()
            await client.close()

    async def test_client_disconnect_cancels_formal_server_waiter(self):
        async with ManagedServer(self.fixture / "model.onnx", batch_size=8, wait_ms=500, capacity=8) as server:
            async with aiohttp.ClientSession() as client:
                task = asyncio.create_task(client.post(server.url + "/v1/infer", json=self.body("disconnect")))
                for _ in range(100):
                    async with client.get(server.url + "/v1/stats") as response:
                        stats = await response.json()
                    if stats["queued"]:
                        break
                    await asyncio.sleep(.001)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                for _ in range(100):
                    async with client.get(server.url + "/v1/stats") as response:
                        stats = await response.json()
                    if stats["occupancy"] == 0:
                        break
                    await asyncio.sleep(.001)
                self.assertEqual(stats["occupancy"], 0)
                self.assertEqual(stats["records"][0]["reason"], "client_disconnect")

    async def test_real_mixed_batch_keeps_rows_and_drains_following_tail(self):
        outer = self
        class GateBackend(ProcessBackend):
            def __init__(self, path):
                super().__init__(path)
                self.gate, self.entered = asyncio.Event(), asyncio.Event()
                outer.backend = self

            async def infer(self, values):
                self.entered.set()
                await self.gate.wait()
                return await super().infer(values)

        client = TestClient(TestServer(create_app(self.fixture / "model.onnx", batch_size=3,
                            wait_ms=20, capacity=6, backend_factory=GateBackend)))
        await client.start_server()
        pending = []
        try:
            async def call(identifier, index, timeout=5000):
                async with client.post("/v1/infer", json=self.body(identifier, index, timeout)) as response:
                    return response.status, await response.json()
            pending = [asyncio.create_task(call("success", 0)),
                       asyncio.create_task(call("timeout", 1, 100)),
                       asyncio.create_task(call("cancel", 2))]
            await asyncio.wait_for(self.backend.entered.wait(), 2)
            pending.append(asyncio.create_task(call("tail", 449)))
            async with client.delete("/v1/requests/cancel") as response:
                self.assertEqual(response.status, 200)
            self.assertEqual((await asyncio.wait_for(asyncio.shield(pending[1]), 2))[0], 504)
            self.assertEqual((await pending[2])[0], 409)
            async with client.get("/v1/stats") as response:
                stats = await response.json()
            self.assertEqual((stats["running"], stats["queued"], stats["occupancy"]), (3, 1, 4))
            self.backend.gate.set()
            results = await asyncio.wait_for(asyncio.gather(*pending), 3)
            for position, index in ((0, 0), (3, 449)):
                self.assertEqual(results[position][0], 200)
                validate_response(results[position][1], self.reference[index])
            for position in (1, 2):
                self.assertNotIn("probabilities", results[position][1])
            async with client.get("/v1/stats") as response:
                stats = await response.json()
            self.assertEqual(stats["occupancy"], 0)
            self.assertEqual(stats["counts"], {"offered": 4, "admitted": 4, "cancelled": 1,
                              "timeout": 1, "computed_rows": 4, "success": 2, "discarded_computed_rows": 2})
            self.assertEqual([b["rows"] for b in stats["batches"]], [3, 1])
            self.assertCountEqual([i for b in stats["batches"] for i in b["ids"]], ["success", "timeout", "cancel", "tail"])
            self.assertEqual(len(stats["records"]), 4)
        finally:
            self.backend.gate.set()
            await asyncio.gather(*pending, return_exceptions=True)
            await client.close()

    async def test_actual_trace_runner_and_summary_use_production_service(self):
        trace = make_trace(25, 200)
        run = await trial(self.fixture / "model.onnx", self.fixture / "fixture.npz", trace,
                          {"batch_size": 16, "wait_ms": 2., "capacity": 64, "audit_limit": 4096})
        summary = summarize(run, trace)
        self.assertEqual(summary["counts"]["success"], 25)
        self.assertEqual(summary["success_latency_ms"]["count"], 25)
        self.assertGreater(summary["peak_paired_rss_bytes"], 0)
        self.assertGreaterEqual(summary["scheduled_success_latency_ms"]["p95"], summary["success_latency_ms"]["p95"])
        self.assertGreater(summary["rss_sample_count"], 1)
        self.assertTrue(all(r["output_checked"] for r in run["client"]))

    def test_reference_validator_rejects_wrong_final_probability(self):
        body = {"probabilities": self.reference[-1].tolist(), "class_id": int(self.reference[-1].argmax())}
        validate_response(body, self.reference[-1])
        body["probabilities"][-1] += .1
        with self.assertRaises(ValueError):
            validate_response(body, self.reference[-1])


if __name__ == "__main__":
    unittest.main()
