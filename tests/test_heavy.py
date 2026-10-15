import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiohttp
import numpy as np
import onnx
from onnx import numpy_helper
from sklearn.datasets import load_digits

from tiny_batch_service.experiment import ManagedServer, validate_response
from tiny_batch_service.report import resolve_run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_heavy_model
from make_model import prepare


class HeavyModelTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.manifest = prepare(cls.directory, "heavy")
        with np.load(cls.directory / "fixture.npz", allow_pickle=False) as archive:
            cls.inputs, cls.reference, cls.labels = [archive[k] for k in ("inputs", "reference", "labels")]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fitting_receives_only_training_rows_and_no_test_validation(self):
        state = json.loads((ROOT / "examples/heavy-digits/model.json").read_text())
        x, y = load_digits(return_X_y=True)
        ids = np.asarray(state["provenance"]["train_ids"])
        test_ids = state["provenance"]["test_ids"]
        self.assertEqual(len(ids), 1347)
        self.assertEqual(len(set(ids) & set(test_ids)), 0)
        with patch.object(train_heavy_model, "MLPClassifier") as estimator:
            train_heavy_model.fit_model()
        args, kwargs = estimator.return_value.fit.call_args
        self.assertEqual(kwargs, {})
        np.testing.assert_array_equal(args[0], (x[ids] / 16).astype(np.float32))
        np.testing.assert_array_equal(args[1], y[ids])
        self.assertFalse(estimator.call_args.kwargs["early_stopping"])
        self.assertEqual(estimator.call_args.kwargs["max_iter"], 60)
        self.assertEqual(estimator.call_args.kwargs["n_iter_no_change"], 61)

    def test_export_uses_exact_frozen_weights_without_extra_compute(self):
        model = onnx.load(self.directory / "model.onnx")
        actual = {a.name: numpy_helper.to_array(a) for a in model.graph.initializer}
        with np.load(ROOT / "examples/heavy-digits/weights.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(actual))
            for key in archive.files:
                np.testing.assert_array_equal(actual[key], archive[key])
        self.assertEqual([n.op_type for n in model.graph.node], ["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Softmax"])
        self.assertEqual(sum(v.size for v in actual.values()), 4274186)
        self.assertEqual(self.manifest["correct"], 444)

    async def test_actual_http_heavy_all_heldout_predictions_and_tail(self):
        async with ManagedServer(self.directory / "model.onnx", batch_size=16, wait_ms=5, capacity=512) as server:
            async with aiohttp.ClientSession() as client:
                async def call(i):
                    async with client.post(server.url + "/v1/infer", json={"id": str(i), "input": self.inputs[i].tolist(), "timeout_ms": 5000}) as response:
                        body = await response.json()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(body["id"], str(i))
                        validate_response(body, self.reference[i])
                        return body["class_id"]
                classes = await asyncio.gather(*(call(i) for i in range(450)))
                self.assertEqual(int(np.sum(np.asarray(classes) == self.labels)), 444)
                async with client.get(server.url + "/v1/stats") as response:
                    audit = await response.json()
                self.assertEqual(audit["counts"]["computed_rows"], 450)
                self.assertEqual(audit["counts"]["success"], 450)
                self.assertEqual(audit["occupancy"], 0)
                self.assertEqual(len({i for b in audit["batches"] for i in b["ids"]}), 450)
                self.assertTrue(any(b["rows"] < 16 for b in audit["batches"]))

    def test_compressed_raw_trial_hash_and_index_must_both_match(self):
        directory = ROOT / "results/heavy-macos-arm64"
        report = json.loads((directory / "report.json").read_text())
        stub = report["cases"][0]["runs"][0]
        run = resolve_run(stub, directory)
        self.assertEqual(len(run["client"]), 6000)
        changed = copy.deepcopy(stub)
        changed["summary"]["success_per_second"] = 1e9
        with self.assertRaises(ValueError):
            resolve_run(changed, directory)
        changed = {**stub, "evidence_file": "../../README.md"}
        with self.assertRaises(ValueError):
            resolve_run(changed, directory)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / stub["evidence_file"]
            path.parent.mkdir()
            data = bytearray((directory / stub["evidence_file"]).read_bytes())
            data[-5] ^= 1
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                resolve_run(stub, temporary)


if __name__ == "__main__":
    unittest.main()
