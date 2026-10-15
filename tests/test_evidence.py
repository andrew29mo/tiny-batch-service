import copy
import json
from pathlib import Path
import unittest

import numpy as np

from tiny_batch_service.experiment import validate_response
from tiny_batch_service.report import verify

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_recorded_runs_and_corruption_rejected_by_formal_verifier(self):
        directory = ROOT / "results/macos-arm64"
        if not (directory / "report.json").exists():
            self.skipTest("Recorded experiment not generated yet")
        report = json.loads((directory / "report.json").read_text())
        fixture = json.loads((directory / "fixture.json").read_text())
        reference = np.asarray(fixture["reference"]["data"], dtype=np.float32)
        verify(report, reference)
        mutations = [
            lambda r: r["cases"][0]["runs"][0]["client"].pop(),
            lambda r: r["cases"][0]["runs"].pop(),
            lambda r: r["cases"][0]["runs"][0]["server_records"].pop(),
            lambda r: r["cases"][0]["runs"][0]["summary"].update(success_per_second=1e9),
            lambda r: r["cases"][0]["runs"][0]["settings"].update(capacity=128),
            lambda r: r["cases"][0]["runs"][0]["rss_samples"][0].update(worker_rss_bytes=-1),
            lambda r: r["cases"][0]["runs"][0]["client"][0]["probabilities"].__setitem__(-1, .5),
            lambda r: r["cases"][0]["runs"][0]["lifecycle"].update(worker_alive_after_http_exit=True),
            lambda r: r["cases"][0]["runs"][0]["audit"].update(audit_evicted=1),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(fault=i):
                changed = copy.deepcopy(report)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify(changed, reference)


if __name__ == "__main__":
    unittest.main()
