import copy
import json
from pathlib import Path
import unittest

import numpy as np

from tiny_batch_service.experiment import summarize
from tiny_batch_service.report import verify

ROOT = Path(__file__).resolve().parents[1]


class EvidenceTests(unittest.TestCase):
    def test_recorded_runs_and_corruption_rejected_by_formal_verifier(self):
        directory = ROOT / "results/macos-arm64"
        self.assertTrue((directory / "report.json").exists(), "Committed experiment is required evidence")
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
            lambda r: r["cases"][0]["runs"][0]["rss_samples"][0].update(at_s=10000),
            lambda r: r["cases"][0]["runs"][0]["rss_samples"][0].update(finished_s=-10000),
            lambda r: r["cases"][0]["runs"][0]["memory_window"].update(end_s=0),
            lambda r: r["cases"][0]["runs"][0]["server_records"][0].update(compute_finished_s=10000),
            lambda r: r["cases"][0]["runs"][0]["batches"][0].update(finished_s=10000),
            lambda r: r["cases"][0]["runs"][0]["client"][0].update(headers_sent_s=-10000),
            lambda r: r["cases"][0]["runs"][0]["client"][0].update(dispatch_s=float("nan")),
            lambda r: r["cases"][0]["runs"][0]["audit"].update(peak_occupancy=63),
            lambda r: r["cases"][0]["runs"][0]["audit"]["counts"].update(cancelled=1),
            lambda r: r["cases"][0]["runs"][0]["workload"].update(loaded_model_sha256="0" * 64),
            lambda r: r["cases"][0]["runs"][0]["workload"].update(fixture_sha256="0" * 64),
            lambda r: r["cases"][0]["comparison"].update(verdict="confirmed_batching_benefit"),
            lambda r: r["cases"][0]["runs"][0].update(rss_samples=r["cases"][0]["runs"][0]["rss_samples"][:1]),
        ]
        for i, mutation in enumerate(mutations):
            with self.subTest(fault=i):
                changed = copy.deepcopy(report)
                mutation(changed)
                with self.assertRaises(ValueError):
                    verify(changed, reference)
        # Do not let an unchanged summary/hash be the only reason timestamp faults fail.
        for mutation in mutations[9:]:
            changed = copy.deepcopy(report)
            mutation(changed)
            with self.assertRaises(ValueError):
                run = changed["cases"][0]["runs"][0]
                run["summary"] = summarize(run, changed["cases"][0]["trace"])
                verify(changed, reference)


if __name__ == "__main__":
    unittest.main()
