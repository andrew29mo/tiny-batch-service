"""Hand-calculated oracles challenge the production summary, including dropped work."""

import copy
import unittest

from tiny_batch_service.experiment import make_trace, summarize


class MetricTests(unittest.TestCase):
    def fixture(self):
        trace = make_trace(4, 100)
        # One success, a running timeout, a queued timeout and a capacity rejection.
        states = [("success", 200, .021, .024, .002, .020),
                  ("timeout", 504, .261, .262, .030, .300),
                  ("timeout", 504, .271, .272, None, None),
                  ("overloaded", 429, .0311, .032, None, None)]
        clients, servers = [], []
        for item, (status, code, terminal, decoded, started, computed) in zip(trace, states):
            clients.append({**item, "status": status, "http_status": code,
                            "dispatch_s": item["at_s"] + .0005, "headers_sent_s": item["at_s"] + .0007,
                            "finished_s": decoded, "output_checked": status == "success"})
            servers.append({"id": item["id"], "status": status, "received_s": item["at_s"] + .001,
                            "deadline_s": item["at_s"] + .251, "terminal_s": terminal,
                            "started_s": started, "compute_finished_s": computed,
                            "phase_at_terminal": "running" if started is not None else "queued",
                            "reason": "request_deadline" if status == "timeout" else None})
        return trace, {"client": clients, "server_records": servers,
                       "settings": {"capacity": 2, "batch_size": 1, "wait_ms": 0.},
                       "batches": [{"ids": ["r000000"], "rows": 1, "started_s": .0021,
                                    "finished_s": .020, "inference_ms": 1.},
                                   {"ids": ["r000001"], "rows": 1, "started_s": .0301,
                                    "finished_s": .300, "inference_ms": 1.}],
                       "audit": {"occupancy": 0, "running": 0, "queued": 0, "audit_evicted": 0,
                                 "batch_evicted": 0, "peak_occupancy": 2, "capacity": 2, "batch_size": 1,
                                 "wait_ms": 0., "healthy": True, "closing": False,
                                 "counts": {"offered": 12, "admitted": 11, "success": 9, "timeout": 2,
                                            "overloaded": 1, "computed_rows": 10, "discarded_computed_rows": 1}},
                       "lifecycle": {"forced_http_kill": False, "worker_alive_after_http_exit": False, "http_exit_code": 0},
                       "memory_window": {"start_s": -.1, "end_s": .31},
                       "rss_samples": [{"at_s": t, "finished_s": t + .0001,
                                        "http_rss_bytes": h, "worker_rss_bytes": w}
                                       for t, h, w in ((-.1, 10, 100), (0., 100, 10), (.15, 20, 20), (.30, 30, 30))]}

    def test_manual_oracle_counts_drain_scheduled_latency_and_paired_rss(self):
        trace, run = self.fixture()
        result = summarize(run, trace)
        self.assertEqual(result["counts"], {"success": 1, "timeout": 2, "overloaded": 1})
        self.assertAlmostEqual(result["observation_s"], .300)  # Last response .272; native compute drains to .300.
        self.assertAlmostEqual(result["success_per_second"], 1 / .300)
        self.assertAlmostEqual(result["success_latency_ms"]["p95"], 23.5)
        self.assertAlmostEqual(result["scheduled_success_latency_ms"]["p95"], 24.)
        self.assertAlmostEqual(result["all_terminal_latency_ms"]["p50"], 137.5)
        self.assertAlmostEqual(result["scheduled_all_terminal_latency_ms"]["p50"], 138.)
        self.assertAlmostEqual(result["queue_ms"]["p50"], 10.)
        self.assertEqual(result["batch_sizes"], {"1": 2})
        self.assertEqual(result["peak_paired_rss_bytes"], 110)  # Independent peaks would incorrectly sum to 200.
        self.assertAlmostEqual(result["max_rss_gap_ms"], 150.)
        self.assertAlmostEqual(result["max_paired_read_ms"], .1)

    def test_recomputed_summary_cannot_hide_incomplete_or_contradictory_raw_records(self):
        trace, baseline = self.fixture()
        faults = [lambda r: r["client"].pop(), lambda r: r["server_records"].pop(),
                  lambda r: r["batches"].pop(),
                  lambda r: r["server_records"][0].update(compute_finished_s=1.),
                  lambda r: r["rss_samples"][0].update(at_s=10000),
                  lambda r: r["client"][0].update(finished_s=float("nan")),
                  lambda r: r["audit"].update(peak_occupancy=1)]
        for i, fault in enumerate(faults):
            with self.subTest(fault=i):
                changed = copy.deepcopy(baseline)
                fault(changed)
                with self.assertRaises(ValueError):
                    summarize(changed, trace)


if __name__ == "__main__":
    unittest.main()
