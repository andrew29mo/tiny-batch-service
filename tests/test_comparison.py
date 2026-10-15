import unittest

from tiny_batch_service.report import comparison, paired_rows


class ComparisonTests(unittest.TestCase):
    def case(self):
        case = {"rate": 6000., "runs": []}
        for trial, ratio in enumerate((2., 1.4, 1.4, 3.)):
            for mode in ("single", "dynamic"):
                case["runs"].append({"trial": trial, "mode": mode,
                    "summary": {"success_per_second": 100. if mode == "single" else 100. * ratio,
                                "arrival_fidelity_passed": trial in (1, 2), "rejection_rate": .2 if mode == "single" else 0.,
                                "scheduled_success_latency_ms": {"p95": 10. if mode == "single" else 3.}}})
        return case

    def test_post_run_passing_subset_cannot_become_benefit_estimate(self):
        case = self.case()
        result = comparison(case, 4)
        self.assertEqual(result["basis"], "all_prespecified_pairs")
        self.assertEqual(result["passing_arrival_pairs"], 2)
        self.assertEqual(result["total_pairs"], 4)
        self.assertEqual(result["verdict"], "inconclusive_arrival_fidelity")
        self.assertEqual([p["descriptive_throughput_ratio"] for p in result["pairs"]], [2., 1.4, 1.4, 3.])
        row = paired_rows({"cases": [case], "repeats": 4})[0]
        self.assertEqual(row[1], "1.700 [1.400, 3.000]")
        self.assertEqual(row[-2], "2/4")
        self.assertEqual(row[-1], "inconclusive_arrival_fidelity")

    def test_all_passing_is_still_only_a_coarse_descriptive_check(self):
        case = self.case()
        for run in case["runs"]:
            run["summary"]["arrival_fidelity_passed"] = True
        self.assertEqual(comparison(case, 4)["verdict"], "descriptive_only_all_pairs_pass_coarse_check")

    def test_missing_pairs_are_rejected_not_silently_excluded(self):
        case = self.case()
        case["runs"].pop()
        with self.assertRaises(ValueError):
            comparison(case, 4)


if __name__ == "__main__":
    unittest.main()
