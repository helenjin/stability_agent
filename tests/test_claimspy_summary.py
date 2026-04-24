from __future__ import annotations

import unittest

from stability_agent.claimspy_summary import summarize_claimspy_source_effects


class ClaimSpySummaryTests(unittest.TestCase):
    def test_summarizes_domains_and_quality_buckets(self) -> None:
        result = {
            "source_summary": [
                {"source_name": "web", "avg_stability": 0.3, "low_n": 1, "high_n": 0},
                {"source_name": "parametric", "avg_stability": 0.9, "low_n": 0, "high_n": 1},
            ],
            "source_pair_summary": [],
            "rows": [
                {"example_id": "0", "source_name": "web", "avg_stability": 0.2},
                {"example_id": "0", "source_name": "parametric", "avg_stability": 0.9},
                {"example_id": "1", "source_name": "web", "avg_stability": 0.8},
            ],
        }
        metadata = {
            "0": {"problem_id": "alloys_0001", "domain": "alloys", "continuous_score": 1.0},
            "1": {"problem_id": "batteries_0001", "domain": "batteries", "continuous_score": 0.0},
        }

        summary = summarize_claimspy_source_effects(result, metadata)

        domains = {(row["domain"], row["source_name"]): row for row in summary["domain_source_summary"]}
        self.assertAlmostEqual(domains[("alloys", "parametric")]["avg_stability"], 0.9)
        self.assertAlmostEqual(domains[("alloys", "web")]["avg_quality_score"], 1.0)
        self.assertEqual(summary["quality_buckets"][0]["bucket"], "high_quality")
        self.assertEqual(summary["source_gap_examples"][0]["problem_id"], "alloys_0001")


if __name__ == "__main__":
    unittest.main()
