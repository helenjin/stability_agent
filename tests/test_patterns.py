from __future__ import annotations

import unittest

from stability_agent.analysis.patterns import compare_source_pairs


class SourcePairComparisonTests(unittest.TestCase):
    def test_compares_sources_on_matched_examples(self) -> None:
        rows = [
            {"example_id": "1", "source_name": "web", "avg_stability": 0.2},
            {"example_id": "1", "source_name": "parametric", "avg_stability": 0.8},
            {"example_id": "2", "source_name": "web", "avg_stability": 0.6},
            {"example_id": "2", "source_name": "parametric", "avg_stability": 0.55},
            {"example_id": "3", "source_name": "web", "avg_stability": 0.9},
        ]

        result = compare_source_pairs(rows)

        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["left_source"], "parametric")
        self.assertEqual(row["right_source"], "web")
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean_delta_left_minus_right"], 0.275)
        self.assertEqual(row["left_more_stable_n"], 1)
        self.assertEqual(row["right_more_stable_n"], 0)
        self.assertEqual(row["similar_n"], 1)
        self.assertEqual(row["large_gap_n"], 1)


if __name__ == "__main__":
    unittest.main()
