from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stability_agent.reporting.popqa_summary import (
    load_popqa_metadata,
    summarize_popqa_source_effects,
)


class PopQASummaryTests(unittest.TestCase):
    def test_loads_csv_and_summarizes_popularity_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "popqa.csv"
            path.write_text(
                "\n".join(
                    [
                        "id,subj,prop,obj,s_pop,o_pop,question,possible_answers",
                        '0,George Rankin,occupation,politician,142,25692,What is George Rankin\'s occupation?,"[""politician""]"',
                        '1,John Mayne,occupation,journalist,236,24952,What is John Mayne\'s occupation?,"[""journalist""]"',
                        '2,Escape,producer,Basil Dean,198,802,Who was the producer of Escape?,"[""Basil Dean""]"',
                    ]
                ),
                encoding="utf-8",
            )
            metadata = load_popqa_metadata(path)

        result = {
            "source_summary": [
                {"source_name": "web", "avg_stability": 0.5, "low_n": 1, "high_n": 1},
                {"source_name": "parametric", "avg_stability": 0.8, "low_n": 0, "high_n": 2},
            ],
            "source_pair_summary": [],
            "rows": [
                {"example_id": "0", "source_name": "web", "avg_stability": 0.2},
                {"example_id": "0", "source_name": "parametric", "avg_stability": 0.9},
                {"example_id": "1", "source_name": "web", "avg_stability": 0.6},
                {"example_id": "1", "source_name": "parametric", "avg_stability": 0.8},
            ],
        }

        summary = summarize_popqa_source_effects(result, metadata)

        self.assertEqual(metadata["0"]["question"], "What is George Rankin's occupation?")
        subject_rows = summary["subject_popularity_summary"]
        self.assertTrue(any(row["bucket"] in {"low", "mid", "high"} for row in subject_rows))
        self.assertEqual(summary["source_gap_examples"][0]["example_id"], "0")


if __name__ == "__main__":
    unittest.main()
