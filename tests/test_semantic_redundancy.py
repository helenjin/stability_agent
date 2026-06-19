from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stability_agent.analysis.semantic_redundancy import (
    analyze_support_texts,
    classify_pair,
    load_support_texts_by_index,
)


class ClassifyPairTests(unittest.TestCase):
    def test_duplicate_or_paraphrase_for_high_overlap(self) -> None:
        self.assertEqual(
            classify_pair(
                "The catalyst remains active after 100 cycles.",
                "The catalyst remains active after 100 cycles.",
            ),
            "duplicate_or_paraphrase",
        )

    def test_entailment_for_containment_with_length_gap(self) -> None:
        self.assertEqual(
            classify_pair(
                "The material is stable at 500 C in air.",
                "The material is stable at 500 C in air during long exposure and repeated thermal cycling.",
            ),
            "entailment",
        )

    def test_same_evidence_role_for_partial_overlap(self) -> None:
        self.assertEqual(
            classify_pair(
                "High ionic conductivity supports fast lithium transport.",
                "Low migration barriers support fast lithium transport.",
            ),
            "same_evidence_role",
        )

    def test_tension_for_negated_high_overlap(self) -> None:
        self.assertEqual(
            classify_pair(
                "The compound is stable in air at room temperature.",
                "The compound is not stable in air at room temperature.",
            ),
            "contradictory_or_tension",
        )

    def test_complementary_for_distinct_support(self) -> None:
        self.assertEqual(
            classify_pair(
                "The alloy has high electrical conductivity.",
                "The oxide scale prevents chemical degradation.",
            ),
            "complementary",
        )


class AnalyzeSupportTextsTests(unittest.TestCase):
    def test_single_text_has_no_redundancy(self) -> None:
        result = analyze_support_texts(["Only one support item is present."])
        self.assertEqual(result["semantic_text_count"], 1)
        self.assertEqual(result["independent_support_count"], 1)
        self.assertEqual(result["redundancy_excess"], 0)
        self.assertEqual(result["semantic_independence_score"], 1.0)

    def test_redundant_cluster_counts_extra_support_items(self) -> None:
        result = analyze_support_texts(
            [
                "The catalyst remains active after 100 cycles.",
                "The catalyst remains active after 100 cycles.",
                "The material is stable at 500 C in air.",
                "The material is stable at 500 C in air during long exposure and repeated thermal cycling.",
                "The synthesis uses inexpensive precursors.",
            ]
        )
        self.assertEqual(result["semantic_text_count"], 5)
        self.assertEqual(result["num_duplicate_pairs"], 1)
        self.assertEqual(result["num_entailment_pairs"], 1)
        self.assertEqual(result["redundancy_cluster_count"], 2)
        self.assertEqual(result["independent_support_count"], 3)
        self.assertEqual(result["redundancy_excess"], 2)
        self.assertAlmostEqual(result["redundant_sentence_fraction"], 4 / 5)
        self.assertAlmostEqual(result["semantic_independence_score"], 3 / 5)

    def test_same_role_transitively_forms_one_cluster(self) -> None:
        result = analyze_support_texts(
            [
                "High ionic conductivity supports fast lithium transport.",
                "Low migration barriers support fast lithium transport.",
                "Fast lithium transport supports high-rate battery performance.",
            ]
        )
        self.assertGreaterEqual(result["num_same_role_pairs"], 1)
        self.assertLess(result["independent_support_count"], result["semantic_text_count"])
        self.assertGreater(result["redundancy_excess"], 0)


class LoadSupportTextsTests(unittest.TestCase):
    def test_loads_both_assessment_schema_variants_and_skips_claim_restatement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a_problem" / "ask_gemini_web_search.json"
            first.parent.mkdir()
            first.write_text(
                json.dumps(
                    {
                        "solution": {
                            "json_output": json.dumps(
                                {
                                    "problem_id": "a_problem",
                                    "explanation": [
                                        {"text": "The claim restates the target.", "evidence": ["claim"]},
                                        {"text": "Support one.", "evidence": ["web1"]},
                                    ],
                                }
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )

            second = root / "b_problem" / "claimspy_v2.json"
            second.parent.mkdir()
            second.write_text(
                json.dumps(
                    {
                        "solution": {
                            "assessment": {
                                "problem_id": "b_problem",
                                "explanation": [
                                    {"text": "Support two.", "evidence": ["E1"]},
                                    {"text": "Support three.", "evidence": ["E2"]},
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_support_texts_by_index(root)

        self.assertEqual(list(loaded), ["0", "1"])
        self.assertEqual(loaded["0"]["problem_id"], "a_problem")
        self.assertEqual(loaded["0"]["support_texts"], ["Support one."])
        self.assertEqual(loaded["1"]["problem_id"], "b_problem")
        self.assertEqual(loaded["1"]["support_texts"], ["Support two.", "Support three."])


if __name__ == "__main__":
    unittest.main()
