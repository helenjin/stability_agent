from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stability_agent.datasets.popqa import (
    normalize_popqa_rows,
    write_popqa_metadata,
    write_popqa_preview,
    write_popqa_questions,
)


class PopQAPrepTests(unittest.TestCase):
    def test_normalize_and_write_outputs(self) -> None:
        rows = normalize_popqa_rows(
            [
                {
                    "id": "0",
                    "subj": "George Rankin",
                    "prop": "occupation",
                    "obj": "politician",
                    "s_pop": "142",
                    "o_pop": "25692",
                    "question": "What is George Rankin's occupation?",
                    "possible_answers": '["politician", "political leader"]',
                },
                {
                    "id": "1",
                    "subj": "John Mayne",
                    "prop": "occupation",
                    "obj": "journalist",
                    "s_pop": "236",
                    "o_pop": "24952",
                    "question": "What is John Mayne's occupation?",
                    "possible_answers": '["journalist"]',
                },
                {
                    "id": "2",
                    "subj": "Escape",
                    "prop": "producer",
                    "obj": "Basil Dean",
                    "s_pop": "198",
                    "o_pop": "802",
                    "question": "Who was the producer of Escape?",
                    "possible_answers": ["Basil Dean"],
                },
            ]
        )

        self.assertEqual(rows[0]["possible_answers"][0], "politician")
        self.assertIn(rows[0]["subject_popularity_bucket"], {"low", "mid", "high"})

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metadata_path = tmp_path / "metadata.json"
            questions_path = tmp_path / "questions.jsonl"
            preview_path = tmp_path / "preview.md"

            write_popqa_metadata(metadata_path, rows)
            write_popqa_questions(questions_path, rows)
            write_popqa_preview(preview_path, rows)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            question_lines = questions_path.read_text(encoding="utf-8").strip().splitlines()
            preview = preview_path.read_text(encoding="utf-8")

        self.assertEqual(len(metadata), 3)
        self.assertEqual(len(question_lines), 3)
        first_question = json.loads(question_lines[0])
        self.assertEqual(first_question["retrieval_query"], "What is George Rankin's occupation?")
        self.assertIn("# PopQA Preview", preview)


if __name__ == "__main__":
    unittest.main()
