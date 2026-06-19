"""Prepare PopQA metadata and question files for stability experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_popqa_rows(
    input_path: Path | None = None,
    *,
    hf_repo: str = "akariasai/PopQA",
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load PopQA rows from a local file or Hugging Face datasets."""

    if input_path is not None:
        return _load_rows_from_file(input_path)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "Loading PopQA from Hugging Face requires `datasets`. "
            "Install with `pip install -e \".[data]\"` or provide --input."
        ) from exc

    dataset = load_dataset(hf_repo)
    if split:
        selected = dataset[split]
    elif hasattr(dataset, "keys"):
        first_split = next(iter(dataset.keys()))
        selected = dataset[first_split]
    else:  # pragma: no cover - defensive
        selected = dataset
    return [dict(row) for row in selected]


def normalize_popqa_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize PopQA rows to a stable local schema."""

    normalized = []
    for idx, row in enumerate(rows):
        normalized_row = {
            "id": str(row.get("id", idx)),
            "question": row.get("question"),
            "subject": row.get("subj") or row.get("subject"),
            "property": row.get("prop") or row.get("property"),
            "object": row.get("obj") or row.get("object"),
            "possible_answers": _parse_answers(row.get("possible_answers")),
            "subject_popularity": _maybe_float(row.get("s_pop") or row.get("subject_popularity")),
            "object_popularity": _maybe_float(row.get("o_pop") or row.get("object_popularity")),
        }
        normalized.append(normalized_row)

    subject_cutoffs = _tertiles(
        [row["subject_popularity"] for row in normalized if row["subject_popularity"] is not None]
    )
    object_cutoffs = _tertiles(
        [row["object_popularity"] for row in normalized if row["object_popularity"] is not None]
    )
    for row in normalized:
        row["subject_popularity_bucket"] = _bucket(row["subject_popularity"], subject_cutoffs)
        row["object_popularity_bucket"] = _bucket(row["object_popularity"], object_cutoffs)
    return normalized


def write_popqa_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def write_popqa_questions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write question-focused JSONL for downstream answer/retrieval pipelines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = {
                "id": row["id"],
                "question": row["question"],
                "possible_answers": row["possible_answers"],
                "subject": row["subject"],
                "property": row["property"],
                "object": row["object"],
                "subject_popularity": row["subject_popularity"],
                "object_popularity": row["object_popularity"],
                "subject_popularity_bucket": row["subject_popularity_bucket"],
                "object_popularity_bucket": row["object_popularity_bucket"],
                "retrieval_query": row["question"],
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_popqa_preview(path: Path, rows: list[dict[str, Any]], limit: int = 8) -> None:
    """Write a short markdown preview to inspect the exported sample."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PopQA Preview",
        "",
        f"Rows: {len(rows)}",
        "",
        "| id | subject | property | object | subject pop | object pop | question |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| {row['id']} | {_truncate(row.get('subject'))} | {_truncate(row.get('property'))} | "
            f"{_truncate(row.get('object'))} | {_fmt(row.get('subject_popularity'))} | "
            f"{_fmt(row.get('object_popularity'))} | {_truncate(row.get('question'), 64)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare PopQA metadata and question files for stability experiments."
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional local PopQA file.")
    parser.add_argument(
        "--hf-repo",
        default="akariasai/PopQA",
        help="Hugging Face dataset repo if --input is omitted.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to load from Hugging Face. Defaults to the first available split.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("reports/popqa_metadata.json"),
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=Path("reports/popqa_questions.jsonl"),
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=Path("reports/popqa_preview.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_popqa_rows(args.input, hf_repo=args.hf_repo, split=args.split)
    rows = normalize_popqa_rows(rows)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    write_popqa_metadata(args.metadata_out, rows)
    write_popqa_questions(args.questions_out, rows)
    write_popqa_preview(args.preview_out, rows)
    print(
        f"Wrote {len(rows)} PopQA rows to "
        f"{args.metadata_out}, {args.questions_out}, and {args.preview_out}."
    )
    return 0


def _load_rows_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return list(data.values())
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported input format: {path}")


def _parse_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tertiles(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 3:
        return None
    ordered = sorted(values)
    first = ordered[len(ordered) // 3]
    second = ordered[(2 * len(ordered)) // 3]
    return (first, second)


def _bucket(value: float | None, cutoffs: tuple[float, float] | None) -> str:
    if value is None or cutoffs is None:
        return "unknown"
    low, high = cutoffs
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return str(value)


def _truncate(value: Any, max_chars: int = 24) -> str:
    text = str(value or "-")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
