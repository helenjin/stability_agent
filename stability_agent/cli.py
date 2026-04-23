"""Command-line entrypoint for stability-agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import AnalysisConfig, StabilityAnalyzer
from .io import write_json, write_text
from .report import render_markdown_report
from .semantic_redundancy import load_support_texts_by_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze soft-stability JSON outputs and compare low/high examples."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Stability JSON files or directories containing numeric JSON files.",
    )
    parser.add_argument("--low-threshold", type=float, default=0.30)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument(
        "--assessment-dir",
        type=Path,
        default=None,
        help="Optional directory of assessment JSON files for semantic redundancy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyzer = StabilityAnalyzer(
        AnalysisConfig(
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            top_k=args.top_k,
        )
    )
    support_texts_by_index = (
        load_support_texts_by_index(args.assessment_dir) if args.assessment_dir else None
    )
    result = analyzer.analyze_paths(args.paths, support_texts_by_index)
    if args.json_out:
        write_json(args.json_out, result)
    markdown = render_markdown_report(result)
    if args.markdown_out:
        write_text(args.markdown_out, markdown)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
