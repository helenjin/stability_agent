"""ClaimSpy / SciFy adapter.

Each assessment file is one scientific-claim feasibility judgment: a claim, a
pool of source-typed evidence units, and a multi-step explanation whose steps
cite those units. This is the only corpus that natively carries per-step,
source-typed evidence, so it drives the canonical schema.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.schema import EvidenceUnit, Example, ReasoningStep, normalize_source
from .base import BaseAdapter


def _parse_assessment(path: Path) -> dict[str, Any] | None:
    """Pull the assessment dict out of one ClaimSpy run file."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    solution = obj.get("solution", {}) if isinstance(obj, dict) else {}
    assessment = solution.get("assessment")
    if assessment is None:
        json_output = solution.get("json_output")
        assessment = json.loads(json_output) if isinstance(json_output, str) else json_output
    return assessment if isinstance(assessment, dict) else None


def _sorted_files(assessment_dir: Path) -> list[Path]:
    files = [
        p for p in assessment_dir.glob("*/*.json")
        if not p.name.startswith("runtime_config")
    ]
    return sorted(files, key=lambda p: (p.parent.name, p.name))


def _domain_from_problem_id(problem_id: Any) -> str:
    if not isinstance(problem_id, str) or "_" not in problem_id:
        return "unknown"
    return problem_id.rsplit("_", 1)[0]


def _evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("source") or "")
    return str(value) if value is not None else ""


class ClaimSpyAdapter(BaseAdapter):
    """Normalize a ClaimSpy assessment corpus into canonical Examples.

    Examples are keyed by sorted-file index (matching the soft-stability run
    ordering, which ``load_claimspy_metadata`` also uses).
    """

    def __init__(self, assessment_dir: Path | str, name: str = "claimspy_v1") -> None:
        self.assessment_dir = Path(assessment_dir)
        self.name = name

    def iter_examples(self) -> Iterator[Example]:
        for idx, path in enumerate(_sorted_files(self.assessment_dir)):
            a = _parse_assessment(path)
            if a is None:
                continue

            evidence: dict[str, EvidenceUnit] = {}
            for key, value in (a.get("evidence") or {}).items():
                evidence[key] = EvidenceUnit(
                    id=key,
                    source=normalize_source(key),
                    text=_evidence_text(value),
                    raw_type=key.rstrip("0123456789_"),
                )

            steps: list[ReasoningStep] = []
            for i, block in enumerate(a.get("explanation") or []):
                if not isinstance(block, dict):
                    continue
                cites = block.get("evidence") or []
                steps.append(ReasoningStep(
                    index=i,
                    text=block.get("text", ""),
                    evidence_ids=tuple(c for c in cites if isinstance(c, str)),
                ))

            problem_id = a.get("problem_id") or path.parent.name
            claim = ""
            claim_unit = (a.get("evidence") or {}).get("claim")
            if isinstance(claim_unit, dict):
                claim = str(claim_unit.get("source") or "")

            yield Example(
                example_id=str(idx),
                dataset=self.name,
                claim=claim,
                evidence=evidence,
                steps=steps,
                domain=_domain_from_problem_id(problem_id),
                verdict=a.get("likert_score"),
                quality_score=a.get("continuous_score"),
                meta={
                    "problem_id": problem_id,
                    "confidence": a.get("confidence"),
                    "path": str(path),
                },
            )


def load_claimspy_metadata(assessment_dir: Path) -> dict[str, dict[str, Any]]:
    """Map sorted ClaimSpy assessment index to lightweight metadata.

    Kept as a standalone loader (used by reporting summaries) alongside the
    richer ``ClaimSpyAdapter``.
    """
    out: dict[str, dict[str, Any]] = {}
    for idx, path in enumerate(_sorted_files(Path(assessment_dir))):
        a = _parse_assessment(path)
        if a is None:
            continue
        problem_id = a.get("problem_id") or path.parent.name
        out[str(idx)] = {
            "problem_id": problem_id,
            "domain": _domain_from_problem_id(problem_id),
            "likert_score": a.get("likert_score"),
            "continuous_score": a.get("continuous_score"),
            "confidence": a.get("confidence"),
            "path": str(path),
        }
    return out
