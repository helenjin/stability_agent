"""Canonical schema for a reasoning-with-evidence example.

Every dataset adapter normalizes its corpus into ``Example`` objects so that
analysis and reporting never touch raw formats. The shape mirrors what the
ClaimSpy assessments encode: a claim/question, a pool of source-typed evidence
units, and an ordered reasoning chain whose steps cite those units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Where an evidence unit comes from — the parametric/web/non-web axis."""

    PARAMETRIC = "parametric"   # model's own knowledge / reasoning / synthesis
    WEB = "web"                 # live web-search results
    ARTIFACT = "artifact"       # provided document / paper text (non-web)
    CLAIM = "claim"             # the claim under assessment itself
    OTHER = "other"


# Prefix-based normalization of the free-form evidence keys seen in the corpora
# (e.g. "web_ti64_melt" -> WEB, "reasoning_diffusion_calc" -> PARAMETRIC).
_SOURCE_PATTERNS: list[tuple[str, SourceType]] = [
    (r"^claim", SourceType.CLAIM),
    (r"^artifact", SourceType.ARTIFACT),
    (r"^web", SourceType.WEB),
    (r"^(reasoning|parametric|dft|simulation|calculation|database|observation|param)", SourceType.PARAMETRIC),
]


def normalize_source(raw_key: str) -> SourceType:
    """Map a raw evidence key to a SourceType by prefix."""
    key = raw_key.strip().lower()
    for pattern, source in _SOURCE_PATTERNS:
        if re.match(pattern, key):
            return source
    return SourceType.OTHER


@dataclass(frozen=True)
class EvidenceUnit:
    id: str                     # original key, e.g. "web1", "reasoning1"
    source: SourceType
    text: str = ""
    raw_type: str = ""          # original key prefix, before normalization


@dataclass(frozen=True)
class ReasoningStep:
    index: int
    text: str
    evidence_ids: tuple[str, ...] = ()   # ids into Example.evidence


@dataclass
class Example:
    """One reasoning instance, normalized across datasets."""

    example_id: str
    dataset: str
    claim: str
    evidence: dict[str, EvidenceUnit] = field(default_factory=dict)
    steps: list[ReasoningStep] = field(default_factory=list)
    domain: str | None = None
    verdict: float | None = None        # the model's judgment (e.g. likert)
    quality_score: float | None = None  # correctness grade (e.g. continuous_score)
    gold_label: float | None = None     # ground truth, if joined
    meta: dict[str, Any] = field(default_factory=dict)

    # --- convenience for per-step source-stability analysis ---
    def step_source_types(self, step: ReasoningStep) -> set[SourceType]:
        return {self.evidence[e].source for e in step.evidence_ids if e in self.evidence}

    def source_type_counts(self) -> dict[SourceType, int]:
        counts: dict[SourceType, int] = {}
        for unit in self.evidence.values():
            counts[unit.source] = counts.get(unit.source, 0) + 1
        return counts

    @property
    def n_steps(self) -> int:
        return len(self.steps)
