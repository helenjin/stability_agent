"""Semantic redundancy heuristics over atomic support sentences."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .redundancy import pearson

SEMANTIC_FEATURES = [
    "semantic_redundancy_score",
    "semantic_independence_score",
    "duplicate_pair_fraction",
    "entailment_pair_fraction",
    "same_role_pair_fraction",
    "complementary_pair_fraction",
    "tension_pair_fraction",
    "redundant_sentence_fraction",
    "support_concentration",
    "effective_support_count",
    "independent_support_count",
    "redundancy_excess",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "by",
    "can",
    "could",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "which",
    "while",
    "with",
    "within",
}

NEGATION = {"no", "not", "never", "without", "unlikely", "infeasible", "unverified"}
POSITIVE = {"improve", "improves", "improved", "feasible", "likely", "stable", "supports"}
NEGATIVE = {
    "degrade",
    "degrades",
    "degraded",
    "infeasible",
    "unlikely",
    "unstable",
    "contradicts",
    "unverified",
}


def load_support_texts_by_index(assessment_dir: Path) -> dict[str, dict[str, Any]]:
    """Load support sentences by sorted assessment-file index."""

    files = [
        path
        for path in assessment_dir.glob("*/*.json")
        if not path.name.startswith("runtime_config")
    ]
    files = sorted(files, key=lambda path: (path.parent.name, path.name))
    out: dict[str, dict[str, Any]] = {}
    for idx, path in enumerate(files):
        assessment = _load_assessment(path)
        if not assessment:
            continue
        support_texts = [
            item.get("text", "")
            for item in assessment.get("explanation", [])
            if item.get("text") and item.get("evidence") != ["claim"]
        ]
        out[str(idx)] = {
            "problem_id": assessment.get("problem_id") or path.parent.name,
            "path": str(path),
            "support_texts": support_texts,
        }
    return out


def semantic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_text = [row for row in rows if row.get("semantic_text_count")]
    return {
        "n_with_text": len(rows_with_text),
        "correlation_with_avg_stability": {
            feature: pearson(
                [row.get(feature) for row in rows_with_text],
                [row.get("avg_stability") for row in rows_with_text],
            )
            for feature in SEMANTIC_FEATURES
        },
        "correlation_with_radius_insensitive_disagreement": {
            feature: pearson(
                [row.get(feature) for row in rows_with_text],
                [_has_pattern(row, "radius_insensitive_disagreement") for row in rows_with_text],
            )
            for feature in SEMANTIC_FEATURES
        },
        "group_means": {
            "radius_insensitive_disagreement": _means(
                [row for row in rows_with_text if _has_pattern(row, "radius_insensitive_disagreement")]
            ),
            "non_radius_insensitive_disagreement": _means(
                [row for row in rows_with_text if not _has_pattern(row, "radius_insensitive_disagreement")]
            ),
            "low_stability": _means(
                [row for row in rows_with_text if _lte(row.get("avg_stability"), 0.30)]
            ),
            "high_stability": _means(
                [row for row in rows_with_text if _gte(row.get("avg_stability"), 0.80)]
            ),
        },
    }


def analyze_support_texts(texts: list[str]) -> dict[str, Any]:
    texts = [text.strip() for text in texts if text and text.strip()]
    n = len(texts)
    if n < 2:
        return _empty_features(n)

    pair_labels: Counter[str] = Counter()
    redundant_edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            label = classify_pair(texts[i], texts[j])
            pair_labels[label] += 1
            if label in {"duplicate_or_paraphrase", "entailment", "same_evidence_role"}:
                redundant_edges.append((i, j))

    total_pairs = n * (n - 1) / 2
    clusters = _clusters(n, redundant_edges)
    redundant_sentences = sum(size for size in clusters if size > 1)
    independent_support_count = len(clusters)
    largest_cluster = max(clusters) if clusters else 0
    semantic_redundancy_score = (
        0.4 * pair_labels["duplicate_or_paraphrase"] / total_pairs
        + 0.3 * pair_labels["entailment"] / total_pairs
        + 0.2 * pair_labels["same_evidence_role"] / total_pairs
        + 0.1 * largest_cluster / n
    )
    return {
        "semantic_text_count": n,
        "num_duplicate_pairs": pair_labels["duplicate_or_paraphrase"],
        "num_entailment_pairs": pair_labels["entailment"],
        "num_same_role_pairs": pair_labels["same_evidence_role"],
        "num_complementary_pairs": pair_labels["complementary"],
        "num_tension_pairs": pair_labels["contradictory_or_tension"],
        "duplicate_pair_fraction": pair_labels["duplicate_or_paraphrase"] / total_pairs,
        "entailment_pair_fraction": pair_labels["entailment"] / total_pairs,
        "same_role_pair_fraction": pair_labels["same_evidence_role"] / total_pairs,
        "complementary_pair_fraction": pair_labels["complementary"] / total_pairs,
        "tension_pair_fraction": pair_labels["contradictory_or_tension"] / total_pairs,
        "redundancy_cluster_count": sum(1 for size in clusters if size > 1),
        "largest_redundancy_cluster_size": largest_cluster,
        "mean_redundancy_cluster_size": mean(clusters) if clusters else 0.0,
        "redundant_sentence_fraction": redundant_sentences / n,
        "independent_support_count": independent_support_count,
        "support_concentration": largest_cluster / n,
        "effective_support_count": _effective_count(clusters),
        "redundancy_excess": n - independent_support_count,
        "semantic_redundancy_score": semantic_redundancy_score,
        "semantic_independence_score": independent_support_count / n,
    }


def classify_pair(left: str, right: str) -> str:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return "complementary"

    jaccard = _jaccard(left_tokens, right_tokens)
    containment = max(
        len(left_tokens & right_tokens) / len(left_tokens),
        len(left_tokens & right_tokens) / len(right_tokens),
    )
    if _has_tension(left_tokens, right_tokens) and jaccard >= 0.18:
        return "contradictory_or_tension"
    if _normalize(left) == _normalize(right) or jaccard >= 0.72:
        return "duplicate_or_paraphrase"
    if containment >= 0.72 and _length_ratio(left_tokens, right_tokens) >= 1.35:
        return "entailment"
    if containment >= 0.88:
        return "duplicate_or_paraphrase"
    if jaccard >= 0.34 or containment >= 0.58:
        return "same_evidence_role"
    return "complementary"


def _load_assessment(path: Path) -> dict[str, Any] | None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    solution = obj.get("solution", {}) if isinstance(obj, dict) else {}
    if "assessment" in solution:
        return solution["assessment"]
    json_output = solution.get("json_output")
    if isinstance(json_output, str):
        return json.loads(json_output)
    if isinstance(json_output, dict):
        return json_output
    return None


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _normalize(text: str) -> str:
    return " ".join(sorted(_tokens(text)))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _length_ratio(left: set[str], right: set[str]) -> float:
    return max(len(left), len(right)) / max(1, min(len(left), len(right)))


def _has_tension(left: set[str], right: set[str]) -> bool:
    left_negated = bool(left & NEGATION)
    right_negated = bool(right & NEGATION)
    polarity_flip = bool(left & POSITIVE and right & NEGATIVE) or bool(
        left & NEGATIVE and right & POSITIVE
    )
    return left_negated != right_negated or polarity_flip


def _clusters(n: int, edges: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for a, b in edges:
        union(a, b)
    counts: Counter[int] = Counter(find(i) for i in range(n))
    return list(counts.values())


def _effective_count(clusters: list[int]) -> float:
    total = sum(clusters)
    if total == 0:
        return 0.0
    entropy = 0.0
    for size in clusters:
        p = size / total
        entropy -= p * math.log(p)
    return math.exp(entropy)


def _empty_features(n: int) -> dict[str, Any]:
    out = {feature: 0.0 for feature in SEMANTIC_FEATURES}
    out.update(
        {
            "semantic_text_count": n,
            "independent_support_count": n,
            "semantic_independence_score": 1.0 if n else 0.0,
            "effective_support_count": float(n),
        }
    )
    return out


def _means(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for feature in ["semantic_text_count", *SEMANTIC_FEATURES]:
        vals = [row.get(feature) for row in rows if row.get(feature) is not None]
        out[feature] = mean(vals) if vals else None
    return out


def _has_pattern(row: dict[str, Any], pattern: str) -> int:
    return int(any(item.get("pattern") == pattern for item in row.get("patterns", [])))


def _lte(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def _gte(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold
