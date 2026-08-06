"""Authoritative stability-index -> problem_id map for the SciFy runs.

The soft-stability runs (``ss_rate_dicts_all*``) iterate the ``scify_icl`` dataset
(``SciFyDataset`` in mythbusters) in the order of the ``potluck-dry-run`` subset
(106 problems). The stability files are named by that positional index
(``<di>.json``) but carry no problem id, so any join to assessment metadata
needs this map.

``SciFyDataset`` always reads exactly one file per problem:
``<assessment_dir>/<problem_id>/ICL-agent-1-000.json`` (see
``mythbusters/src/exp_helpers/datasets/scify.py``). That directory is
``mythbusters/data/scify/v1/system-outputs-dry-run-&-potluck/potluck-dry-run-raw-outputs``
-- registered as ``scify_potluck_raw`` in ``data/registry.yaml``. It is NOT
``claimspy_v1`` (``claimspy_v1_ICL_Agent_2``): that is a different corpus with a
disjoint problem-id namespace (``alloys_0001`` vs. ``matsci_db_001``) that happens
to share the same assessment JSON schema. Passing ``claimspy_v1`` as
``assessment_dir`` here will silently match 0 problems.

The ordering is verified by matching each run's stored ``true_label`` against the
problem's own ``ICL-agent-1-000.json`` likert score in potluck order: 103/106
exact matches. Do NOT use ``claimspy_v1`` sorted-filename order: it is a join
against the wrong corpus entirely, not merely a misordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .claimspy import _domain_from_problem_id, _parse_assessment

_SUBSET = "subsets/potluck-dry-run/sprint1-problems.jsonl"
_GOLD = "subsets/potluck-dry-run/sprint1-gold.jsonl"
_ASSESSMENT_FILENAME = "ICL-agent-1-000.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def potluck_order(eval_data_dir: Path) -> list[dict[str, Any]]:
    """Return the 106 problems in stability-index (``di``) order."""
    problems = _read_jsonl(Path(eval_data_dir) / _SUBSET)
    gold = {g["problem_id"]: g.get("likert_score") for g in _read_jsonl(Path(eval_data_dir) / _GOLD)}
    out = []
    for di, prob in enumerate(problems):
        pid = prob["problem_id"]
        out.append({
            "di": di,
            "problem_id": pid,
            "domain": _domain_from_problem_id(pid),
            "claim": prob.get("claim"),
            "gold_likert": gold.get(pid),
        })
    return out


def build_index_map(eval_data_dir: Path, assessment_dir: Path) -> list[dict[str, Any]]:
    """di -> problem_id with claim, gold, and the matching assessment (if any).

    Reads ``<assessment_dir>/<problem_id>/ICL-agent-1-000.json`` directly, the
    one file ``SciFyDataset`` reads per problem. Earlier this globbed every
    ``*.json`` in the problem's folder and kept whichever sorted last
    alphabetically -- some problems have multiple system outputs (e.g. a
    ``PoVE-multiagent.json`` alongside ``ICL-agent-1-000.json``), so that
    silently picked the wrong assessment for any problem with more than one
    file on disk.
    """
    rows = potluck_order(eval_data_dir)
    for row in rows:
        path = Path(assessment_dir) / row["problem_id"] / _ASSESSMENT_FILENAME
        a = _parse_assessment(path) if path.exists() else None
        row["assessed"] = a is not None
        if a is not None:
            row["assessment_path"] = str(path)
            row["likert_score"] = a.get("likert_score")
            row["continuous_score"] = a.get("continuous_score")
            row["confidence"] = a.get("confidence")
    return rows


def validate_against_run(index_map: list[dict[str, Any]], stability_run_dir: Path) -> tuple[int, int]:
    """Cross-check di->gold against a run's stored true_label. Returns (matches, n)."""
    matches = total = 0
    for row in index_map:
        f = Path(stability_run_dir) / f"{row['di']}.json"
        if not f.exists():
            continue
        obj = json.loads(f.read_text())
        true_label = next(
            (obj[k].get("true_label") for k in sorted(obj, key=int) if obj[k].get("true_label") is not None),
            None,
        )
        if true_label is None or row["gold_likert"] is None:
            continue
        total += 1
        matches += int(true_label == row["gold_likert"])
    return matches, total


def write_index_map(out_path: Path, eval_data_dir: Path, assessment_dir: Path) -> list[dict[str, Any]]:
    rows = build_index_map(eval_data_dir, assessment_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def load_scify_index_metadata(eval_data_dir: Path, assessment_dir: Path) -> dict[str, dict[str, Any]]:
    """Map stability-index (``di``) to metadata, in the same shape as
    ``claimspy.load_claimspy_metadata`` -- a drop-in replacement for reporting
    code that joins on ``str(example_id)``, but sourced from the corpus the
    stability runs actually iterate (see module docstring) instead of the
    unrelated ``claimspy_v1`` corpus.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in build_index_map(eval_data_dir, assessment_dir):
        if not row["assessed"]:
            continue
        out[str(row["di"])] = {
            "problem_id": row["problem_id"],
            "domain": row["domain"],
            "likert_score": row.get("likert_score"),
            "continuous_score": row.get("continuous_score"),
            "confidence": row.get("confidence"),
            "path": row.get("assessment_path"),
        }
    return out
