"""Regenerate claimspy_followup_summary and claimspy_gap_audit_systematic.

Uses the corrected stability-index join (stability_agent.datasets.scify_index)
against scify_potluck_raw instead of the unrelated claimspy_v1 corpus. See
data/registry.yaml and stability_agent/datasets/scify_index.py for why.
"""

from __future__ import annotations

import json
from pathlib import Path

from stability_agent.datasets.scify_index import load_scify_index_metadata
from stability_agent.io import write_json, write_text
from stability_agent.reporting.claimspy_summary import (
    build_gap_audit,
    render_claimspy_markdown,
    render_gap_audit_markdown,
    summarize_claimspy_source_effects,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    result = json.loads((ROOT / "reports" / "source_comparison_balanced.json").read_text())
    metadata = load_scify_index_metadata(
        ROOT / "data" / "raw" / "evaluation_data",
        ROOT / "data" / "raw" / "scify_potluck_raw",
    )
    print(f"metadata rows: {len(metadata)}")

    followup = summarize_claimspy_source_effects(result, metadata)
    write_json(ROOT / "reports" / "claimspy_followup_summary.json", followup)
    write_text(ROOT / "reports" / "claimspy_followup_summary.md", render_claimspy_markdown(followup))

    audit = build_gap_audit(result, metadata)
    write_json(ROOT / "reports" / "claimspy_gap_audit_systematic.json", audit)
    write_text(ROOT / "reports" / "claimspy_gap_audit_systematic.md", render_gap_audit_markdown(audit))

    print(f"gap cases: {audit['n_gap_cases']} / {audit['n_total']}")
    print("done")


if __name__ == "__main__":
    main()
