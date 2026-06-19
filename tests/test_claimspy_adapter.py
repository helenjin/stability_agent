"""Validate the ClaimSpy adapter produces well-formed canonical Examples.

Runs against the real corpus via the data/ symlink; skips if it is absent.
"""

import pytest

from stability_agent import data_paths as dp
from stability_agent.core.schema import Example, SourceType, normalize_source
from stability_agent.datasets.claimspy import ClaimSpyAdapter


def test_normalize_source_prefixes():
    assert normalize_source("web1") is SourceType.WEB
    assert normalize_source("web_ti64_melt") is SourceType.WEB
    assert normalize_source("reasoning1") is SourceType.PARAMETRIC
    assert normalize_source("parametric_knowledge") is SourceType.PARAMETRIC
    assert normalize_source("artifact1") is SourceType.ARTIFACT
    assert normalize_source("claim") is SourceType.CLAIM
    assert normalize_source("mystery_key") is SourceType.OTHER


@pytest.fixture
def corpus_dir():
    path = dp.raw_corpus("claimspy_v1")
    if not path.exists():
        pytest.skip("claimspy_v1 corpus not available (run scripts/setup_data.sh)")
    return path


def test_adapter_yields_examples(corpus_dir):
    examples = ClaimSpyAdapter(corpus_dir).examples()
    assert len(examples) > 0
    ex = examples[0]
    assert isinstance(ex, Example)
    assert ex.dataset == "claimspy_v1"
    assert ex.claim                      # claim text recovered
    assert ex.n_steps >= 1               # multi-step reasoning present
    assert ex.evidence                   # evidence pool populated


def test_steps_cite_typed_evidence(corpus_dir):
    examples = ClaimSpyAdapter(corpus_dir).examples()
    # Every step should cite >=1 evidence id that resolves to a typed unit.
    total_steps = cited_steps = 0
    sources_seen: set[SourceType] = set()
    for ex in examples:
        for step in ex.steps:
            total_steps += 1
            resolved = ex.step_source_types(step)
            if resolved:
                cited_steps += 1
                sources_seen |= resolved
    assert total_steps > 0
    # The corpus has per-step evidence on ~all steps.
    assert cited_steps / total_steps > 0.9
    # And the source contrast is present at the evidence level.
    assert {SourceType.WEB, SourceType.PARAMETRIC} <= sources_seen
