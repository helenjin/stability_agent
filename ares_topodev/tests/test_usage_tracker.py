"""Tests for ares_topodev.eval_harness.usage_tracker.

Uses a stub in place of the real `openai_client.chat.completions.create` --
never a real network call -- to verify the monkey-patch records exact usage
fields and is idempotent (doesn't double-wrap on repeated `enable()` calls).
"""
import json
from types import SimpleNamespace

from ares_topodev.eval_harness import _bootstrap  # noqa: F401
from ares_topodev.eval_harness import usage_tracker


def _fake_response(model="gpt-4o-mini-2024-07-18", prompt_tokens=100, completion_tokens=2):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens
        ),
    )


def _reset_tracker_state():
    usage_tracker._enabled = False
    usage_tracker._log_path = None


def test_enable_records_real_usage_fields(tmp_path, monkeypatch):
    _reset_tracker_state()
    from exp_helpers.models.openai_llm import openai_client

    calls = []

    def stub_create(*args, **kwargs):
        calls.append(kwargs)
        return _fake_response(prompt_tokens=1837, completion_tokens=3)

    monkeypatch.setattr(openai_client.chat.completions, "create", stub_create)
    log_path = str(tmp_path / "usage.jsonl")
    usage_tracker.enable(log_path)

    result = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])

    assert len(calls) == 1  # the original stub was actually invoked, not bypassed
    assert result.usage.prompt_tokens == 1837  # the real response is returned unmodified

    records = [json.loads(line) for line in open(log_path)]
    assert len(records) == 1
    assert records[0]["model_requested"] == "gpt-4o-mini"
    assert records[0]["model_resolved"] == "gpt-4o-mini-2024-07-18"
    assert records[0]["prompt_tokens"] == 1837
    assert records[0]["completion_tokens"] == 3
    assert records[0]["total_tokens"] == 1840
    _reset_tracker_state()


def test_enable_is_idempotent(tmp_path, monkeypatch):
    _reset_tracker_state()
    from exp_helpers.models.openai_llm import openai_client

    call_count = [0]

    def stub_create(*args, **kwargs):
        call_count[0] += 1
        return _fake_response()

    monkeypatch.setattr(openai_client.chat.completions, "create", stub_create)
    log_path = str(tmp_path / "usage.jsonl")

    usage_tracker.enable(log_path)
    usage_tracker.enable(log_path)  # second call must be a no-op, not a second layer of wrapping
    usage_tracker.enable(str(tmp_path / "other.jsonl"))  # even with a different path

    openai_client.chat.completions.create(model="gpt-4o-mini", messages=[])

    assert call_count[0] == 1  # not called twice by a double-wrapped patch
    records = [json.loads(line) for line in open(log_path)]
    assert len(records) == 1  # exactly one record, not duplicated
    _reset_tracker_state()


def test_summarize_aggregates_by_resolved_model(tmp_path):
    log_path = str(tmp_path / "usage.jsonl")
    with open(log_path, "w") as f:
        for pt, ct in [(1837, 3), (1364, 2), (2000, 5)]:
            f.write(json.dumps({
                "model_requested": "gpt-4o-mini",
                "model_resolved": "gpt-4o-mini-2024-07-18",
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
            }) + "\n")

    summary = usage_tracker.summarize(log_path)
    assert set(summary.keys()) == {"gpt-4o-mini-2024-07-18"}
    agg = summary["gpt-4o-mini-2024-07-18"]
    assert agg["calls"] == 3
    assert agg["prompt_tokens"] == 1837 + 1364 + 2000
    assert agg["completion_tokens"] == 3 + 2 + 5
    assert agg["total_tokens"] == agg["prompt_tokens"] + agg["completion_tokens"]


def test_summarize_missing_file_returns_empty():
    assert usage_tracker.summarize("/nonexistent/path/usage.jsonl") == {}
