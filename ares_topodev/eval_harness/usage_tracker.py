"""Records real, exact per-call token usage for every request the vendored
`OpenAILLM` makes -- taken straight from OpenAI's own `response.usage` field
(prompt_tokens/completion_tokens/total_tokens), not estimated from
reconstructed prompts or a local tokenizer.

Does not modify `vendor/ares/src`: `openai_llm.py` discards `response.usage`
immediately after reading `.choices[0].message.content` (see its
`single_generate`/`cached_openai_generate`/`non_cached_openai_generate`), so
there's no vendored hook to read it from. Instead this monkey-patches the
one shared `openai_client` object that module already constructs at import
time -- the same non-invasive pattern `_bootstrap.py` already uses to
neutralize that module's stray `pdb.set_trace()` call: wrapping a vendored
object's method from our own code, never editing the vendored file itself.

Usage: call `enable(log_path)` once, before any real (non-dry-run) API calls
are made -- `run_experiment.build_entailment_model` does this. Idempotent:
safe to call multiple times (e.g. once per recipe-concurrency thread) --
only the first call actually patches anything.
"""
import json
import os
import threading

_lock = threading.Lock()
_enabled = False
_log_path = None


def enable(log_path: str) -> None:
    global _enabled, _log_path
    with _lock:
        if _enabled:
            return
        _enabled = True
        _log_path = log_path

        from exp_helpers.models.openai_llm import openai_client

        original_create = openai_client.chat.completions.create

        def patched_create(*args, **kwargs):
            response = original_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                record = {
                    "model_requested": kwargs.get("model"),
                    "model_resolved": getattr(response, "model", None),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
                _append(record)
            return response

        openai_client.chat.completions.create = patched_create


def _append(record: dict) -> None:
    with _lock:
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        with open(_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def summarize(log_path: str) -> dict:
    """Aggregate the usage log by resolved model. Real per-call records
    only -- never fabricates a number for calls made before tracking was
    enabled (those simply aren't in the log)."""
    by_model = {}
    if not os.path.exists(log_path):
        return by_model
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            model = r["model_resolved"] or r["model_requested"] or "unknown"
            agg = by_model.setdefault(
                model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            agg["calls"] += 1
            agg["prompt_tokens"] += r["prompt_tokens"]
            agg["completion_tokens"] += r["completion_tokens"]
            agg["total_tokens"] += r["total_tokens"]
    return by_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default="ares_topodev/results/usage_log.jsonl")
    args = parser.parse_args()

    summary = summarize(args.log_path)
    if not summary:
        print(f"no usage recorded yet at {args.log_path}")
    for model, agg in sorted(summary.items()):
        print(
            f"{model:30s} calls={agg['calls']:8d}  "
            f"prompt_tokens={agg['prompt_tokens']:12d}  "
            f"completion_tokens={agg['completion_tokens']:10d}  "
            f"total_tokens={agg['total_tokens']:12d}"
        )
