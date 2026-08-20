"""A deterministic stand-in LLM for `--dry-run`, so the orchestration, caching,
node-identity remapping, and aggregation code paths can be exercised end-to-end
with zero API calls / zero cost. Never used to produce reported results.
"""
import hashlib
from typing import Any, List

_LABELS = [
    "Very Likely",
    "Likely",
    "Somewhat Likely",
    "Neutral",
    "Somewhat Unlikely",
    "Unlikely",
    "Very Unlikely",
]


class MockLLM:
    model_name = "mock-llm"

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> List[str]:
        outputs = []
        for prompt in prompts:
            # Deterministic but prompt-dependent, so different premise
            # prefixes (i.e. different orderings) can produce different
            # scores -- this is what lets a dry run sanity-check that
            # TopoDev aggregation actually detects nonzero deviation when
            # present, rather than trivially always being zero.
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % len(_LABELS)
            outputs.append(_LABELS[idx])
        return outputs
