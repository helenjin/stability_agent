"""A deterministic stand-in LLM for `--dry-run`, so the orchestration, caching,
node-identity remapping, and aggregation code paths can be exercised end-to-end
with zero API calls / zero cost. Never used to produce reported results.
"""
import hashlib
import json
import re
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

# llm_judge_whole's prompt embeds "Derived Claims:\n<json array>" (see
# methods/llm_judge_whole.py's get_stability_rate). Recognizing it lets the
# mock return a well-formed JSON response of the right length instead of a
# bare Likert label, so a dry run can also exercise that scorer's success
# path (JSON parsing, per-claim mapping, length-matches-order check) --
# without this, every dry-run call to it would only ever hit its parse-
# failure branch, which is a weaker structural check.
_DERIVED_CLAIMS_PATTERN = re.compile(r"Derived Claims:\n(\[[\s\S]*?\n\])", re.MULTILINE)


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
            match = _DERIVED_CLAIMS_PATTERN.search(prompt)
            if match:
                outputs.append(self._mock_llm_judge_whole_response(prompt, match.group(1)))
                continue
            # Deterministic but prompt-dependent, so different premise
            # prefixes (i.e. different orderings) can produce different
            # scores -- this is what lets a dry run sanity-check that
            # TopoDev aggregation actually detects nonzero deviation when
            # present, rather than trivially always being zero.
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % len(_LABELS)
            outputs.append(_LABELS[idx])
        return outputs

    @staticmethod
    def _mock_llm_judge_whole_response(prompt: str, derived_claims_json: str) -> str:
        derived_claims = json.loads(derived_claims_json)
        entries = []
        for claim in derived_claims:
            digest = hashlib.sha256((prompt[:64] + claim).encode("utf-8")).hexdigest()
            entailed = "YES" if int(digest[:8], 16) % 2 == 0 else "NO"
            entries.append({"claim": claim, "reasoning": "mock", "entailed": entailed})
        return "```json\n" + json.dumps({"raw_claims": [], "derived_claims": entries}) + "\n```"
