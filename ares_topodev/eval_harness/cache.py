"""A disk cache for LLM calls, keyed on the exact rendered prompt text.

This does NOT modify vendored ares code. `exp_helpers.models.openai_llm`'s own
`cached_openai_generate` has its `@cache.memoize()` decorator commented out in
the released repo (see VENDOR.md), so there's no working cache to reuse or
accidentally misuse. Instead, `CachingLLM` wraps any `BaseLLM` instance at the
`.generate(prompts, ...)` boundary that `EntailmentModel.forward` calls
(`self.llm.generate(...)`) -- so we pass `CachingLLM(real_llm)` into
`EntailmentModel(llm=...)` instead of the raw LLM, with zero changes to
`entailment_model.py` or `openai_llm.py`.

Why this is safe across orderings: the cache key is the exact prompt string
(which already embeds the premises + hypothesis for a specific step under a
specific ordering) plus model name and temperature. Two calls only hit the
same cache entry if they are, verbatim, the same premises-plus-hypothesis
prompt -- which can legitimately happen (e.g. a node whose overcomplete premise
prefix happens to coincide across two different orderings, or re-running after
an interrupted experiment), but never conflates two *different* prompts. No
ordering-dependent context is erased: a different prefix produces a different
prompt string produces a different key.
"""
import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional


class DiskPromptCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._store: Dict[str, str] = {}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._store[entry["key"]] = entry["value"]

    @staticmethod
    def make_key(*, model_name: str, prompt: str, temperature: float) -> str:
        payload = json.dumps(
            {"model_name": model_name, "prompt": prompt, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._store:
                return
            self._store[key] = value
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")

    def __len__(self) -> int:
        return len(self._store)


class CachingLLM:
    """Wraps any exp_helpers BaseLLM-compatible object, adding prompt-level
    disk caching in front of `.generate(...)`. Never used to bypass ARES's
    scoring logic -- only to avoid re-paying for an identical prompt string."""

    def __init__(self, llm, cache: DiskPromptCache):
        self.llm = llm
        self.cache = cache
        self.model_name = getattr(llm, "model_name", "unknown-model")
        self.hits = 0
        self.misses = 0

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> List[str]:
        results: List[Optional[str]] = [None] * len(prompts)
        keys = [
            self.cache.make_key(model_name=self.model_name, prompt=p, temperature=temperature)
            for p in prompts
        ]
        miss_indices = []
        for i, key in enumerate(keys):
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
                self.hits += 1
            else:
                miss_indices.append(i)

        if miss_indices:
            miss_prompts = [prompts[i] for i in miss_indices]
            fresh_outputs = self.llm.generate(
                miss_prompts, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs
            )
            self.misses += len(miss_indices)
            for i, output in zip(miss_indices, fresh_outputs):
                results[i] = output
                self.cache.put(keys[i], output)

        return results  # type: ignore[return-value]

    def __getattr__(self, item):
        # Delegate any attribute EntailmentModel/methods might read off the
        # underlying LLM (e.g. isinstance checks against PRMModel elsewhere
        # don't apply to us -- we only wrap OpenAILLM in this experiment).
        return getattr(self.llm, item)
