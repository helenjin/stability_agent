"""Non-invasive fix for two bugs in the vendored `QwenLLM`
(`vendor/ares/src/exp_helpers/models/qwen_llm.py`), found while probing it
as a candidate backbone:

1. `batch_generate` always passes `do_sample=True` to the HF pipeline, even
   when `temperature=0.0` (our experiments' standard "temp0" setting).
   Current `transformers` raises on that combination ("temperature has to be
   a strictly positive float ... set do_sample=False"). The vendored
   `except Exception` swallows this and returns `""` for every prompt in
   the failing batch -- silently degenerate, not a crash, so easy to miss.
2. The HF pipeline's default `return_full_text=True` means `generated_text`
   includes the entire input prompt, not just the completion -- never
   verified because (1) always threw first.

Fixed here by subclassing rather than editing the vendored file (same
pattern as `cache.CachingLLM` wrapping any `BaseLLM`, or `usage_tracker`
monkey-patching `openai_client`): `do_sample` is now derived from whether
`temperature > 0` (greedy decoding at temperature=0, matching every other
backbone's "temp0" behavior in this codebase), and `return_full_text=False`
is passed explicitly.
"""
from typing import Any, List

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)
from exp_helpers.models.qwen_llm import QwenLLM


class QwenLLMFixed(QwenLLM):
    def batch_generate(self, prompts: List[str], batch_size: int = 8, **kwargs: Any) -> List[str]:
        outputs = []
        params = {**self.default_params, **kwargs}
        temperature = params.get("temperature", 0.0)
        do_sample = temperature > 0

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            generate_kwargs = dict(
                max_new_tokens=params.get("max_new_tokens", 500),
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                return_full_text=False,
            )
            if do_sample:  # top_p/repetition_penalty are meaningless (and can warn) under greedy decoding
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = params.get("top_p", 0.8)
                generate_kwargs["repetition_penalty"] = params.get("repetition_penalty", 1.1)

            try:
                batch_outputs = self.pipeline(batch, **generate_kwargs)
                for output in batch_outputs:
                    outputs.append(output[0]["generated_text"])
            except Exception as e:
                print(f"Error generating text: {e}")
                outputs.extend(["" for _ in range(len(batch))])
        return outputs
