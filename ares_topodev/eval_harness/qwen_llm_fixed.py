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
3. `__init__` hardcodes `device_map="auto"` for both the model and the
   pipeline, with no way to pin an instance to a specific GPU. Qwen2.5-7B in
   bfloat16 (~14-15GB) comfortably fits on a single 20GB GPU, so "auto"
   always places the whole model on one device anyway -- meaning loading
   several instances (e.g. one per recipe-processing thread, for real
   parallelism across this machine's multiple idle GPUs) would pile them
   all onto the same device instead of spreading across them.
4. `batch_generate` sends the raw prompt string straight to the
   text-generation pipeline with NO chat template applied. Qwen2.5-7B-
   Instruct only behaves like an instruction-following assistant when its
   input is wrapped in its chat template (`<|im_start|>user\n...<|im_end|>\n
   <|im_start|>assistant\n`); fed a bare string, it's just a base LM
   continuing arbitrary text. Verified directly: an un-templated "hi"
   produced an unrelated coding-help continuation; the same string run
   through `tokenizer.apply_chat_template(...)` produced a normal reply.
   This affects every real prompt sent to Qwen, not just toy examples.

Fixed here by subclassing rather than editing the vendored file (same
pattern as `cache.CachingLLM` wrapping any `BaseLLM`, or `usage_tracker`
monkey-patching `openai_client`): `do_sample` is now derived from whether
`temperature > 0` (greedy decoding at temperature=0, matching every other
backbone's "temp0" behavior in this codebase), `return_full_text=False` is
passed explicitly, `__init__` accepts an optional `device` (e.g. `"cuda:1"`)
to pin a specific instance to a specific GPU instead of always "auto", and
each prompt is now run through the chat template as a single "user" turn
before generation -- matching `openai_llm.py`'s `cached_openai_generate`,
which also sends the ENTIRE constructed prompt (system instructions +
context + hypothesis all together) as one `{"role": "user", ...}` message,
not split into separate system/user roles. Matching that structure here
keeps the ARES-vs-SAGER comparison about the backbone model, not an
incidental difference in how each backbone's prompt is structured.
"""
import os
from typing import Any, List, Optional

import torch
import transformers

from ares_topodev.eval_harness import _bootstrap  # noqa: F401  (sys.path + OPENAI_API_KEY placeholder)
from exp_helpers.models.qwen_llm import QwenLLM


class QwenLLMFixed(QwenLLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        # Deliberately does NOT call QwenLLM.__init__ (which hardcodes
        # device_map="auto") -- mirrors it exactly otherwise, going one
        # level up to BaseLLM.__init__ instead.
        from exp_helpers.models.base_llm import BaseLLM

        BaseLLM.__init__(self, model_name, **kwargs)

        if cache_dir:
            os.environ["HF_HOME"] = cache_dir

        self.device = device
        # A plain device string (not the {"": device} dict form some HF
        # versions document/accept) -- verified empirically against the
        # transformers version actually installed here: the dict form is
        # silently mishandled (falls back to placing everything on cuda:0
        # regardless of the requested device), while the bare string
        # correctly pins the whole model to that one device.
        device_map = device if device else "auto"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_new_tokens": kwargs.get("max_new_tokens", 500),
            "top_p": kwargs.get("top_p", 0.8),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        # CRITICAL, verified empirically: constructing transformers.pipeline()
        # with an already-loaded, already-correctly-placed `model=` object
        # MOVES that model -- to cuda:0, regardless of where it actually was
        # -- unless given an explicit integer `device=` index. This happens
        # even with NO device_map/device kwarg at all (that's why the "auto"
        # case looked fine before this was caught: auto-placement already
        # puts a single-GPU-sized model on cuda:0, so the erroneous move was
        # invisible). Passing device_map=<string> here (what an earlier,
        # broken version of this file did) does NOT prevent the move either
        # -- only the integer `device=` pipeline kwarg does.
        pipeline_kwargs = dict(model=self.model, tokenizer=self.tokenizer, **self.default_params)
        if device:
            pipeline_kwargs["device"] = int(device.split(":")[1]) if device.startswith("cuda:") else device
        self.pipeline = transformers.pipeline("text-generation", **pipeline_kwargs)

    def _apply_chat_template(self, prompt: str) -> str:
        """Wraps a raw prompt as a single user turn and renders it through
        Qwen's chat template -- see fix (4) in the module docstring. Without
        this, generation is base-model continuation, not instruction
        following."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def batch_generate(self, prompts: List[str], batch_size: int = 8, **kwargs: Any) -> List[str]:
        outputs = []
        params = {**self.default_params, **kwargs}
        temperature = params.get("temperature", 0.0)
        do_sample = temperature > 0
        templated_prompts = [self._apply_chat_template(p) for p in prompts]

        for i in range(0, len(templated_prompts), batch_size):
            batch = templated_prompts[i : i + batch_size]
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
