"""Inert stub for the optional `vllm` dependency.

The vendored ARES code (`vendor/ares/src/exp_helpers/models/vllm.py` and
`qwen_prm_vllm.py`) does `from vllm import LLM, SamplingParams` at import time,
even though our experiment only ever exercises the OpenAI backend
(`exp_helpers.models.openai_llm`). `vllm` is a large, GPU/CUDA-oriented package
that this experiment has no functional use for.

This stub satisfies the import so `exp_helpers.models` (and therefore
`get_stability_scorer`, `EntailmentModel`, etc.) can be imported without
installing real vllm. If any code path actually tries to *use* `LLM` or
`SamplingParams`, it will fail loudly here rather than silently doing the
wrong thing.

Not part of vendored ARES; not used unless this stub package directory is
placed ahead of the real `vllm` (or in its absence) on `sys.path` -- see
`ares_topodev/eval_harness/_bootstrap.py`.
"""


class LLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "vllm.LLM is stubbed out in this experiment (ares_topodev only uses "
            "the OpenAI backend). Install real vllm if you need the vllm/Qwen3 "
            "backbone."
        )


class SamplingParams:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "vllm.SamplingParams is stubbed out in this experiment (ares_topodev "
            "only uses the OpenAI backend). Install real vllm if you need the "
            "vllm/Qwen3 backbone."
        )
