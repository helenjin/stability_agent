"""Tests for QwenLLMFixed's two fixes over the vendored QwenLLM -- mocking
the HF pipeline, never loading real model weights or touching a GPU."""
from unittest.mock import MagicMock

from ares_topodev.eval_harness.qwen_llm_fixed import QwenLLMFixed


def _make_fixed_llm():
    llm = QwenLLMFixed.__new__(QwenLLMFixed)  # bypass __init__ (which loads real weights)
    llm.default_params = {"temperature": 0.0, "max_new_tokens": 500, "top_p": 0.8, "repetition_penalty": 1.1}
    llm.tokenizer = MagicMock(eos_token_id=0)
    llm.pipeline = MagicMock(return_value=[[{"generated_text": "Very Likely"}]])
    return llm


def test_temperature_zero_uses_greedy_decoding_not_do_sample():
    llm = _make_fixed_llm()
    llm.generate(["prompt"], temperature=0.0)

    _, kwargs = llm.pipeline.call_args
    assert kwargs["do_sample"] is False  # the vendored code always passed True here, which crashes at temperature=0
    assert "temperature" not in kwargs  # meaningless (and can warn) under greedy decoding
    assert "top_p" not in kwargs


def test_temperature_positive_uses_sampling_with_params():
    llm = _make_fixed_llm()
    llm.generate(["prompt"], temperature=0.7)

    _, kwargs = llm.pipeline.call_args
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
    assert "top_p" in kwargs


def test_return_full_text_false_is_always_passed():
    llm = _make_fixed_llm()
    llm.generate(["prompt"], temperature=0.0)

    _, kwargs = llm.pipeline.call_args
    assert kwargs["return_full_text"] is False


def test_pipeline_exception_falls_back_to_empty_strings_not_crash():
    llm = _make_fixed_llm()
    llm.pipeline.side_effect = RuntimeError("out of memory")

    outputs = llm.batch_generate(["a", "b"], batch_size=2, temperature=0.0)
    assert outputs == ["", ""]
