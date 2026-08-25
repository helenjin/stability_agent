"""Tests for QwenLLMFixed's fixes over the vendored QwenLLM -- mocking the
HF pipeline/model loading, never loading real model weights or touching a
GPU."""
from unittest.mock import MagicMock, patch

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


def test_explicit_device_pins_a_single_gpu_instead_of_auto():
    with patch("ares_topodev.eval_harness.qwen_llm_fixed.transformers") as mock_transformers, \
         patch("ares_topodev.eval_harness.qwen_llm_fixed.torch"):
        QwenLLMFixed(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:1")

        model_call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        pipeline_call_kwargs = mock_transformers.pipeline.call_args.kwargs
        # from_pretrained wants a bare device string (verified empirically:
        # the {"": device} dict form is silently mishandled by the installed
        # transformers version). pipeline() is different again: passing it
        # ANY device_map/device kwarg other than an integer index -- or even
        # omitting the kwarg entirely -- was verified to MOVE an
        # already-placed model back to cuda:0. Only an int index prevents that.
        assert model_call_kwargs["device_map"] == "cuda:1"
        assert pipeline_call_kwargs["device"] == 1
        assert "device_map" not in pipeline_call_kwargs


def test_no_device_falls_back_to_auto():
    with patch("ares_topodev.eval_harness.qwen_llm_fixed.transformers") as mock_transformers, \
         patch("ares_topodev.eval_harness.qwen_llm_fixed.torch"):
        QwenLLMFixed(model_name="Qwen/Qwen2.5-7B-Instruct")

        model_call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        pipeline_call_kwargs = mock_transformers.pipeline.call_args.kwargs
        assert model_call_kwargs["device_map"] == "auto"
        assert "device" not in pipeline_call_kwargs  # no explicit pin requested -> don't force one
