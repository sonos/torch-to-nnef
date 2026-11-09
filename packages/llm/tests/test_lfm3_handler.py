import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal
from torch_to_nnef_llm.models.handlers import (
    Lfm3ArchitectureHandler,
    get_handler,
)
from torch_to_nnef_llm.models.handlers.lfm3 import _ExportCache


def test_lfm3_cache_is_functional():
    config = SimpleNamespace(layer_types=["conv", "full_attention"])
    conv = torch.randn(1, 8, 6)
    key = torch.randn(1, 2, 3, 4)
    states = (conv, key, key.clone())
    original = [t.clone() for t in states]
    cache = _ExportCache(config, states)
    new = torch.randn(1, 8, 2)
    joined = cache.update_conv_state(new, 0, conv_kernel_size=7)
    torch.testing.assert_close(joined, torch.cat((conv, new), -1))
    torch.testing.assert_close(cache.layers[0].conv_states[0], joined[..., -6:])
    cache.update(key, key, 1)
    assert cache.get_seq_length() == 6
    for before, after in zip(original, states, strict=True):
        torch.testing.assert_close(before, after)
    assert get_handler("lfm3") is Lfm3ArchitectureHandler


@pytest.fixture
def tiny_model(request):
    # The model lives in pinned Hub remote code, not in transformers itself.
    source = os.environ.get("LFM3_MODEL_DIR")
    if not source:
        pytest.skip("set LFM3_MODEL_DIR to the pinned example snapshot")
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.vocab_size = 128
    if getattr(request, "param", 3) == 3:
        config.num_hidden_layers = 3
        config.layer_types = ["conv", "full_attention", "sliding_attention"]
    config.sliding_window = 4
    config.pad_token_id = 0
    torch.manual_seed(0)
    return (
        AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        .float()
        .eval()
    )


def _spec(model, past=0):
    return Lfm3ArchitectureHandler().build_input_spec(
        tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.arange(1, 33).reshape(1, -1)
        ),
        config_helper=HFConfigHelper(model.config),
        inputs_dtype=torch.float32,
        sample_text="",
        n_input_tokens=3,
        n_past_input_tokens=past,
    )


def test_lfm3_streaming_matches_original(tiny_model):
    model = tiny_model
    Lfm3ArchitectureHandler().prepare_model_for_export(model)
    wrapper = BaseCausal(
        model, handler=Lfm3ArchitectureHandler(), num_logits_to_keep=0
    )
    state = _spec(model).inputs[1:]
    seen = []
    with torch.no_grad():
        # Cross the window boundary and the width-seven convolution boundary.
        for ids in (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]]),
            torch.tensor([[5, 6, 7, 8, 9]]),
            torch.tensor([[10]]),
        ):
            seen.append(ids)
            outputs = wrapper(ids, *state)
            state = outputs[1:]
            reference = model(torch.cat(seen, dim=1), use_cache=False).logits
            torch.testing.assert_close(
                outputs[0],
                reference[:, -ids.shape[1] :],
                atol=1e-5,
                rtol=1e-4,
            )


@pytest.mark.parametrize("tiny_model", [3, 24], indirect=True)
def test_lfm3_dynamic_nnef(tiny_model, tmp_path):
    binary = os.environ.get("TRACT_BIN")
    if not binary:
        pytest.skip("set TRACT_BIN for dynamic NNEF runtime validation")
    Lfm3ArchitectureHandler().prepare_model_for_export(tiny_model)
    cli = TractCli(Path(binary))
    spec = _spec(tiny_model, past=2)
    wrapper = BaseCausal(
        tiny_model, handler=Lfm3ArchitectureHandler(), num_logits_to_keep=0
    )
    path = tmp_path / "model.nnef.tgz"
    target = TractNNEF(
        version=cli.version,
        specific_tract_binary_path=cli.tract_path,
        dynamic_axes=spec.dynamic_axes,
        check_io=True,
    )
    with torch.no_grad():
        export_model_to_nnef(
            model=wrapper,
            args=spec.inputs,
            file_path_export=path,
            inference_target=target,
            input_names=spec.input_names,
            output_names=spec.output_names,
        )
        state = _spec(tiny_model).inputs[1:]
        for count in (3, 1, 5, 1):
            inputs = (torch.arange(1, count + 1).reshape(1, -1), *state)
            outputs = wrapper(*inputs)
            state = outputs[1:]
            in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
            for names, tensors, bundle in (
                (spec.input_names, inputs, in_path),
                (spec.output_names, outputs, out_path),
            ):
                np.savez(
                    bundle,
                    **{
                        k: v.numpy()
                        for k, v in zip(names, tensors, strict=True)
                    },
                )
            cli.assert_io(
                path,
                in_path,
                out_path,
                check_tolerance=target.check_io_tolerance,
            )


@pytest.mark.parametrize("logits_to_keep", [1, "dynamic"])
def test_lfm3_cli_probes_carry_mixed_state(
    tiny_model, tmp_path, logits_to_keep
):
    from torch_to_nnef_llm.exporter import LLMExporter

    tokenizer = lambda *a, **kw: SimpleNamespace(  # noqa: E731
        input_ids=torch.arange(1, 33).reshape(1, -1)
    )
    exporter = LLMExporter(
        tiny_model, tokenizer, num_logits_to_keep=logits_to_keep
    )
    with torch.no_grad():
        bundles = exporter.dump_all_io_npz_kind(tmp_path, size=9)
    assert len(bundles) == 3
    for inputs_path, outputs_path in bundles:
        with np.load(inputs_path) as inputs, np.load(outputs_path) as outputs:
            assert "in_cache_conv_0" in inputs
            assert "out_cache_conv_0" in outputs
            assert outputs["out_cache_conv_0"].shape == (1, 32, 6)
            assert outputs["out_cache_key_1"].shape[2] == 9
            if inputs_path.name != "prompt_inputs.npz":
                assert np.any(inputs["in_cache_conv_0"] != 0)
