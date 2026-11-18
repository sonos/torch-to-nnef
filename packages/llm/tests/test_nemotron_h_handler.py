import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef.op.custom_extractors.mamba2 import MAMBA2_MIN_TRACT_VERSION
from torch_to_nnef.op.custom_extractors.moe import (
    MOE_FFN_NEMOTRON_H_MIN_TRACT_VERSION,
)
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal
from torch_to_nnef_llm.models.handlers import (
    NemotronHArchitectureHandler,
    get_handler,
)

nemotron_h = pytest.importorskip(
    "transformers.models.nemotron_h.modeling_nemotron_h"
)


@pytest.fixture
def tiny_model():
    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=128,
        hidden_size=32,
        layers_block_type=[
            "linear_attention",
            "moe",
            "full_attention",
            "mlp",
            "linear_attention",
        ],
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=48,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=40,
        n_routed_experts=8,
        num_experts_per_tok=3,
        routed_scaling_factor=2.5,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = NemotronHForCausalLM(config).float().eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nemotron_h.NemotronHTopkRouter):
                module.e_score_correction_bias.normal_(std=0.3)
            if isinstance(module, nemotron_h.NemotronHMamba2Mixer):
                # HF clamps the time step in its chunked prefill only; the
                # export, like NVIDIA's model and HF decode, never does
                module.time_step_limit = (0.0, float("inf"))
    return model


def _spec(model, past=0):
    return NemotronHArchitectureHandler().build_input_spec(
        tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.arange(1, 33).reshape(1, -1)
        ),
        config_helper=HFConfigHelper(model.config),
        inputs_dtype=torch.float32,
        sample_text="",
        n_input_tokens=3,
        n_past_input_tokens=past,
    )


def test_nemotron_h_handler_is_registered():
    assert get_handler("nemotron_h") is NemotronHArchitectureHandler


def test_nemotron_h_spec(tiny_model):
    spec = _spec(tiny_model, past=2)
    assert spec.input_names == [
        "input_ids",
        "in_cache_conv_0",
        "in_cache_ssm_0",
        "in_cache_key_2",
        "in_cache_value_2",
        "in_cache_conv_4",
        "in_cache_ssm_4",
    ]
    assert spec.output_names[0] == "outputs"
    assert [list(t.shape) for t in spec.inputs[1:3]] == [
        [1, 32 + 2 * 2 * 16, 4],
        [1, 4, 8, 16],
    ]
    assert spec.inputs[2].dtype == torch.float32
    assert spec.dynamic_axes["in_cache_key_2"] == {2: "P"}
    assert "in_cache_conv_0" not in spec.dynamic_axes


def test_nemotron_h_streaming_matches_hf(tiny_model):
    """Chunks of any length carrying states match HF on the whole sequence.

    The chunks cross the conv kernel width and the HF Mamba chunk size.
    """
    model = tiny_model
    NemotronHArchitectureHandler().prepare_model_for_export(model)
    wrapper = BaseCausal(
        model, handler=NemotronHArchitectureHandler(), num_logits_to_keep=0
    )
    state = _spec(model).inputs[1:]
    seen = []
    with torch.no_grad():
        for ids in (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]]),
            torch.tensor([[5, 6, 7, 8, 9, 10]]),
            torch.tensor([[11]]),
        ):
            seen.append(ids)
            outputs = wrapper(ids, *state)
            state = outputs[1:]
            reference = model(torch.cat(seen, dim=1), use_cache=False).logits
            torch.testing.assert_close(
                outputs[0],
                reference[:, -ids.shape[1] :],
                atol=1e-4,
                rtol=1e-4,
            )


def test_nemotron_h_dynamic_nnef(tiny_model, tmp_path):
    binary = os.environ.get("TRACT_BIN")
    if not binary:
        pytest.skip("set TRACT_BIN for dynamic NNEF runtime validation")
    cli = TractCli(Path(binary))
    required = max(
        MAMBA2_MIN_TRACT_VERSION, MOE_FFN_NEMOTRON_H_MIN_TRACT_VERSION
    )
    if cli.version < required:
        pytest.skip(f"NemotronH requires tract >= {required}")
    NemotronHArchitectureHandler().prepare_model_for_export(tiny_model)
    spec = _spec(tiny_model, past=2)
    wrapper = BaseCausal(
        tiny_model, handler=NemotronHArchitectureHandler(), num_logits_to_keep=0
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
