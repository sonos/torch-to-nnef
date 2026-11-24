"""Inkling streaming parity, including loop passes and convolution history."""

import copy
import os
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.inference_target.tract import TractCli
from torch_to_nnef_llm.config import HFConfigHelper
from torch_to_nnef_llm.models.base import BaseCausal, update_forward_signature
from torch_to_nnef_llm.models.handlers import get_handler
from torch_to_nnef_llm.models.handlers.inkling2 import (
    Inkling2ArchitectureHandler,
    Inkling2TextArchitectureHandler,
    _cache_layer_ids,
    _InklingCache,
)


def test_registration_and_loop_cache_order():
    assert get_handler("inkling2_loop") is Inkling2ArchitectureHandler
    assert get_handler("inkling2_loop_text") is Inkling2TextArchitectureHandler
    config = SimpleNamespace(
        num_hidden_layers=24,
        loop_layer_index_start=6,
        loop_layer_index_end=18,
        loop_num_passes=3,
        layer_types=["hybrid_sliding"] * 24,
    )
    assert _cache_layer_ids(config) == list(range(24)) + list(range(6, 18)) * 2
    states = []
    for _ in range(48):
        states.extend([torch.zeros(1, 2, 0, 4)] * 2)
        states.extend([torch.zeros(1, 8, 3)] * 4)
    cache = _InklingCache(config, states)
    new = torch.ones(1, 8, 2)
    cache.update_conv_state(new, 24, 0, 4)
    assert cache.layers[24].conv_states[0].sum() == 16
    assert cache.layers[6].conv_states[0].sum() == 0
    assert all(state.sum() == 0 for state in states)


@pytest.fixture
def tiny_model(request):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Inkling2LoopForCausalLM"):
        pytest.skip("requires transformers with Inkling 2 support")
    config = transformers.Inkling2LoopTextConfig(
        vocab_size=32,
        unpadded_vocab_size=32,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        swa_num_attention_heads=2,
        swa_num_key_value_heads=1,
        swa_head_dim=8,
        sliding_window_size=4,
        rel_extent=4,
        d_rel=2,
        n_layers_per_attn_res_block=1,
        loop_layer_index_start=1,
        loop_layer_index_end=3,
        loop_num_passes=3,
        loop_gate_rank=4,
        conv_kernel_size=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"],
        eos_token_id=31,
        pad_token_id=0,
    )
    if getattr(request, "param", 4) == 24:
        config.num_hidden_layers = 24
        config.n_layers_per_attn_res_block = 6
        config.loop_layer_index_start = 6
        config.loop_layer_index_end = 18
        config.layer_types = [
            "hybrid" if (i + 1) % 6 == 0 else "hybrid_sliding"
            for i in range(24)
        ]
    if getattr(
        request.node, "callspec", None
    ) and request.node.callspec.params.get("quantize"):
        # Exercise eligible projections as well as expert tables: Q4_0's
        # reduction dimension must be a multiple of 32.
        config.hidden_size = 32
        config.head_dim = 16
        config.swa_head_dim = 16
        config.moe_intermediate_size = 32
        config.loop_gate_rank = 32
    torch.manual_seed(7)
    model = transformers.Inkling2LoopForCausalLM(config).float().eval()
    # HF initializes message gates and convolution weights to zero. Exercise
    # actual state and inter-pass messaging rather than identity operations.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if (
                "conv" in name
                or "message_gate" in name
                or "forget_gate" in name
            ):
                parameter.normal_(std=0.05)
    return model


def _spec(model, past=0):
    return Inkling2TextArchitectureHandler().build_input_spec(
        tokenizer=lambda *a, **kw: SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]])
        ),
        config_helper=HFConfigHelper(model.config),
        inputs_dtype=torch.float32,
        sample_text="",
        n_input_tokens=3,
        n_past_input_tokens=past,
    )


def _wrapper(model):
    handler = Inkling2TextArchitectureHandler()
    handler.prepare_model_for_export(model)
    wrapper = BaseCausal(model, handler=handler, num_logits_to_keep=0)
    update_forward_signature(wrapper, _spec(model))
    return wrapper


@pytest.mark.parametrize("tiny_model", [4, 24], indirect=True)
def test_streaming_matches_upstream(tiny_model):
    reference = copy.deepcopy(tiny_model)
    wrapper = _wrapper(tiny_model)
    state = _spec(tiny_model).inputs[1:]
    seen = []
    with torch.no_grad():
        for ids in (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[4]]),
            torch.tensor([[5, 6, 7, 8, 9]]),
            torch.tensor([[10]]),
        ):
            seen.append(ids)
            outputs = wrapper(ids, *state)
            state = outputs[1:]
            expected = reference(torch.cat(seen, dim=1), use_cache=False).logits
            torch.testing.assert_close(
                outputs[0],
                expected[:, -ids.shape[1] :],
                atol=1e-5,
                rtol=1e-4,
            )


def test_trace_remains_dynamic(tiny_model):
    wrapper = _wrapper(tiny_model)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, _spec(tiny_model, past=2).inputs)
        state = _spec(tiny_model).inputs[1:]
        for count in (3, 1, 5, 1):
            inputs = (torch.arange(1, count + 1).reshape(1, -1), *state)
            expected = wrapper(*inputs)
            actual = traced(*inputs)
            for result, ref in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, ref, atol=1e-5, rtol=1e-4)
            state = actual[1:]


def test_loader_accepts_upstream_dequantized_experts(tiny_model, tmp_path):
    from torch_to_nnef_llm.loader import load_model

    tiny_model.config.quantization_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }
    tiny_model.save_pretrained(tmp_path)
    loaded = load_model(local_dir=tmp_path, upcast_quant=["any"])
    with torch.no_grad():
        ids = torch.tensor([[1, 2]])
        torch.testing.assert_close(
            loaded(ids, use_cache=False).logits,
            tiny_model(ids, use_cache=False).logits,
        )


def test_prepare_is_idempotent(tiny_model):
    handler = Inkling2TextArchitectureHandler()
    handler.prepare_model_for_export(tiny_model)
    count = len(list(tiny_model.modules()))
    handler.prepare_model_for_export(tiny_model)
    assert len(list(tiny_model.modules())) == count


def test_loader_does_not_accept_packed_experts_as_dense(tiny_model):
    from torch_to_nnef.exceptions import T2NErrorMisuse
    from torch_to_nnef_llm.loader import assert_upcast_dense

    tiny_model.config.quantization_config = {"quant_method": "modelopt"}
    experts = tiny_model.model.layers[0].mlp.experts
    experts.gate_up_proj = torch.nn.Parameter(
        experts.gate_up_proj.detach().to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    with pytest.raises(T2NErrorMisuse, match="did not fully dequantize"):
        assert_upcast_dense(tiny_model, ["any"])


def test_multimodal_checkpoint_text_path(tiny_model):
    from transformers import (
        Inkling2LoopConfig,
        Inkling2LoopForConditionalGeneration,
    )

    config = Inkling2LoopConfig(
        text_config=tiny_model.config.to_dict(),
        vision_config={
            "model_type": "inkling2_vision",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "text_hidden_size": 16,
            "projection_hidden_dim": 16,
            "posemb_grid_size": 2,
        },
    )
    model = Inkling2LoopForConditionalGeneration(config).float().eval()
    handler = Inkling2ArchitectureHandler()
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3]])
        expected = model(ids, use_cache=False).logits
        handler.prepare_model_for_export(model)
        wrapper = BaseCausal(model, handler=handler, num_logits_to_keep=0)
        actual = wrapper(ids, *_spec(model).inputs[1:])[0]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_exporter_probes_carry_all_state(tiny_model, tmp_path):
    from torch_to_nnef_llm.exporter import LLMExporter

    tokenizer = lambda *a, **kw: SimpleNamespace(  # noqa: E731
        input_ids=torch.arange(1, 17).reshape(1, -1)
    )
    exporter = LLMExporter(tiny_model, tokenizer)
    with torch.no_grad():
        bundles = exporter.dump_all_io_npz_kind(tmp_path, size=9)
    assert len(bundles) == 3
    for inputs_path, outputs_path in bundles:
        with np.load(inputs_path) as inputs, np.load(outputs_path) as outputs:
            assert outputs["out_cache_key_7"].shape[2] == 9
            assert outputs["out_cache_conv3_7"].shape == (1, 16, 2)
            if inputs_path.name != "prompt_inputs.npz":
                assert np.any(inputs["in_cache_conv3_7"] != 0)


@pytest.mark.parametrize("quantize", [False, True])
def test_dynamic_nnef(tiny_model, tmp_path, quantize):
    binary = os.environ.get("TRACT_BIN")
    if not binary:
        pytest.skip("set TRACT_BIN for NNEF runtime validation")
    wrapper = _wrapper(tiny_model)
    if quantize:
        from torch_to_nnef.compress import DEFAULT_COMPRESSION
        from torch_to_nnef.op.custom_extractors.moe import SelectedGatedExperts
        from torch_to_nnef.tensor.quant import QTensor

        wrapper = DEFAULT_COMPRESSION[
            "min_max_q4_0_with_embeddings_and_experts"
        ](wrapper)
        experts = [
            m for m in wrapper.modules() if isinstance(m, SelectedGatedExperts)
        ]
        assert experts
        for module in experts:
            assert isinstance(module.gate_up_proj, QTensor)
            assert isinstance(module.down_proj, QTensor)
    spec = _spec(tiny_model, past=2)
    cli = TractCli(Path(binary))
    if quantize and cli.version < "0.23.10":
        pytest.skip(
            "Inkling Q4_0 requires tract 0.23.10 quantized-slice support"
        )
    target = TractNNEF(
        version=cli.version,
        specific_tract_binary_path=cli.tract_path,
        dynamic_axes=spec.dynamic_axes,
        # Quantized matmul also rounds activations in tract. Check a narrow
        # explicit tolerance below against the dequantized PyTorch reference.
        check_io=not quantize,
    )
    path = tmp_path / "inkling.nnef.tgz"
    with torch.no_grad():
        export_model_to_nnef(
            model=wrapper,
            args=spec.inputs,
            file_path_export=path,
            inference_target=target,
            input_names=spec.input_names,
            output_names=spec.output_names,
        )
        if quantize:
            from torch_to_nnef.nnef_io.tensor import DatBinHeader

            with tarfile.open(path) as archive:
                members = [
                    m
                    for m in archive.getmembers()
                    if m.name.endswith(
                        ("experts.gate_up_proj.dat", "experts.down_proj.dat")
                    )
                ]
                assert len(members) == 2 * len(experts)
                for idx, member in enumerate(members):
                    weight_path = tmp_path / f"expert_{idx}.dat"
                    weight_path.write_bytes(archive.extractfile(member).read())
                    header = DatBinHeader.from_dat(weight_path)
                    assert (
                        header.torch_dtype_or_custom
                        == DatBinHeader.TractCustomTypes.Q40
                    )
        state = _spec(tiny_model).inputs[1:]
        for count in (3, 1, 5, 1):
            inputs = (torch.arange(1, count + 1).reshape(1, -1), *state)
            expected = wrapper(*inputs)
            state = expected[1:]
            in_path, out_path = tmp_path / "in.npz", tmp_path / "out.npz"
            for names, tensors, bundle in (
                (spec.input_names, inputs, in_path),
                (spec.output_names, expected, out_path),
            ):
                np.savez(
                    bundle,
                    **{
                        name: value.numpy()
                        for name, value in zip(names, tensors, strict=True)
                    },
                )
            if quantize:
                actual_path = tmp_path / "actual.npz"
                subprocess.run(
                    cli.run_save_outputs_cmd_str(path, in_path, actual_path),
                    check=True,
                    capture_output=True,
                )
                with np.load(actual_path) as actual:
                    for name, reference in zip(
                        spec.output_names, expected, strict=True
                    ):
                        np.testing.assert_allclose(
                            actual[name],
                            reference.numpy(),
                            atol=2e-3,
                            rtol=2e-3,
                        )
            else:
                cli.assert_io(
                    path,
                    in_path,
                    out_path,
                    check_tolerance=target.check_io_tolerance,
                )
