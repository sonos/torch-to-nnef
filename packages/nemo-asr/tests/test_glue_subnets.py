"""Tests for the generic glue-subnet mechanism and the prompt_kernel head.

These do not require NeMo: ``glue_subnets`` depends only on torch, and the
export test uses a tiny stand-in ``prompt_kernel``.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_to_nnef_nemo.glue_subnets import (
    DEFAULT_PROMPT_LANG_ID,
    GLUE_SUBNET_BUILDERS,
    FusedGlueSubnet,
    PromptKernelSubnet,
    _builder_for_model,
    build_prompt_kernel_subnets,
    fuse_glues_into,
    iter_all_glue_subnets,
    iter_glue_subnets_after,
    register_glue_subnet,
)


class _FakeEncoder(nn.Module):
    """Minimal encoder-like native subnet: [B, mel, T] -> ([B, D, T], [B])."""

    input_names = ["audio_signal", "length"]
    output_names = ["outputs", "encoded_lengths"]
    input_types: dict = {}

    def dynamic_shapes_for_export(self, use_dynamo=False):
        return {"audio_signal": [0, 2], "length": [0]}

    def forward(self, audio_signal, length):
        b, _, t = audio_signal.shape
        return audio_signal.new_zeros(b, D, t) + 0.5, length


D, P, H = 16, 8, 32


def _prompt_kernel(d=D, p=P, h=H):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d + p, h), nn.ReLU(), nn.Linear(h, d)).eval()


class _EncStub(nn.Module):
    output_types: dict = {}


class EncDecRNNTBPEModelWithPrompt(nn.Module):
    """Stand-in whose *name* matches the registered key."""

    def __init__(self, lang_auto=101):
        super().__init__()
        self.encoder = _EncStub()
        self.prompt_kernel = _prompt_kernel()
        self.cfg = type(
            "C", (), {"prompt_dictionary": {"auto": lang_auto, "en-US": 0}}
        )()


def test_builder_resolves_by_class_name_via_mro():
    assert _builder_for_model(EncDecRNNTBPEModelWithPrompt()) is (
        build_prompt_kernel_subnets
    )

    # subclasses resolve too
    class Sub(EncDecRNNTBPEModelWithPrompt):
        pass

    assert _builder_for_model(Sub()) is build_prompt_kernel_subnets
    # unrelated model -> no builder
    assert _builder_for_model(nn.Linear(2, 2)) is None


def test_build_prompt_kernel_subnets_infers_dims_and_lang():
    glues = build_prompt_kernel_subnets(EncDecRNNTBPEModelWithPrompt())
    assert len(glues) == 1
    sub = glues[0]
    assert isinstance(sub, PromptKernelSubnet)
    assert sub.name == "prompt" and sub.after_subnet == "encoder"
    assert sub.d_model == D and sub.num_prompts == P
    assert sub.default_lang_id == 101
    assert sub.input_names == ["encoder_outputs", "lang_id"]
    assert sub.output_names == ["outputs"]
    assert sub.dynamic_shapes_for_export() == {"encoder_outputs": [0, 2]}


def test_build_prompt_kernel_subnets_defaults_when_no_dictionary():
    m = EncDecRNNTBPEModelWithPrompt()
    m.cfg = type("C", (), {})()  # no prompt_dictionary
    assert build_prompt_kernel_subnets(m)[0].default_lang_id == (
        DEFAULT_PROMPT_LANG_ID
    )


def test_build_prompt_kernel_subnets_skips_when_absent_or_malformed():
    m = EncDecRNNTBPEModelWithPrompt()
    del m.prompt_kernel
    assert build_prompt_kernel_subnets(m) == []
    # one-hot width would be non-positive
    m2 = EncDecRNNTBPEModelWithPrompt()
    m2.prompt_kernel = nn.Sequential(
        nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D)
    )
    assert build_prompt_kernel_subnets(m2) == []


def test_iter_glue_subnets_after_placement_and_filter():
    m = EncDecRNNTBPEModelWithPrompt()
    after_enc = list(iter_glue_subnets_after(m, "encoder"))
    assert [g[0] for g in after_enc] == ["prompt"]
    # nothing attaches after decoder_joint
    assert list(iter_glue_subnets_after(m, "decoder_joint")) == []
    # only_subnets filter on the glue's own name
    assert list(iter_glue_subnets_after(m, "encoder", allow={"encoder"})) == []
    assert (
        len(list(iter_glue_subnets_after(m, "encoder", allow={"prompt"}))) == 1
    )
    # unrelated model yields nothing
    assert list(iter_glue_subnets_after(nn.Linear(2, 2), "encoder")) == []


def test_iter_all_glue_subnets_builds_once_and_filters():
    m = EncDecRNNTBPEModelWithPrompt()
    emitted = list(iter_all_glue_subnets(m))
    assert [e[0] for e in emitted] == ["prompt"]
    name, glue, example, dyn = emitted[0]
    assert name == "prompt"
    assert [tuple(t.shape) for t in example] == [(1, D, 16), (1,)]
    assert dyn == {"encoder_outputs": [0, 2]}
    # allow filter
    assert list(iter_all_glue_subnets(m, allow={"encoder"})) == []
    assert len(list(iter_all_glue_subnets(m, allow={"prompt"}))) == 1
    # unrelated model -> nothing
    assert list(iter_all_glue_subnets(nn.Linear(2, 2))) == []


def test_warns_when_prompt_kernel_present_but_no_builder(caplog):
    """Warn when a prompt model's class name isn't registered (else silent)."""

    class _RenamedPromptModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_kernel = _prompt_kernel()

    with caplog.at_level("WARNING"):
        assert list(iter_all_glue_subnets(_RenamedPromptModel())) == []
    assert any("prompt_kernel" in r.message for r in caplog.records)


def test_out_of_range_default_lang_warns(caplog):
    with caplog.at_level("WARNING"):
        PromptKernelSubnet(
            _prompt_kernel(), num_prompts=P, d_model=D, default_lang_id=P + 5
        )
    assert any("outside" in r.message for r in caplog.records)


def test_fuse_glues_into_wraps_native_and_appends_extra_input():
    native = _FakeEncoder().eval()
    glue = PromptKernelSubnet(_prompt_kernel(), num_prompts=P, d_model=D)
    audio = torch.randn(2, 4, 6)
    length = torch.tensor([6, 6])
    fused, example, matched = fuse_glues_into(
        native, "encoder", [audio, length], [glue]
    )
    assert matched
    assert isinstance(fused, FusedGlueSubnet)
    assert fused.input_names == ["audio_signal", "length", "lang_id"]
    assert fused.output_names == ["outputs", "encoded_lengths"]
    assert len(example) == 3 and tuple(example[2].shape) == (1,)
    # input_types + dyn delegate to the native encoder
    assert fused.dynamic_shapes_for_export() == {
        "audio_signal": [0, 2],
        "length": [0],
    }
    with torch.no_grad():
        out = fused(audio, length, torch.tensor([7]))
        enc, _ = native(audio, length)
        ref = glue(enc, torch.tensor([7]))
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.allclose(out[0], ref)
    assert torch.equal(out[1], length)  # non-first outputs pass through


def test_fuse_glues_into_no_match_returns_native():
    native = _FakeEncoder().eval()
    glue = PromptKernelSubnet(_prompt_kernel(), num_prompts=P, d_model=D)
    audio, length = torch.randn(1, 4, 5), torch.tensor([5])
    module, example, matched = fuse_glues_into(
        native, "decoder_joint", [audio, length], [glue]
    )
    assert not matched
    assert module is native
    assert example == [audio, length]


def test_register_glue_subnet_roundtrip():
    sentinel = "._t2n_test_glue_model_"

    @register_glue_subnet(sentinel)
    def _b(_model):  # pragma: no cover - trivial
        return []

    try:
        assert GLUE_SUBNET_BUILDERS[sentinel] is _b
    finally:
        GLUE_SUBNET_BUILDERS.pop(sentinel, None)


def test_prompt_subnet_forward_matches_reference():
    pk = _prompt_kernel()
    sub = PromptKernelSubnet(pk, num_prompts=P, d_model=D, default_lang_id=3)
    enc, lang = sub.input_example(max_batch=2, seq_len=5)
    assert tuple(enc.shape) == (2, D, 5) and tuple(lang.shape) == (1,)
    with torch.no_grad():
        out = sub(enc, lang)
        feats = enc.transpose(1, 2)
        onehot = torch.zeros(feats.shape[0], feats.shape[1], P)
        onehot[..., 3] = 1.0  # default_lang_id
        ref = pk(torch.cat([feats, onehot], dim=-1)).transpose(1, 2)
    assert out.shape == enc.shape
    assert torch.allclose(out, ref, atol=1e-6)


def test_prompt_subnet_lang_id_changes_output():
    pk = _prompt_kernel()
    sub = PromptKernelSubnet(pk, num_prompts=P, d_model=D)
    enc = torch.randn(1, D, 4)
    with torch.no_grad():
        a = sub(enc, torch.tensor([0]))
        b = sub(enc, torch.tensor([5]))
    assert not torch.allclose(a, b)


def test_prompt_subnet_exports_and_check_io_exact():
    """The in-graph one-hot + concat + MLP round-trips through tract exactly."""
    tract = pytest.importorskip("torch_to_nnef.inference_target")
    from torch_to_nnef import export_model_to_nnef

    sub = PromptKernelSubnet(_prompt_kernel(), num_prompts=P, d_model=D).eval()
    example = sub.input_example(max_batch=2, seq_len=6)
    target = tract.TractNNEF.latest().with_dynamic_axes(
        {"encoder_outputs": {0: "B", 2: "T"}}
    )
    with tempfile.TemporaryDirectory() as d:
        export_model_to_nnef(
            model=sub,
            args=example,
            inference_target=target,
            input_names=sub.input_names,
            output_names=sub.output_names,
            file_path_export=Path(d) / "prompt.nnef.tgz",
            check_io_names_qte_match=False,
        )
