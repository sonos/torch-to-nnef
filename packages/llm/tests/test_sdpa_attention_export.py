import sys
from types import SimpleNamespace

import pytest

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef_llm import cli, exporter, loader


class _FakeModel:
    config = SimpleNamespace(model_type="fake")

    def to(self, _dtype):
        return self


class _FakeAutoModel:
    calls = []

    @classmethod
    def from_config(cls, _config, **kwargs):
        cls.calls.append(kwargs)
        return _FakeModel()


class _FakeTransformers:
    __version__ = "5.12.1"
    AutoModelForCausalLM = _FakeAutoModel


def _load_fake_model(monkeypatch, *, attn_implementation=None):
    _FakeAutoModel.calls = []
    monkeypatch.setitem(
        loader.CUSTOM_CONFIGS,
        "fake/model",
        SimpleNamespace(model_type="phi"),
    )
    monkeypatch.setattr(
        loader,
        "resolve_auto_model_class",
        lambda _model_type: _FakeAutoModel,
    )
    return loader.load_model.__wrapped__(
        "fake/model",
        trust_remote_code=False,
        attn_implementation=attn_implementation,
        transformers=_FakeTransformers,
    )


def test_load_model_keeps_safe_eager_default_for_transformers_5(monkeypatch):
    _load_fake_model(monkeypatch)
    assert _FakeAutoModel.calls[-1]["attn_implementation"] == "eager"


def test_load_model_accepts_explicit_sdpa_attention(monkeypatch):
    _load_fake_model(monkeypatch, attn_implementation="sdpa")
    assert _FakeAutoModel.calls[-1]["attn_implementation"] == "sdpa"


def test_load_model_treats_auto_attention_as_default(monkeypatch):
    _load_fake_model(monkeypatch, attn_implementation="auto")
    assert _FakeAutoModel.calls[-1]["attn_implementation"] == "eager"


def test_load_model_rejects_unknown_attention_implementation(monkeypatch):
    with pytest.raises(T2NErrorMisuse, match="attn_implementation"):
        _load_fake_model(monkeypatch, attn_implementation="definitely-not-real")


def test_reify_sdpa_operator_implies_sdpa_attention():
    assert exporter._resolve_attn_implementation(None, True) == "sdpa"
    assert exporter._resolve_attn_implementation("auto", True) == "sdpa"
    assert exporter._resolve_attn_implementation("auto", False) is None


def test_reify_sdpa_operator_rejects_eager_attention():
    with pytest.raises(T2NErrorMisuse, match="requires"):
        exporter._resolve_attn_implementation("eager", True)


def test_dump_llm_routes_reified_sdpa_to_loader(monkeypatch, tmp_path):
    captured_load = {}
    captured_dump = {}

    class _Exporter:
        def dump(self, **kwargs):
            captured_dump.update(kwargs)

    def fake_load(*args, **kwargs):
        captured_load.update(kwargs)
        return _Exporter()

    monkeypatch.setattr(exporter.LLMExporter, "load", staticmethod(fake_load))
    exporter.dump_llm(
        "fake/model",
        export_dirpath=tmp_path / "export",
        reify_sdpa_operator=True,
    )

    assert captured_load["attn_implementation"] == "sdpa"
    assert captured_dump["reify_sdpa_operator"] is True


def test_cli_reify_sdpa_operator_implies_sdpa_attention(monkeypatch, tmp_path):
    captured = {}

    def fake_dump_llm(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "dump_llm", fake_dump_llm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch-to-nnef-llm",
            "-s",
            "fake/model",
            "-e",
            str(tmp_path / "export"),
            "--reify-sdpa-operator",
        ],
    )

    cli.main()

    assert captured["attn_implementation"] == "sdpa"


def test_cli_accepts_transformers_attention_implementation(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_dump_llm(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "dump_llm", fake_dump_llm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "torch-to-nnef-llm",
            "-s",
            "fake/model",
            "-e",
            str(tmp_path / "export"),
            "--transformers-attn-implementation",
            "eager",
        ],
    )

    cli.main()

    assert captured["attn_implementation"] == "eager"
