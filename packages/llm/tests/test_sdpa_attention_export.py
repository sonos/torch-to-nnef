import sys
from types import SimpleNamespace

import pytest

from torch_to_nnef.exceptions import T2NErrorMisuse
from torch_to_nnef_llm import cli, exporter, loader


class _FakeModel:
    def __init__(self, config=None):
        self.config = config or SimpleNamespace(model_type="fake")

    def to(self, _dtype):
        return self


class _FakeAutoModel:
    calls = []

    @classmethod
    def from_config(cls, config, **kwargs):
        cls.calls.append(kwargs)
        return _FakeModel(config)


class _FakeTransformers:
    __version__ = "5.12.1"
    AutoModelForCausalLM = _FakeAutoModel
    integrations = SimpleNamespace(
        moe=SimpleNamespace(
            ALL_EXPERTS_FUNCTIONS=SimpleNamespace(
                valid_keys=lambda: ["batched_mm", "grouped_mm"]
            )
        )
    )


class _FakeTransformersWithoutMoe:
    """Stand-in for a supported Transformers predating the experts registry."""

    __version__ = "4.52.0"
    AutoModelForCausalLM = _FakeAutoModel
    integrations = SimpleNamespace()


def _load_fake_model(
    monkeypatch,
    *,
    attn_implementation=None,
    experts_implementation="auto",
    has_moe_experts=True,
    transformers=_FakeTransformers,
):
    _FakeAutoModel.calls = []
    model_config = SimpleNamespace(model_type="phi")
    if has_moe_experts:
        model_config._experts_implementation = "grouped_mm"
    monkeypatch.setitem(
        loader.CUSTOM_CONFIGS,
        "fake/model",
        model_config,
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
        experts_implementation=experts_implementation,
        transformers=transformers,
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


def test_load_model_accepts_explicit_experts_implementation(monkeypatch):
    model = _load_fake_model(monkeypatch, experts_implementation="batched_mm")

    assert model.config._experts_implementation == "batched_mm"


def test_load_model_uses_export_safe_auto_experts_default(monkeypatch):
    model = _load_fake_model(monkeypatch, experts_implementation="auto")

    assert model.config._experts_implementation == "batched_mm"


def test_load_model_can_keep_model_experts_default(monkeypatch):
    model = _load_fake_model(monkeypatch, experts_implementation="model")

    assert model.config._experts_implementation == "grouped_mm"


def test_load_model_ignores_experts_implementation_without_moe(monkeypatch):
    model = _load_fake_model(
        monkeypatch,
        experts_implementation="auto",
        has_moe_experts=False,
    )

    assert not hasattr(model.config, "_experts_implementation")


def test_load_model_rejects_unknown_experts_implementation(monkeypatch):
    with pytest.raises(T2NErrorMisuse, match="experts_implementation"):
        _load_fake_model(
            monkeypatch, experts_implementation="definitely-not-real"
        )


def test_load_model_auto_is_noop_without_experts_registry(monkeypatch):
    # Supported Transformers predating the experts registry must still load;
    # the default `auto` degrades to a no-op instead of failing the export.
    model = _load_fake_model(
        monkeypatch,
        experts_implementation="auto",
        transformers=_FakeTransformersWithoutMoe,
    )

    assert model.config._experts_implementation == "grouped_mm"


def test_load_model_rejects_explicit_experts_without_registry(monkeypatch):
    with pytest.raises(T2NErrorMisuse, match="ALL_EXPERTS_FUNCTIONS"):
        _load_fake_model(
            monkeypatch,
            experts_implementation="batched_mm",
            transformers=_FakeTransformersWithoutMoe,
        )


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


def test_dump_llm_routes_experts_implementation_to_loader(
    monkeypatch, tmp_path
):
    captured_load = {}

    class _Exporter:
        def dump(self, **kwargs):
            del kwargs

    def fake_load(*args, **kwargs):
        captured_load.update(kwargs)
        return _Exporter()

    monkeypatch.setattr(exporter.LLMExporter, "load", staticmethod(fake_load))
    exporter.dump_llm(
        "fake/model",
        export_dirpath=tmp_path / "export",
        experts_implementation="batched_mm",
    )

    assert captured_load["experts_implementation"] == "batched_mm"


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


def test_cli_accepts_transformers_experts_implementation(monkeypatch, tmp_path):
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
            "--transformers-experts-implementation",
            "batched_mm",
        ],
    )

    cli.main()

    assert captured["experts_implementation"] == "batched_mm"
