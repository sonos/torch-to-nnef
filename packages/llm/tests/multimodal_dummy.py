"""Helpers for systematic multimodal export tests on shrunk dummy configs.

The real per-architecture checkpoints are large, so the default tests build a
tiny random-weight model from a shrunk `transformers` config and run it through
the *real* :class:`MultiModalExporter` (encoder + decoder + manifest), including
the tract ``check_io`` (which raises on numerical mismatch). A shrunk model
accumulates almost no rounding, so even the loosest preset the exporter
selects is a meaningful drift guard here, and it needs no asset download.
f32 exports at the decoder/encoder default tolerance; f16 exports at the
exporter's fp16 preset (``ULTRA``) like every fp16 model, but the tiny model
barely diverges regardless.
"""

import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import torch

from torch_to_nnef_llm.exporter import LM_VAR_SCHEME, LLMExporter
from torch_to_nnef_llm.multimodal_exporter import (
    MANIFEST_NAME,
    MultiModalExporter,
)

DECODER_DIRNAME = "decoder"


class DummyTokenizer:
    """Minimal tokenizer stub.

    The decoder handler only uses the tokenizer to obtain an ``input_ids``
    tensor of the right shape; it then overwrites the values (placing the
    modality placeholder token itself), so any long-enough tensor works.
    """

    def __call__(self, text, return_tensors=None):  # noqa: D401,ARG002
        return SimpleNamespace(input_ids=torch.zeros(1, 128, dtype=torch.long))


def build_dummy_exporter(config, model_cls, dtype: str) -> MultiModalExporter:
    """Build a `MultiModalExporter` around a fresh random-weight tiny model."""
    torch.manual_seed(0)
    model = model_cls(config).eval()
    if dtype == "f16":
        model = model.half()
    decoder_exporter = LLMExporter(
        hf_model_causal=model,
        tokenizer=DummyTokenizer(),
        force_module_dtype=dtype,
    )
    return MultiModalExporter(decoder_exporter)


def _graph_nnef(archive_path: Path) -> str:
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.name.endswith("graph.nnef"):
                extracted = tf.extractfile(member)
                assert extracted is not None
                return extracted.read().decode("utf-8")
    raise AssertionError(f"graph.nnef not found in {archive_path}")


def _assert_manifest_matches_contracts(
    export_dirpath: Path, exporter: MultiModalExporter
) -> None:
    """Check ``multimodal.json`` wires each contract to its decoder input."""
    manifest = json.loads((export_dirpath / MANIFEST_NAME).read_text())
    assert manifest["decoder"]["path"] == f"{DECODER_DIRNAME}/model.nnef.tgz"

    config = exporter.hf_model_causal.config
    contracts = exporter.contracts
    entries_by_output = {
        out["name"]: (entry, out)
        for entry in manifest["encoders"]
        for out in entry["outputs"]
    }
    assert len(entries_by_output) == len(contracts), (
        f"manifest has {len(entries_by_output)} encoder output(s) for "
        f"{len(contracts)} contract(s)"
    )
    for contract in contracts:
        assert contract.output_name in entries_by_output, (
            f"{contract.output_name} missing from manifest"
        )
        entry, out = entries_by_output[contract.output_name]
        assert out["feeds"] == contract.input_name
        assert out["shape"] == [contract.dynamic_axis, contract.hidden_size]
        assert entry["placeholder_token_id"] == getattr(
            config, contract.placeholder_token_id_attr
        )
        if contract.injection_layers:
            deepstack = entry["deepstack"]
            assert [d["layer"] for d in deepstack] == list(
                contract.injection_layers
            )
            for i, stream in enumerate(deepstack):
                assert stream["name"] == contract.deepstack_output_name(i)
                assert stream["feeds"] == contract.deepstack_input_name(i)
        else:
            assert "deepstack" not in entry


def assert_dummy_multimodal_export(
    config, model_cls, dtype: str, export_dirpath: Path
) -> Path:
    """Export a tiny dummy model and assert graphs + manifest + check_io.

    ``export`` runs the tract ``check_io`` and raises on mismatch, so reaching
    the assertions means the numerical check passed. We then assert the
    ``multimodal.json`` manifest matches the exporter's embedding contracts
    (so a wiring regression is caught even when both graphs export cleanly). For
    ``f16`` we additionally assert the encoder graph carries the SDPA +
    f32-accumulation rewrite that makes fp16 towers verifiable (guards
    `_prefer_sdpa_attention` / the f32 flags in ``export``).
    """
    exporter = build_dummy_exporter(config, model_cls, dtype)
    target = exporter.decoder_exporter.build_inference_target(no_verify=False)
    if export_dirpath.exists():
        shutil.rmtree(export_dirpath)
    exporter.export(export_dirpath, target, LM_VAR_SCHEME)

    assert (export_dirpath / DECODER_DIRNAME / "model.nnef.tgz").exists()
    assert (export_dirpath / MANIFEST_NAME).exists()
    encoder_dirs = [
        p
        for p in export_dirpath.iterdir()
        if p.is_dir() and p.name != DECODER_DIRNAME
    ]
    assert encoder_dirs, "no encoder graph produced"

    _assert_manifest_matches_contracts(export_dirpath, exporter)

    if dtype == "f16":
        for enc_dir in encoder_dirs:
            graph = _graph_nnef(enc_dir / "model.nnef.tgz")
            assert (
                "scaled_dot_product_attention" in graph
                or "tract_transformers_sdpa" in graph
            ), f"{enc_dir.name}: fp16 encoder not routed through SDPA"
            assert "tract_core_cast" in graph and "to = 'f32'" in graph, (
                f"{enc_dir.name}: fp16 encoder missing f32 accumulation cast"
            )
    return export_dirpath
