"""Helpers for systematic multimodal export tests on shrunk dummy configs.

The real per-architecture checkpoints are large, so the default tests build a
tiny random-weight model from a shrunk `transformers` config and run it through
the *real* :class:`MultiModalExporter` (encoder + decoder + manifest), including
the tract ``check_io``. A shrunk model accumulates almost no fp16 rounding, so
its export can be checked at a tight tolerance -- a stronger drift guard than a
loose check on a real deep model, and it needs no asset download.
"""

import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import torch

from torch_to_nnef_llm.exporter import LM_VAR_SCHEME, LLMExporter
from torch_to_nnef_llm.multimodal_exporter import MultiModalExporter

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


def assert_dummy_multimodal_export(
    config, model_cls, dtype: str, export_dirpath: Path
) -> Path:
    """Export a tiny dummy model and assert graphs + manifest + check_io.

    ``export`` runs the tract ``check_io`` and raises on mismatch, so reaching
    the assertions means the numerical check passed. For ``f16`` we additionally
    assert the encoder graph carries the SDPA + f32-accumulation rewrite that
    makes fp16 towers verifiable (guards `_prefer_sdpa_attention` / the f32
    flags in ``export``).
    """
    exporter = build_dummy_exporter(config, model_cls, dtype)
    target = exporter.decoder_exporter.build_inference_target(no_verify=False)
    if export_dirpath.exists():
        shutil.rmtree(export_dirpath)
    exporter.export(export_dirpath, target, LM_VAR_SCHEME)

    assert (export_dirpath / DECODER_DIRNAME / "model.nnef.tgz").exists()
    assert (export_dirpath / "multimodal.json").exists()
    encoder_dirs = [
        p
        for p in export_dirpath.iterdir()
        if p.is_dir() and p.name != DECODER_DIRNAME
    ]
    assert encoder_dirs, "no encoder graph produced"

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
