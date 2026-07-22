"""End-to-end ``dump_multimodal`` (the CLI entry point's worker).

Marked ``experimental`` (needs ``--run-experimental``): downloads
SmolVLM-256M and builds the decoder graph, the vision encoder graph, and the
``multimodal.json`` manifest. Runs with ``no_verify=True`` so it exercises the
full plumbing (load -> build both graphs -> write manifest) without spawning a
tract check_io subprocess.
"""

import json
from pathlib import Path

import pytest

from torch_to_nnef_llm.exporter import dump_multimodal

pytestmark = pytest.mark.experimental

MODEL_SLUG = "HuggingFaceTB/SmolVLM-256M-Instruct"


def test_dump_multimodal_writes_graphs_and_manifest(tmp_path):
    out = tmp_path / "mm"
    path, exporter = dump_multimodal(
        model_slug=MODEL_SLUG,
        export_dirpath=str(out),
        no_verify=True,
        force_module_dtype="f32",
    )

    assert Path(path) == out
    assert (out / "decoder" / "model.nnef.tgz").exists()
    assert (out / "vision" / "model.nnef.tgz").exists()

    manifest = json.loads((out / "multimodal.json").read_text())
    assert manifest["decoder"]["path"] == "decoder/model.nnef.tgz"

    assert len(manifest["encoders"]) == 1
    enc = manifest["encoders"][0]
    assert enc["modality"] == "image"
    assert enc["path"] == "vision/model.nnef.tgz"
    # placeholder token id is read from the HF config (the <image> token)
    assert isinstance(enc["placeholder_token_id"], int)

    out0 = enc["outputs"][0]
    # the encoder output feeds the decoder input of the same contract
    assert out0["name"] == "out_image_embeddings"
    assert out0["feeds"] == "in_image_embeddings"
    assert out0["shape"][-1] > 0  # hidden size
