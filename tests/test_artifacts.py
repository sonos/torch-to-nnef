import tarfile
import tempfile
from pathlib import Path

import torch
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import TractNNEF


class Tiny(nn.Module):
    def forward(self, x):  # noqa: D401
        return x * 2


def _export_and_assert(model, x, out_path: Path, compression_level):
    model = model.eval()
    exported = export_model_to_nnef(
        model=model,
        args=x,
        file_path_export=out_path,
        input_names=["inp"],
        output_names=["out"],
        inference_target=TractNNEF.latest(),
        compression_level=compression_level,
    )
    assert exported.exists(), exported
    return exported


def test_artifacts_for_nnef_base_path():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "model.nnef"
        x = torch.rand(1, 2)
        m = Tiny()

        # None -> directory
        exported = _export_and_assert(m, x, base, compression_level=None)
        assert exported.suffix == ".nnef" and exported.is_dir()
        assert (exported / "graph.nnef").exists()

        # 0 -> tar
        exported = _export_and_assert(m, x, base, compression_level=0)
        assert exported.suffix == ".tar"
        with tarfile.open(exported, "r:") as tf:
            assert any(m.name.endswith("graph.nnef") for m in tf.getmembers())

        # 1 -> tgz
        exported = _export_and_assert(m, x, base, compression_level=1)
        assert exported.suffix == ".tgz"
        with tarfile.open(exported, "r:*") as tf:
            assert any(m.name.endswith("graph.nnef") for m in tf.getmembers())


def test_artifacts_for_nneftgz_path():
    with tempfile.TemporaryDirectory() as td:
        tgz_path = Path(td) / "model.nnef.tgz"
        x = torch.rand(1, 2)
        m = Tiny()

        # None -> directory at base name
        exported = _export_and_assert(m, x, tgz_path, compression_level=None)
        assert exported.suffix == ".nnef" and exported.is_dir()
        assert (exported / "graph.nnef").exists()

        # 0 -> tgz (store) honoring suffix intent
        exported = _export_and_assert(m, x, tgz_path, compression_level=0)
        assert exported.suffix == ".tgz"
        with tarfile.open(exported, "r:*") as tf:
            assert any(m.name.endswith("graph.nnef") for m in tf.getmembers())

        # 3 -> tgz (compressed)
        exported = _export_and_assert(m, x, tgz_path, compression_level=3)
        assert exported.suffix == ".tgz"
        with tarfile.open(exported, "r:*") as tf:
            assert any(m.name.endswith("graph.nnef") for m in tf.getmembers())

