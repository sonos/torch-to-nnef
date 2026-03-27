import torch

from torch_to_nnef.remodeler.adapter import BoundaryAdapter


class _Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_names = ["a"]
        self.output_names = ["o"]

    def forward(self, x):  # pragma: no cover - not used in this test
        return x


def test_dynamic_axes_symbol_renames_applied():
    m = _Dummy().eval()
    # External example with rank 2, dynamic axes declared on input 'a'
    ex = [torch.zeros(1, 2, dtype=torch.float32)]
    dyn = {"a": {0: "B", 1: "U"}}
    # Rename sources: B -> BATCH
    rename = {"BATCH": ["B"]}
    ba = BoundaryAdapter(
        m,
        "s",
        ex,
        dyn,
        collapse_by_input={},
        binds_by_input={},
        renamed_map=rename,
        outputs_keep=[],
    )
    out_dyn = ba.dynamic_shapes_for_export()
    assert out_dyn["a"][0] == "BATCH"
    assert out_dyn["a"][1] == "U"
