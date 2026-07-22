"""Constant-folding of ops that carry an INFER_RULE must bake REAL values.

``IrOpNode.realise_output_type_and_size`` constant-folds any op whose inputs
are all compile-time constants (``has_constant_inputs``) by calling
``data_node.set_data(result)``. For ops with an ``INFER_RULE`` (e.g.
``aten::embedding``) the fast trace path (``_infer_trace_result(approx=True)``)
only infers shape/dtype and returns a ``torch.empty`` placeholder
(uninitialised memory). Folding that placeholder bakes GARBAGE as the constant
-- a structurally valid graph with silently wrong values.

This is the Qwen3-VL vision tower bug: ``fast_pos_embed_interpolate`` does
``pos_embed(idx)`` where ``idx`` is a constant derived from a baked grid and the
``pos_embed`` weight is an offloaded (lazily-loaded) param, so the embedding
folded to garbage and corrupted ~90% of the encoder outputs.

Two conditions must coincide, both reproduced below:

- the index is a *folded constant* (built with ``F.embedding`` in-scope; a
  submodule ``nn.Embedding`` would re-wrap the index as a fresh non-constant
  input and hide the bug), so ``aten::embedding`` has ``has_constant_inputs``
  and is folded, and
- the weight is *opaque* (an offloaded param -> ``meta`` tensor during tracing),
  so the embedding cannot execute while tracing and its output carries no data
  at fold time, which is what routes the fold through the placeholder.

We assert on the folded IR constant directly (rather than ``check_io``): with an
opaque weight the reference forward is itself degenerate, so an end-to-end
tolerance check cannot see the corruption -- but the baked constant must equal
the real gathered rows.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.opaque import set_opaque_tensor_in_params_as_ref
from torch_to_nnef.torch_graph.ir_graph import module_tracer_into_ir_graph
from torch_to_nnef.torch_graph.ir_module_tracer import TorchModuleTracer
from torch_to_nnef.torch_graph.torch_const import ATEN_EMBEDDING, ATEN_GATHER


class _EmbeddingOpaqueWeightConstIndex(nn.Module):
    """F.embedding(constant idx, opaque weight): folded ruled op."""

    def __init__(self, offload_dir):
        super().__init__()
        weight = torch.arange(40.0).reshape(8, 5)
        off = OffloadedTensor.from_original_tensor(
            weight, "emb_weight", offload_dir=Path(offload_dir)
        )
        self.weight = nn.Parameter(off, requires_grad=False)
        self.register_buffer(
            "base",
            torch.tensor([7, 0, 3, 5, 1, 10, 6, 12, 2, 9], dtype=torch.long),
        )

    def forward(self, x):
        idx = self.base.remainder(8)  # folded constant index (values 0..7)
        rows = F.embedding(idx, self.weight)
        return x + rows.sum(0)


def test_embedding_opaque_weight_const_index_folds_real_values():
    with tempfile.TemporaryDirectory() as od:
        mod = _EmbeddingOpaqueWeightConstIndex(od).eval()
        set_opaque_tensor_in_params_as_ref(mod)
        x = torch.zeros(5)

        graph = module_tracer_into_ir_graph(TorchModuleTracer(mod, args=(x,)))

        emb_ops = [op for op in graph.op_nodes if op.kind == ATEN_EMBEDDING]
        assert len(emb_ops) == 1
        out = emb_ops[0].outputs[0]

        # the embedding has all-constant inputs -> it is constant-folded; the
        # baked value must be the REAL gathered rows, not a torch.empty
        # placeholder.
        assert out.data is not None, "embedding output was not folded"
        weight = torch.arange(40.0).reshape(8, 5)
        idx = torch.tensor([7, 0, 3, 5, 1, 10, 6, 12, 2, 9]).remainder(8)
        expected = F.embedding(idx, weight)
        assert torch.equal(out.data.float(), expected), (
            "constant-folded embedding baked garbage instead of real rows"
        )


class _GatherOpaqueSourceConstIndex(nn.Module):
    """torch.gather(opaque source, constant index): folded ruled op.

    Folding this op now runs ``call_op(aten::gather)``; its ``sparse_grad``
    kwarg-only arg must be routed so overload resolution keeps the ``int dim``
    overload instead of flipping to ``dimname`` (str dim) and raising.
    """

    def __init__(self, offload_dir):
        super().__init__()
        src = torch.arange(12.0).reshape(3, 4)
        off = OffloadedTensor.from_original_tensor(
            src, "gather_src", offload_dir=Path(offload_dir)
        )
        self.src = nn.Parameter(off, requires_grad=False)
        self.register_buffer(
            "base",
            torch.tensor(
                [[3, 5, 4, 3], [5, 4, 3, 5], [4, 3, 5, 4]], dtype=torch.long
            ),
        )

    def forward(self, x):
        idx = self.base.remainder(3)  # folded constant index (values 0..2)
        picked = torch.gather(self.src, 0, idx)
        return x + picked.sum(0)


def test_gather_opaque_source_const_index_folds_real_values():
    with tempfile.TemporaryDirectory() as od:
        mod = _GatherOpaqueSourceConstIndex(od).eval()
        set_opaque_tensor_in_params_as_ref(mod)
        x = torch.zeros(4)

        graph = module_tracer_into_ir_graph(TorchModuleTracer(mod, args=(x,)))

        gather_ops = [op for op in graph.op_nodes if op.kind == ATEN_GATHER]
        assert len(gather_ops) == 1
        out = gather_ops[0].outputs[0]

        assert out.data is not None, "gather output was not folded"
        src = torch.arange(12.0).reshape(3, 4)
        idx = torch.tensor(
            [[3, 5, 4, 3], [5, 4, 3, 5], [4, 3, 5, 4]]
        ).remainder(3)
        expected = torch.gather(src, 0, idx)
        assert torch.equal(out.data.float(), expected), (
            "constant-folded gather baked garbage instead of real values"
        )
