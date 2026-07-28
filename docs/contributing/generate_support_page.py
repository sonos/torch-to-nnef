"""Generate supported operators markdown page.

Allow to compare supported operators in `torch_to_nnef` and `ONNX` builtin
support against core PyTorch operators as per PyTorch IR documentation.

Disclaimer: this is a best effort script that may not reflect 100% reality
of operator support in all cases. It is meant to give a general idea
of the coverage level of `torch_to_nnef` regarding PyTorch operators.
"""

import argparse
import json
import re
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

# When invoked as ``python docs/contributing/generate_support_page.py``
# Python seeds ``sys.path[0]`` with this script's directory, which means
# ``import torch_to_nnef`` resolves against ``site-packages`` (any stale
# install in ``.tox/<env>/lib/.../site-packages/`` wins) instead of the
# in-tree source. Prepend the repo root so the live source always wins.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bs4  # noqa: E402
import requests as rq  # noqa: E402
import rich.progress  # noqa: E402

# The graded ONNX artifact is produced by the proptest sweep
# (`tests/test_primitive_proptest_onnx.py`), and its grading rules live
# with it. Importing them keeps one definition of what `partial` means
# instead of a second copy here that could drift from the measurement.
from tests.proptest.onnx_report import (  # noqa: E402
    GRADE_BLOCKED,
    GRADE_FULL,
    GRADE_NONE,
    GRADE_PARTIAL,
    GRADE_UNTESTED,
    SCHEMA_VERSION,
    merge_op_records,
)
from torch_to_nnef.op.aten import aten_ops_registry  # noqa: E402

#: Floor for the `aten::` source grep. torch 2.13 yields 1159 raw names
#: (581 page rows after pruning), and no supported release is anywhere
#: near this low, so a smaller result means the extraction broke.
_MIN_PLAUSIBLE_ATEN_NAMES = 400

#: Display-only refinement of `untested`: no spec covers the operator, but
#: the retired TorchScript listing claimed it was supported. Kept out of
#: `onnx_report`'s grade vocabulary because it is not a measurement:
#: it is an unverified historical claim, and the star is what says so.
DISPLAY_UNTESTED_DOCUMENTED = "untested-documented"

#: Glyph per export grade. `blocked` is not an ONNX verdict: `torch.export`
#: could not capture the module, so the exporter never saw it.
GRADE_GLYPH = {
    GRADE_FULL: "✅",
    GRADE_PARTIAL: "🟡",
    GRADE_NONE: "❌",
    GRADE_BLOCKED: "⚠️",
    GRADE_UNTESTED: "-",
    DISPLAY_UNTESTED_DOCUMENTED: "✅*",
}

#: Filter radio buttons per section flavour. The binary listing keeps the
#: historical three modes verbatim so its rendered section is unchanged.
BINARY_FILTER_MODES = (
    ("all", "All"),
    ("supported", "Supported only"),
    ("unsupported", "Unsupported only"),
)

#: Cross-section filter: operators the *other* tab credits and this one
#: does not. On the `TractNNEF` tab that is the implementation shortlist,
#: which is the one question the two tables can answer together and
#: neither can answer alone.
CROSS_GAP_MODE = "cross-gap"

#: Class marking a row the other tab credits as supported.
CROSS_OK_CLASS = "cross-ok"
#: Labels say what the filter selects, not what the grade is called. Next
#: to "All", a radio labelled "None" reads as "select nothing", and
#: "Blocked" does not say blocked by what (`torch.export`, never ONNX).
GRADED_FILTER_MODES = (
    ("all", "All"),
    (GRADE_FULL, "Exports fully"),
    (GRADE_PARTIAL, "Exports partially"),
    (GRADE_NONE, "Never exports"),
    (GRADE_BLOCKED, "Blocked before ONNX"),
    (DISPLAY_UNTESTED_DOCUMENTED, "Claimed, unverified"),
    (GRADE_UNTESTED, "Untested, no data"),
)


def headline_bars(core_ratio: str, total_ratio: str) -> str:
    """The two headline progress bars, in the order every section uses.

    Shared so the tabs stay comparable. Both take the same denominators
    (core opset size, then the full `aten::` listing) and state them in
    the same order, otherwise the reader is put in front of two bars that
    look like a comparison and are not one.
    """
    return (
        "- core PyTorch opset:\n\n"
        f'[={core_ratio} "{core_ratio}"]\n\n'
        "-  and support from full `aten::`: \n\n"
        f'[={total_ratio} "{total_ratio}"]\n\n'
    )


class MeasuredOnnxSupport:
    """Per-operator ONNX grades, keyed by the page's own row names.

    The artifact is keyed by the aten names the proptest specs declare.
    Those are chosen to be page-visible, but the page still applies two
    normalizations of its own (aliases collapse onto their canonical name,
    and in-place variants merge into the base name), so a measured name
    can land on a row under a different spelling, and two measured names
    can land on the same row. Records are merged by summing counters, so
    the row's grade is derived from every example that reached it.
    """

    def __init__(
        self,
        payload: dict,
        alias_manager: "AliasManager",
        page_rows: Set[str],
    ):
        schema = payload.get("schema")
        if schema != SCHEMA_VERSION:
            # Grades are re-derived from the record's counters, so an
            # artifact written against a different schema can silently
            # regrade (missing counters read as zero, which turns a
            # `none` into `blocked`). Refuse instead.
            raise ValueError(
                f"ONNX artifact schema {schema!r} != expected "
                f"{SCHEMA_VERSION}; regenerate it with "
                "`tox -e proptest_onnx --onnx-no-reuse`"
            )
        self.payload = payload
        self.regen_index = payload.get("regen_index", 0)
        self.measurements = payload.get("measurements", {})
        by_row: dict = defaultdict(list)
        self.unmapped: Set[str] = set()
        for op_name, backends in payload.get("ops", {}).items():
            record = backends.get("onnx")
            if record is None:
                continue
            row = self._normalize(op_name, alias_manager)
            if row not in page_rows:
                # Measured but absent from the page's operator list: the
                # `aten::*` source grep drops `_`-prefixed names, so e.g.
                # `_upsample_nearest_exact2d` has no row. Recorded so the
                # generator can report it rather than dropping it silently.
                self.unmapped.add(op_name)
                continue
            by_row[row].append(record)
        self.rows = {
            row: merge_op_records(records) for row, records in by_row.items()
        }

    @staticmethod
    def _normalize(op_name: str, alias_manager: "AliasManager") -> str:
        """Map a measured op name onto the page's row name."""
        name = op_name
        if alias_manager.is_alias(name):
            # `absolute` is listed under `abs`, etc.
            for canonical, aliases in alias_manager.ref_alias.items():
                if name in aliases:
                    name = canonical
                    break
        if name.endswith("_"):
            # In-place variants are merged into their base name.
            name = name[:-1]
        return name

    def record(self, row: str) -> Optional[dict]:
        return self.rows.get(row)

    def grade(self, row: str) -> str:
        record = self.rows.get(row)
        if record is None:
            return GRADE_UNTESTED
        return record.get("export", GRADE_UNTESTED)

    def axis(self, row: str, axis: str) -> str:
        record = self.rows.get(row)
        if record is None:
            return "-"
        value = record.get(axis, "-")
        return "-" if value == "not_reached" else value

    @property
    def newest_measurement(self) -> dict:
        """The measurement entry describing this regeneration."""
        if not self.measurements:
            return {}
        key = f"m{self.regen_index}"
        if key in self.measurements:
            return self.measurements[key]
        return list(self.measurements.values())[-1]

    def counts(self) -> "Counter":
        return Counter(
            record.get("export", GRADE_UNTESTED)
            for record in self.rows.values()
        )


class LinkToTorchDocCache:
    UNK = "unk"

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_dic = self.load()

    def load(self):
        base = defaultdict(set)
        if self.cache_path.exists():
            with self.cache_path.open("r", encoding="utf8") as fh:
                for pat, elms in json.load(fh).items():
                    for elm in elms:
                        base[pat].add(elm)
        return base

    def save(self):
        with self.cache_path.open("w", encoding="utf8") as fh:
            json.dump(
                {k: sorted(list(v)) for k, v in self.cache_dic.items()},
                fh,
                indent=4,
            )

    def add(self, pattern: str, op_name: str, exclusive_pattern: bool = True):
        for k, v in self.cache_dic.items():
            # `==` not `is`: the UNK key survives a JSON round-trip as
            # an equal-but-not-identical string, so `is self.UNK` lets
            # the UNK bucket short-circuit the early return on every
            # op (everything ends up in UNK on first probe), defeating
            # the purpose of additional URL-pattern fallbacks.
            if k == self.UNK:
                continue
            if op_name in v and exclusive_pattern:
                return
        if rq.get(pattern.format(op_name), timeout=20).status_code == 200:
            self.cache_dic[pattern].add(op_name)
            if op_name in self.cache_dic[self.UNK]:
                self.cache_dic[self.UNK].remove(op_name)
        else:
            self.cache_dic[self.UNK].add(op_name)

    def get_url(self, op_name) -> Optional[str]:
        for k, v in self.cache_dic.items():
            if op_name in v and k != self.UNK:
                return k.format(op_name)
        return None


class AliasManager:
    def __init__(self, alias_tups: Set[tuple[str, ...]]):
        self._alias_tups = alias_tups
        self._aliases = set([_[0] for _ in alias_tups])
        self.ref_alias = defaultdict(list)
        for k, v in self._alias_tups:
            self.ref_alias[v].append(k)

        # sorted aliases for consistent output
        for k, v in self.ref_alias.items():
            self.ref_alias[k] = sorted(v)

    def is_alias(self, op_name: str) -> bool:
        return op_name in self._aliases

    def get_aliases(self, op_name: str) -> List[str]:
        return self.ref_alias.get(op_name, [])


#: Last torch version whose ``torch.compiler_ir.html`` page enumerates the
#: core ATen IR ops in scrapeable form. Starting at 2.10 the page was
#: emptied (the published version is ~1 KB of boilerplate), so we fall
#: back to this one to keep the "is core" column populated.
LAST_TORCH_VERSION_WITH_IR_DOC = "2.9"

# =============================================================================
# Noise filters.
#
# `get_aten_torch_from_code` greps PyTorch source for every ``aten::*``
# string it can find. That sweep is intentionally broad and catches a
# long tail of identifiers that are *not* tensor ops -- Python builtin
# method names referenced in TorchScript scaffolding, distributed-RPC
# primitives, sparse-tensor machinery, etc. These cannot surface in any
# JIT-traced inference graph and so cannot be meaningful targets for a
# `torch_to_nnef` emitter (not even as a no-op map).
#
# Each subgroup below is documented with the reason we are 100%
# confident the listed names will never appear in an inference trace.
# Random ops, loss ops, training-only ops, `*_copy` functionalisation
# variants etc. are deliberately *not* excluded: a model could plausibly
# use them, even if rarely, and a no-op / decomposition mapping might be
# valuable.
# =============================================================================

_EXCLUDED_PYTHON_BUILTIN_FALSE_POSITIVES = frozenset(
    {
        # The `rg "aten::"` sweep also reaches TorchScript builtins exposed
        # for scripted Python compatibility (`str.capitalize`, `dict.keys`,
        # `list.pop`, etc.). They are not tensor ops and cannot land in a
        # traced graph.
        "capitalize",
        "center",
        "chr",
        "clear",
        "dict",
        "endswith",
        "expandtabs",
        "extend",
        "find",
        "format",
        "get",
        "getelem",
        "hash",
        "hex",
        "isalnum",
        "isalpha",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
        "items",
        "join",
        "keys",
        "ljust",
        "lower",
        "lstrip",
        "oct",
        "ord",
        "partition",
        "popitem",
        "rfind",
        "rindex",
        "rjust",
        "rpartition",
        "rsplit",
        "rstrip",
        "setdefault",
        "sorted",
        "splitlines",
        "startswith",
        "strip",
        "swapcase",
        "title",
        "update",
        "upper",
        "values",
        "zfill",
    }
)

_EXCLUDED_TEST_FIXTURES = frozenset(
    {
        # Test harness / placeholder / debugging identifiers that ship in
        # PyTorch source for unit tests or symbolic-shape scaffolding.
        "confirmed_by_owner",
        "foo",
        "mathremainder",
        "percentFormat",
        "pointwise_placeholder",
        "symbolic_b",
        "test",
        "test_symbol",
        "test_vartype",
        "test_vartype2",
        "unknown",
        "view_expand_placeholder",
        "your_op",
    }
)

_EXCLUDED_DISTRIBUTED_RPC = frozenset(
    {
        # `torch.distributed` collectives / RPC primitives. By definition
        # cross-process, never recorded in a single-rank inference trace.
        "all_gather_into_tensor",
        "all_reduce",
        "fork",
        "get_gradients",
        "is_owner",
        "local_value",
        "owner",
        "owner_name",
        "reduce_scatter_tensor",
        "to_here",
        "wait",
        "wait_tensor",
        "warn",
        "warns",
    }
)

_EXCLUDED_BACKEND_SPECIFIC = frozenset(
    {
        # Backend-specific dispatch shims (`cudnn_*` / `miopen_*` /
        # `mkldnn_*` / `mps_*`). The framework picks one at compile time
        # based on device + build flags; the resulting JIT trace always
        # records the generic op (`convolution`, `linear`, ...). t2n
        # exports run device-agnostic NNEF, so these can never reach us.
        "cudnn_affine_grid_generator",
        "cudnn_batch_norm",
        "cudnn_convolution",
        "cudnn_convolution_add_relu",
        "cudnn_convolution_relu",
        "cudnn_convolution_transpose",
        "cudnn_grid_sampler",
        "cudnn_is_acceptable",
        "miopen_batch_norm",
        "miopen_convolution",
        "miopen_convolution_add_relu",
        "miopen_convolution_relu",
        "miopen_convolution_transpose",
        "miopen_ctc_loss",
        "miopen_depthwise_convolution",
        "miopen_rnn",
        "mkldnn_adaptive_avg_pool2d",
        "mkldnn_convolution",
        "mkldnn_linear",
        "mkldnn_max_pool2d",
        "mkldnn_max_pool3d",
        "mkldnn_reorder_conv2d_weight",
        "mkldnn_reorder_conv3d_weight",
        "mkldnn_rnn_layer",
        "mps_linear",
        "to_mkldnn",
    }
)

_EXCLUDED_TENSOR_METADATA = frozenset(
    {
        # Return a Python value (int / bool / dtype / device), not a
        # tensor. They cannot appear *as* graph nodes that produce a
        # tensor output; downstream graph ops see the resolved literal.
        "can_cast",
        "data",
        "dense_dim",
        "device",
        "dtype",
        "element_size",
        "enable_grad",
        "get_autocast_dtype",
        "get_device",
        "get_pool_ceil_padding",
        "grad",
        "has_torch_function",
        "iinfo",
        "initial_seed",
        "int_repr",
        "is_autocast_cpu_enabled",
        "is_autocast_enabled",
        "is_coalesced",
        "is_complex",
        "is_conj",
        "is_contiguous",
        "is_cuda",
        "is_grad_enabled",
        "is_leaf",
        "is_non_overlapping_and_dense",
        "is_nonzero",
        "is_pinned",
        "is_same_size",
        "is_scripting",
        "is_set_to",
        "is_signed",
        "is_strides_like_format",
        "manual_seed",
        "node",
        "op",
        "op_name",
        "output_nr",
        "pin_memory",
        "promote_types",
        "q_per_channel_axis",
        "q_per_channel_scales",
        "q_per_channel_zero_points",
        "q_scale",
        "q_zero_point",
        "qscheme",
        "record_stream",
        "refine_names",
        "rename",
        "requires_grad_",
        "result_type",
        "retain_grad",
        "retains_grad",
        "save",
        "seed",
        "set_data",
        "set_grad_enabled",
        "set_source_Tensor_storage_offset",
        "sparse_dim",
        "storage_offset",
        "stride",
    }
)

_EXCLUDED_NAMED_TENSORS = frozenset(
    {
        # PyTorch's named-tensor API: `align_as`, `align_to`, ... rely on
        # axis names that JIT trace strips before recording the graph, so
        # these can never reach the converter.
        "align_as",
        "align_tensors",
        "align_to",
    }
)

_EXCLUDED_SPARSE_ONLY = frozenset(
    {
        # NNEF / tract are dense-only inference targets. PyTorch's sparse
        # tensor machinery (COO / CSR / CSC factories, layout accessors,
        # sparse-only matmul kernels) has no representation in our IR.
        "ccol_indices",
        "ccol_indices_copy",
        "col_indices",
        "col_indices_copy",
        "copy_sparse_to_sparse",
        "crow_indices",
        "crow_indices_copy",
        "coalesce",
        "hspmm",
        "indices",
        "indices_copy",
        "nested_to_padded_tensor",
        "row_indices",
        "row_indices_copy",
        "smm",
        "sparse_compressed_tensor",
        "sparse_coo_tensor",
        "sparse_mask",
        "sparse_resize",
        "sparse_resize_and_clear",
        "sparse_sampled_addmm",
        "sspaddmm",
        "to_dense",
        "to_padded_tensor",
        "values_copy",
    }
)

_EXCLUDED_FUNCTIONALIZATION_COPY_SCATTER = frozenset(
    {
        # Functionalization `*_copy` / `*_scatter` variants. The
        # functionalization pass (used by FX / AOT export pipelines)
        # emits these as out-of-place / strided-write surrogates for
        # views. The JIT trace that t2n consumes skips functionalization
        # and records the underlying view op (`view`, `slice`, `select`,
        # `narrow`, `permute`, `unbind`, `as_strided`, ...), so the
        # `_copy` / `_scatter` aliases never reach us.
        "alias_copy",
        "as_strided_copy",
        "as_strided_scatter",
        "detach_copy",
        "diagonal_copy",
        "diagonal_scatter",
        "lift",
        "lift_fresh",
        "lift_fresh_copy",
        "narrow_copy",
        "permute_copy",
        "select_copy",
        "slice_copy",
        "slice_inverse",
        "split_copy",
        "unbind_copy",
        "unfold_copy",
        "view_as_complex_copy",
        "view_as_real_copy",
        "view_copy",
    }
)

_EXCLUDED_BATCH_NORM_TRAINING_INTERNALS = frozenset(
    {
        # Training-only / backward-only batch-norm internals: running
        # stats computation (`*_stats`, `*_update_stats`,
        # `*_gather_stats`) and the legit-mode forward that returns
        # `(out, save_mean, save_invstd)` for the backward pass.
        # Inference paths resolve through `aten::batch_norm`; these
        # never escape into a JIT trace of an inference graph.
        "batch_norm_elemt",
        "batch_norm_gather_stats",
        "batch_norm_gather_stats_with_counts",
        "batch_norm_stats",
        "batch_norm_update_stats",
        "native_batch_norm",
        "native_norm",
        "norm_except_dim",
    }
)

_EXCLUDED_LINALG_EX_AND_LEGACY = frozenset(
    {
        # `linalg_*_ex` paired-output kernels return `(result, info)`
        # for per-batch error codes; the public `torch.linalg.*` Python
        # wrappers strip `info` and record the bare op, so these
        # variants never reach the trace.
        "linalg_cholesky_ex",
        "linalg_inv_ex",
        "linalg_ldl_factor_ex",
        "linalg_lu_factor_ex",
        # Deprecated pre-`torch.linalg` wrappers: `torch.eig`, `lstsq`,
        # `symeig`, `svd`, `pinverse` etc. live in `torch/_tensor.py` /
        # `torch/functional.py` as shims forwarding to `torch.linalg.*`.
        # JIT trace records the `linalg_*` callee, not the deprecation
        # shim.
        "eig",
        "lstsq",
        "matrix_rank",
        "pinv",
        "pinverse",
        "solve",
        "svd",
        "symeig",
    }
)

_EXCLUDED_FAKE_QUANT_TRAINING = frozenset(
    {
        # QAT `*_cachemask` outputs carry the mask tensor consumed only
        # by the backward pass; `fused_moving_avg_obs_fake_quant` and
        # `choose_qparams_optimized` are observer-fusion ops used to
        # *learn* qparams during training. Inference quantized graphs
        # call `quantize_per_*` / `dequantize` directly.
        "choose_qparams_optimized",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_channel_affine_cachemask",
        "fake_quantize_per_tensor_affine",
        "fake_quantize_per_tensor_affine_cachemask",
        "fused_moving_avg_obs_fake_quant",
    }
)

_EXCLUDED_SLOW_CONV_FALLBACK = frozenset(
    {
        # `slow_conv*` / `thnn_conv*` are dispatcher-fallback CPU
        # kernels (autograd-friendly, no fast backend). The JIT trace
        # always records the generic dispatched name (`convolution` /
        # `_convolution`); these names never appear in the recorded
        # graph regardless of which backend actually runs.
        "conv_depthwise3d",
        "slow_conv3d",
        "slow_conv3d_forward",
        "slow_conv_dilated2d",
        "slow_conv_dilated3d",
        "slow_conv_transpose2d",
        "slow_conv_transpose3d",
        "thnn_conv2d",
    }
)

_EXCLUDED_REGEX_SCRAPE_ARTIFACTS = frozenset(
    {
        # Phantom names produced by `get_aten_torch_from_code`'s regex
        # against pytorch source. The sweep captures everything matching
        # `aten::([a-zA-Z0-9_]*)`, so an f-string like
        # `f"aten::conv{dim}d("` in
        # `test/quantization/jit/test_quantize_jit.py` reports a bare
        # `conv` -- no such aten op exists. Actual ops are
        # `conv1d` / `conv2d` / `conv3d` / `_convolution` /
        # `_convolution_mode`.
        "conv",
    }
)

_EXCLUDED_PYTHON_SCALAR_BUILTINS_EXTRA = frozenset(
    {
        # More Python / TorchScript scalar builtins routed through
        # `aten::*` for scripting compatibility (see
        # `torch/csrc/jit/runtime/register_prim_ops.cpp` and
        # `ir_emitter.cpp` scaffolding). Unary scalar (`float`/`int`)
        # math + container-protocol method names that the `rg "aten::"`
        # sweep picks up but cannot produce tensor outputs in a graph.
        "append",
        "bin",
        "count",
        "cpu",
        "cuda",
        "degrees",
        "divmod",
        "dim",
        "equal",
        "fabs",
        "factorial",
        "insert",
        "is_floating_point",
        "item",
        "len",
        "list",
        "list_with_default",
        # `aten::modf(float a) -> (float, float)` is a scalar-only
        # TorchScript builtin (binds `math.modf`); there is no tensor
        # `aten::modf`. `torch.frac` covers the tensor case.
        "modf",
        "neq",
        "pop",
        "radians",
        "remove",
        "replace",
        "reverse",
        "str",
        "tensor",
    }
)

_EXCLUDED_INPLACE_STORAGE_MUTATORS = frozenset(
    {
        # Inplace storage / metadata mutators with no value semantics
        # in a functional graph -- JIT trace strips inplace ops via
        # `remove_inplace_ops_for_onnx.cpp`; named-tensor inplace
        # (`rename_`) cannot reach us because names are erased before
        # trace recording. `from_file` constructs a tensor by loading
        # disk data outside the graph entirely.
        "fill_diagonal_",
        "float_power_",
        "rename_",
        "resize",
        "resize_as_",
        "resize_as_sparse",
        "set",
        "sparse_resize_",
        "sparse_resize_and_clear_",
    }
)

_EXCLUDED_AUTOGRAD_TRAINING_INTERNALS = frozenset(
    {
        # Backward-only / dynamo-autograd internals:
        # * `sum_to` is emitted by `python_compiled_autograd.cpp` for
        #   broadcast-aware grad accumulation.
        # * `*_forward` (paired-output) are autograd forward helpers
        #   that return `(output, saved_tensor)`; inference uses the
        #   bare forward.
        # * `*_functional` are FX-style pure-output variants only used
        #   by AOT autograd.
        # * `embedding_renorm` fires only when
        #   `nn.Embedding.max_norm is not None` (training-only).
        # * `from_file` constructs a tensor from disk outside the graph.
        "embedding_renorm",
        "from_file",
        "glu_jvp",
        "log_sigmoid_forward",
        "multilabel_margin_loss_forward",
        "nll_loss_forward",
        "normal_functional",
        "rrelu_with_noise_functional",
        "rowwise_prune",
        "sum_to",
    }
)

#: Combined exclusion table keyed by group label -- preserved so the
#: support-page header can document the rationale to readers and we can
#: surface a quick summary in the warning block.
NEVER_IN_INFERENCE_TRACE = {
    "Python builtin / scripting false positives": (
        _EXCLUDED_PYTHON_BUILTIN_FALSE_POSITIVES
    ),
    "Test harness / placeholder identifiers": _EXCLUDED_TEST_FIXTURES,
    "Distributed / RPC primitives": _EXCLUDED_DISTRIBUTED_RPC,
    "Backend-specific dispatch shims (cudnn / miopen / mkldnn / mps)": (
        _EXCLUDED_BACKEND_SPECIFIC
    ),
    "Tensor metadata accessors (return a Python value, not a tensor)": (
        _EXCLUDED_TENSOR_METADATA
    ),
    "Named-tensor API (names stripped before JIT trace)": (
        _EXCLUDED_NAMED_TENSORS
    ),
    "Sparse-tensor machinery (NNEF / tract are dense-only)": (
        _EXCLUDED_SPARSE_ONLY
    ),
    "Functionalization `*_copy` / `*_scatter` variants": (
        _EXCLUDED_FUNCTIONALIZATION_COPY_SCATTER
    ),
    "Batch-norm training / backward-only internals": (
        _EXCLUDED_BATCH_NORM_TRAINING_INTERNALS
    ),
    "`linalg_*_ex` paired-output variants + deprecated linalg wrappers": (
        _EXCLUDED_LINALG_EX_AND_LEGACY
    ),
    "QAT `fake_quantize_*` training-only ops": (_EXCLUDED_FAKE_QUANT_TRAINING),
    "`slow_conv*` / `thnn_conv*` dispatcher-fallback kernels": (
        _EXCLUDED_SLOW_CONV_FALLBACK
    ),
    "Python / TorchScript scalar builtins (extra)": (
        _EXCLUDED_PYTHON_SCALAR_BUILTINS_EXTRA
    ),
    "Inplace storage / metadata mutators stripped by JIT": (
        _EXCLUDED_INPLACE_STORAGE_MUTATORS
    ),
    "Backward / dynamo-autograd internals": (
        _EXCLUDED_AUTOGRAD_TRAINING_INTERNALS
    ),
    "Regex-scrape artifacts (phantom names from f-strings)": (
        _EXCLUDED_REGEX_SCRAPE_ARTIFACTS
    ),
}

EXCLUDED_NEVER_IN_INFERENCE_TRACE = frozenset(
    name for group in NEVER_IN_INFERENCE_TRACE.values() for name in group
)

#: Last torch version that ships
#: ``onnx_torchscript_supported_aten_ops.html``. Starting at 2.9 the
#: TorchScript ONNX exporter was retired and that page 404s; fall back
#: to this one so the ONNX comparison column doesn't simply disappear.
LAST_TORCH_VERSION_WITH_ONNX_DOC = "2.8"


class FetchFromTorchVersion:
    def __init__(self, torch_version: str):
        self.torch_version = torch_version
        # Set by `get_core_ir`: the URL that actually yielded the core
        # IR list (fallback or not). Used by the markdown header so the
        # `is core` link points at the page that was scraped.
        self.resolved_ir_url: Optional[str] = None
        # Set by `get_onnx_support`: the URL that yielded the ONNX
        # support listing. `None` if no ONNX data could be fetched at
        # all (the section is then omitted).
        self.resolved_onnx_url: Optional[str] = None

    @property
    def url_ir(self) -> str:
        return f"https://docs.pytorch.org/docs/{self.torch_version}/torch.compiler_ir.html"

    @property
    def url_ir_fallback(self) -> str:
        return (
            "https://docs.pytorch.org/docs/"
            f"{LAST_TORCH_VERSION_WITH_IR_DOC}/torch.compiler_ir.html"
        )

    @property
    def onnx_support_url(self) -> str:
        return (
            f"https://docs.pytorch.org/docs/{self.torch_version}/"
            "onnx_torchscript_supported_aten_ops.html"
        )

    @property
    def onnx_support_url_fallback(self) -> str:
        return (
            "https://docs.pytorch.org/docs/"
            f"{LAST_TORCH_VERSION_WITH_ONNX_DOC}/"
            "onnx_torchscript_supported_aten_ops.html"
        )

    @staticmethod
    def _parse_ir_page(html: bytes) -> tuple[Set[str], List[str]]:
        soup = bs4.BeautifulSoup(html, "html.parser")
        res = soup.find_all("span", {"class": "pre"})
        official_aten_names = {
            r.text.split(".")[1]
            for r in res
            if r.text.startswith("aten")
            if "backward" not in r.text
        }
        official_prim_names = sorted(
            [r.text.split(".")[1] for r in res if r.text.startswith("prim")]
        )
        return official_aten_names, official_prim_names

    def get_core_ir(self) -> tuple[Set[str], List[str]]:
        resp = rq.get(self.url_ir, timeout=20)
        assert resp.status_code == 200
        official_aten_names, official_prim_names = self._parse_ir_page(
            resp.content
        )
        self.resolved_ir_url = self.url_ir
        if not official_aten_names:
            warnings.warn(
                f"{self.url_ir} no longer enumerates the core ATen IR "
                "(emptied in torch 2.10+); falling back to "
                f"{self.url_ir_fallback} for the 'is core' column.",
                stacklevel=2,
            )
            fallback = rq.get(self.url_ir_fallback, timeout=20)
            assert fallback.status_code == 200
            official_aten_names, official_prim_names = self._parse_ir_page(
                fallback.content
            )
            self.resolved_ir_url = self.url_ir_fallback
        return official_aten_names, official_prim_names

    @staticmethod
    def _parse_onnx_page(html: bytes) -> tuple[Set[str], Set[str]]:
        soup = bs4.BeautifulSoup(html, "html.parser")
        supported_ops = {
            _.text.replace("aten::", "")
            for _ in soup.find(id="id1").find_all("span", {"class": "pre"})
            if "aten::" in _.text
        }
        unsupported_ops = {
            _.text.replace("aten::", "")
            for _ in soup.find(id="id2").find_all("span", {"class": "pre"})
            if "aten::" in _.text
        }
        return supported_ops, unsupported_ops

    def get_onnx_support(self) -> tuple[Set[str], Set[str]]:
        """Fetch the TorchScript-ONNX per-op support page.

        PyTorch removed this page after torch 2.8 (TorchScript ONNX
        export was deprecated in favour of `torch.onnx.export(dynamo=
        True)`), and the new dynamo path doesn't ship a tabular
        per-op page. Fall back to the last torch version that still
        ships the page so the ONNX comparison column survives; the
        link in the section header points at whichever URL actually
        served the data.
        """
        resp = rq.get(self.onnx_support_url, timeout=20)
        if resp.status_code == 200:
            self.resolved_onnx_url = self.onnx_support_url
            return self._parse_onnx_page(resp.content)
        if resp.status_code == 404:
            warnings.warn(
                f"ONNX support page not found at {self.onnx_support_url} "
                "(removed in torch 2.9+); falling back to "
                f"{self.onnx_support_url_fallback} for the ONNX column.",
                stacklevel=2,
            )
            fallback = rq.get(self.onnx_support_url_fallback, timeout=20)
            if fallback.status_code == 200:
                self.resolved_onnx_url = self.onnx_support_url_fallback
                return self._parse_onnx_page(fallback.content)
        # Both the requested version and the fallback failed; let the
        # caller drop the ONNX section.
        self.resolved_onnx_url = None
        return set(), set()

    def get_aten_torch_from_code(self) -> List[str]:
        aten_torch_from_code = sorted(
            subprocess.check_output(
                "cd /tmp ; "
                "git clone -q git@github.com:pytorch/pytorch.git || "
                "git -C 'pytorch' pull; "
                "cd /tmp/pytorch ;"
                f"git checkout v{self.torch_version}.0; "
                'rg "aten::" | '
                'sed "s|.*aten::\\([a-zA-Z0-9_]*\\).*|\\1|g"|sort|uniq',
                shell=True,
            )
            .decode("utf8")
            .split("\n")
        )
        names = [_ for _ in aten_torch_from_code if not _.startswith("_")]
        # `check_output` only checks the exit code of the last command in
        # the pipeline, so a failed clone/checkout upstream of `rg` yields
        # an empty (or tiny) list rather than an error, and the page then
        # renders with no operator rows at all. Every torch release has
        # hundreds of `aten::` names, so anything this small means the
        # checkout is broken, not that PyTorch shrank.
        if len(names) < _MIN_PLAUSIBLE_ATEN_NAMES:
            raise RuntimeError(
                f"only {len(names)} aten names extracted from "
                f"/tmp/pytorch at v{self.torch_version}.0, expected at "
                f"least {_MIN_PLAUSIBLE_ATEN_NAMES}. The checkout is "
                "probably in a bad state; remove /tmp/pytorch and retry."
            )
        return names

    def get_aliases_from_code(self) -> AliasManager:
        aliases = sorted(
            subprocess.check_output(
                "cd /tmp ; "
                "git -C 'pytorch' pull || "
                "git clone -q git@github.com:pytorch/pytorch.git; "
                "cd /tmp/pytorch ;"
                f"git checkout v{self.torch_version}.0; "
                "cat ./torch/csrc/jit/passes/normalize_ops.cpp",
                shell=True,
            )
            .decode("utf8")
            .split("\n")
        )
        return AliasManager(
            {
                tuple(
                    x.replace("aten::", "") for x in a.strip()[1:-2].split(", ")
                )
                for a in aliases
                if "{" in a and "}" in a and "aten::" in a
            }
        )

    def get_cache_url(
        self,
        aten_torch_from_code: List[str],
    ) -> LinkToTorchDocCache:
        cache_path = (
            Path(__file__).parent / f"torch_{self.torch_version}_doc_urls.json"
        )
        cache_url = LinkToTorchDocCache(cache_path)
        # Probed namespaces, in priority order. `LinkToTorchDocCache.add`
        # stops at the first hit, so put the most specific / canonical
        # ones first; `torch.Tensor.{}` catches tensor-method ops that
        # don't have a free-function form (e.g. `to_dense`, `index_put`,
        # `masked_scatter`).
        url_tails = (
            "torch.nn.functional.{}.html",
            "torch.{}.html",
            "torch.Tensor.{}.html",
            "torch.linalg.{}.html",
        )
        for a_from_code in rich.progress.track(
            aten_torch_from_code,
            total=len(aten_torch_from_code),
            description=f"Caching torch doc links in '{cache_path.name}'",
        ):
            for tail in url_tails:
                cache_url.add(
                    f"https://docs.pytorch.org/docs/{self.torch_version}"
                    f"/generated/{tail}",
                    a_from_code,
                )
        return cache_url


def print_t(text, file):
    """Print tabbed."""
    if text:
        if "\n" in text:
            lines = text.split("\n")
            new_lines = []
            for line in lines:
                new_line = f"    {line}" if line.strip() else line
                new_lines.append(new_line)
            text = "\n".join(new_lines)
        else:
            text = f"    {text}"
        print(text, file=file)
    else:
        print("", file=file)


def _md_link(text: str, href: Optional[str]) -> str:
    """Inline-anchor or plain text fallback."""
    if not href:
        return text
    return f'<a href="{href}">{text}</a>'


def _format_aliases(aliases: List[str]) -> str:
    return ", ".join(aliases)


def _write_measured_summary(
    support_target_msg: str,
    measured: MeasuredOnnxSupport,
    aten_torch_from_code: List[str],
    full_qte: int,
    full_core: int,
    measured_core: int,
    qte_core: int,
    untested_documented: int,
    untested_documented_core: int,
    fh,
):
    """Headline stats for a measured section.

    Two deliberate choices make this bar comparable to the binary tab's.

    Same **denominators** (core opset size, full `aten::` listing): a bar
    over a measured-only denominator answers a different question ("of
    what we tested, how much passed") and silently rescales against the
    tab next to it.

    Same reading of the **numerator**, which here means crediting the
    retired listing's claims (`✅*`) alongside our own measurements. Those
    operators are unverified, but the reason they are unverified is our
    missing spec coverage, and scoring that as an ONNX gap would understate
    a competing exporter for our own shortfall. The caption keeps the two
    populations separate so the measured-only ratio stays one line away.
    """
    counts = measured.counts()
    total_rows = len(aten_torch_from_code)
    measured_rows = sum(counts.values())
    no_data = total_rows - measured_rows - untested_documented
    entry = measured.newest_measurement
    env = ", ".join(
        f"{label} {entry[key]}"
        for key, label in (
            ("torch", "torch"),
            ("onnxruntime", "onnxruntime"),
            ("opset", "opset"),
        )
        if entry.get(key) is not None
    )
    print_t(
        f"{support_target_msg}\n\n"
        "Total operators exportable, over the same denominators as the "
        "`TractNNEF` tab:\n\n"
        + headline_bars(
            f"{full_core + untested_documented_core}/{qte_core}",
            f"{full_qte + untested_documented}/{total_rows}",
        )
        + " (**both bars credit unverified claims**: "
        f"{full_core} core / {full_qte} overall are measured fully "
        f"exportable here, plus {untested_documented_core} core / "
        f"{untested_documented} overall that no spec of ours covers and "
        "that are counted on the retired listing's word alone (✅*). "
        "Leaving those out would report our own missing coverage as an "
        "ONNX gap."
        f" Of the {measured_rows} operators actually measured: "
        f"{counts.get(GRADE_FULL, 0)} full, "
        f"{counts.get(GRADE_PARTIAL, 0)} partial, "
        f"{counts.get(GRADE_NONE, 0)} none, "
        f"{counts.get(GRADE_BLOCKED, 0)} blocked before ONNX, so "
        f"{full_core}/{measured_core} of the measured core operators and "
        f"{counts.get(GRADE_FULL, 0)}/{measured_rows} of all measured ones "
        "export fully."
        f" The {no_data} rows with neither a measurement nor a claim (`-`) "
        "stay out of the numerator."
        f" Measured with {env}.)",
        file=fh,
    )


#: Binary-section filter values that select a grade under another name.
_MODE_ALIASES = {"supported": GRADE_FULL, "unsupported": GRADE_NONE}


def counted_modes(modes, display_counts, total: int, cross_gap_qte: int):
    """Label each filter with its row count, and drop the empty ones.

    A radio that selects nothing is worse than absent: it invites a click
    that empties the table and says nothing about why. `blocked` is 0
    whenever `torch.export` captured every module, which is the normal
    case, so the mode list has to follow the data rather than be fixed.

    The counts double as the section's summary: they say how the rows
    split without the reader clicking through every mode.
    """
    counted = []
    for value, label in modes:
        if value == "all":
            count = total
        elif value == CROSS_GAP_MODE:
            count = cross_gap_qte
        else:
            count = display_counts.get(_MODE_ALIASES.get(value, value), 0)
        if count:
            counted.append((value, f"{label} ({count})"))
    return tuple(counted)


def _filter_widget(filter_id: str, modes) -> str:
    """Radio buttons for one section's row filter."""
    labels = "".join(
        f'<label><input type="radio" name="{filter_id}" '
        f'value="{value}"{" checked" if value == "all" else ""}> '
        f"{label}</label>\n"
        for value, label in modes
    )
    return (
        '<div class="op-filter-container" markdown="0">\n'
        '<form class="op-filter-form">\n' + labels + "</form>\n"
    )


def write_operator_support(
    support_target_name: str,
    support_target_msg: str,
    aten_torch_from_code: List[str],
    supported_opset: Set[str],
    alias_manager: AliasManager,
    official_aten_names: Set[str],
    fh,
    cache_url: LinkToTorchDocCache,
    support_inplace: Set[str],
    support_n_ops_label: str,
    measured: Optional[MeasuredOnnxSupport] = None,
    include_documented: bool = True,
    legend: Optional[str] = None,
    cross_support: Optional[Set[str]] = None,
    cross_gap_label: str = "",
):
    """Emit one tabbed section.

    The table is raw HTML (not a markdown pipe table) so each `<tr>`
    can carry a `supported`/`unsupported` class hooked up by the inline
    filter widget at the top of the section.

    With `measured`, the first column becomes a four-way export grade from
    the proptest sweep instead of a binary "is there an entry for this
    name", two informational columns are added (does onnxruntime run the
    exported graph, do its outputs match torch), and `supported_opset`
    demotes to a `documented` column so operators no spec covers still
    carry what the retired PyTorch doc page said about them.

    With `cross_support` (the set of names the other section credits),
    rows also carry a marker class and the filter gains a "supported
    there, missing here" mode.
    """
    row_items: List[tuple] = []
    qte_core = 0
    qte_supported_core = 0
    matched_qte = 0
    measured_core = 0
    untested_documented = 0
    untested_documented_core = 0
    cross_gap_qte = 0

    print(f'=== "{support_target_name}"', file=fh)
    print("", file=fh)
    for a_from_code in rich.progress.track(
        aten_torch_from_code,
        total=len(aten_torch_from_code),
        description="Generating support table",
    ):
        if alias_manager.is_alias(a_from_code):
            continue
        is_core = a_from_code in official_aten_names
        is_core_official_str = "✅" if is_core else "-"

        documented = a_from_code in supported_opset
        if measured is None:
            grade = GRADE_FULL if documented else GRADE_NONE
        else:
            grade = measured.grade(a_from_code)
        exist_in_support = grade == GRADE_FULL
        # Display state may refine the grade; the measured breakdown always
        # uses `grade`, so a starred row never inflates it. The headline
        # bars do count starred rows, from these separate tallies.
        display = grade
        if grade == GRADE_UNTESTED and documented:
            display = DISPLAY_UNTESTED_DOCUMENTED
            untested_documented += 1
            if is_core:
                untested_documented_core += 1

        if is_core:
            qte_core += 1
            if exist_in_support:
                qte_supported_core += 1
            if measured is not None and grade != GRADE_UNTESTED:
                measured_core += 1

        mapped_in_support_str = GRADE_GLYPH[display]
        if exist_in_support:
            matched_qte += 1

        cross_ok = cross_support is not None and a_from_code in cross_support
        if cross_ok and not exist_in_support:
            cross_gap_qte += 1

        inplace_str = "✅" if a_from_code in support_inplace else "❌"
        alias_str = _format_aliases(alias_manager.get_aliases(a_from_code))
        torch_url_doc = cache_url.get_url(a_from_code)
        op_name_html = _md_link(a_from_code, torch_url_doc)
        extra_cells: List[str] = []
        if measured is not None:
            extra_cells = [
                measured.axis(a_from_code, "runtime"),
                measured.axis(a_from_code, "numerics"),
            ]
            if include_documented:
                extra_cells.append("✅" if documented else "-")
        row_items.append(
            (
                exist_in_support,
                is_core,
                op_name_html,
                alias_str,
                inplace_str,
                is_core_official_str,
                mapped_in_support_str,
                display,
                extra_cells,
                cross_ok,
            )
        )

    # Core ops first to keep the historical sort, then unsupported core.
    row_items.sort(key=lambda r: -int(r[1]))

    print_t("", file=fh)
    if measured is None:
        support_n_ops = len([_ for _ in supported_opset if not _.endswith("_")])
        ratio_total_str = f"{matched_qte}/{len(aten_torch_from_code)}"
        print_t(
            f"Total matched operators in {support_target_msg} compared to:"
            "\n\n"
            + headline_bars(f"{qte_supported_core}/{qte_core}", ratio_total_str)
            + f" (total operators listed as supported by {support_n_ops_label} "
            f"being {support_n_ops})",
            file=fh,
        )
    else:
        _write_measured_summary(
            support_target_msg,
            measured,
            aten_torch_from_code,
            matched_qte,
            qte_supported_core,
            measured_core,
            qte_core,
            untested_documented,
            untested_documented_core,
            fh,
        )
    print_t("", file=fh)
    if legend:
        print_t(legend, file=fh)
        print_t("", file=fh)

    # Filter widget + raw HTML table. The filter scope is a single
    # `.op-filter-container`, so multiple sections (TractNNEF, ONNX) on
    # the same page each get their own independent toggle state.
    filter_id = f"op-filter-{support_target_name}"
    if measured is None:
        header_cells = (
            "<th>export&amp;run</th><th>aten name</th><th>aliases</th>"
            "<th>can in-place</th><th>is core</th>"
        )
        modes = BINARY_FILTER_MODES
    else:
        documented_header = "<th>documented</th>" if include_documented else ""
        header_cells = (
            "<th>export</th><th>aten name</th><th>aliases</th>"
            "<th>runtime</th><th>numerics</th>"
            f"{documented_header}"
            "<th>can in-place</th><th>is core</th>"
        )
        modes = GRADED_FILTER_MODES
    if cross_support is not None:
        modes = modes + ((CROSS_GAP_MODE, cross_gap_label),)
    # Trailing unpack rather than a fixed index: the row tuple grows, and
    # a stale offset here would silently count the wrong field.
    display_counts = Counter(
        display for *_, display, _cells, _cross_ok in row_items
    )
    modes = counted_modes(modes, display_counts, len(row_items), cross_gap_qte)
    print_t(
        _filter_widget(filter_id, modes)
        + '<table class="op-table">\n'
        + "<thead><tr>"
        + header_cells
        + "</tr></thead>\n"
        + "<tbody>",
        file=fh,
    )
    for (
        exist_in_support,
        _is_core,
        op_name_html,
        alias_str,
        inplace_str,
        is_core_official_str,
        mapped_in_support_str,
        display,
        extra_cells,
        cross_ok,
    ) in row_items:
        # Binary sections keep the historical two-state class so their
        # rendered output is unchanged. Graded sections carry only
        # `grade-<name>`: calling an `untested` row `unsupported` in the
        # DOM would assert exactly what the grade refuses to assert.
        if measured is None:
            klass = "supported" if exist_in_support else "unsupported"
        else:
            klass = f"grade-{display}"
        if cross_ok:
            klass += f" {CROSS_OK_CLASS}"
        middle = f"<td>{alias_str}</td>" + "".join(
            f"<td>{cell}</td>" for cell in extra_cells
        )
        print_t(
            f'<tr class="op-row {klass}">'
            f"<td>{mapped_in_support_str}</td>"
            f"<td>{op_name_html}</td>"
            f"{middle}"
            f"<td>{inplace_str}</td>"
            f"<td>{is_core_official_str}</td>"
            "</tr>",
            file=fh,
        )
    print_t("</tbody>\n</table>\n</div>", file=fh)
    print_t("", file=fh)


FILTER_SCRIPT = """\
<script>
(function () {
  function applyFilter(form) {
    var mode = form.querySelector('input[type="radio"]:checked').value;
    var rows = form.parentElement.querySelectorAll('tr.op-row');
    rows.forEach(function (tr) {
      var sup = tr.classList.contains('supported');
      // Graded sections carry `grade-<name>`; binary ones do not, so they
      // fall back to the two-state reading. Without this, a `partial` row
      // would show up under "Unsupported only" purely for lacking the
      // `supported` class.
      // `[a-z-]` so the hyphenated `grade-untested-documented` state
      // matches in full; `[a-z]+` would truncate it to `untested` and
      // make the two untested filters indistinguishable.
      // `grade-` prefix required: the cross-section marker `cross-ok` is a
      // separate class and must not be read as a grade.
      var graded = tr.className.match(/grade-([a-z-]+)/);
      var grade = graded ? graded[1] : (sup ? 'full' : 'none');
      var crossOk = tr.classList.contains('cross-ok');
      var keep =
        mode === 'all' ||
        mode === grade ||
        (mode === 'supported' && grade === 'full') ||
        (mode === 'unsupported' && grade === 'none') ||
        // Supported by the other tab, missing here: the shortlist.
        (mode === 'cross-gap' && crossOk && grade !== 'full');
      tr.style.display = keep ? '' : 'none';
    });
  }
  document.querySelectorAll('.op-filter-form').forEach(function (form) {
    form.addEventListener('change', function () { applyFilter(form); });
  });
})();
</script>
"""


def _measured_onnx_section_msg(measured: MeasuredOnnxSupport) -> str:
    """Intro paragraph for the measured ONNX section.

    Only how the numbers were produced. What each glyph means is a legend,
    not an intro, so it goes in its own admonition next to the table it
    describes (`_measured_onnx_legend`) rather than as a wall of prose
    between the reader and the bars.
    """
    entry = measured.newest_measurement
    profile = entry.get("profile", "unknown")
    examples = entry.get("examples", "?")
    return (
        "`ONNX` support **measured** by exporting real modules with "
        "`torch.onnx.export(dynamo=True)`, one per operator, over "
        f"{examples} generated examples each (hypothesis `{profile}` "
        "profile). Produced by `tox -e proptest_onnx`; see "
        "[how to refresh it](./onnx_support_page.md)."
    )


def _measured_onnx_legend(onnx_url: str, has_documented: bool) -> str:
    """Column legend for the measured ONNX table, as its own box.

    Body lines carry 4 spaces of their own: `print_t` adds the 4 the
    tabbed block needs, and the admonition body must sit 4 further in.
    """
    rows = [
        ("✅", "every generated example exported"),
        ("🟡", "some examples exported, others raised"),
        ("❌", "no example exported"),
        (
            "⚠️",
            "`torch.export` could not capture the module, so the ONNX "
            "exporter never ran. **Not** an ONNX verdict",
        ),
        (
            "✅*",
            "no spec covers it: **not** verified here, but the retired "
            "listing claimed it was supported, and the bars above take "
            "that claim at face value",
        ),
        ("`-`", "no spec covers it and nothing was ever claimed either way"),
    ]
    table = "\n".join(f"    | {glyph} | {meaning} |" for glyph, meaning in rows)
    msg = (
        '!!! info "How to read this table"\n\n'
        "    The `export` column grades **operator coverage only**:\n\n"
        "    | | |\n"
        "    | --- | --- |\n"
        f"{table}\n\n"
        "    Only ✅ / 🟡 / ❌ are measurements. ✅* is an unverified "
        "historical claim: the headline bars count it, so that operators "
        "we never wrote a spec for are not scored against ONNX, but the "
        "measured breakdown in the caption excludes it.\n\n"
        "    `runtime` (does onnxruntime load and run the exported graph) "
        "and `numerics` (do its outputs match PyTorch) are separate "
        "columns on purpose: a graph that exports but diverges "
        "numerically is usually a property of the kernel that ran, not a "
        "missing operator."
    )
    if has_documented:
        msg += (
            "\n\n    The `documented` column is what the retired "
            f"TorchScript-exporter listing ([this page]({onnx_url})) said, "
            "kept for the operators no spec covers yet."
        )
    return msg


def build_markdown_header(
    fetcher, measured: Optional[MeasuredOnnxSupport] = None
) -> str:
    date = datetime.now().strftime("%d %b %Y")
    ir_url = fetcher.resolved_ir_url or fetcher.url_ir
    ir_note = (
        f"[PyTorch IR documentation page]({ir_url})"
        if ir_url == fetcher.url_ir
        else (
            f"[PyTorch IR documentation page]({ir_url}) "
            f"(the page for torch {fetcher.torch_version} was emptied "
            f"upstream; falling back to "
            f"torch {LAST_TORCH_VERSION_WITH_IR_DOC} which is the last "
            "version that still enumerates the core ATen IR)"
        )
    )
    measured_note = ""
    if measured is not None:
        entry = measured.newest_measurement
        measured_note = (
            " The `ONNX` section is **not** scraped from PyTorch's docs: "
            "it is measured by actually exporting one module per operator "
            f"with torch {entry.get('torch', '?')} at opset "
            f"{entry.get('opset', '?')} (see that section for how to read "
            "its columns)."
        )
    return (
        "!!! note\n"
        "    This table and page are auto generated by "
        "`docs/contributing/generate_support_page.py` and reflect the "
        "PyTorch reference docs at the time of generation."
        f" Targeted torch version: **{fetcher.torch_version}**. "
        f"Generated on **{date}**."
        f"{measured_note}\n\n"
        "!!! warning\n"
        "     Take these results with a grain of salt: many of the listed "
        "operators never appear in the torch IR graph that "
        "`torch_to_nnef` traces (they get remapped to more generic ops "
        "upstream), and some uncommon operators are rare in real models "
        "so support may be lacking even when marked unsupported. "
        "**SONOS maintains operators on a per-need basis**, "
        "and contributions are always welcome "
        "[see how](./add_new_aten_op.md)."
        "\n\n"
        f"\n 'is core' column refers to this {ir_note}.\n\n"
        "We filter out 'backward' and 'sym' operators from the listing "
        "since they are unwanted in an inference engine. "
        "In-place operations are merged with their non-inplace "
        "counterparts since that distinction is an inference "
        "implementation detail."
        "\n\n"
        "We also exclude a long tail of identifiers that the `aten::*` "
        "source-grep picks up but that can never surface in an inference "
        "JIT trace ([see the full list at the bottom of this "
        "page](#excluded-aten-names)). This trims the unsupported column "
        "to the names where a `torch_to_nnef` emitter (or a deliberate "
        "no-op map) would actually be meaningful."
    )


def build_excluded_appendix() -> str:
    """Trailing collapsible appendix listing every excluded aten name.

    Renders as a mkdocs-material collapsible admonition (`???`) so the
    100+ identifiers don't drown the page; readers who want to audit
    the exclusion sets can click to expand.
    """
    lines = [
        '<h2 id="excluded-aten-names" style="margin-top:2rem;">',
        "Appendix: identifiers excluded from this page",
        "</h2>",
        "",
        "These names are filtered out of the support tables above because "
        "they cannot surface in an inference JIT trace. Each group has a "
        "documented rationale; click any to expand the full member list.",
        "",
    ]
    for label, group in NEVER_IN_INFERENCE_TRACE.items():
        lines.append(f'??? note "{label} ({len(group)} names)"')
        members = ", ".join(f"`{n}`" for n in sorted(group))
        lines.append(f"    {members}")
        lines.append("")
    return "\n".join(lines)


def _load_measured_onnx(
    onnx_report_path: Optional[Path],
    alias_manager: AliasManager,
    page_rows: Set[str],
) -> Optional[MeasuredOnnxSupport]:
    """Load the graded ONNX artifact, if one was supplied."""
    if onnx_report_path is None:
        return None
    if not onnx_report_path.exists():
        raise FileNotFoundError(
            f"--onnx-report {onnx_report_path} does not exist. Generate it "
            "with `tox -e proptest_onnx`."
        )
    with onnx_report_path.open("r", encoding="utf8") as fh:
        payload = json.load(fh)
    measured = MeasuredOnnxSupport(payload, alias_manager, page_rows)
    if measured.unmapped:
        warnings.warn(
            "measured ONNX operators with no row on this page (the "
            "`aten::*` source grep drops `_`-prefixed names): "
            f"{sorted(measured.unmapped)}",
            stacklevel=2,
        )
    return measured


def onnx_credited_names(
    aten_rows: List[str],
    measured: Optional[MeasuredOnnxSupport],
    onnx_supported: Set[str],
) -> Set[str]:
    """Rows the ONNX section counts as supported.

    Same rule as its headline bars: measured `full`, or claimed by the
    retired listing and never contradicted by a measurement of ours. A
    measured `partial`/`none`/`blocked` is evidence and overrides the
    claim, so those never make the set.
    """
    credited = set()
    for name in aten_rows:
        if measured is None:
            if name in onnx_supported:
                credited.add(name)
            continue
        grade = measured.grade(name)
        if grade == GRADE_FULL or (
            grade == GRADE_UNTESTED and name in onnx_supported
        ):
            credited.add(name)
    return credited


def build_markdown_page(
    torch_version: str, onnx_report_path: Optional[Path] = None
):
    """Build supported operators markdown page."""
    fetcher = FetchFromTorchVersion(torch_version)
    official_aten_names, official_prim_names = fetcher.get_core_ir()
    t2n_aten = set(list(aten_ops_registry._registry.keys()))
    onnx_supported, onnx_unsupported = fetcher.get_onnx_support()
    aten_torch_from_code = fetcher.get_aten_torch_from_code()

    aliases_manager = fetcher.get_aliases_from_code()

    support_inplace = set()
    offset = 0
    for ix, a in enumerate(aten_torch_from_code[:]):
        if (  # pylint: disable-next=too-many-boolean-expressions
            a.endswith("_")
            and a[:-1] in aten_torch_from_code
            or aliases_manager.is_alias(a)
            or a.strip() == ""
            or (len(a) and a[0].isupper())
            or "backward" in a
            or a.startswith("sym_")
            or a in EXCLUDED_NEVER_IN_INFERENCE_TRACE
        ):
            del aten_torch_from_code[ix - offset]
            offset += 1
            support_inplace.add(a[:-1])

    measured = _load_measured_onnx(
        onnx_report_path, aliases_manager, set(aten_torch_from_code)
    )

    cache_url = fetcher.get_cache_url(aten_torch_from_code)
    with (Path(__file__).parent / "./supported_operators.md").open(
        "w", encoding="utf8"
    ) as fh:
        print(
            build_markdown_header(fetcher, measured),
            file=fh,
        )
        write_operator_support(
            "TractNNEF",
            "`torch_to_nnef`",
            aten_torch_from_code,
            t2n_aten,
            official_aten_names=official_aten_names,
            alias_manager=aliases_manager,
            fh=fh,
            cache_url=cache_url,
            support_inplace=support_inplace,
            support_n_ops_label="`torch_to_nnef`",
            cross_support=onnx_credited_names(
                aten_torch_from_code, measured, onnx_supported
            ),
            cross_gap_label="Missing here, credited by ONNX",
        )
        if measured is not None or onnx_supported or onnx_unsupported:
            onnx_url = fetcher.resolved_onnx_url or fetcher.onnx_support_url
            onnx_legend = None
            if measured is not None:
                onnx_section_msg = _measured_onnx_section_msg(measured)
                onnx_legend = _measured_onnx_legend(
                    onnx_url, bool(onnx_supported)
                )
            else:
                onnx_section_msg = (
                    f"builtin PyTorch `ONNX` support based on "
                    f"[this page]({onnx_url})"
                )
                if onnx_url != fetcher.onnx_support_url:
                    onnx_section_msg += (
                        f" (the page for torch {fetcher.torch_version} was "
                        f"removed upstream in torch 2.9+; falling back to "
                        f"torch {LAST_TORCH_VERSION_WITH_ONNX_DOC} which is "
                        "the last version that still ships the TorchScript "
                        "ONNX support listing)"
                    )
            write_operator_support(
                "ONNX",
                onnx_section_msg,
                aten_torch_from_code,
                onnx_supported,
                official_aten_names=official_aten_names,
                alias_manager=aliases_manager,
                fh=fh,
                cache_url=cache_url,
                support_inplace=support_inplace,
                support_n_ops_label="PyTorch's TorchScript ONNX exporter",
                measured=measured,
                include_documented=bool(onnx_supported),
                legend=onnx_legend,
            )
        print(FILTER_SCRIPT, file=fh)
        print(build_excluded_appendix(), file=fh)
    cache_url.save()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate supported operators markdown page."
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default=None,
        help=(
            "Target PyTorch version (X.Y) to generate the report for. "
            "Defaults to the installed torch, which is also what the ONNX "
            "sweep measured, so the two cannot silently disagree."
        ),
    )
    parser.add_argument(
        "--onnx-report",
        type=Path,
        default=None,
        help=(
            "graded ONNX support artifact from `tox -e proptest_onnx`. "
            "Given, the ONNX section reports measured export grades; "
            "omitted, it falls back to scraping PyTorch's retired "
            "TorchScript-exporter listing."
        ),
    )
    return parser.parse_args()


def installed_torch_version() -> str:
    """`X.Y` of the installed torch, for use as the default target.

    The ONNX sweep measures whichever torch is installed, so defaulting to
    it keeps the page's operator list and its measured column describing
    the same release instead of leaving that to a hand-typed flag.
    """
    import torch  # noqa: PLC0415

    major, minor = torch.__version__.split(".")[:2]
    return f"{major}.{minor}"


if __name__ == "__main__":
    args = parse_args()
    torch_version = args.torch_version or installed_torch_version()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        assert len(re.findall(r"\.", torch_version)) == 1, (
            "expect X.Y format for torch version"
        )
    assert torch_version.replace(".", "").isdigit(), (
        "expect X.Y format for torch version"
    )
    build_markdown_page(
        torch_version=torch_version,
        onnx_report_path=args.onnx_report,
    )
