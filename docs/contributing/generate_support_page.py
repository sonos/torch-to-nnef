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
from collections import defaultdict
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

from torch_to_nnef.op.aten import aten_ops_registry  # noqa: E402


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
        return [_ for _ in aten_torch_from_code if not _.startswith("_")]

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
):
    """Emit one tabbed section.

    The table is raw HTML (not a markdown pipe table) so each `<tr>`
    can carry a `supported`/`unsupported` class hooked up by the inline
    filter widget at the top of the section.
    """
    row_items: List[tuple] = []
    qte_core = 0
    qte_supported_core = 0
    matched_qte = 0

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

        exist_in_support = a_from_code in supported_opset

        if is_core:
            qte_core += 1
            if exist_in_support:
                qte_supported_core += 1

        mapped_in_support_str = "✅" if exist_in_support else "❌"
        if exist_in_support:
            matched_qte += 1

        inplace_str = "✅" if a_from_code in support_inplace else "❌"
        alias_str = _format_aliases(alias_manager.get_aliases(a_from_code))
        torch_url_doc = cache_url.get_url(a_from_code)
        op_name_html = _md_link(a_from_code, torch_url_doc)
        row_items.append(
            (
                exist_in_support,
                is_core,
                op_name_html,
                alias_str,
                inplace_str,
                is_core_official_str,
                mapped_in_support_str,
            )
        )

    # Core ops first to keep the historical sort, then unsupported core.
    row_items.sort(key=lambda r: -int(r[1]))

    print_t("", file=fh)
    support_n_ops = len([_ for _ in supported_opset if not _.endswith("_")])
    ratio_total_str = f"{matched_qte}/{len(aten_torch_from_code)}"
    print_t(
        f"Total matched operators in {support_target_msg} compared to:\n\n"
        f"- core PyTorch opset:\n\n"
        f"[={qte_supported_core}/{qte_core} "
        f'"{qte_supported_core}/{qte_core}"]\n\n'
        "-  and support from full `aten::`: \n\n"
        f'[={ratio_total_str} "{ratio_total_str}"]\n\n'
        f" (total operators listed as supported by {support_n_ops_label} "
        f"being {support_n_ops})",
        file=fh,
    )
    print_t("", file=fh)

    # Filter widget + raw HTML table. The filter scope is a single
    # `.op-filter-container`, so multiple sections (TractNNEF, ONNX) on
    # the same page each get their own independent toggle state.
    filter_id = f"op-filter-{support_target_name}"
    print_t(
        '<div class="op-filter-container" markdown="0">\n'
        '<form class="op-filter-form">\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="all" checked> All</label>\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="supported"> Supported only</label>\n'
        '<label><input type="radio" name="' + filter_id + '" '
        'value="unsupported"> Unsupported only</label>\n'
        "</form>\n"
        '<table class="op-table">\n'
        "<thead><tr>"
        "<th>translated</th><th>aten name</th><th>aliases</th>"
        "<th>can in-place</th><th>is core</th>"
        "</tr></thead>\n"
        "<tbody>",
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
    ) in row_items:
        klass = "supported" if exist_in_support else "unsupported"
        print_t(
            f'<tr class="op-row {klass}">'
            f"<td>{mapped_in_support_str}</td>"
            f"<td>{op_name_html}</td>"
            f"<td>{alias_str}</td>"
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
      var keep =
        mode === 'all' ||
        (mode === 'supported' && sup) ||
        (mode === 'unsupported' && !sup);
      tr.style.display = keep ? '' : 'none';
    });
  }
  document.querySelectorAll('.op-filter-form').forEach(function (form) {
    form.addEventListener('change', function () { applyFilter(form); });
  });
})();
</script>
"""


def build_markdown_header(fetcher) -> str:
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
    return (
        "!!! note\n"
        "    This table and page are auto generated by "
        "`docs/contributing/generate_support_page.py` and reflect the "
        "PyTorch reference docs at the time of generation."
        f" Targeted torch version: **{fetcher.torch_version}**. "
        f"Generated on **{date}**.\n\n"
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


def build_markdown_page(torch_version: str):
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

    cache_url = fetcher.get_cache_url(aten_torch_from_code)
    with (Path(__file__).parent / "./supported_operators.md").open(
        "w", encoding="utf8"
    ) as fh:
        print(
            build_markdown_header(fetcher),
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
        )
        if onnx_supported or onnx_unsupported:
            onnx_url = fetcher.resolved_onnx_url or fetcher.onnx_support_url
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
        required=True,
        help="Target PyTorch version to generate the report for.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        assert len(re.findall("\.", args.torch_version)) == 1, (
            "expect X.Y format for torch version"
        )
    assert args.torch_version.replace(".", "").isdigit(), (
        "expect X.Y format for torch version"
    )
    build_markdown_page(torch_version=args.torch_version)
