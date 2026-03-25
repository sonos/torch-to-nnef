import logging
import typing as T

import torch

from torch_to_nnef._optional_types import (
    InjectedNemoModule,
    InjectedTorchaudioModule,
)
from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.nemo_tract.axes import collapse_dynamic_axes_mapping  # legacy import (unused)
from torch_to_nnef.nemo_tract.constants import (
    DEFAULT_TIME,
    INPUT_STATE_TUPLE_NAME,
    LENGTH_INPUT_NAMES,
    LENGTH_OUTPUT_NAMES,
    OUT_STATE_NAME,
    STATE_INPUT_NAMES,
    is_length_name,
)
from torch_to_nnef.nemo_tract.dynaxes import (
    filter_dynamic_axes_by_ranks,
    symbols_from_input_types,
)
from torch_to_nnef.nemo_tract.utils import map_args_to_kwargs_by_names
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


def decoder_fix_input_example_batch_size(
    input_example: T.List[torch.Tensor],
    batch_size: int,
) -> T.List[torch.Tensor]:
    """Fix the batch size of the input example for decoder models."""

    def expand_dim(x, dim, size):
        shape = list(x.shape)
        shape[dim] = size
        return x.expand(*shape)

    assert batch_size > 0, "Batch size must be positive."
    input_ids, *encoder_outputs = input_example
    if batch_size != 1 and input_ids.size(0) == 1:
        input_ids = expand_dim(input_ids, 0, batch_size)
    return [input_ids] + list(encoder_outputs)


class WrapAudioPreprocessor(torch.nn.Module):
    """Wraps the AudioPreprocessor to fix input_example empty."""

    def __init__(self, preprocessor: torch.nn.Module):
        super().__init__()
        self.preprocessor = preprocessor

    @require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="torchaudio")
    def input_example(
        self,
        max_batch: int = 2,
        *,
        torchaudio: InjectedTorchaudioModule = INJECTED,
    ):
        results = self.preprocessor.input_example(max_batch=max_batch)
        if results is not None:
            LOGGER.warning(
                "AudioPreprocessor input_example is not empty; using it as-is. "
                "If you encounter issues with dynamic axes during export,"
                " consider overriding the input_example method "
                "to return a stable example."
            )
            return results
        # Build a stable example from input_types, irrespective of the
        # underlying module's own (possibly varying) input_example.
        input_types = self.preprocessor.input_types
        if self.preprocessor.featurizer is None:
            LOGGER.warning(
                "AudioPreprocessor has no featurizer."
                "This is unknown behavior for T2N maintainer."
            )
        if not isinstance(
            self.preprocessor.featurizer,
            (torchaudio.transforms.MelSpectrogram, torchaudio.transforms.MFCC),
        ):
            cls = (
                type(self.preprocessor.featurizer).__name__
                if self.preprocessor.featurizer is not None
                else "None"
            )
            LOGGER.info(
                "AudioPreprocessor featurizer is %s (not MelSpectrogram/MFCC). "
                "Export continues via wrapper; verify dynamic axes if needed.",
                cls,
            )
        batch_size = max_batch
        # safe default for time axis
        default_time = (
            self.preprocessor.featurizer.sample_rate
            if self.preprocessor.featurizer
            and hasattr(self.preprocessor.featurizer, "sample_rate")
            else DEFAULT_TIME
        )

        example = []
        for _, neural_type in input_types.items():
            axes = neural_type.axes  # tuple of axis descriptors
            shape = []
            dtype = torch.float32  # default

            for axis in axes:
                axis_name = getattr(axis, "kind", axis)
                axis_name = str(axis_name).lower()
                if "batch" in axis_name or axis_name == "b":
                    shape.append(batch_size)
                elif "time" in axis_name or axis_name == "t":
                    shape.append(default_time)
                else:
                    shape.append(1)

            element_type_name = type(neural_type.elements_type).__name__.lower()
            if "length" in element_type_name:
                dtype = torch.long
                tensor = torch.full(
                    (batch_size,),
                    default_time,
                    dtype=dtype,
                )
            else:
                tensor = torch.zeros(*shape, dtype=dtype)

            example.append(tensor)

        return tuple(example)

    def dynamic_shapes_for_export(self, *args, **kwargs):
        return symbols_from_input_types(self.preprocessor.input_types)

    @property
    def input_names(self):
        return list(self.preprocessor.input_types.keys())

    @property
    def input_types(self):
        return self.preprocessor.input_types

    @property
    def output_names(self):
        return list(self.preprocessor.output_types.keys())

    def _args_to_kwargs(self, args, kwargs):
        try:
            names = list(self.preprocessor.input_types.keys())
        except AttributeError:  # pragma: no cover - defensive
            names = list(getattr(self.preprocessor, "input_names", []) or [])
        return (
            map_args_to_kwargs_by_names(args, kwargs, names)
            if names
            else kwargs
        )

    def forward(self, *args, **kwargs):
        # NeMo typed modules enforce kwargs-only; translate when necessary
        call_kwargs = self._args_to_kwargs(args, kwargs)
        return self.preprocessor(**call_kwargs)


class WrapPreprocessorCast(torch.nn.Module):
    """Wraps the preprocessor to add a cast to float16/32 at the output."""

    def __init__(self, preprocessor: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.preprocessor = preprocessor
        self.dtype = dtype

    def _args_to_kwargs(self, args, kwargs):
        try:
            names = list(self.preprocessor.input_types.keys())
        except AttributeError:  # pragma: no cover - defensive
            names = list(getattr(self.preprocessor, "input_names", []) or [])
        return (
            map_args_to_kwargs_by_names(args, kwargs, names)
            if names
            else kwargs
        )

    def forward(self, *args, **kwargs):
        # Ensure kwargs-only dispatch to preprocessor
        call_kwargs = self._args_to_kwargs(args, kwargs)
        x = self.preprocessor(**call_kwargs)
        return tuple([x[0].to(self.dtype)] + list(x)[1:])

    def input_example(self):
        return self.preprocessor.input_example()

    def _export_teardown(self):
        self.preprocessor._export_teardown()

    def _prepare_for_export(self, *args, **kwargs):
        self.preprocessor._prepare_for_export(*args, **kwargs)

    def dynamic_shapes_for_export(self, *args, **kwargs):
        return self.preprocessor.dynamic_shapes_for_export(*args, **kwargs)

    @property
    def input_names(self):
        return self.preprocessor.input_names

    @property
    def output_names(self):
        return self.preprocessor.output_names


class RenameOutputs(torch.nn.Module):
    """Wrapper that renames output tensor names for export-time only.

    Leaves computation unchanged and preserves input names.
    Useful to avoid name collisions between inputs and outputs
    (e.g., both named 'length').
    """

    def __init__(self, module: torch.nn.Module, rename_map: T.Dict[str, str]):
        super().__init__()
        self.module = module
        self._rename_map = dict(rename_map or {})

    @property
    def input_names(self):
        return getattr(self.module, "input_names", [])

    @property
    def output_names(self):
        base = list(getattr(self.module, "output_names", []) or [])
        return [self._rename_map.get(n, n) for n in base]

    def dynamic_shapes_for_export(self, *args, **kwargs):
        # outputs renaming does not affect dynamic input axes
        if hasattr(self.module, "dynamic_shapes_for_export"):
            return self.module.dynamic_shapes_for_export(*args, **kwargs)
        return {}

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class DecoderWithoutTargetLength(torch.nn.Module):
    """Wrap decoder/joint+decoder to remove 'target_length' argument/output."""

    FILTER_ARGUMENT = "target_length"
    FILTER_OUTPUT = "prednet_lengths"

    @require_extra_decorator(
        extra=T2NExtra.NEMO_TRACT, module="nemo.collections.asr", kw="nemo_asr"
    )
    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        nemo_asr: InjectedNemoModule = INJECTED,
    ):
        super().__init__()
        self.decoder = decoder
        self.active_fitering = isinstance(
            decoder,
            (
                nemo_asr.modules.rnnt.RNNTDecoderJoint,
                nemo_asr.modules.rnnt.RNNTDecoder,
            ),
        )

    def _infer_batch_size(self, args, kwargs):
        for v in args:
            if torch.is_tensor(v):
                return v.shape[0], v
        for v in kwargs.values():
            if torch.is_tensor(v):
                return v.shape[0], v

        raise T2NErrorInvalidArgument(
            "Cannot infer batch size: no Tensor inputs found"
        )

    @property
    def input_names(self):
        if self.active_fitering:
            return [
                name
                for name in self.decoder.input_names
                if name != self.FILTER_ARGUMENT
            ]
        return self.decoder.input_names

    @property
    def output_names(self):
        def rename_state(name: str) -> str:
            return OUT_STATE_NAME if name == INPUT_STATE_TUPLE_NAME else name

        if self.active_fitering:
            return [
                rename_state(_)
                for _ in self.decoder.output_names
                if _ != self.FILTER_OUTPUT
            ]
        return self.decoder.output_names

    @property
    def index_arg_to_remove(self) -> int:
        if self.active_fitering:
            for idx, name in enumerate(self.decoder.input_names):
                if name == self.FILTER_ARGUMENT:
                    return idx
        raise T2NErrorInvalidArgument(
            f"Cannot find argument named {self.FILTER_ARGUMENT} to remove"
        )

    @property
    def index_output_to_remove(self) -> int:
        if self.active_fitering:
            for idx, name in enumerate(self.decoder.output_names):
                if name == self.FILTER_OUTPUT:
                    return idx
        raise T2NErrorInvalidArgument(
            f"Cannot find output named {self.FILTER_OUTPUT} to remove"
        )

    def input_example(self, *args, **kwargs):
        if not self.active_fitering:
            return self.decoder.input_example(*args, **kwargs)
        return self.filter_original_input_example(
            self.decoder.input_example(*args, **kwargs)
        )

    def filter_original_input_example(
        self, inputs: T.List[torch.Tensor]
    ) -> T.List[torch.Tensor]:
        filtered_inputs = []
        for name, tensor in zip(self.decoder.input_names, inputs):
            if name != self.FILTER_ARGUMENT:
                filtered_inputs.append(tensor)
        return filtered_inputs

    def forward(self, *args, **kwargs):
        if not self.active_fitering:
            return self.decoder(*args, **kwargs)

        assert self.FILTER_ARGUMENT not in kwargs
        batch_size, ref_tensor = self._infer_batch_size(args, kwargs)

        # Ensure target_length is int64 (Tract-friendly for TDim casting)
        # Shape as (B,) for compatibility with NeMo decoders
        target_length = torch.ones(
            (batch_size,), device=ref_tensor.device, dtype=torch.long
        )
        to_rm_in_idx = self.index_arg_to_remove
        if len(args) > to_rm_in_idx:
            args = list(args)
            args.insert(to_rm_in_idx, target_length)
            args = tuple(args)
        else:
            kwargs = dict(kwargs)
            kwargs[self.FILTER_ARGUMENT] = target_length

        outs = self.decoder(*args, **kwargs)

        to_rm_out_idx = self.index_output_to_remove
        return tuple(
            list(outs[:to_rm_out_idx]) + list(outs[to_rm_out_idx + 1 :])
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)


class _REMOVE_LEGACY_PLACEHOLDER:  # kept to preserve file structure; no-op
    pass


class BoundaryAdapter(torch.nn.Module):
    """Export-time boundary adapter applying per-input tuple flattening and collapse.

    - Flattens tuple inputs to `name_0`, `name_1`, ... for external IO.
    - Applies per-input collapse of dynamic axes (supports batch-only v1).
    - Re-inserts collapsed axes before invoking the inner module.
    - Recomputes dynamic input axes to match the external interface.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        subnet_name: str,
        input_example: list,
        dynamic_axes: dict[str, dict[int, str]] | None,
        collapse_by_input: dict[str, set[str]] | None,
        binds_by_input: dict[str, str] | None = None,
        renamed_map: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.subnet_name = subnet_name
        self._orig_input_names = list(getattr(module, "input_names", []) or [])
        self._orig_output_names = list(getattr(module, "output_names", []) or [])
        self._orig_input_example = list(input_example or [])
        self._dyn_axes = dict(dynamic_axes or {})

        # Build external names by flattening tuple inputs
        initial_ext_names: list[str] = []
        initial_ext_map: list[tuple[str, int | None]] = []  # (base, idx)
        for nm, val in zip(self._orig_input_names, self._orig_input_example):
            if isinstance(val, (list, tuple)) and val:
                for k, _ in enumerate(val):
                    initial_ext_names.append(f"{nm}_{k}")
                    initial_ext_map.append((nm, k))
            else:
                initial_ext_names.append(nm)
                initial_ext_map.append((nm, None))

        # Resolve collapse rules per external name (qualified keys)
        self._collapse_idx: dict[str, list[int]] = {}
        self._rename_map = {
            str(t).upper(): [str(s).upper() for s in (srcs or [])]
            for t, srcs in (renamed_map or {}).items()
        }
        for ext in initial_ext_names:
            q = f"{subnet_name}.{ext}"
            want = set((collapse_by_input or {}).get(q, set()))
            if not want:
                continue
            dyn = self._dyn_axes.get(ext, {})
            # Support collapsing any requested dynamic symbols present on this input
            drop = []
            for i, s in dyn.items():
                su = str(s).upper()
                if su in want:
                    drop.append(i)
                    continue
                # Also allow alias: if wanted contains a target alias and current symbol is in its sources
                for w in want:
                    srcs = self._rename_map.get(w)
                    if srcs and su in srcs:
                        drop.append(i)
                        break
            if drop:
                self._collapse_idx[ext] = sorted(drop)

        # Resolve bindings per external name and mark targets as removed
        self._bind_for_ext: dict[str, tuple[str, str]] = {}
        if binds_by_input:
            for ext in list(initial_ext_names):
                q = f"{subnet_name}.{ext}"
                val = binds_by_input.get(q)
                if not isinstance(val, str) or "." not in val:
                    continue
                src_q, _, sym = val.rpartition(".")
                src_ext = src_q.split(".", 1)[1] if "." in src_q else src_q
                self._bind_for_ext[ext] = (src_ext, str(sym).strip().upper())

        # Final external names exclude bound targets
        self._ext_names: list[str] = []
        self._ext_map: list[tuple[str, int | None]] = []
        self._bound_targets: dict[str, tuple[str, str, str, int | None]] = {}
        for ext, mapping in zip(initial_ext_names, initial_ext_map):
            if ext in self._bind_for_ext:
                src_ext, sym = self._bind_for_ext[ext]
                base, idx = mapping
                self._bound_targets[ext] = (src_ext, sym, base, idx)
                continue
            self._ext_names.append(ext)
            self._ext_map.append(mapping)

    @property
    def input_names(self) -> list[str]:
        return list(self._ext_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._orig_output_names)

    def _ext_example_for(self, name: str, val) -> object:
        drop = set(self._collapse_idx.get(name, []))
        t = val
        if torch.is_tensor(t) and drop:
            for ax in sorted(drop, reverse=True):
                if 0 <= ax < t.dim():
                    # select 0 or squeeze for batch-like axis
                    t = t.squeeze(ax) if t.size(ax) == 1 else t.select(ax, 0)
        return t

    def input_example(self):
        out: list[object] = []
        for (base, idx), ext in zip(self._ext_map, self._ext_names):
            if idx is None:
                val = next((v for n, v in zip(self._orig_input_names, self._orig_input_example) if n == base), None)
            else:
                base_val = next((v for n, v in zip(self._orig_input_names, self._orig_input_example) if n == base), None)
                val = base_val[idx] if isinstance(base_val, (list, tuple)) else None
            out.append(self._ext_example_for(ext, val))
        return tuple(out)

    def dynamic_shapes_for_export(self) -> dict[str, dict[int, str]]:
        # Filter dynamic axes by removed indices
        ext_dyn: dict[str, dict[int, str]] = {}
        for ext in self._ext_names:
            axes = dict(self._dyn_axes.get(ext, {}) or {})
            if not axes:
                continue
            drop = set(self._collapse_idx.get(ext, []))
            if not drop:
                ext_dyn[ext] = axes
                continue
            remap: dict[int, str] = {}
            shift = 0
            for ax in sorted(axes.keys()):
                if ax in drop:
                    shift += 1
                    continue
                remap[ax - shift] = axes[ax]
            if remap:
                ext_dyn[ext] = remap
        return ext_dyn

    def _rebuild_internal_args(self, args) -> list:
        # Reinsert collapsed axes, and repack tuples
        by_base: dict[str, object | list] = {}
        ext_val_map: dict[str, object] = {ext: val for ext, val in zip(self._ext_names, args)}
        for (base, idx), ext, val in zip(self._ext_map, self._ext_names, args):
            t = val
            # Reinsert collapsed axes as size-1 dims
            for ax in sorted(self._collapse_idx.get(ext, [])):
                if torch.is_tensor(t):
                    t = t.unsqueeze(ax)
            if idx is None:
                by_base[base] = t
            else:
                lst = by_base.get(base)
                if not isinstance(lst, list):
                    lst = []
                # ensure size
                while len(lst) <= idx:
                    lst.append(None)
                lst[idx] = t
                by_base[base] = lst

        # Inject bound scalars for removed external inputs
        for tgt_ext, (src_ext, sym, base, idx) in self._bound_targets.items():
            src_val = ext_val_map.get(src_ext)
            if not torch.is_tensor(src_val):
                raise T2NErrorInvalidArgument(
                    f"binding source '{src_ext}' is not a tensor"
                )
            # Find original axis index for symbol on source, then remap after collapse
            axes = self._dyn_axes.get(src_ext, {}) or {}
            # Resolve symbol on source: exact match or match through rename alias
            old_ax = None
            for i, s in axes.items():
                if str(s).upper() == sym:
                    old_ax = i
                    break
                srcs = self._rename_map.get(sym)
                if srcs and str(s).upper() in srcs:
                    old_ax = i
                    break
            if old_ax is None:
                raise T2NErrorInvalidArgument(
                    f"binding symbol '{sym}' not found on source '{src_ext}'"
                )
            drop = sorted(self._collapse_idx.get(src_ext, []))
            shift = sum(1 for d in drop if d < old_ax)
            new_ax = old_ax - shift
            if not (0 <= new_ax < src_val.dim()):
                raise T2NErrorInvalidArgument(
                    f"binding axis {new_ax} out of range for '{src_ext}'"
                )
            # Keep dynamism: prefer aten::size (may return Tensor in tracing)
            dim_val = src_val.size(new_ax)
            if torch.is_tensor(dim_val):
                bound_scalar = dim_val.to(dtype=torch.long, device=src_val.device)
            else:
                bound_scalar = torch.scalar_tensor(
                    dim_val, dtype=torch.long, device=src_val.device
                )
            # If the target originally had collapsed axes (e.g., BATCH for length),
            # reinsert them so the inner module receives the expected rank (e.g., [B]).
            tgt_drop = sorted(self._collapse_idx.get(tgt_ext, [])) if hasattr(self, "_collapse_idx") else []
            tval = bound_scalar
            for ax in tgt_drop:
                tval = tval.unsqueeze(ax)
            if idx is None:
                by_base[base] = tval
            else:
                lst = by_base.get(base)
                if not isinstance(lst, list):
                    lst = []
                while len(lst) <= idx:
                    lst.append(None)
                lst[idx] = tval
                by_base[base] = lst
        # Order by original input_names; replace lists with tuples where needed
        ordered: list = []
        for nm in self._orig_input_names:
            val = by_base.get(nm)
            if isinstance(val, list):
                ordered.append(tuple(val))
            else:
                ordered.append(val)
        return ordered

    def forward(self, *args, **kwargs):
        assert not kwargs, "BoundaryAdapter expects positional args only"
        internal_args = self._rebuild_internal_args(list(args))
        return self.module(*internal_args)


@require_extra_decorator(extra=T2NExtra.NEMO_TRACT, module="nemo")
def use_pytorch_sdpa(
    model: torch.nn.Module, *, nemo: InjectedNemoModule = INJECTED
):
    """Modify the model to use PyTorch sdpa implementations where applicable."""
    nemo_submod = nemo.collections.asr.parts.submodules
    mha = nemo_submod.multi_head_attention.MultiHeadAttention
    for module in model.modules():
        if isinstance(module, mha):
            if not hasattr(module, "use_pytorch_sdpa"):
                raise T2NErrorInvalidArgument(
                    (
                        "The provided model's MultiHeadAttention module does "
                        "not have the 'use_pytorch_sdpa' attribute. Cannot "
                        "apply PyTorch SDPA. Please ensure a compatible NeMo "
                        f"version (yours: '{nemo.__version__}', required: "
                        "'2.1.0' or later) with PyTorch SDPA support."
                    )
                )
            module.use_pytorch_sdpa = True
