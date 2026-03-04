import logging
import typing as T

import torch

from torch_to_nnef._optional_types import InjectedNemoModule
from torch_to_nnef.nemo_tract.axes import collapse_dynamic_axes_mapping
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
        self, max_batch: int = 2, *, torchaudio: InjectedNemoModule = INJECTED
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
            LOGGER.warning(
                "AudioPreprocessor featurizer is not a MelSpectrogram/MFCC."
                "This is unknown behavior for T2N maintainer and may lead to "
                "suboptimal export results."
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
        raise RuntimeError("Cannot infer batch size: no Tensor inputs found")

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
        raise RuntimeError(
            f"Cannot find argument named {self.FILTER_ARGUMENT} to remove"
        )

    @property
    def index_output_to_remove(self) -> int:
        if self.active_fitering:
            for idx, name in enumerate(self.decoder.output_names):
                if name == self.FILTER_OUTPUT:
                    return idx
        raise RuntimeError(
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


class CollapseBatchDimWrapper(torch.nn.Module):
    """Wrap a NeMo exportable subnet to remove batch from its interface."""

    LENGTH_INPUTS = LENGTH_INPUT_NAMES
    LENGTH_OUTPUTS = LENGTH_OUTPUT_NAMES

    def __init__(
        self,
        module: torch.nn.Module,
        sym_dynamic_axes: T.Dict[str, T.Dict[int, str]],
    ):
        super().__init__()
        self.module = module
        self._sym_dynamic_axes = sym_dynamic_axes or {}
        self._orig_input_names = list(getattr(module, "input_names", []))
        self._orig_output_names = list(getattr(module, "output_names", []))

        self._ext_input_names = [
            n for n in self._orig_input_names if not is_length_name(n)
        ]
        self._collapsed_axes = collapse_dynamic_axes_mapping(
            sym_dynamic_axes or {}, self._ext_input_names
        )

        self._expected_rank_no_batch: T.Dict[str, int] = {}
        self._b_axes: T.Dict[str, T.List[int]] = {}
        self._init_expected_ranks_from_example()
        self._init_batch_axes(sym_dynamic_axes)

    def _get_module_input_example(self):
        if hasattr(self.module, "input_example"):
            return self.module.input_example()
        if hasattr(self.module, "input_module"):
            return self.module.input_module.input_example()
        return None

    def _init_expected_ranks_from_example(self) -> None:
        ex = self._get_module_input_example()
        if not isinstance(ex, (list, tuple)):
            return
        for name, t in zip(self._orig_input_names, ex):
            if is_length_name(name):
                continue
            if (
                name == INPUT_STATE_TUPLE_NAME
                and isinstance(t, (list, tuple))
                and t
            ):
                ts = t[0]
                if torch.is_tensor(ts):
                    self._expected_rank_no_batch[name] = (
                        max(0, ts.dim() - 1) if ts.dim() > 0 else 0
                    )
                else:
                    self._expected_rank_no_batch[name] = 0
            elif torch.is_tensor(t):
                self._expected_rank_no_batch[name] = (
                    max(0, t.dim() - 1) if t.dim() > 0 else 0
                )
            else:
                self._expected_rank_no_batch[name] = 0

    def _init_batch_axes(self, sym_dynamic_axes) -> None:
        def _is_b(sym: object) -> bool:
            s = str(sym).upper()
            return s == "B" or "BATCH" in s

        for name in self._orig_input_names:
            axes = (sym_dynamic_axes or {}).get(name, {})
            bpos = sorted([i for i, s in axes.items() if _is_b(s)])
            self._b_axes[name] = bpos if bpos else [0]
        for sname in STATE_INPUT_NAMES:
            if sname in self._orig_input_names:
                self._b_axes[sname] = [1]

    @property
    def input_names(self) -> T.List[str]:
        return list(self._ext_input_names)

    @property
    def output_names(self) -> T.List[str]:
        return [
            n for n in self._orig_output_names if n not in self.LENGTH_OUTPUTS
        ]

    def input_example(self) -> T.Tuple[torch.Tensor, ...]:
        ex = self._get_module_input_example()
        if ex is None:
            return ()
        return self._process_input_example(ex)

    def _process_input_example(self, ex) -> T.Tuple[torch.Tensor, ...]:
        out: T.List[T.Any] = []
        for name, t in zip(self._orig_input_names, ex):
            if is_length_name(name):
                continue
            # Proactively squeeze batch axes wherever they may be located
            # (some NeMo modules expose [T, B] instead of [B, T]).
            if torch.is_tensor(t):
                for bpos in sorted(self._b_axes.get(name, []), reverse=True):
                    if 0 <= bpos < t.dim() and t.size(bpos) == 1:
                        t = t.squeeze(bpos)
            if name in ("input_states_1", "input_states_2") and torch.is_tensor(
                t
            ):
                if t.dim() > 1 and t.size(1) == 1:
                    t = t.squeeze(1)
            elif name == INPUT_STATE_TUPLE_NAME and isinstance(
                t, (list, tuple)
            ):
                proc = []
                for s in t:
                    if torch.is_tensor(s) and s.dim() > 1 and s.size(1) == 1:
                        s = s.squeeze(1)
                    proc.append(s)
                t = tuple(proc)
            elif torch.is_tensor(t) and t.dim() > 0 and t.size(0) == 1:
                t = t.squeeze(0)
            out.append(t)
        return tuple(out)

    def dynamic_shapes_for_export(self) -> T.Dict[str, T.Dict[int, str]]:
        # Filter any indices that exceed the rank of the external interface
        ex = self.input_example()
        ranks = {
            n: (t.dim() if torch.is_tensor(t) else 0)
            for n, t in zip(self.input_names, ex or ())  # type: ignore[arg-type]
        }
        return filter_dynamic_axes_by_ranks(self._collapsed_axes or {}, ranks)

    def _infer_time_from_visible_inputs(
        self, visible: T.Sequence[torch.Tensor]
    ) -> int:
        """Infer a reasonable time dimension from external inputs.

        - Ignores any length-only inputs (already filtered from ``visible``).
        - Skips state inputs entirely, as their trailing dimension is
          hidden size and does not represent time.
        - Handles tuples/lists of tensors and 0-D tensors safely.
        """
        times: T.List[int] = []
        collapsed_index = 0
        for name in self._orig_input_names:
            if is_length_name(name):
                continue
            if collapsed_index >= len(visible):
                break
            t = visible[collapsed_index]
            collapsed_index += 1

            # Do not use state tensors to infer time
            if name in STATE_INPUT_NAMES:
                continue

            def push_time(x: T.Any):
                if torch.is_tensor(x) and x.dim() > 0:
                    times.append(int(x.shape[-1]))

            if isinstance(t, (list, tuple)):
                for s in t:
                    push_time(s)
            else:
                push_time(t)

        return int(max(times)) if times else 1

    def _synthesize_length_tensor(
        self,
        ref_device: torch.device,
        ref_dtype: torch.dtype,
        length: int,
        name: str,
    ) -> torch.Tensor:
        if name == "target_length":
            # For collapsed-batch invocation, a single-sample vector of length 1
            # is enough; DecoderWithoutTargetLength will expand per batch later.
            return torch.tensor([length], device=ref_device, dtype=torch.long)
        return torch.tensor([length], device=ref_device, dtype=torch.long)

    def forward(self, *args, **kwargs):
        assert not kwargs, (
            "CollapseBatchDimWrapper expects positional args only"
        )
        visible = list(args)
        ref_device, ref_dtype = self._select_device_dtype(visible)
        time_len = self._infer_time_from_visible_inputs(visible)
        full = self._build_full_inputs(visible, ref_device, ref_dtype, time_len)
        outs = self.module(*tuple(full))
        return self._filter_and_squeeze_outputs(outs)

    def _select_device_dtype(self, visible):
        ref_tensor = next((a for a in visible if torch.is_tensor(a)), None)
        ref_device = (
            ref_tensor.device
            if torch.is_tensor(ref_tensor)
            else torch.device("cpu")
        )
        ref_dtype = (
            ref_tensor.dtype if torch.is_tensor(ref_tensor) else torch.float32
        )
        return ref_device, ref_dtype

    def _build_full_inputs(
        self,
        visible: T.Sequence[T.Any],
        ref_device: torch.device,
        ref_dtype: torch.dtype,
        time_len: int,
    ) -> T.List[T.Any]:
        full: T.List[T.Any] = []
        vis_iter = iter(visible)
        for name in self._orig_input_names:
            if is_length_name(name):
                full.append(
                    self._synthesize_length_tensor(
                        ref_device, ref_dtype, time_len, name
                    )
                )
                continue
            val = next(vis_iter)
            # For state tensors, default to explicit batch axis at dim 1.
            if name == INPUT_STATE_TUPLE_NAME and isinstance(
                val, (list, tuple)
            ):
                proc = []
                for s in val:
                    t = s
                    b_axes = self._b_axes.get(name, [1])
                    for offset, ax in enumerate(b_axes):
                        t = t.unsqueeze(dim=ax + offset)
                    proc.append(t)
                full.append(tuple(proc))
                continue
            if torch.is_tensor(val):
                t = val
                # Detect token-like integer inputs (e.g., decoder targets)
                is_token = not is_length_name(name) and t.dtype in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
                if is_token:
                    # For token IDs, prefer explicit [B, U] with B inserted
                    # at dim 0
                    if t.dim() == 0:
                        t = t.unsqueeze(0)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    # Do not add further unsqueezes based on dynamic map
                    # for tokens
                    full.append(t)
                    continue

                expected_rank = self._expected_rank_no_batch.get(
                    name, max(1, t.dim())
                )
                if name not in ("input_states_1", "input_states_2"):
                    while t.dim() > max(1, expected_rank):
                        t = t.select(dim=0, index=0)
                b_axes = self._b_axes.get(name, [0])
                for offset, ax in enumerate(b_axes):
                    t = t.unsqueeze(dim=ax + offset)
                full.append(t)
            else:
                full.append(val)
        return full

    def _filter_and_squeeze_outputs(self, outs) -> T.Tuple[T.Any, ...]:
        if not isinstance(outs, tuple):
            outs = (outs,)
        keep_indices = [
            i
            for i, n in enumerate(self._orig_output_names)
            if n not in self.LENGTH_OUTPUTS
        ]
        proc: T.List[T.Any] = []
        for i in keep_indices:
            o = outs[i]
            if torch.is_tensor(o) and o.dim() > 0 and o.size(0) == 1:
                o = o.squeeze(0)
            proc.append(o)
        return tuple(proc)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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
                raise RuntimeError(
                    "The provided model's MultiHeadAttention module "
                    "does not have the 'use_pytorch_sdpa' attribute. "
                    "Cannot apply PyTorch SDPA."
                    " Please ensure you are using a compatible NeMo version"
                    f"(yours: '{nemo.__version__}', required: '2.1.0' or later)"
                    " with PyTorch SDPA support."
                )
            module.use_pytorch_sdpa = True
