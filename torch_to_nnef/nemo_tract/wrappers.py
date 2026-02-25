import logging
import typing as T

import torch

from torch_to_nnef._optional_types import InjectedNemoModule
from torch_to_nnef.utils import INJECTED, T2NExtra, require_extra_decorator

LOGGER = logging.getLogger(__name__)


def decoder_fix_input_example_batch_size(
    input_example: T.Tuple[torch.Tensor, ...],
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

    def input_example(self, max_batch: int = 1):
        # return a dummy input example if the original is empty
        ie = self.preprocessor.input_example()
        if ie is None or len(ie) == 0:
            input_types = self.preprocessor.input_types
            batch_size = max_batch
            default_time = 16000  # safe default for time axis

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

                element_type_name = type(
                    neural_type.elements_type
                ).__name__.lower()
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
        return ie

    def dynamic_shapes_for_export(self, *args, **kwargs):
        name_map = {
            "batch": "B",
            "time": "T",
            "stream": "S",
        }
        return {
            k: {ix: name_map[str(ax.kind)] for ix, ax in enumerate(v.axes)}
            for k, v in self.preprocessor.input_types.items()
        }

    @property
    def input_names(self):
        return list(self.preprocessor.input_types.keys())

    @property
    def output_names(self):
        return list(self.preprocessor.output_types.keys())

    def forward(self, *args, **kwargs):
        return self.preprocessor(*args, **kwargs)


class WrapPreprocessorCast(torch.nn.Module):
    """Wraps the preprocessor to add a cast to float16/32 at the output."""

    def __init__(self, preprocessor: torch.nn.Module, dtype: torch.dtype):
        super().__init__()
        self.preprocessor = preprocessor
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        x = self.preprocessor(*args, **kwargs)
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
            if name == "states":
                return "out_states"
            return name

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

        target_length = torch.ones(
            (batch_size, 1),
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
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

    LENGTH_INPUTS = {
        "length",
        "target_length",
        "processed_length",
        "audio_signal_length",
    }
    LENGTH_OUTPUTS = {
        "encoded_lengths",
        "prednet_lengths",
        "length",
        "processed_length",
        "audio_signal_length",
        "input_length",
    }

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

        def _is_len_name(n: str) -> bool:
            nl = n.lower()
            return (
                nl in self.LENGTH_INPUTS
                or nl in self.LENGTH_OUTPUTS
                or ("length" in nl)
            )

        self._ext_input_names = [
            n for n in self._orig_input_names if not _is_len_name(n)
        ]
        # Build collapsed axes for external interface (no batch, hide lengths)
        self._collapsed_axes = collapse_dynamic_axes_mapping(
            sym_dynamic_axes or {}, self._ext_input_names
        )
        # Infer expected rank (without batch) per visible input from an example
        self._expected_rank_no_batch: T.Dict[str, int] = {}
        self._b_axes: T.Dict[str, T.List[int]] = {}
        try:
            ex = self.module.input_example(max_batch=1)
        except Exception:  # pragma: no cover - defensive
            ex = None
        if isinstance(ex, (list, tuple)):
            for name, t in zip(self._orig_input_names, ex):
                if name in self.LENGTH_INPUTS or "length" in name.lower():
                    continue
                if name == "states" and isinstance(t, (list, tuple)) and t:
                    ts = t[0]
                    if torch.is_tensor(ts):
                        self._expected_rank_no_batch[name] = (
                            max(0, ts.dim() - 1) if ts.dim() > 0 else 0
                        )
                    else:
                        self._expected_rank_no_batch[name] = 0
                elif torch.is_tensor(t):
                    # assume batch is first dim when present
                    self._expected_rank_no_batch[name] = (
                        max(0, t.dim() - 1) if t.dim() > 0 else 0
                    )
                else:
                    self._expected_rank_no_batch[name] = 0
        # Determine batch axes per input from symbols mapping or default to [0]
        for name in self._orig_input_names:
            axes = (sym_dynamic_axes or {}).get(name, {})
            bpos = sorted([i for i, s in axes.items() if s == "B"])
            self._b_axes[name] = bpos if bpos else [0]
        # Special-case known state inputs: batch axis is 1 (layout: [L, B, H])
        for sname in ("input_states_1", "input_states_2", "states"):
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
        ex = None
        try:
            ex = self.module.input_example(max_batch=1)
        except AttributeError:
            if hasattr(self.module, "input_module"):
                ex = self.module.input_module.input_example(max_batch=1)
        if ex is None:
            return ()
        # Remove any length-like inputs and squeeze batch where applicable
        out: T.List[T.Any] = []
        for name, t in zip(self._orig_input_names, ex):
            if name in self.LENGTH_INPUTS or "length" in name.lower():
                continue
            if name in ("input_states_1", "input_states_2") and torch.is_tensor(t):
                if t.dim() > 1 and t.size(1) == 1:
                    t = t.squeeze(1)  # remove batch axis at dim 1
            elif name == "states" and isinstance(t, (list, tuple)):
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
        return dict(self._collapsed_axes)

    def _infer_time_from_visible_inputs(
        self, visible: T.Sequence[torch.Tensor]
    ):
        times = []
        collapsed_index = 0
        for name in self._orig_input_names:
            if name in self.LENGTH_INPUTS or "length" in name.lower():
                continue
            t = visible[collapsed_index]
            if torch.is_tensor(t):
                times.append(t.shape[-1])
            collapsed_index += 1
        return max(times) if times else 1

    def _synthesize_length_tensor(
        self,
        ref_device: torch.device,
        ref_dtype: torch.dtype,
        length: int,
        name: str,
    ) -> torch.Tensor:
        if name == "target_length":
            return torch.tensor([[length]], device=ref_device, dtype=torch.long)
        return torch.tensor([length], device=ref_device, dtype=torch.long)

    def forward(self, *args, **kwargs):
        assert not kwargs, (
            "CollapseBatchDimWrapper expects positional args only"
        )
        visible = list(args)
        full: T.List[T.Any] = []

        ref_tensor = next((a for a in visible if torch.is_tensor(a)), None)
        ref_device = (
            ref_tensor.device
            if torch.is_tensor(ref_tensor)
            else torch.device("cpu")
        )
        ref_dtype = (
            ref_tensor.dtype if torch.is_tensor(ref_tensor) else torch.float32
        )
        time_len = self._infer_time_from_visible_inputs(visible)

        vis_iter = iter(visible)
        for name in self._orig_input_names:
            if ("length" in name.lower()) or name in self.LENGTH_INPUTS:
                full.append(
                    self._synthesize_length_tensor(
                        ref_device, ref_dtype, time_len, name
                    )
                )
                continue
            val = next(vis_iter)
            # Handle combined states passed as a tuple (h, c)
            if name == "states" and isinstance(val, (list, tuple)):
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
                expected_rank = self._expected_rank_no_batch.get(
                    name, max(1, val.dim())
                )
                t = val
                # Avoid reducing layer dimension for RNNT state tensors
                if name not in ("input_states_1", "input_states_2"):
                    while t.dim() > max(1, expected_rank):
                        t = t.select(dim=0, index=0)
                b_axes = self._b_axes.get(name, [0])
                for offset, ax in enumerate(b_axes):
                    t = t.unsqueeze(dim=ax + offset)
                full.append(t)
            else:
                full.append(val)

        outs = self.module(*tuple(full))
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


def collapse_dynamic_axes_mapping(
    nemo_dynamic_axes: T.Dict[str, T.Dict[int, str]],
    input_names: T.Sequence[str],
) -> T.Dict[str, T.Dict[int, str]]:
    """Local import to avoid circulars (reused by wrapper)."""
    from .export import collapse_dynamic_axes_mapping as _collapse

    return _collapse(nemo_dynamic_axes, input_names)


def use_pytorch_sdpa(model: torch.nn.Module):
    """Modify the model to use PyTorch sdpa implementations where applicable."""
    # pylint: disable=import-outside-toplevel
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        MultiHeadAttention,
    )

    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.use_pytorch_sdpa = True
