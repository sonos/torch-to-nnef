import logging
import typing as T

import torch

from torch_to_nnef._optional_types import (
    InjectedNemoModule,
    InjectedTorchaudioModule,
)
from torch_to_nnef.exceptions import T2NErrorInvalidArgument
from torch_to_nnef.nemo_tract.constants import (
    DEFAULT_TIME,
)
from torch_to_nnef.nemo_tract.dynaxes import (
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
                if "batch" in axis_name:
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
