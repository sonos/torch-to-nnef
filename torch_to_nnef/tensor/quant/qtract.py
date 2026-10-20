import filecmp
import logging
import operator
import platform
import tempfile
import typing as T
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.opaque import (
    OFFLOAD_STATE_KEY,
    SupportsOffloadState,
)
from torch_to_nnef.tensor.quant.base import (
    QScalePerGroupF16,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)

LOGGER = logging.getLogger(__name__)


class QTensorTract(QTensor):
    """All QTensorTract implementations."""


class QTensorTractBlockQuant(QTensorTract, SupportsOffloadState):
    """Common base of tract GGML-like block quantization formats.

    Values are quantized per group of `GROUP_SIZE` elements sharing a f16
    scale. Each concrete format only defines its tract item type and how a
    chunk of groups is packed into bytes.

    """

    FORMAT_NAME: str
    N_BITS: int
    GROUP_SIZE = 32
    # calibration default, see qscale_per_group_f16_min_max_calibration
    RESTRICTED_RANGE = False
    MIN_TRACT_VERSION: T.Optional[str] = None
    qscheme: QScalePerGroupF16  # type notation for mypy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.qscheme, QScalePerGroupF16), self.qscheme
        # tract limited support of packing
        assert self.qscheme.n_bits == self.N_BITS, self.qscheme.n_bits
        assert self.qscheme.group_size == self.GROUP_SIZE, (
            self.qscheme.group_size
        )
        self.decompressed_shape = self.decompress_to_u8().shape
        assert len(self.decompressed_shape) in [2, 3, 4], (
            self.decompressed_shape
        )
        assert (
            reduce(operator.mul, self.decompressed_shape[1:]) % self.GROUP_SIZE
            == 0
        ), self.decompressed_shape

    @classmethod
    def _item_type(
        cls, inference_target: TractNNEF
    ) -> DatBinHeader.TractCustomTypes:
        raise T2NErrorNotImplemented()

    @classmethod
    def _pack_groups(
        cls,
        u8_groups: torch.Tensor,
        scale: torch.Tensor,
        inference_target: TractNNEF,
    ) -> np.ndarray:
        """Pack (n_groups, GROUP_SIZE) u8 values + (n_groups,) f16 scales."""
        raise T2NErrorNotImplemented()

    def to_offload_state(self) -> T.Dict[str, T.Any]:
        return {
            OFFLOAD_STATE_KEY: self.OFFLOAD_STATE_TAG,
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "u8_blob": self.u8_blob,
            "u8_compressors": self.u8_compressors,
            "qscheme": self.qscheme,
            "dequant_to_dtype": self.dequant_to_dtype,
            "nnef_name": self.nnef_name,
            "decompressed_shape": tuple(self.decompressed_shape),
        }

    def _load_extra_offload_state(self, state):
        """Restore format specific attributes from offload state."""

    @classmethod
    def from_offload_state(cls, state, target_device=None):
        assert cls.is_offload_state(state), state
        # Fallback reload path only: rebuild the Tensor subclass shell on CPU
        # from the compact state. The fp-shaped buffer is an uninitialized
        # placeholder (never read: decompression uses u8_blob), so it stays
        # cheap. Large NNEF exports skip this entirely via the direct
        # offload-state writer (write_offload_state_in_file).
        fp_placeholder = torch.empty(state["shape"], dtype=state["dtype"])
        qtensor = cls.__new__(
            cls,
            fp_placeholder,
            state["qscheme"],
            dequant_to_dtype=state["dequant_to_dtype"],
        )
        qtensor.u8_compressors = state["u8_compressors"]
        qtensor.u8_blob = state["u8_blob"]
        qtensor.qscheme = state["qscheme"]
        qtensor.dequant_to_dtype = state["dequant_to_dtype"]
        qtensor.nnef_name = state["nnef_name"]
        qtensor.requires_grad = False
        qtensor.decompressed_shape = state["decompressed_shape"]
        qtensor._load_extra_offload_state(state)
        if target_device is not None and qtensor.device != target_device:
            qtensor.to_device(target_device)
        return qtensor

    @classmethod
    def _check_inference_target_support(cls, inference_target: TractNNEF):
        if (
            cls.MIN_TRACT_VERSION is not None
            and inference_target.version < cls.MIN_TRACT_VERSION
        ):
            raise T2NErrorNotImplemented(
                f"tract {cls.FORMAT_NAME} export requires tract >= "
                f"{cls.MIN_TRACT_VERSION} but got {inference_target.version}"
            )

    @classmethod
    def _build_binary_dat_header_for_shape(
        cls, decompressed_shape, inference_target: TractNNEF
    ) -> bytes:
        return DatBinHeader.build_tract_qtensor(
            cls._item_type(inference_target), decompressed_shape
        ).to_bytes()

    @staticmethod
    def _decompress_u8_blob(u8_blob, u8_compressors):
        decompress_u8 = u8_blob
        for u8_compressor in reversed(u8_compressors or []):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        return decompress_u8

    @classmethod
    def _iter_binary_dat_content_chunks(
        cls,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        inference_target: TractNNEF,
        chunk_groups: int = 1 << 16,
    ):
        group_size = cls.GROUP_SIZE
        tensor_flat = u8_blob.detach().cpu().flatten()
        assert tensor_flat.numel() % group_size == 0, tensor_flat.shape
        group_count = tensor_flat.numel() // group_size
        scale_flat = scale.detach().cpu().reshape(group_count)

        for start in range(0, group_count, chunk_groups):
            end = min(start + chunk_groups, group_count)
            u8_groups = tensor_flat[start * group_size : end * group_size]
            b_all = cls._pack_groups(
                u8_groups.reshape(end - start, group_size),
                scale_flat[start:end].contiguous(),
                inference_target,
            )
            assert b_all.dtype == np.uint8 and b_all.shape[0] == end - start
            yield b_all.tobytes()

    def _build_binary_dat_content(self, inference_target: TractNNEF) -> bytes:
        return b"".join(
            self._iter_binary_dat_content_chunks(
                self.decompress_to_u8(), self.qscheme.scale, inference_target
            )
        )

    @classmethod
    def _write_binary_dat_from_parts(
        cls,
        path: Path,
        decompressed_shape,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        inference_target: TractNNEF,
    ):
        with path.open("wb") as fh:
            fh.write(
                cls._build_binary_dat_header_for_shape(
                    decompressed_shape, inference_target
                )
            )
            for chunk in cls._iter_binary_dat_content_chunks(
                u8_blob, scale, inference_target
            ):
                fh.write(chunk)

    @staticmethod
    def _assert_same_as_existing_dat(path: Path, write_fn) -> None:
        """Raise if `write_fn(dirpath)` would not reproduce existing `path`."""
        with tempfile.TemporaryDirectory() as _td:
            td = Path(_td)
            write_fn(td)
            new_path = td / path.name
            assert new_path.exists(), new_path
            if filecmp.cmp(path, new_path):
                return
        raise T2NErrorNotImplemented(
            "At least 2 variables in the NNEF graph, "
            f"share same Parameters: '{path.stem}' but they try "
            "to use different data-type (likely quantization format). "
            "This variable collision as no resolution strategy yet."
        )

    @classmethod
    def write_offload_state_in_file(
        cls,
        state,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: T.Optional[InferenceTarget] = None,
    ):
        assert cls.is_offload_state(state), state
        if inference_target is None:
            inference_target = TractNNEF.latest()
        path = Path(dirpath) / f"{label}.dat"
        if path.exists():
            cls._assert_same_as_existing_dat(
                path,
                lambda td: cls.write_offload_state_in_file(
                    state, td, label, inference_target=inference_target
                ),
            )
            return
        assert isinstance(inference_target, TractNNEF), inference_target
        cls._check_inference_target_support(inference_target)
        assert state["qscheme"].group_size == cls.GROUP_SIZE, state["qscheme"]
        cls._write_binary_dat_from_parts(
            path,
            state["decompressed_shape"],
            cls._decompress_u8_blob(state["u8_blob"], state["u8_compressors"]),
            state["qscheme"].scale,
            inference_target,
        )

    def write_in_file(
        self,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: T.Optional[InferenceTarget] = None,
    ):
        if inference_target is None:
            inference_target = TractNNEF.latest()
        path = Path(dirpath) / f"{label}.dat"
        # already created a variable dump with that name:
        # check we would produce identical serialized data
        if path.exists():
            self._assert_same_as_existing_dat(
                path,
                lambda td: self.write_in_file(
                    td, label, inference_target=inference_target
                ),
            )
            return
        assert isinstance(inference_target, TractNNEF), inference_target
        self._check_inference_target_support(inference_target)
        self._write_binary_dat_from_parts(
            path,
            self.decompressed_shape,
            self.decompress_to_u8(),
            self.qscheme.scale,
            inference_target,
        )


class QTensorTractScaleOnly(QTensorTractBlockQuant):
    """Tract data format it serializes to: Q4_0.

    Each group of 32 values is stored as 18 bytes:
    f16 scale, then 32 packed 4bit values.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_scale_only.v1"
    FORMAT_NAME = "Q4_0"
    N_BITS = 4

    def __init__(
        self, *args, specific_machine: T.Optional[str] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.specific_machine = specific_machine

    def decompress(self):
        """Tract Q4_0 dequantization depends on hardware.

        Typically dequantization happen with ops in f16
        on ARM and f32 (scale directly casted) on others
        so we overwrite the function to be consistant
        with tract.

        """
        assert self.qscheme is not None
        decompress_u8 = self.decompress_to_u8()
        if (self.specific_machine and "arm" in self.specific_machine) or (
            self.specific_machine is None and "arm" in platform.machine()
        ):
            return self.qscheme.dequantize(
                decompress_u8, target_dtype=torch.float16
            ).to(self.dequant_to_dtype)
        return self.qscheme.dequantize(
            decompress_u8, target_dtype=self.dequant_to_dtype
        )

    def to_offload_state(self) -> T.Dict[str, T.Any]:
        state = super().to_offload_state()
        state["specific_machine"] = self.specific_machine
        return state

    def _load_extra_offload_state(self, state):
        self.specific_machine = state["specific_machine"]

    @classmethod
    def _item_type(
        cls, inference_target: TractNNEF
    ) -> DatBinHeader.TractCustomTypes:
        if inference_target.version >= "0.21.11":
            return DatBinHeader.TractCustomTypes.Q40
        return DatBinHeader.TractCustomTypes.Q40_LEGACY

    @staticmethod
    def _pack_nibbles(
        u8_groups: torch.Tensor, post_tract_21_11: bool
    ) -> np.ndarray:
        if post_tract_21_11:
            tensor_per_group = u8_groups.reshape(-1, 2, 16)
            return (
                tensor_per_group[:, 0, :] | (tensor_per_group[:, 1, :] << 4)
            ).numpy()
        tensor_per_group = u8_groups.reshape(-1, 16, 2)
        return (
            tensor_per_group[:, :, 0] | (tensor_per_group[:, :, 1] << 4)
        ).numpy()

    @classmethod
    def _pack_groups(
        cls,
        u8_groups: torch.Tensor,
        scale: torch.Tensor,
        inference_target: TractNNEF,
    ) -> np.ndarray:
        n_groups = u8_groups.shape[0]
        post_tract_21_11 = inference_target.version >= "0.21.11"
        b_all = np.empty((n_groups, 2 + cls.GROUP_SIZE // 2), dtype=np.uint8)
        b_all[:, :2] = scale.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 2:] = (
            cls._pack_nibbles(u8_groups, post_tract_21_11)
            .view(np.uint8)
            .reshape(n_groups, cls.GROUP_SIZE // 2)
        )
        return b_all


class QTensorTractQ8_1(QTensorTractBlockQuant):  # noqa: N801
    """Tract data format it serializes to: Q8_1 (GGML style).

    Each group of 32 values is stored as 36 bytes:
    f16 scale, f16 sum of dequantized values, then 32 int8.

    Warning:
        tract (<= 0.23.9) only supports Q8_1 weights in embedding/gathering
        operations: there is no matmul or convolution kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q8_1.v1"
    FORMAT_NAME = "Q8_1"
    N_BITS = 8
    # GGML Q8 convention: d = absmax / 127, int8 values in [-127, 127]
    RESTRICTED_RANGE = True
    MIN_TRACT_VERSION = "0.23.0"

    @classmethod
    def _item_type(
        cls, inference_target: TractNNEF
    ) -> DatBinHeader.TractCustomTypes:
        return DatBinHeader.TractCustomTypes.Q81

    @classmethod
    def _pack_groups(
        cls,
        u8_groups: torch.Tensor,
        scale: torch.Tensor,
        inference_target: TractNNEF,
    ) -> np.ndarray:
        n_groups = u8_groups.shape[0]
        # u8 storage is offset by 2**(n_bits-1)
        i8_vals = (u8_groups.to(torch.int16) - 2 ** (cls.N_BITS - 1)).to(
            torch.int8
        )
        group_sum = i8_vals.sum(dim=1).to(torch.float32) * scale.to(
            torch.float32
        )
        f16_max = torch.finfo(torch.float16).max
        if group_sum.abs().max() > f16_max:
            # tract does not read it, but avoid writing inf in the .dat
            LOGGER.warning(
                "Q8_1 group sum exceed f16 range: clamped to +/-%s", f16_max
            )
            group_sum = group_sum.clamp(-f16_max, f16_max)
        group_sum = group_sum.to(torch.float16)
        b_all = np.empty((n_groups, 4 + cls.GROUP_SIZE), dtype=np.uint8)
        b_all[:, :2] = scale.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 2:4] = group_sum.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 4:] = i8_vals.numpy().view(np.uint8)
        return b_all


TQTensorTractBlockQuant = T.TypeVar(
    "TQTensorTractBlockQuant", bound=QTensorTractBlockQuant
)


def _fp_to_tract_block_quant_with_min_max_calibration(
    fp_tensor,
    qtensor_cls: T.Type[TQTensorTractBlockQuant],
    percentile: float,
) -> TQTensorTractBlockQuant:
    format_name = qtensor_cls.FORMAT_NAME
    group_size = qtensor_cls.GROUP_SIZE
    if isinstance(fp_tensor, torch.nn.Parameter):
        fp_tensor = fp_tensor.data

    if len(fp_tensor.shape) not in [2, 3, 4]:
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} does only support weight "
            f"of shape 2d, 3d or 4d but found {fp_tensor.shape}"
        )

    multiple_axis = 1
    if reduce(operator.mul, fp_tensor.shape[multiple_axis:]) % group_size != 0:
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} does only support weight with "
            f"dim={multiple_axis} divisible by {group_size} but "
            f"found {fp_tensor.shape[multiple_axis:]}"
        )
    with torch.no_grad():
        q_scheme = qscale_per_group_f16_min_max_calibration(
            fp_tensor,
            n_bits=qtensor_cls.N_BITS,
            group_size=group_size,
            percentile=percentile,
            restricted_range=qtensor_cls.RESTRICTED_RANGE,
        )
        return qtensor_cls(
            fp_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
        )


def fp_to_tract_q4_0_with_min_max_calibration(
    fp_tensor, percentile: float = 1.0
) -> QTensorTractScaleOnly:
    """Min-Max method to quantize float tensor to tract supported Q4_0."""
    return _fp_to_tract_block_quant_with_min_max_calibration(
        fp_tensor, QTensorTractScaleOnly, percentile
    )


def fp_to_tract_q8_1_with_min_max_calibration(
    fp_tensor, percentile: float = 1.0
) -> QTensorTractQ8_1:
    """Min-Max method to quantize float tensor to tract supported Q8_1.

    Only use it on embedding weights (see `QTensorTractQ8_1`).
    """
    return _fp_to_tract_block_quant_with_min_max_calibration(
        fp_tensor, QTensorTractQ8_1, percentile
    )
