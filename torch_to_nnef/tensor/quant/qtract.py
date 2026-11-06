"""tract quantization formats serialized as custom `.dat` items.

`QTensorTractDat` is the base of every tract custom `.dat` format: offload
state, tract version gate and `.dat` writing workflow, with no assumption on
the serialized layout.

`QTensorTractBlockQuant` is the base of every ggml style block format: its
u8 storage holds one quantized code per value, its qscheme the block
parameters as torch tensors, and tract bytes are only packed when written.
This module holds the formats with one f16 scale per block of 32 values
(Q4_0, Q8_1, Q2_0_T, `QScalePerGroupF16` that a torch calibration sets:
min max, percentile, or external methods like GPTQ). Formats whose
calibration reproduces a llama.cpp reference quantizer (Q8_0, K-quants,
IQ formats) live in `qtract_ggml`.
"""

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
    T2NErrorMisuse,
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
    QScheme,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)

LOGGER = logging.getLogger(__name__)


class QTensorTract(QTensor):
    """All QTensorTract implementations."""


def check_block_quant_input(
    fp_tensor: torch.Tensor, block_len: int, format_name: str
) -> None:
    """Reject weights a tract block quantization format can not store.

    tract reads block quantized `.dat` as `shape[0]` rows of
    `prod(shape[1:])` values split in blocks of `block_len`.

    Raises:
        T2NErrorImpossibleQuantization: non floating, empty, rank not in
            2..4, `prod(shape[1:])` not a multiple of `block_len` or
            non finite values.
    """
    if not fp_tensor.is_floating_point():
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} requires floating weights "
            f"but found {fp_tensor.dtype}"
        )
    if fp_tensor.numel() == 0:
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} requires nonempty weights "
            f"but found shape {tuple(fp_tensor.shape)}"
        )
    if fp_tensor.ndim not in (2, 3, 4):
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} does only support weight "
            f"of rank 2 to 4 but found shape {tuple(fp_tensor.shape)}"
        )
    if reduce(operator.mul, fp_tensor.shape[1:]) % block_len != 0:
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} requires flattened rows "
            f"(prod(shape[1:])) divisible by {block_len} but found "
            f"shape {tuple(fp_tensor.shape)}"
        )
    if not torch.isfinite(fp_tensor).all():
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} requires finite weights"
        )


def check_f16_scales(
    scales: torch.Tensor,
    format_name: str,
    nonzero_mask: T.Optional[torch.Tensor] = None,
    scale_name: str = "scale",
) -> None:
    """Reject f16 rounded block scales out of f16 range.

    Args:
        scales: block scales already rounded to f16.
        format_name: format named in the error message.
        nonzero_mask: same shape as `scales`, True where the scale
            represents nonzero weights. When given, such scales rounded
            to 0 are rejected (f16 underflow would erase those weights).
        scale_name: scale named in the error message.

    Raises:
        T2NErrorImpossibleQuantization: on f16 overflow (inf) or underflow.
    """
    scales = scales.detach()
    if not torch.isfinite(scales).all():
        raise T2NErrorImpossibleQuantization(
            f"tract {format_name} block {scale_name} overflows f16: "
            "weights magnitude is too large"
        )
    if nonzero_mask is not None:
        underflow = nonzero_mask.to(torch.bool) & (scales == 0)
        if underflow.any():
            raise T2NErrorImpossibleQuantization(
                f"tract {format_name} block {scale_name} underflows f16: "
                f"{int(underflow.sum())} nonzero block(s) would dequantize "
                "to zeros (weights magnitude is too small)"
            )


class QTensorTractDat(QTensorTract, SupportsOffloadState):
    """Common base of QTensor serialized as tract custom `.dat` item types.

    It handles offload state, tract version gating and the `.dat` file
    writing workflow. Subclasses implement
    `_write_binary_dat_from_qtensor_parts` which dumps header and content
    from a (decompressed) u8 storage and its qscheme.

    It makes no assumption on the serialized layout (block quantization,
    vector quantization codebooks, ...): each format computes its own
    `.dat` data size. Subclasses set `decompressed_shape` to the float
    tensor shape at construction. Their u8 storage layout is format
    specific: `numel` must stay the float tensor one.

    """

    FORMAT_NAME: str
    MIN_TRACT_VERSION: T.Optional[str] = None
    # appended to the version gate warning when set
    VERSION_GATE_NOTE: T.Optional[str] = None
    decompressed_shape: T.Tuple[int, ...]

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
    def _check_offload_state(cls, state):
        """Format specific sanity checks of an offload state."""

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
        """Warn with an upgrade hint then raise below `MIN_TRACT_VERSION`.

        Called before any tensor data is written.
        """
        if (
            cls.MIN_TRACT_VERSION is not None
            and inference_target.version < cls.MIN_TRACT_VERSION
        ):
            msg = (
                f"{cls.FORMAT_NAME} export requires tract >= "
                f"{cls.MIN_TRACT_VERSION}; target is "
                f"{inference_target.version}. Upgrade tract before "
                f"exporting {cls.FORMAT_NAME} weights."
            )
            if cls.VERSION_GATE_NOTE:
                msg = f"{msg} {cls.VERSION_GATE_NOTE}"
            LOGGER.warning(msg)
            # same upgrade hint for callers that only see the exception
            raise T2NErrorNotImplemented(f"tract {msg}")

    @staticmethod
    def _decompress_u8_blob(u8_blob, u8_compressors):
        decompress_u8 = u8_blob
        for u8_compressor in reversed(u8_compressors or []):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        return decompress_u8

    @classmethod
    def _write_binary_dat_from_qtensor_parts(
        cls,
        path: Path,
        decompressed_shape,
        u8_blob: torch.Tensor,
        qscheme: QScheme,
        inference_target: TractNNEF,
    ):
        raise T2NErrorNotImplemented()

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
        cls._check_offload_state(state)
        cls._write_binary_dat_from_qtensor_parts(
            path,
            state["decompressed_shape"],
            cls._decompress_u8_blob(state["u8_blob"], state["u8_compressors"]),
            state["qscheme"],
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
        self._write_binary_dat_from_qtensor_parts(
            path,
            self.decompressed_shape,
            self.decompress_to_u8(),
            self.qscheme,
            inference_target,
        )


class QTensorTractBlockQuant(QTensorTractDat):
    """Common base of tract (ggml style) block quantization formats.

    Values are split in blocks of `BLOCK_LEN` consecutive values (in
    row-major order), each serialized as `BLOCK_BYTES` bytes.

    The u8 storage holds one quantized code per value (same shape as the
    float tensor) and the qscheme holds the block parameters (scales, mins,
    ...) as torch tensors. tract bytes are only made when written, by
    `_pack_blocks`.

    Subclass contract:
        - `OFFLOAD_STATE_TAG`, `FORMAT_NAME`, `MIN_TRACT_VERSION`
        - `BLOCK_LEN`, `BLOCK_BYTES`: serialized block layout
        - `QSCHEME_CLS`: its qscheme class, checked by `_check_qscheme`
          (`QScalePerGroupF16` by default: one f16 scale per block of
          `N_BITS` codes)
        - `_item_type(inference_target)`: tract `.dat` item type
        - `_block_params(qscheme, n_blocks)`: qscheme tensors with a
          leading `n_blocks` dimension, sliced per chunk of blocks
        - `_pack_blocks(codes, params, inference_target)`: pack u8 codes
          `(n, BLOCK_LEN)` and their params chunk into u8 `(n, BLOCK_BYTES)`

    """

    BLOCK_LEN: int
    BLOCK_BYTES: int
    QSCHEME_CLS: T.Type[QScheme]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_block_layout()

    def _init_block_layout(self):
        self._check_qscheme(self.qscheme)
        self.decompressed_shape = tuple(self.decompress_to_u8().shape)
        assert len(self.decompressed_shape) in [2, 3, 4], (
            self.decompressed_shape
        )
        assert (
            reduce(operator.mul, self.decompressed_shape[1:]) % self.BLOCK_LEN
            == 0
        ), self.decompressed_shape

    @classmethod
    def _from_codes(
        cls,
        fp_tensor: torch.Tensor,
        codes: torch.Tensor,
        qscheme: QScheme,
        dequant_to_dtype: torch.dtype,
    ):
        """Build from the u8 codes a calibration already computed.

        Same result as `cls(fp_tensor, qscheme, ...)`, without running
        `qscheme.quantize_as_u8` again (costly for search based formats).
        Only for formats without extra constructor state.
        """
        assert codes.dtype == torch.uint8, codes.dtype
        assert codes.shape == fp_tensor.shape, (codes.shape, fp_tensor.shape)
        qtensor = cls.__new__(
            cls, fp_tensor, qscheme, dequant_to_dtype=dequant_to_dtype
        )
        qtensor.u8_compressors = []
        qtensor.u8_blob = codes
        qtensor.qscheme = qscheme
        qtensor.dequant_to_dtype = dequant_to_dtype
        qtensor.nnef_name = None
        qtensor.requires_grad = False
        qtensor._init_block_layout()
        return qtensor

    def _rebuild_with(self, fp_tensor, qscheme, dequant_to_dtype):
        """Reuse the u8 codes without quantizing again (see `QTensor.to`).

        Formats with extra constructor state copy it in their override.
        """
        return self._from_codes(
            fp_tensor, self.decompress_to_u8(), qscheme, dequant_to_dtype
        )

    @classmethod
    def data_size_bytes(cls, shape: T.Tuple[int, ...]) -> int:
        """Serialized size of a tensor of `shape`."""
        vol = reduce(operator.mul, shape, 1)
        if vol % cls.BLOCK_LEN != 0:
            raise T2NErrorMisuse(
                f"tensor volume {vol} of shape {tuple(shape)} is not "
                f"a multiple of block length {cls.BLOCK_LEN}"
            )
        return vol // cls.BLOCK_LEN * cls.BLOCK_BYTES

    @classmethod
    def _check_qscheme(cls, qscheme: QScheme) -> None:
        assert isinstance(qscheme, cls.QSCHEME_CLS), qscheme

    @classmethod
    def _item_type(
        cls, inference_target: TractNNEF
    ) -> DatBinHeader.TractCustomTypes:
        raise T2NErrorNotImplemented()

    @classmethod
    def _block_params(
        cls, qscheme: QScheme, n_blocks: int
    ) -> T.Dict[str, torch.Tensor]:
        """Block parameters of `qscheme`, each with a leading `n_blocks` dim."""
        raise T2NErrorNotImplemented()

    @classmethod
    def _pack_blocks(
        cls,
        codes: torch.Tensor,
        params: T.Dict[str, torch.Tensor],
        inference_target: TractNNEF,
    ) -> np.ndarray:
        """Pack (n, BLOCK_LEN) u8 codes and their block params into bytes."""
        raise T2NErrorNotImplemented()

    @classmethod
    def _check_offload_state(cls, state):
        cls._check_qscheme(state["qscheme"])

    @classmethod
    def _build_binary_dat_header_for_shape(
        cls, decompressed_shape, inference_target: TractNNEF
    ) -> bytes:
        return DatBinHeader.build_tract_qtensor(
            cls._item_type(inference_target),
            tuple(decompressed_shape),
            data_size_bytes=cls.data_size_bytes(tuple(decompressed_shape)),
        ).to_bytes()

    @classmethod
    def _iter_binary_dat_content_chunks(
        cls,
        u8_blob: torch.Tensor,
        qscheme: QScheme,
        inference_target: TractNNEF,
        chunk_blocks: int = 1 << 16,
    ):
        block_len = cls.BLOCK_LEN
        codes = u8_blob.detach().cpu().flatten()
        assert codes.numel() % block_len == 0, codes.shape
        n_blocks = codes.numel() // block_len
        codes = codes.reshape(n_blocks, block_len)
        params = {
            k: v.detach().cpu()
            for k, v in cls._block_params(qscheme, n_blocks).items()
        }
        for start in range(0, n_blocks, chunk_blocks):
            end = min(start + chunk_blocks, n_blocks)
            b_all = cls._pack_blocks(
                codes[start:end],
                {k: v[start:end].contiguous() for k, v in params.items()},
                inference_target,
            )
            assert b_all.dtype == np.uint8 and b_all.shape == (
                end - start,
                cls.BLOCK_BYTES,
            ), b_all.shape
            yield b_all.tobytes()

    def _build_binary_dat_content(self, inference_target: TractNNEF) -> bytes:
        return b"".join(
            self._iter_binary_dat_content_chunks(
                self.decompress_to_u8(), self.qscheme, inference_target
            )
        )

    @classmethod
    def _write_binary_dat_from_qtensor_parts(
        cls,
        path: Path,
        decompressed_shape,
        u8_blob: torch.Tensor,
        qscheme: QScheme,
        inference_target: TractNNEF,
    ):
        with path.open("wb") as fh:
            fh.write(
                cls._build_binary_dat_header_for_shape(
                    decompressed_shape, inference_target
                )
            )
            for chunk in cls._iter_binary_dat_content_chunks(
                u8_blob, qscheme, inference_target
            ):
                fh.write(chunk)


class QTensorTractScalePerGroupF16(QTensorTractBlockQuant):
    """Common base of formats with one f16 scale per block of 32 codes.

    Its qscheme is a `QScalePerGroupF16` (Q4_0, Q8_1, Q2_0_T) that a torch
    calibration sets: min max, percentile, or external methods like GPTQ.
    """

    BLOCK_LEN = 32
    # former name of `BLOCK_LEN`
    GROUP_SIZE = BLOCK_LEN
    QSCHEME_CLS: T.Type[QScheme] = QScalePerGroupF16
    # bits per code and calibration default
    # (see qscale_per_group_f16_min_max_calibration)
    N_BITS: int
    RESTRICTED_RANGE = False
    qscheme: QScalePerGroupF16  # type notation for mypy

    @classmethod
    def _check_qscheme(cls, qscheme: QScheme) -> None:
        super()._check_qscheme(qscheme)
        assert isinstance(qscheme, QScalePerGroupF16), qscheme
        # tract limited support of packing
        assert qscheme.n_bits == cls.N_BITS, qscheme.n_bits
        assert qscheme.group_size == cls.BLOCK_LEN, qscheme.group_size

    @classmethod
    def _block_params(cls, qscheme, n_blocks):
        assert isinstance(qscheme, QScalePerGroupF16), qscheme
        return {"scale": qscheme.scale.reshape(n_blocks)}


class QTensorTractScaleOnly(QTensorTractScalePerGroupF16):
    """Tract data format it serializes to: Q4_0.

    Each group of 32 values is stored as 18 bytes:
    f16 scale, then 32 packed 4bit values.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_scale_only.v1"
    FORMAT_NAME = "Q4_0"
    N_BITS = 4
    BLOCK_BYTES = 18

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

    def _rebuild_with(self, fp_tensor, qscheme, dequant_to_dtype):
        new_obj = super()._rebuild_with(fp_tensor, qscheme, dequant_to_dtype)
        new_obj.specific_machine = self.specific_machine
        return new_obj

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
    def _pack_blocks(
        cls,
        codes: torch.Tensor,
        params: T.Dict[str, torch.Tensor],
        inference_target: TractNNEF,
    ) -> np.ndarray:
        u8_groups, scale = codes, params["scale"]
        n_groups = u8_groups.shape[0]
        post_tract_21_11 = inference_target.version >= "0.21.11"
        b_all = np.empty((n_groups, cls.BLOCK_BYTES), dtype=np.uint8)
        b_all[:, :2] = scale.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 2:] = (
            cls._pack_nibbles(u8_groups, post_tract_21_11)
            .view(np.uint8)
            .reshape(n_groups, cls.BLOCK_LEN // 2)
        )
        return b_all


class _TernaryScalePerGroupF16(QScalePerGroupF16):
    def quantize_as_u8(self, fp_tensor):
        groups = fp_tensor.detach().reshape(-1, 32)
        magnitudes = groups.abs().amax(dim=1, keepdim=True)
        if (
            not torch.isfinite(groups).all()
            or not ((groups == 0) | (groups.abs() == magnitudes)).all()
        ):
            raise T2NErrorImpossibleQuantization(
                "Q2_0_T requires finite ternary weights per group of 32"
            )
        if not torch.equal(magnitudes.to(torch.float16), self.scale):
            raise T2NErrorImpossibleQuantization("Q2_0_T scale mismatch")
        check_f16_scales(self.scale, "Q2_0_T", nonzero_mask=magnitudes != 0)
        return (fp_tensor.sign() + 1).to(torch.uint8)

    def dequantize(self, u8_tensor, target_dtype):
        values = u8_tensor.reshape(-1, 32).to(target_dtype) - 1
        return (values * self.scale.to(target_dtype)).reshape(u8_tensor.shape)


class QTensorTractQ2_0_T(QTensorTractScalePerGroupF16):  # noqa: N801
    """Tract ternary weights: f16 scale and 32 two-bit codes per block.

    Use `ternary_to_tract_q2_0_t` to construct from existing ternary weights.
    Codes 0, 1 and 2 represent -scale, zero and +scale. Each block occupies
    10 bytes, with four consecutive codes packed low bits first per byte.
    Export requires tract >= 0.23.11; older targets warn and are rejected.
    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q2_0_t.v1"
    FORMAT_NAME = "Q2_0_T"
    N_BITS = 2
    BLOCK_BYTES = 10
    MIN_TRACT_VERSION = "0.23.11"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.qscheme, _TernaryScalePerGroupF16):
            raise T2NErrorImpossibleQuantization(
                "Q2_0_T requires its ternary scale scheme"
            )

    @classmethod
    def _item_type(cls, inference_target):
        return DatBinHeader.TractCustomTypes.Q20T

    @classmethod
    def _pack_blocks(cls, codes, params, inference_target):
        scale = params["scale"]
        codes = codes.numpy().reshape(-1, 8, 4)
        blocks = np.empty((len(codes), cls.BLOCK_BYTES), dtype=np.uint8)
        blocks[:, :2] = (
            scale.numpy().astype("<f2").view(np.uint8).reshape(-1, 2)
        )
        blocks[:, 2:] = (
            codes[:, :, 0]
            | (codes[:, :, 1] << 2)
            | (codes[:, :, 2] << 4)
            | (codes[:, :, 3] << 6)
        )
        return blocks


def ternary_to_tract_q2_0_t(fp_tensor: torch.Tensor) -> QTensorTractQ2_0_T:
    """Pack existing ternary weights, rounding only their scales to f16.

    Accepts floating tensors of rank 2 to 4 whose flattened rows are a
    multiple of 32. Each consecutive group of 32 must contain only zero
    and a single absolute nonzero value. Larger source groups (e.g. 128)
    naturally split into blocks of 32. All-zero blocks are supported.

    Raises:
        T2NErrorImpossibleQuantization: Invalid shape, non-ternary values,
            or nonzero scales that overflow or underflow in f16.
    """
    check_block_quant_input(
        fp_tensor, QTensorTractQ2_0_T.BLOCK_LEN, QTensorTractQ2_0_T.FORMAT_NAME
    )
    with torch.no_grad():
        fp_tensor = fp_tensor.detach()
        scale = fp_tensor.reshape(-1, 32).abs().amax(dim=1, keepdim=True)
        scheme = _TernaryScalePerGroupF16(32, scale, 2)
        return QTensorTractQ2_0_T(
            fp_tensor, qscheme=scheme, dequant_to_dtype=fp_tensor.dtype
        )


class QTensorTractQ8_1(QTensorTractScalePerGroupF16):  # noqa: N801
    """Tract data format it serializes to: Q8_1 (GGML style).

    Each group of 32 values is stored as 36 bytes:
    f16 scale, f16 sum of dequantized values, then 32 int8.

    Warning:
        tract (<= 0.23.10) only supports Q8_1 weights in embedding/gathering
        operations: there is no matmul or convolution kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q8_1.v1"
    FORMAT_NAME = "Q8_1"
    N_BITS = 8
    BLOCK_BYTES = 36
    # GGML Q8 convention: d = absmax / 127, int8 values in [-127, 127]
    RESTRICTED_RANGE = True
    MIN_TRACT_VERSION = "0.23.0"

    @classmethod
    def _item_type(
        cls, inference_target: TractNNEF
    ) -> DatBinHeader.TractCustomTypes:
        if inference_target.version >= "0.23.10":
            return DatBinHeader.TractCustomTypes.Q81
        return DatBinHeader.TractCustomTypes.Q81_LEGACY

    @classmethod
    def _pack_blocks(
        cls,
        codes: torch.Tensor,
        params: T.Dict[str, torch.Tensor],
        inference_target: TractNNEF,
    ) -> np.ndarray:
        u8_groups, scale = codes, params["scale"]
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
        b_all = np.empty((n_groups, cls.BLOCK_BYTES), dtype=np.uint8)
        b_all[:, :2] = scale.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 2:4] = group_sum.numpy().view(np.uint8).reshape(n_groups, 2)
        b_all[:, 4:] = i8_vals.numpy().view(np.uint8)
        return b_all


TQTensorTractBlockQuant = T.TypeVar(
    "TQTensorTractBlockQuant", bound=QTensorTractScalePerGroupF16
)


def _fp_to_tract_block_quant_with_min_max_calibration(
    fp_tensor,
    qtensor_cls: T.Type[TQTensorTractBlockQuant],
    percentile: float,
) -> TQTensorTractBlockQuant:
    format_name = qtensor_cls.FORMAT_NAME
    group_size = qtensor_cls.BLOCK_LEN
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
