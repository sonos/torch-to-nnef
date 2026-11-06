"""tract formats stored as raw ggml block bytes (K-quants, IQ formats).

These formats store each block of `BLOCK_LEN` values as an opaque
`BLOCK_BYTES` bytes chunk laid out exactly like llama.cpp `block_*`
structures. The QTensor u8 storage is these raw ggml bytes, produced by a
numpy port of the ggml reference quantizer and written as is in the tract
`.dat` file.

Formats with a single f16 scale per group of 32 values that a torch
calibration can set (Q4_0, Q8_1, Q2_0_T) live in `qtract` instead: their
u8 storage holds one quantized value per byte, packed only when written.

Warning:
    tract (<= 0.23.10) has no matmul or convolution kernel for these formats:
    only embedding/gathering operations are supported.

"""

import logging
import operator
import typing as T
from functools import lru_cache, reduce
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant._iq2_s_grid import KGRID_2BIT_1024
from torch_to_nnef.tensor.quant.base import QScheme, U8Compressor
from torch_to_nnef.tensor.quant.qtract import (
    QTensorTractDat,
    TractBlockLayout,
    check_block_quant_input,
    check_f16_scales,
)

LOGGER = logging.getLogger(__name__)

# numbers of blocks processed at once by numpy (bounds peak memory)
GGML_CHUNK_BLOCKS = 1 << 14
# ggml K-quants super-block length
QK_K = 256
# blocks erased by f16 scale underflow are tolerated (with a warning) when
# their weights are below this fraction of the tensor largest weight
GGML_UNDERFLOW_REL_TOL = float(np.finfo(np.float16).eps)


class QScaleGGMLBlock(QScheme):
    """Opaque ggml block scheme: the u8 storage are raw ggml block bytes.

    `QTensor` quantizes through its scheme, so this scheme only runs the
    chunked numpy loop and delegates the format logic (block quantization,
    dequantization and checks) to its `fmt` class. Each format is then a
    single `QTensorTractGGMLBlock` subclass, with no state in the scheme
    besides that class.

    """

    def __init__(self, fmt: T.Type["QTensorTractGGMLBlock"]):
        self.fmt = fmt

    def quantize_as_torch(self, fp_tensor):
        raise T2NErrorNotImplemented(
            f"native torch does not support ggml {self.fmt.FORMAT_NAME}"
        )

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Quantize into a (n_blocks, BLOCK_BYTES) u8 tensor."""
        fmt = self.fmt
        if fp_tensor.numel() % fmt.BLOCK_LEN != 0:
            raise T2NErrorImpossibleQuantization(
                f"tensor volume {fp_tensor.numel()} is not a multiple of "
                f"{fmt.FORMAT_NAME} block length {fmt.BLOCK_LEN}"
            )
        fp_blocks = (
            fp_tensor.detach()
            .cpu()
            .to(torch.float32)
            .contiguous()
            .numpy()
            .reshape((-1, fmt.BLOCK_LEN))
        )
        out = np.empty((fp_blocks.shape[0], fmt.BLOCK_BYTES), dtype=np.uint8)
        absmax = 0.0
        erased_count = 0
        erased_absmax = 0.0
        for start in range(0, fp_blocks.shape[0], GGML_CHUNK_BLOCKS):
            end = start + GGML_CHUNK_BLOCKS
            fp_chunk = fp_blocks[start:end]
            chunk = fmt.quantize_blocks(fp_chunk)
            assert chunk.dtype == np.uint8, chunk.dtype
            assert chunk.shape == (len(fp_chunk), fmt.BLOCK_BYTES)
            erased = fmt.check_quantized_blocks(fp_chunk, chunk)
            chunk_absmax = np.abs(fp_chunk).max(axis=1)
            absmax = max(absmax, float(chunk_absmax.max()))
            if erased.any():
                erased_count += int(erased.sum())
                erased_absmax = max(
                    erased_absmax, float(chunk_absmax[erased].max())
                )
            out[start:end] = chunk
        fmt.check_underflow(erased_count, erased_absmax, absmax)
        return torch.from_numpy(out).to(fp_tensor.device)

    def dequantize(self, u8_tensor: torch.Tensor, target_dtype):
        """Dequantize a (n_blocks, BLOCK_BYTES) u8 tensor into flat fp.

        Like tract: values are computed in f32 then casted to target dtype.
        """
        fmt = self.fmt
        u8_blocks = (u8_tensor.detach().cpu().contiguous().numpy()).reshape(
            (-1, fmt.BLOCK_BYTES)
        )
        out = np.empty((u8_blocks.shape[0], fmt.BLOCK_LEN), dtype=np.float32)
        for start in range(0, u8_blocks.shape[0], GGML_CHUNK_BLOCKS):
            end = start + GGML_CHUNK_BLOCKS
            out[start:end] = fmt.dequantize_blocks(u8_blocks[start:end])
        return (
            torch.from_numpy(out.reshape(-1))
            .to(target_dtype)
            .to(u8_tensor.device)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fmt={self.fmt.FORMAT_NAME})"


class QTensorTractGGMLBlock(TractBlockLayout, QTensorTractDat):
    """Common base of tract formats stored as raw ggml block bytes.

    Storage is the raw ggml block bytes (u8 tensor of shape
    `(n_blocks, BLOCK_BYTES)`, on the float tensor device), blocks follow
    the row-major order of the float tensor, and are dumped as is after the
    tract `.dat` header. `numel` is the float tensor one, not the bytes
    count.

    Subclass contract:
        - `OFFLOAD_STATE_TAG`: unique offload state tag
        - `FORMAT_NAME`: ggml format name (like "Q4_K")
        - `BLOCK_LEN`: number of values per block
        - `BLOCK_BYTES`: number of bytes per block
        - `ITEM_TYPE`: `DatBinHeader.TractCustomTypes` member
        - `MIN_TRACT_VERSION`: minimum tract version reading it
        - `F16_SCALE_FIELDS`: name to byte offset of each f16 block scale
          (like `{"d": 0, "dmin": 2}`), checked for f16 overflow
        - `quantize_blocks(fp32 (n, BLOCK_LEN)) -> u8 (n, BLOCK_BYTES)`
        - `dequantize_blocks(u8 (n, BLOCK_BYTES)) -> fp32 (n, BLOCK_LEN)`
          which must reproduce tract dequantization bit exactly.

    Validation shared by all formats:
        - `check_fp_tensor`: input weights (called at construction)
        - `check_quantized_blocks`: f16 scales overflow (called on each
          quantized chunk), then `check_underflow` on blocks erased by
          f16 scales underflow
        - `QTensorTractDat._check_inference_target_support`: tract version
          gate, warns with an upgrade hint then raises before any data
          is written

    Warning:
        tract (<= 0.23.10) only supports these formats in
        embedding/gathering operations.

    """

    ITEM_TYPE: DatBinHeader.TractCustomTypes
    MIN_TRACT_VERSION: T.Optional[str] = "0.23.10"
    F16_SCALE_FIELDS: T.Dict[str, int] = {}
    qscheme: QScaleGGMLBlock  # type notation for mypy

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        qscheme: QScaleGGMLBlock,
        dequant_to_dtype=torch.float32,
        u8_compressors: T.Optional[T.List[U8Compressor]] = None,
    ):
        assert isinstance(qscheme, QScaleGGMLBlock), qscheme
        assert qscheme.fmt is type(self), (qscheme.fmt, type(self))
        self.check_fp_tensor(fp_tensor)
        super().__init__(
            fp_tensor,
            qscheme,
            dequant_to_dtype=dequant_to_dtype,
            u8_compressors=u8_compressors,
        )
        self.decompressed_shape = tuple(fp_tensor.shape)

    @classmethod
    def check_fp_tensor(cls, fp_tensor: torch.Tensor):
        """Reject weights this format can not represent.

        See `check_block_quant_input`, subclasses may extend it.
        """
        check_block_quant_input(fp_tensor, cls.BLOCK_LEN, cls.FORMAT_NAME)

    @classmethod
    def from_fp(
        cls, fp_tensor: torch.Tensor, dequant_to_dtype=None
    ) -> "QTensorTractGGMLBlock":
        """Quantize a float tensor in this format."""
        if isinstance(fp_tensor, torch.nn.Parameter):
            fp_tensor = fp_tensor.data
        with torch.no_grad():
            return cls(
                fp_tensor,
                qscheme=QScaleGGMLBlock(cls),
                dequant_to_dtype=dequant_to_dtype or fp_tensor.dtype,
            )

    @classmethod
    def check_quantized_blocks(
        cls, fp_blocks: np.ndarray, u8_blocks: np.ndarray
    ) -> np.ndarray:
        """Reject f16 scales overflow, find blocks erased by underflow.

        Overflow: any `F16_SCALE_FIELDS` scale (or dequantized value) is
        not finite. Underflow: a block with nonzero weights dequantizes
        to all zeros (f16 scales rounded to 0). A partial underflow (like
        Q4_K `d` rounded to 0 while `dmin` is not) is not reported: it only
        drops variations far below the remaining f16 scale, within the
        quantization error.

        Returns:
            bool (n,) mask of the nonzero blocks dequantized to all zeros,
            judged by `check_underflow` over the whole tensor.

        Raises:
            T2NErrorImpossibleQuantization: on f16 overflow.
        """
        for name, offset in cls.F16_SCALE_FIELDS.items():
            check_f16_scales(
                u8_blocks[:, offset : offset + 2].copy().view(np.float16),
                cls.FORMAT_NAME,
                scale_name=f"scale `{name}`",
            )
        with np.errstate(over="ignore", invalid="ignore"):
            deq = cls.dequantize_blocks(u8_blocks)
        # per block dequantized magnitude: inf/nan on overflow, 0 when all
        # the f16 scales of a nonzero block underflow
        with np.errstate(invalid="ignore"):
            block_absmax = np.abs(deq).max(axis=1)
        check_f16_scales(block_absmax, cls.FORMAT_NAME, scale_name="scales")
        return (fp_blocks != 0).any(axis=1) & (block_absmax == 0)

    @classmethod
    def check_underflow(
        cls, erased_count: int, erased_absmax: float, absmax: float
    ) -> None:
        """Judge the blocks erased to zeros by f16 scales underflow.

        Like llama.cpp and tract quantizers, such blocks are kept as zeros
        (with a warning) when their weights are negligible against the
        tensor largest weight (below `GGML_UNDERFLOW_REL_TOL` of it, the
        f16 precision): this is a quantization error below what the f16
        dequantized tensor can represent. Otherwise the weights are too
        small as a whole for this format f16 scales.

        Raises:
            T2NErrorImpossibleQuantization: when erased weights are not
                negligible.
        """
        if erased_count == 0:
            return
        msg = (
            f"tract {cls.FORMAT_NAME} block scales underflow f16: "
            f"{erased_count} nonzero block(s) dequantize to zeros "
            f"(largest erased weight magnitude {erased_absmax:.3g}, "
            f"tensor largest {absmax:.3g})"
        )
        if erased_absmax > GGML_UNDERFLOW_REL_TOL * absmax:
            raise T2NErrorImpossibleQuantization(
                f"{msg}: weights magnitude is too small"
            )
        LOGGER.warning("%s: kept as zeros", msg)

    @classmethod
    def quantize_blocks(cls, fp_blocks: np.ndarray) -> np.ndarray:
        """Quantize fp32 (n, BLOCK_LEN) into u8 (n, BLOCK_BYTES)."""
        raise T2NErrorNotImplemented()

    @classmethod
    def dequantize_blocks(cls, u8_blocks: np.ndarray) -> np.ndarray:
        """Dequantize u8 (n, BLOCK_BYTES) into fp32 (n, BLOCK_LEN)."""
        raise T2NErrorNotImplemented()

    def numel(self) -> int:
        return reduce(operator.mul, self.decompressed_shape, 1)

    def decompress(self):
        return (
            self.qscheme.dequantize(
                self.decompress_to_u8(), target_dtype=torch.float32
            )
            .reshape(self.decompressed_shape)
            .to(self.dequant_to_dtype)
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
        assert isinstance(qscheme, QScaleGGMLBlock), qscheme
        assert qscheme.fmt is cls, (qscheme.fmt, cls)
        header = DatBinHeader.build_tract_qtensor(
            cls.ITEM_TYPE,
            tuple(decompressed_shape),
            data_size_bytes=cls.data_size_bytes(tuple(decompressed_shape)),
        )
        content = u8_blob.detach().cpu().contiguous().numpy()
        assert content.dtype == np.uint8, content.dtype
        assert content.size == header.data_size_bytes, (
            content.size,
            header.data_size_bytes,
        )
        with path.open("wb") as fh:
            fh.write(header.to_bytes())
            fh.write(memoryview(content.reshape(-1)))


# ggml shared helpers ######################################################


def ggml_seq_sum(arr: np.ndarray) -> np.ndarray:
    """f32 sum over last axis accumulated sequentially, like ggml loops."""
    return np.cumsum(arr, axis=-1, dtype=np.float32)[..., -1]


def ggml_nearest_int(arr: np.ndarray) -> np.ndarray:
    """Port of ggml `nearest_int` (f32 mantissa trick, round half to even).

    Exact like ggml for |x| <= 2**22 (ggml asserts it), and platform
    independent beyond, unlike a float to int32 cast.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        biased = arr.astype(np.float32) + np.float32(12582912.0)
    return (biased.view(np.int32) & 0x7FFFFF) - 0x400000


def ggml_make_qkx2_quants(
    x: np.ndarray,
    weights: np.ndarray,
    nmax: int,
    rmin: float,
    rdelta: float,
    nstep: int,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ggml `make_qkx2_quants` vectorized over rows of `x` (n, group).

    Returns:
        (scale (n,), levels u8 (n, group) in [0, nmax], the_min (n,))
    """
    f32 = np.float32
    x = x.astype(f32, copy=False)
    weights = weights.astype(f32, copy=False)
    xmin = x.min(axis=1)
    xmin = np.where(xmin > 0, f32(0), xmin)
    xmax = x.max(axis=1)
    sum_w = ggml_seq_sum(weights)
    sum_x = ggml_seq_sum(weights * x)
    degenerate = xmax == xmin
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        iscale = f32(nmax) / (xmax - xmin)
        scale = f32(1.0) / iscale
        levels = np.clip(
            ggml_nearest_int(iscale[:, None] * (x - xmin[:, None])), 0, nmax
        )
        diff = scale[:, None] * levels.astype(f32) + xmin[:, None] - x
        best_error = ggml_seq_sum(weights * diff * diff)
        for step in range(nstep + 1):
            iscale = (f32(rmin) + f32(rdelta) * f32(step) + f32(nmax)) / (
                xmax - xmin
            )
            laux = np.clip(
                ggml_nearest_int(iscale[:, None] * (x - xmin[:, None])),
                0,
                nmax,
            )
            laux_f = laux.astype(f32)
            sum_l = ggml_seq_sum(weights * laux_f)
            sum_l2 = ggml_seq_sum(weights * (laux * laux).astype(f32))
            sum_xl = ggml_seq_sum(weights * laux_f * x)
            det = sum_w * sum_l2 - sum_l * sum_l
            valid = det > 0
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / det
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det
            pos_min = this_min > 0
            this_min = np.where(pos_min, f32(0), this_min)
            this_scale = np.where(pos_min, sum_xl / sum_l2, this_scale)
            diff = this_scale[:, None] * laux_f + this_min[:, None] - x
            cur_error = ggml_seq_sum(weights * diff * diff)
            update = valid & (cur_error < best_error) & ~degenerate
            levels = np.where(update[:, None], laux, levels)
            best_error = np.where(update, cur_error, best_error)
            scale = np.where(update, this_scale, scale)
            xmin = np.where(update, this_min, xmin)
    levels = np.where(degenerate[:, None], 0, levels).astype(np.uint8)
    scale = np.where(degenerate, f32(0), scale).astype(f32)
    return scale, levels, (-xmin).astype(f32)


def ggml_unpack_scale_min_k4(
    scales: np.ndarray,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Port of ggml `get_scale_min_k4` for the 12 bytes of all blocks (n, 12).

    Returns:
        (sub-block scales u8 (n, 8), sub-block mins u8 (n, 8))
    """
    q_lo, q_mid, q_hi = scales[:, 0:4], scales[:, 4:8], scales[:, 8:12]
    sc = np.concatenate(
        [q_lo & 63, (q_hi & 0xF) | ((q_lo >> 6) << 4)], axis=1
    ).astype(np.uint8)
    mn = np.concatenate(
        [q_mid & 63, (q_hi >> 4) | ((q_mid >> 6) << 4)], axis=1
    ).astype(np.uint8)
    return sc, mn


def ggml_quant_k_scales_mins(
    x: np.ndarray, nmax: int, rmin: float, nstep: int
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Q4_K/Q5_K shared ggml header quantization (`quant_scales_mins`).

    Args:
        x: fp32 (n, 256) super-blocks
        nmax: maximum level (15 for Q4_K, 31 for Q5_K)
        rmin: `make_qkx2_quants` search start
        nstep: `make_qkx2_quants` search steps

    Returns:
        (header u8 (n, 16): f16 d, f16 dmin, 12 bytes packed scales/mins,
         levels u8 (n, 256) in [0, nmax])
    """
    f32 = np.float32
    n_blocks = x.shape[0]
    assert x.shape[1] == QK_K, x.shape
    sub = x.reshape((n_blocks * 8, 32)).astype(f32, copy=False)
    av_x = np.sqrt(ggml_seq_sum(sub * sub) / f32(32.0))
    weights = av_x[:, None] + np.abs(sub)
    scales, levels, mins = ggml_make_qkx2_quants(
        sub, weights, nmax, rmin, 0.1, nstep
    )
    scales = scales.reshape((n_blocks, 8))
    mins = mins.reshape((n_blocks, 8))
    levels = levels.reshape((n_blocks, QK_K))
    # comparisons as in ggml: -0.0 mins never win over 0.0
    max_scale = np.where(scales > 0, scales, f32(0)).max(axis=1)
    max_min = np.where(mins > 0, mins, f32(0)).max(axis=1)
    with np.errstate(divide="ignore"):
        inv_scale = np.where(max_scale > 0, f32(63.0) / max_scale, f32(0))
        inv_min = np.where(max_min > 0, f32(63.0) / max_min, f32(0))
    # `as u8` wraps like in tract before the min
    ls = np.minimum(
        ggml_nearest_int(inv_scale[:, None] * scales) & 0xFF, 63
    ).astype(np.uint8)
    lm = np.minimum(
        ggml_nearest_int(inv_min[:, None] * mins) & 0xFF, 63
    ).astype(np.uint8)
    sc = np.zeros((n_blocks, 12), dtype=np.uint8)
    sc[:, 0:4] = ls[:, :4] | ((ls[:, 4:] >> 4) << 6)
    sc[:, 4:8] = lm[:, :4] | ((lm[:, 4:] >> 4) << 6)
    sc[:, 8:12] = (ls[:, 4:] & 0xF) | ((lm[:, 4:] & 0xF) << 4)
    # f16 overflow to inf is rejected by `check_quantized_blocks`
    with np.errstate(over="ignore"):
        d_f16 = (max_scale / f32(63.0)).astype(np.float16)
        dmin_f16 = (max_min / f32(63.0)).astype(np.float16)
    header = np.empty((n_blocks, 16), dtype=np.uint8)
    header[:, 0:2] = d_f16.reshape((n_blocks, 1)).view(np.uint8)
    header[:, 2:4] = dmin_f16.reshape((n_blocks, 1)).view(np.uint8)
    header[:, 4:16] = sc
    # final levels with f16 rounded super scales
    sc_u, m_u = ggml_unpack_scale_min_k4(sc)
    d = d_f16.astype(f32)[:, None] * sc_u.astype(f32)
    dm = dmin_f16.astype(f32)[:, None] * m_u.astype(f32)
    x8 = x.reshape((n_blocks, 8, 32)).astype(f32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        new_levels = np.clip(
            ggml_nearest_int((x8 + dm[:, :, None]) / d[:, :, None]), 0, nmax
        )
    levels = np.where(
        (d != 0)[:, :, None], new_levels, levels.reshape((n_blocks, 8, 32))
    ).astype(np.uint8)
    return header, levels.reshape((n_blocks, QK_K))


def ggml_dequant_k_scales_mins(
    header: np.ndarray,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Decode Q4_K/Q5_K (n, 16) header into per sub-block fp32 terms.

    Returns:
        (d * scale (n, 8), dmin * min (n, 8)) so value = d_sc * q - dmin_m
    """
    f32 = np.float32
    d = header[:, 0:2].copy().view(np.float16).astype(f32)
    dmin = header[:, 2:4].copy().view(np.float16).astype(f32)
    sc, mn = ggml_unpack_scale_min_k4(header[:, 4:16])
    return d * sc.astype(f32), dmin * mn.astype(f32)


# formats ##################################################################


class QTensorTractQ4_K(QTensorTractGGMLBlock):  # noqa: N801
    """Tract data format it serializes to: Q4_K (ggml `block_q4_K`).

    Each super-block of 256 values is stored as 144 bytes:
    f16 `d`, f16 `dmin`, 12 bytes of eight 6bit scales and mins
    (one per sub-block of 32 values), then 256 4bit levels (4.5bpw).
    Value is `d * scale * level - dmin * min`.

    Quantization port llama.cpp `quantize_row_q4_K_ref` (no importance
    matrix), dequantization matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports Q4_K weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q4_k.v1"
    FORMAT_NAME = "Q4_K"
    BLOCK_LEN = QK_K
    BLOCK_BYTES = 144
    ITEM_TYPE = DatBinHeader.TractCustomTypes.Q4K
    F16_SCALE_FIELDS = {"d": 0, "dmin": 2}

    @classmethod
    def quantize_blocks(cls, fp_blocks: np.ndarray) -> np.ndarray:
        n_blocks = fp_blocks.shape[0]
        header, levels = ggml_quant_k_scales_mins(
            fp_blocks, nmax=15, rmin=-1.0, nstep=20
        )
        # per 64 values: low nibbles from first 32, high from next 32
        chunks = levels.reshape((n_blocks, 4, 2, 32))
        qs = chunks[:, :, 0, :] | (chunks[:, :, 1, :] << 4)
        out = np.empty((n_blocks, cls.BLOCK_BYTES), dtype=np.uint8)
        out[:, :16] = header
        out[:, 16:] = qs.reshape((n_blocks, 128))
        return out

    @classmethod
    def dequantize_blocks(cls, u8_blocks: np.ndarray) -> np.ndarray:
        n_blocks = u8_blocks.shape[0]
        d_sc, dmin_m = ggml_dequant_k_scales_mins(u8_blocks[:, :16])
        qs = u8_blocks[:, 16:].reshape((n_blocks, 4, 1, 32))
        levels = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(
            (n_blocks, 8, 32)
        )
        # same f32 ops order as tract: (d * sc) * q - (dmin * m)
        out = d_sc[:, :, None] * levels.astype(np.float32) - dmin_m[:, :, None]
        return out.reshape((n_blocks, cls.BLOCK_LEN))


def fp_to_tract_q4_k(fp_tensor: torch.Tensor) -> QTensorTractQ4_K:
    """Quantize float tensor to tract supported Q4_K (ggml reference).

    Only use it on embedding weights (see `QTensorTractQ4_K`).
    """
    qtensor = QTensorTractQ4_K.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractQ4_K)
    return qtensor


# IQ2_S ####################################################################

# ggml `GROUP_MAX_EPS_IQ2_S`
IQ2_S_GROUP_MAX_EPS = 1e-8
# packed 2bit codes whose digits are at most 2 (43690 is eight 2 digits)
IQ2_S_KMAP_SIZE = 43692
# ggml `iq2s_grid` dequant bytes for levels 0, 1, 2
IQ2_S_CODE_BYTE = np.array([0x08, 0x19, 0x2B], dtype=np.float32)
# octets processed at once by the off grid neighbour search
IQ2_S_NEIGHBOUR_CHUNK = 1 << 14
# nearest shell padding widths (largest shell holds 70 grid points)
IQ2_S_SHELL_WIDTHS = (1, 2, 4, 8, 16, 32)


def _iq2_s_unpack_levels(codes: np.ndarray) -> np.ndarray:
    """Unpack (...) 2bit packed codes into (..., 8) levels."""
    shifts = 2 * np.arange(8, dtype=np.int64)
    return ((codes.astype(np.int64)[..., None] >> shifts) & 3).astype(np.int8)


def _iq2_s_pack_levels(levels: np.ndarray) -> np.ndarray:
    """Pack (..., 8) levels in [0, 2] into (...) 2bit codes."""
    shifts = 2 * np.arange(8, dtype=np.int64)
    return (levels.astype(np.int64) << shifts).sum(axis=-1)


@lru_cache(maxsize=1)
def _iq2_s_search() -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of tract `build_search` (ggml `iq2_s` init, nearest shell only).

    Returns:
        (grid odd levels i8 (1024, 8) in {1, 3, 5},
         map i32 (KMAP_SIZE,): packed code to grid index or -1 off grid,
         neighbours i32 (KMAP_SIZE, max_shell): grid indices of the
         nearest shell in index order, padded with -1)
    """
    grid_levels = _iq2_s_unpack_levels(KGRID_2BIT_1024)
    assert grid_levels.max() <= 2
    grid = (2 * grid_levels + 1).astype(np.int8)
    kmap = np.full(IQ2_S_KMAP_SIZE, -1, dtype=np.int32)
    kmap[KGRID_2BIT_1024.astype(np.int64)] = np.arange(1024, dtype=np.int32)
    # snapped levels are clamped to [0, 2]: only those codes are looked up
    all_levels = np.stack(
        np.meshgrid(*[np.arange(3)] * 8, indexing="ij"), axis=-1
    ).reshape((-1, 8))
    all_codes = _iq2_s_pack_levels(all_levels)
    off = kmap[all_codes] < 0
    odd = (2 * all_levels[off] + 1).astype(np.int32)
    d2 = ((odd[:, None, :] - grid[None].astype(np.int32)) ** 2).sum(axis=-1)
    is_best = d2 == d2.min(axis=1, keepdims=True)
    max_shell = int(is_best.sum(axis=1).max())
    # shell kept in grid index order (ggml qsort order), integer ops only
    rows, cols = np.nonzero(is_best)
    slots = np.cumsum(is_best, axis=1)[rows, cols] - 1
    shell = np.full((is_best.shape[0], max_shell), -1, dtype=np.int32)
    shell[rows, slots] = cols
    neighbours = np.full((IQ2_S_KMAP_SIZE, max_shell), -1, dtype=np.int32)
    neighbours[all_codes[off]] = shell
    return grid, kmap, neighbours


def _iq2_s_find_best_neighbour(
    codes: np.ndarray, xval: np.ndarray, waux: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Port of tract `find_best_neighbour` for off grid octets.

    Args:
        codes: (m,) off grid packed codes
        xval: fp32 (m, 8) absolute values
        waux: fp32 (m, 8) weights
        scale: fp32 (m,) trial scale

    Returns:
        levels i8 (m, 8) of the best on grid neighbour
    """
    f32 = np.float32
    grid, _, neighbours = _iq2_s_search()
    grid_t = grid.T.astype(f32)
    shell_len = (neighbours[codes] >= 0).sum(axis=1)
    out = np.empty((codes.shape[0], 8), dtype=np.int8)
    # shells are padded to a few widths to bound the wasted work
    prev_width = 0
    for width in IQ2_S_SHELL_WIDTHS + (neighbours.shape[1],):
        sel_all = np.nonzero((shell_len > prev_width) & (shell_len <= width))[0]
        prev_width = width
        for start in range(0, sel_all.size, IQ2_S_NEIGHBOUR_CHUNK):
            sel = sel_all[start : start + IQ2_S_NEIGHBOUR_CHUNK]
            cands = neighbours[codes[sel], :width]
            valid = cands >= 0
            cands_ix = np.where(valid, cands, 0)
            sc = scale[sel, None]
            # tract order: d2 += (w * diff) * diff, sequentially
            d2 = np.zeros(cands.shape, dtype=f32)
            for k in range(8):
                diff = sc * grid_t[k][cands_ix] - xval[sel, k, None]
                d2 += waux[sel, k, None] * diff * diff
            # tract strict `d2 < best` scan: first minimum wins on ties
            # (no `np.argmin`: its SIMD tie breaking is platform dependent)
            best_d2 = d2[:, 0]
            pick = cands[:, 0]
            for j in range(1, width):
                update = valid[:, j] & (d2[:, j] < best_d2)
                best_d2 = np.where(update, d2[:, j], best_d2)
                pick = np.where(update, cands[:, j], pick)
            out[sel] = (grid[pick] - 1) // 2
    return out


def _iq2_s_snap(
    xval: np.ndarray,
    waux: np.ndarray,
    inv_scale: np.ndarray,
    scale: np.ndarray,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Port of tract `snap_octet` vectorized over (m, 8) octets.

    Returns:
        (levels i8 (m, 8) in [0, 2] on grid, on_grid bool (m,) before
         the neighbour search)
    """
    f32 = np.float32
    _, kmap, _ = _iq2_s_search()
    levels = np.clip(
        ggml_nearest_int(f32(0.5) * (inv_scale[:, None] * xval - f32(1.0))),
        0,
        2,
    ).astype(np.int8)
    codes = _iq2_s_pack_levels(levels)
    on_grid = kmap[codes] >= 0
    off = np.nonzero(~on_grid)[0]
    if off.size:
        levels[off] = _iq2_s_find_best_neighbour(
            codes[off], xval[off], waux[off], scale[off]
        )
    return levels, on_grid


def _iq2_s_sums(
    weight: np.ndarray, xval: np.ndarray, levels: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Group `sumqx` and `sumq2` over (g, 16) values, like tract."""
    f32 = np.float32
    q = f32(2.0) * levels.astype(f32) + f32(1.0)
    sumqx = ggml_seq_sum(weight * xval * q)
    sumq2 = ggml_seq_sum(weight * q * q)
    return sumqx, sumq2


def _iq2_s_quantize_groups(
    xb: np.ndarray, sigma2: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of tract IQ2_S `quantize_group` vectorized over (g, 16) groups.

    Returns:
        (scale fp32 (g,), grid index (g, 2), sign bytes u8 (g, 2))
        skipped groups (below `GROUP_MAX_EPS_IQ2_S`) get 0 everywhere.
    """
    f32 = np.float32
    _, kmap, _ = _iq2_s_search()
    n_groups = xb.shape[0]
    weight = f32(0.25) * sigma2[:, None] + xb * xb
    waux = np.sqrt(weight)
    neg = xb < 0
    xval = np.where(neg, -xb, xb)
    bits = (1 << np.arange(8)).astype(np.uint8)
    signs = (neg.reshape((n_groups, 2, 8)) * bits).sum(axis=-1).astype(np.uint8)
    max_x = xval.max(axis=1)
    skipped = max_x < f32(IQ2_S_GROUP_MAX_EPS)
    max_x = np.where(skipped, f32(1.0), max_x)
    xval_o = xval.reshape((n_groups * 2, 8))
    waux_o = waux.reshape((n_groups * 2, 8))
    levels = np.zeros((n_groups, 16), dtype=np.int8)
    best = np.zeros(n_groups, dtype=f32)
    scale = max_x / f32(5.0)
    on_grid = np.ones((n_groups, 2), dtype=bool)
    # huge weights overflow sums like in tract (rejected afterwards)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for step in range(-9, 10):
            inv_scale = (f32(5.0) + f32(step) * f32(0.1)) / max_x
            this_scale = f32(1.0) / inv_scale
            aux, on_aux = _iq2_s_snap(
                xval_o,
                waux_o,
                np.repeat(inv_scale, 2),
                np.repeat(this_scale, 2),
            )
            aux = aux.reshape((n_groups, 16))
            sumqx, sumq2 = _iq2_s_sums(weight, xval, aux)
            update = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
            new_scale = sumqx / sumq2
            scale = np.where(update, new_scale, scale)
            best = np.where(update, new_scale * sumqx, best)
            levels = np.where(update[:, None], aux, levels)
            on_grid = np.where(
                update[:, None], on_aux.reshape((-1, 2)), on_grid
            )
        # off grid octets are snapped again with the best scale
        refit = (~on_grid).any(axis=1) & (scale > 0)
        redo = np.nonzero((~on_grid & refit[:, None]).reshape(-1))[0]
        if redo.size:
            scale_o = np.repeat(scale, 2)[redo]
            levels_o = levels.reshape((-1, 8))
            levels_o[redo], _ = _iq2_s_snap(
                xval_o[redo], waux_o[redo], f32(1.0) / scale_o, scale_o
            )
            levels = levels_o.reshape((n_groups, 16))
            sumqx, sumq2 = _iq2_s_sums(weight, xval, levels)
            scale = np.where(refit & (sumq2 > 0), sumqx / sumq2, scale)
    flip = scale < 0
    scale = np.where(flip, -scale, scale)
    signs = np.where(flip[:, None], ~signs, signs)
    grid_index = kmap[_iq2_s_pack_levels(levels.reshape((n_groups, 2, 8)))]
    assert (grid_index[~skipped] >= 0).all(), "iq2_s point is not on the grid"
    scale = np.where(skipped, f32(0.0), scale).astype(f32)
    grid_index = np.where(skipped[:, None], 0, grid_index)
    signs = np.where(skipped[:, None], 0, signs).astype(np.uint8)
    return scale, grid_index, signs


class QTensorTractIQ2_S(QTensorTractGGMLBlock):  # noqa: N801
    """Tract data format it serializes to: IQ2_S (ggml `block_iq2_s`).

    Each super-block of 256 values is stored as 82 bytes (2.5625bpw):
    f16 `d`, 32 low grid index bytes then 32 sign bytes (one per octet
    of 8 values), 8 bytes of 2bit high grid index parts, then sixteen
    4bit scales (one per group of 16 values).
    Each octet is a point of the ggml 1024 entries `kgrid_2bit_1024`
    grid (magnitudes in {8, 25, 43}), value is
    `d * (0.5 + scale) * 0.25 * magnitude * sign`.

    Quantization port llama.cpp `quantize_row_iq2_s_ref` without
    importance matrix (grid nearest neighbour search with a weighted
    scale fit), dequantization matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports IQ2_S weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_iq2_s.v1"
    FORMAT_NAME = "IQ2_S"
    BLOCK_LEN = QK_K
    BLOCK_BYTES = 82
    ITEM_TYPE = DatBinHeader.TractCustomTypes.IQ2S
    F16_SCALE_FIELDS = {"d": 0}

    @classmethod
    def quantize_blocks(cls, fp_blocks: np.ndarray) -> np.ndarray:
        f32 = np.float32
        n_blocks = fp_blocks.shape[0]
        x = fp_blocks.astype(f32, copy=False)
        sigma2 = f32(2.0) * ggml_seq_sum(x * x) / f32(QK_K)
        scales, grid_index, signs = _iq2_s_quantize_groups(
            x.reshape((n_blocks * 16, 16)), np.repeat(sigma2, 16)
        )
        scales = scales.reshape((n_blocks, 16))
        grid_index = grid_index.reshape((n_blocks, 32))
        max_scale = scales.max(axis=1)
        zero = max_scale == 0
        # nibbles fit against max / 31, stored d is that times 0.9875
        d = np.where(zero, f32(1.0), max_scale / f32(31.0)).astype(f32)
        nibbles = np.clip(
            ggml_nearest_int(
                f32(0.5) * ((f32(1.0) / d)[:, None] * scales - f32(1.0))
            ),
            0,
            15,
        ).astype(np.uint8)
        # two high index bits per octet, four octets per qh byte
        hi = (grid_index >> 8).reshape((n_blocks, 8, 4))
        qh = (hi << (2 * np.arange(4))).sum(axis=2)
        out = np.empty((n_blocks, cls.BLOCK_BYTES), dtype=np.uint8)
        # f16 overflow to inf is rejected by `check_quantized_blocks`
        with np.errstate(over="ignore"):
            d_f16 = (d * f32(0.9875)).astype(np.float16)
        out[:, 0:2] = d_f16.reshape((n_blocks, 1)).view(np.uint8)
        out[:, 2:34] = grid_index & 0xFF
        out[:, 34:66] = signs.reshape((n_blocks, 32))
        out[:, 66:74] = qh
        out[:, 74:82] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        out[zero] = 0
        return out

    @classmethod
    def dequantize_blocks(cls, u8_blocks: np.ndarray) -> np.ndarray:
        f32 = np.float32
        n_blocks = u8_blocks.shape[0]
        d = u8_blocks[:, 0:2].copy().view(np.float16).astype(f32)
        qs = u8_blocks[:, 2:34].astype(np.int64)
        signs = u8_blocks[:, 34:66]
        qh = u8_blocks[:, 66:74].astype(np.int64)
        sc = u8_blocks[:, 74:82]
        # group 2k scale is the low nibble of byte k, group 2k+1 the high
        nib = np.stack([sc & 0xF, sc >> 4], axis=2).reshape((n_blocks, 16))
        # same f32 ops order as tract: (d * (0.5 + nibble)) * 0.25
        db = d * (f32(0.5) + nib.astype(f32)) * f32(0.25)
        hi = (qh[:, :, None] >> (2 * np.arange(4))) & 3
        index = qs | (hi.reshape((n_blocks, 32)) << 8)
        mags = IQ2_S_CODE_BYTE[_iq2_s_unpack_levels(KGRID_2BIT_1024[index])]
        neg = (signs[:, :, None] >> np.arange(8, dtype=np.uint8)) & 1
        sign = np.where(neg == 1, f32(-1.0), f32(1.0))
        # (dl * magnitude) * sign, dl being the octet group scale
        out = np.repeat(db, 2, axis=1)[:, :, None] * mags * sign
        return out.reshape((n_blocks, cls.BLOCK_LEN))


def fp_to_tract_iq2_s(fp_tensor: torch.Tensor) -> QTensorTractIQ2_S:
    """Quantize float tensor to tract supported IQ2_S (ggml reference).

    Only use it on embedding weights (see `QTensorTractIQ2_S`).
    """
    qtensor = QTensorTractIQ2_S.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractIQ2_S)
    return qtensor
