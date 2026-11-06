"""tract ggml super-block quantization formats (K-quants, IQ formats).

These formats store each block of `BLOCK_LEN` values as an opaque
`BLOCK_BYTES` bytes chunk laid out exactly like llama.cpp `block_*`
structures. The QTensor u8 storage is these raw ggml bytes, written as is
in the tract `.dat` file.

Warning:
    tract (<= 0.23.10) has no matmul or convolution kernel for these formats:
    only embedding/gathering operations are supported.

"""

import operator
import typing as T
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant.base import QScheme, U8Compressor
from torch_to_nnef.tensor.quant.qtract import QTensorTractDat

# numbers of blocks processed at once by numpy (bounds peak memory)
GGML_CHUNK_BLOCKS = 1 << 14
# ggml K-quants super-block length
QK_K = 256


class QScaleGGMLBlock(QScheme):
    """Opaque ggml block scheme: the u8 storage are raw ggml block bytes.

    All the format logic lives in its `QTensorTractGGMLBlock` subclass.

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
            .reshape(-1, fmt.BLOCK_LEN)
        )
        out = np.empty((fp_blocks.shape[0], fmt.BLOCK_BYTES), dtype=np.uint8)
        for start in range(0, fp_blocks.shape[0], GGML_CHUNK_BLOCKS):
            end = start + GGML_CHUNK_BLOCKS
            chunk = fmt.quantize_blocks(fp_blocks[start:end])
            assert chunk.dtype == np.uint8, chunk.dtype
            assert chunk.shape == (len(fp_blocks[start:end]), fmt.BLOCK_BYTES)
            out[start:end] = chunk
        return torch.from_numpy(out)

    def dequantize(self, u8_tensor: torch.Tensor, target_dtype):
        """Dequantize a (n_blocks, BLOCK_BYTES) u8 tensor into flat fp.

        Like tract: values are computed in f32 then casted to target dtype.
        """
        fmt = self.fmt
        u8_blocks = (u8_tensor.detach().cpu().contiguous().numpy()).reshape(
            -1, fmt.BLOCK_BYTES
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


class QTensorTractGGMLBlock(QTensorTractDat):
    """Common base of tract ggml super-block formats.

    Storage is the raw ggml block bytes (u8 tensor of shape
    `(n_blocks, BLOCK_BYTES)`), blocks follow the row-major order of the
    float tensor, and are dumped as is after the tract `.dat` header.

    Subclass contract:
        - `OFFLOAD_STATE_TAG`: unique offload state tag
        - `FORMAT_NAME`: ggml format name (like "Q4_K")
        - `BLOCK_LEN`: number of values per block
        - `BLOCK_BYTES`: number of bytes per block
        - `ITEM_TYPE`: `DatBinHeader.TractCustomTypes` member
        - `MIN_TRACT_VERSION`: minimum tract version reading it
        - `quantize_blocks(fp32 (n, BLOCK_LEN)) -> u8 (n, BLOCK_BYTES)`
        - `dequantize_blocks(u8 (n, BLOCK_BYTES)) -> fp32 (n, BLOCK_LEN)`
          which must reproduce tract dequantization bit exactly.

    Warning:
        tract (<= 0.23.10) only supports these formats in
        embedding/gathering operations.

    """

    BLOCK_LEN: int
    BLOCK_BYTES: int
    ITEM_TYPE: DatBinHeader.TractCustomTypes
    MIN_TRACT_VERSION: T.Optional[str] = "0.23.10"
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
        self.check_fp_shape(fp_tensor.shape)
        super().__init__(
            fp_tensor,
            qscheme,
            dequant_to_dtype=dequant_to_dtype,
            u8_compressors=u8_compressors,
        )
        self.decompressed_shape = tuple(fp_tensor.shape)

    @classmethod
    def check_fp_shape(cls, shape):
        if len(shape) not in [2, 3, 4]:
            raise T2NErrorImpossibleQuantization(
                f"tract {cls.FORMAT_NAME} does only support weight "
                f"of shape 2d, 3d or 4d but found {tuple(shape)}"
            )
        # tract splits the tensor into shape[0] rows of blocks
        if reduce(operator.mul, shape[1:]) % cls.BLOCK_LEN != 0:
            raise T2NErrorImpossibleQuantization(
                f"tract {cls.FORMAT_NAME} does only support weight with "
                f"dim=1.. volume divisible by {cls.BLOCK_LEN} but "
                f"found {tuple(shape[1:])}"
            )

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
            block_len=cls.BLOCK_LEN,
            block_bytes=cls.BLOCK_BYTES,
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
    """Port of ggml `nearest_int`: round half to even (for |x| < 2**22)."""
    return np.rint(arr).astype(np.int32)


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
    sub = x.reshape(n_blocks * 8, 32).astype(f32, copy=False)
    av_x = np.sqrt(ggml_seq_sum(sub * sub) / f32(32.0))
    weights = av_x[:, None] + np.abs(sub)
    scales, levels, mins = ggml_make_qkx2_quants(
        sub, weights, nmax, rmin, 0.1, nstep
    )
    scales = scales.reshape(n_blocks, 8)
    mins = mins.reshape(n_blocks, 8)
    levels = levels.reshape(n_blocks, QK_K)
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
    d_f16 = (max_scale / f32(63.0)).astype(np.float16)
    dmin_f16 = (max_min / f32(63.0)).astype(np.float16)
    header = np.empty((n_blocks, 16), dtype=np.uint8)
    header[:, 0:2] = d_f16.reshape(n_blocks, 1).view(np.uint8)
    header[:, 2:4] = dmin_f16.reshape(n_blocks, 1).view(np.uint8)
    header[:, 4:16] = sc
    # final levels with f16 rounded super scales
    sc_u, m_u = ggml_unpack_scale_min_k4(sc)
    d = d_f16.astype(f32)[:, None] * sc_u.astype(f32)
    dm = dmin_f16.astype(f32)[:, None] * m_u.astype(f32)
    x8 = x.reshape(n_blocks, 8, 32).astype(f32, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        new_levels = np.clip(
            ggml_nearest_int((x8 + dm[:, :, None]) / d[:, :, None]), 0, nmax
        )
    levels = np.where(
        (d != 0)[:, :, None], new_levels, levels.reshape(n_blocks, 8, 32)
    ).astype(np.uint8)
    return header, levels.reshape(n_blocks, QK_K)


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

    @classmethod
    def quantize_blocks(cls, fp_blocks: np.ndarray) -> np.ndarray:
        n_blocks = fp_blocks.shape[0]
        header, levels = ggml_quant_k_scales_mins(
            fp_blocks, nmax=15, rmin=-1.0, nstep=20
        )
        # per 64 values: low nibbles from first 32, high from next 32
        chunks = levels.reshape(n_blocks, 4, 2, 32)
        qs = chunks[:, :, 0, :] | (chunks[:, :, 1, :] << 4)
        out = np.empty((n_blocks, cls.BLOCK_BYTES), dtype=np.uint8)
        out[:, :16] = header
        out[:, 16:] = qs.reshape(n_blocks, 128)
        return out

    @classmethod
    def dequantize_blocks(cls, u8_blocks: np.ndarray) -> np.ndarray:
        n_blocks = u8_blocks.shape[0]
        d_sc, dmin_m = ggml_dequant_k_scales_mins(u8_blocks[:, :16])
        qs = u8_blocks[:, 16:].reshape(n_blocks, 4, 1, 32)
        levels = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(
            n_blocks, 8, 32
        )
        # same f32 ops order as tract: (d * sc) * q - (dmin * m)
        out = d_sc[:, :, None] * levels.astype(np.float32) - dmin_m[:, :, None]
        return out.reshape(n_blocks, cls.BLOCK_LEN)


def fp_to_tract_q4_k(fp_tensor: torch.Tensor) -> QTensorTractQ4_K:
    """Quantize float tensor to tract supported Q4_K (ggml reference).

    Only use it on embedding weights (see `QTensorTractQ4_K`).
    """
    qtensor = QTensorTractQ4_K.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractQ4_K)
    return qtensor


class QTensorTractQ5_K(QTensorTractGGMLBlock):  # noqa: N801
    """Tract data format it serializes to: Q5_K (ggml `block_q5_K`).

    Each super-block of 256 values is stored as 176 bytes:
    f16 `d`, f16 `dmin`, 12 bytes of eight 6bit scales and mins
    (one per sub-block of 32 values), 32 bytes of level high bits,
    then 256 low 4bit levels (5.5bpw).
    Value is `d * scale * level - dmin * min` with 5bit levels.

    Quantization port llama.cpp `quantize_row_q5_K_ref` (no importance
    matrix), dequantization matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports Q5_K weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q5_k.v1"
    FORMAT_NAME = "Q5_K"
    BLOCK_LEN = QK_K
    BLOCK_BYTES = 176
    ITEM_TYPE = DatBinHeader.TractCustomTypes.Q5K

    # bit of `qh` holding the 5th bit of level `i` of each (pair, half)
    _QH_SHIFTS = np.arange(8, dtype=np.uint8).reshape(1, 4, 2, 1)

    @classmethod
    def quantize_blocks(cls, fp_blocks: np.ndarray) -> np.ndarray:
        n_blocks = fp_blocks.shape[0]
        header, levels = ggml_quant_k_scales_mins(
            fp_blocks, nmax=31, rmin=-0.5, nstep=15
        )
        # per 64 values: (pair, half) where half picks first or next 32
        chunks = levels.reshape(n_blocks, 4, 2, 32)
        low = chunks & 0xF
        ql = low[:, :, 0, :] | (low[:, :, 1, :] << 4)
        qh = np.bitwise_or.reduce(
            ((chunks >> 4) << cls._QH_SHIFTS).reshape(n_blocks, 8, 32),
            axis=1,
        )
        out = np.empty((n_blocks, cls.BLOCK_BYTES), dtype=np.uint8)
        out[:, :16] = header
        out[:, 16:48] = qh
        out[:, 48:] = ql.reshape(n_blocks, 128)
        return out

    @classmethod
    def dequantize_blocks(cls, u8_blocks: np.ndarray) -> np.ndarray:
        n_blocks = u8_blocks.shape[0]
        d_sc, dmin_m = ggml_dequant_k_scales_mins(u8_blocks[:, :16])
        qh = u8_blocks[:, 16:48].reshape(n_blocks, 1, 1, 32)
        ql = u8_blocks[:, 48:].reshape(n_blocks, 4, 1, 32)
        low = np.concatenate([ql & 0xF, ql >> 4], axis=2)
        high = (qh >> cls._QH_SHIFTS) & 1
        levels = (low | (high << 4)).reshape(n_blocks, 8, 32)
        # same f32 ops order as tract: (d * sc) * q - (dmin * m)
        out = d_sc[:, :, None] * levels.astype(np.float32) - dmin_m[:, :, None]
        return out.reshape(n_blocks, cls.BLOCK_LEN)


def fp_to_tract_q5_k(fp_tensor: torch.Tensor) -> QTensorTractQ5_K:
    """Quantize float tensor to tract supported Q5_K (ggml reference).

    Only use it on embedding weights (see `QTensorTractQ5_K`).
    """
    qtensor = QTensorTractQ5_K.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractQ5_K)
    return qtensor
