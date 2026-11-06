"""tract ggml K-quant and IQ block formats (Q4_K, IQ2_S).

Like the formats of `qtract`, the u8 storage holds one quantized code per
value and the qscheme holds the block parameters as torch tensors. tract
bytes, laid out exactly like llama.cpp `block_*` structures, are only made
when written.

Quantization ports the ggml reference quantizers in torch: every f32
operation runs in ggml order (sums accumulate sequentially, no fused
multiply add), so on CPU the written bytes match llama.cpp reference
quantizers built without floating point contraction (`-ffp-contract=off`;
compilers fusing multiply adds give slightly different bytes).

Warning:
    tract (<= 0.23.10) has no matmul or convolution kernel for these formats:
    only embedding/gathering operations are supported.

"""

import logging
import typing as T
from functools import lru_cache

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorMisuse,
    T2NErrorNotImplemented,
)
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant._iq2_s_grid import KGRID_2BIT_1024
from torch_to_nnef.tensor.quant.base import QScheme
from torch_to_nnef.tensor.quant.qtract import (
    QTensorTractBlockQuant,
    check_block_quant_input,
    check_f16_scales,
)

LOGGER = logging.getLogger(__name__)

# numbers of blocks processed at once (bounds peak memory)
GGML_CHUNK_BLOCKS = 1 << 14
# ggml K-quants super-block length
QK_K = 256
# blocks erased by f16 scale underflow are tolerated (with a warning) when
# their weights are below this fraction of the tensor largest weight
GGML_UNDERFLOW_REL_TOL = torch.finfo(torch.float16).eps


# ggml shared helpers ######################################################
# Ports of ggml scalar code over rows of torch tensors.


def ggml_f32(value: float) -> torch.Tensor:
    """f32 scalar, to keep ggml f32 arithmetic on python constants."""
    return torch.tensor(value, dtype=torch.float32)


def ggml_seq_sum(arr: torch.Tensor) -> torch.Tensor:
    """f32 sum over last dim accumulated sequentially, like ggml loops.

    `torch.sum` (pairwise) and CPU `torch.cumsum` (double accumulator)
    round differently.
    """
    acc = arr[..., 0]
    for i in range(1, arr.shape[-1]):
        acc = acc + arr[..., i]
    return acc


def ggml_nearest_int(arr: torch.Tensor) -> torch.Tensor:
    """Port of ggml `nearest_int` (f32 mantissa trick, round half to even).

    Exact like ggml for |x| <= 2**22 (ggml asserts it), and platform
    independent beyond, unlike a float to int32 cast.
    """
    biased = arr.to(torch.float32) + ggml_f32(12582912.0)
    return (biased.view(torch.int32) & 0x7FFFFF) - 0x400000


def ggml_f16_to_bytes(values: torch.Tensor) -> torch.Tensor:
    """(n,) f32 to (n, 2) u8: little endian f16 bytes (ggml_fp16_t)."""
    return values.to(torch.float16).reshape((-1, 1)).view(torch.uint8)


def ggml_make_qkx2_quants(
    x: torch.Tensor,
    weights: torch.Tensor,
    nmax: int,
    rmin: float,
    rdelta: float,
    nstep: int,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of ggml `make_qkx2_quants` vectorized over rows of `x` (n, group).

    Returns:
        (scale (n,), levels u8 (n, group) in [0, nmax], the_min (n,))
    """
    zero = ggml_f32(0.0)
    x = x.to(torch.float32)
    weights = weights.to(torch.float32)
    xmin = x.amin(dim=1)
    xmin = torch.where(xmin > 0, zero, xmin)
    xmax = x.amax(dim=1)
    sum_w = ggml_seq_sum(weights)
    sum_x = ggml_seq_sum(weights * x)
    degenerate = xmax == xmin
    iscale = ggml_f32(nmax) / (xmax - xmin)
    scale = ggml_f32(1.0) / iscale
    levels = torch.clamp(
        ggml_nearest_int(iscale[:, None] * (x - xmin[:, None])), 0, nmax
    )
    diff = scale[:, None] * levels.to(torch.float32) + xmin[:, None] - x
    best_error = ggml_seq_sum(weights * (diff * diff))
    for step in range(nstep + 1):
        iscale = (
            ggml_f32(rmin) + ggml_f32(rdelta) * ggml_f32(step) + ggml_f32(nmax)
        ) / (xmax - xmin)
        laux = torch.clamp(
            ggml_nearest_int(iscale[:, None] * (x - xmin[:, None])), 0, nmax
        )
        laux_f = laux.to(torch.float32)
        sum_l = ggml_seq_sum(weights * laux_f)
        sum_l2 = ggml_seq_sum(weights * laux_f * laux_f)
        sum_xl = ggml_seq_sum(weights * laux_f * x)
        det = sum_w * sum_l2 - sum_l * sum_l
        valid = det > 0
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / det
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det
        pos_min = this_min > 0
        this_min = torch.where(pos_min, zero, this_min)
        this_scale = torch.where(pos_min, sum_xl / sum_l2, this_scale)
        diff = this_scale[:, None] * laux_f + this_min[:, None] - x
        cur_error = ggml_seq_sum(weights * (diff * diff))
        update = valid & (cur_error < best_error) & ~degenerate
        levels = torch.where(update[:, None], laux, levels)
        best_error = torch.where(update, cur_error, best_error)
        scale = torch.where(update, this_scale, scale)
        xmin = torch.where(update, this_min, xmin)
    levels = torch.where(degenerate[:, None], 0, levels).to(torch.uint8)
    scale = torch.where(degenerate, zero, scale)
    return scale, levels, -xmin


class GGMLUnderflowCheck:
    """Blocks erased to zeros by f16 scales underflow, over a whole tensor.

    Like llama.cpp and tract quantizers, such blocks are kept as zeros
    (with a warning) when their weights are negligible against the tensor
    largest weight (below `GGML_UNDERFLOW_REL_TOL` of it, the f16
    precision): this is a quantization error below what the f16
    dequantized tensor can represent. Otherwise the weights are too small
    as a whole for the format f16 scales. A partial underflow (like Q4_K
    `d` rounded to 0 while `dmin` is not) is not reported: it only drops
    variations far below the remaining f16 scale, within the quantization
    error.

    Feed each chunk of blocks to `update`, then call `check`.
    """

    def __init__(self, format_name: str):
        self.format_name = format_name
        self.count = 0
        self.erased_absmax = 0.0
        self.absmax = 0.0

    def update(self, fp_blocks: torch.Tensor, deq_blocks: torch.Tensor):
        """Account (n, block) fp weights and their dequantized values.

        Scales overflow must be rejected before (`check_f16_scales`).
        """
        deq_absmax = deq_blocks.reshape(fp_blocks.shape).abs().amax(dim=1)
        fp_absmax = fp_blocks.abs().amax(dim=1)
        self.absmax = max(self.absmax, float(fp_absmax.max()))
        erased = (fp_absmax != 0) & (deq_absmax == 0)
        if erased.any():
            self.count += int(erased.sum())
            self.erased_absmax = max(
                self.erased_absmax, float(fp_absmax[erased].max())
            )

    def check(self):
        """Warn on negligible erased blocks, raise otherwise.

        Raises:
            T2NErrorImpossibleQuantization: when erased weights are not
                negligible.
        """
        if self.count == 0:
            return
        msg = (
            f"tract {self.format_name} block scales underflow f16: "
            f"{self.count} nonzero block(s) dequantize to zeros "
            f"(largest erased weight magnitude {self.erased_absmax:.3g}, "
            f"tensor largest {self.absmax:.3g})"
        )
        if self.erased_absmax > GGML_UNDERFLOW_REL_TOL * self.absmax:
            raise T2NErrorImpossibleQuantization(
                f"{msg}: weights magnitude is too small"
            )
        LOGGER.warning("%s: kept as zeros", msg)


def ggml_iter_blocks(fp_tensor: torch.Tensor, block_len: int):
    """Yield (start, f32 chunk (n, block_len)) of `fp_tensor` blocks.

    Blocks follow the row-major order; only one chunk at a time is casted
    to f32 (bounds peak memory).
    """
    blocks = fp_tensor.detach().reshape((-1, block_len))
    for start in range(0, blocks.shape[0], GGML_CHUNK_BLOCKS):
        chunk = blocks[start : start + GGML_CHUNK_BLOCKS]
        yield start, chunk.to(torch.float32).contiguous()


# ggml super-block scheme and tensor bases ##################################


class QScaleGGMLK(QScheme):
    """Common base of ggml super-block schemes (K-quants, IQ formats).

    Holds, as torch tensors with a leading super-block dimension (n,),
    a f16 super-block scale `d` and per sub-block of `SUB_LEN` values
    integer scales, plus format specific parameters (`PARAM_NAMES`).
    A code dequantizes in f32 to
    `sub_scales * levels(code) - sub_offsets` (tract op order), then is
    casted.

    Subclass contract:
        - `SUB_LEN`, `PARAM_NAMES` (constructor tensor arguments, in
          order), `F16_PARAMS` (checked for f16 overflow)
        - `_settings()`: non tensor constructor arguments
        - `fit_blocks(x, **settings) -> (scheme, codes)`: ggml reference
          quantizer of f32 (n, 256) super-blocks
        - `quantize_blocks(x) -> codes`: codes of (n, 256) super-blocks
          with this scheme parameters
        - `sub_scales()`, optionally `sub_offsets()` and `levels(codes)`
    """

    SUB_LEN: int
    PARAM_NAMES: T.Tuple[str, ...]
    F16_PARAMS: T.Tuple[str, ...] = ("d",)
    d: torch.Tensor

    @property
    def n_blocks(self) -> int:
        return self.d.shape[0]

    def _settings(self) -> T.Dict[str, T.Any]:
        return {}

    def params(self) -> T.Dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in self.PARAM_NAMES}

    def slice_blocks(self, start: int, end: int) -> "QScaleGGMLK":
        return type(self)(
            **{k: v[start:end] for k, v in self.params().items()},
            **self._settings(),
        )

    @classmethod
    def fit_blocks(
        cls, x: torch.Tensor, **settings
    ) -> T.Tuple["QScaleGGMLK", torch.Tensor]:
        """Fit ggml reference parameters and u8 codes of (n, 256) blocks."""
        raise T2NErrorNotImplemented()

    def quantize_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """U8 codes (n, 256) of f32 (n, 256) blocks with these params."""
        raise T2NErrorNotImplemented()

    def sub_scales(self) -> torch.Tensor:
        """F32 (n, 256 // SUB_LEN) scale of each sub-block."""
        raise T2NErrorNotImplemented()

    def sub_offsets(self) -> T.Optional[torch.Tensor]:
        """F32 (n, 256 // SUB_LEN) offset of each sub-block, if any."""
        return None

    def levels(self, codes: torch.Tensor) -> torch.Tensor:
        """F32 levels of (n, 256 // SUB_LEN, SUB_LEN) u8 codes."""
        return codes.to(torch.float32)

    @classmethod
    def calibrate(
        cls, fp_tensor: torch.Tensor, format_name: str, **settings
    ) -> T.Tuple["QScaleGGMLK", torch.Tensor]:
        """Fit ggml reference parameters and codes of `fp_tensor`.

        Returns:
            (scheme, u8 codes of `fp_tensor` shape)

        Raises:
            T2NErrorImpossibleQuantization: on f16 overflow of a
                `F16_PARAMS` scale, or not negligible f16 underflow.
        """
        chunks = []
        codes = torch.empty(
            fp_tensor.shape, dtype=torch.uint8, device=fp_tensor.device
        )
        flat_codes = codes.view((-1, QK_K))
        underflow = GGMLUnderflowCheck(format_name)
        for start, x in ggml_iter_blocks(fp_tensor, QK_K):
            chunk, chunk_codes = cls.fit_blocks(x, **settings)
            for name in cls.F16_PARAMS:
                check_f16_scales(
                    getattr(chunk, name), format_name, scale_name=f"`{name}`"
                )
            underflow.update(x, chunk.dequantize(chunk_codes, torch.float32))
            chunks.append(chunk)
            flat_codes[start : start + x.shape[0]] = chunk_codes
        underflow.check()
        scheme = cls(
            **{
                k: torch.cat([getattr(c, k) for c in chunks])
                for k in cls.PARAM_NAMES
            },
            **chunks[0]._settings(),
        )
        return scheme, codes

    def quantize_as_torch(self, fp_tensor):
        raise T2NErrorNotImplemented(
            f"native torch does not support {self.__class__.__name__}"
        )

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Codes of `fp_tensor` with these parameters."""
        assert fp_tensor.numel() == self.n_blocks * QK_K, fp_tensor.shape
        codes = torch.empty(
            fp_tensor.shape, dtype=torch.uint8, device=fp_tensor.device
        )
        flat_codes = codes.view((-1, QK_K))
        for start, x in ggml_iter_blocks(fp_tensor, QK_K):
            end = start + x.shape[0]
            flat_codes[start:end] = self.slice_blocks(
                start, end
            ).quantize_blocks(x)
        return codes

    def dequantize(self, u8_tensor: torch.Tensor, target_dtype):
        """F32 `sub_scales * levels - sub_offsets` like tract, then cast."""
        n_blocks = u8_tensor.numel() // QK_K
        assert n_blocks == self.n_blocks, (u8_tensor.shape, self.n_blocks)
        n_sub = QK_K // self.SUB_LEN
        codes = u8_tensor.to(self.d.device).reshape(
            (n_blocks, n_sub, self.SUB_LEN)
        )
        out = self.sub_scales()[:, :, None] * self.levels(codes)
        offsets = self.sub_offsets()
        if offsets is not None:
            out = out - offsets[:, :, None]
        return out.reshape(u8_tensor.shape).to(target_dtype)

    def __repr__(self) -> str:
        settings = "".join(f", {k}={v}" for k, v in self._settings().items())
        return f"{self.__class__.__name__}(n_blocks={self.n_blocks}{settings})"


class QTensorTractGGMLK(QTensorTractBlockQuant):
    """Common base of tract ggml super-block formats (`QScaleGGMLK`).

    Subclasses define `FORMAT_NAME`, `OFFLOAD_STATE_TAG`, `BLOCK_BYTES`,
    `ITEM_TYPE`, `QSCHEME_CLS`, optionally `_scheme_settings()`, and
    `_pack_blocks`.
    """

    BLOCK_LEN = QK_K
    MIN_TRACT_VERSION: T.Optional[str] = "0.23.10"
    ITEM_TYPE: DatBinHeader.TractCustomTypes
    QSCHEME_CLS: T.Type[QScaleGGMLK]

    @classmethod
    def _scheme_settings(cls) -> T.Dict[str, T.Any]:
        return {}

    @classmethod
    def from_fp(cls, fp_tensor: torch.Tensor) -> "QTensorTractGGMLK":
        """Quantize a float tensor with the ggml reference quantizer."""
        if isinstance(fp_tensor, torch.nn.Parameter):
            fp_tensor = fp_tensor.data
        check_block_quant_input(fp_tensor, cls.BLOCK_LEN, cls.FORMAT_NAME)
        with torch.no_grad():
            fp_tensor = fp_tensor.detach()
            qscheme, codes = cls.QSCHEME_CLS.calibrate(
                fp_tensor, cls.FORMAT_NAME, **cls._scheme_settings()
            )
            return cls._from_codes(fp_tensor, codes, qscheme, fp_tensor.dtype)

    @classmethod
    def _check_qscheme(cls, qscheme):
        super()._check_qscheme(qscheme)
        assert qscheme._settings() == cls._scheme_settings(), qscheme

    @classmethod
    def _item_type(cls, inference_target):
        return cls.ITEM_TYPE

    @classmethod
    def _block_params(cls, qscheme, n_blocks):
        assert qscheme.n_blocks == n_blocks, (qscheme.n_blocks, n_blocks)
        return qscheme.params()


# K-quants with mins (Q4_K, Q5_K) ##########################################


def ggml_pack_scale_min_k4(sc: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Pack 6bit (n, 8) sub-block scales and mins in ggml 12 bytes (n, 12).

    Inverse of ggml `get_scale_min_k4`.
    """
    return torch.cat(
        [
            sc[:, :4] | ((sc[:, 4:] >> 4) << 6),
            m[:, :4] | ((m[:, 4:] >> 4) << 6),
            (sc[:, 4:] & 0xF) | ((m[:, 4:] & 0xF) << 4),
        ],
        dim=1,
    )


def ggml_k_min_sub_weights(sub: torch.Tensor) -> torch.Tensor:
    """Weights of ggml K-quants `make_qkx2_quants` for (n, 32) sub-blocks."""
    av_x = torch.sqrt(ggml_seq_sum(sub * sub) / ggml_f32(32.0))
    return av_x[:, None] + sub.abs()


def ggml_quant_k_scales_mins(
    x: torch.Tensor, nmax: int, rmin: float, nstep: int
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q4_K/Q5_K super-block parameters (ggml `quantize_row_q*_K_ref`).

    Args:
        x: fp32 (n, 256) super-blocks
        nmax: maximum level (15 for Q4_K, 31 for Q5_K)
        rmin: `make_qkx2_quants` search start
        nstep: `make_qkx2_quants` search steps

    Returns:
        (d f16 (n,), dmin f16 (n,), 6bit scales u8 (n, 8),
         6bit mins u8 (n, 8))
    """
    zero = ggml_f32(0.0)
    n_blocks = x.shape[0]
    assert x.shape[1] == QK_K, x.shape
    sub = x.reshape((n_blocks * 8, 32)).to(torch.float32)
    scales, _, mins = ggml_make_qkx2_quants(
        sub, ggml_k_min_sub_weights(sub), nmax, rmin, 0.1, nstep
    )
    scales = scales.reshape((n_blocks, 8))
    mins = mins.reshape((n_blocks, 8))
    # comparisons as in ggml: -0.0 mins never win over 0.0
    max_scale = torch.where(scales > 0, scales, zero).amax(dim=1)
    max_min = torch.where(mins > 0, mins, zero).amax(dim=1)
    inv_scale = torch.where(max_scale > 0, ggml_f32(63.0) / max_scale, zero)
    inv_min = torch.where(max_min > 0, ggml_f32(63.0) / max_min, zero)
    # `as u8` wraps like in tract before the min
    ls = torch.clamp(
        ggml_nearest_int(inv_scale[:, None] * scales) & 0xFF, max=63
    ).to(torch.uint8)
    lm = torch.clamp(
        ggml_nearest_int(inv_min[:, None] * mins) & 0xFF, max=63
    ).to(torch.uint8)
    # f16 overflow to inf is rejected by the calibration checks
    d_f16 = (max_scale / ggml_f32(63.0)).to(torch.float16)
    dmin_f16 = (max_min / ggml_f32(63.0)).to(torch.float16)
    return d_f16, dmin_f16, ls, lm


class QScaleGGMLKMin(QScaleGGMLK):
    """ggml K-quant super-block scheme with mins (Q4_K, Q5_K).

    Per super-block of 256 values: f16 `d` and `dmin` (n,), and per
    sub-block of 32 values 6bit scales `sc` and mins `m` (n, 8). A code
    `q` in [0, nmax] dequantizes to `(d * sc) * q - (dmin * m)`.

    `nmax`, `rmin` and `nstep` are the format `make_qkx2_quants` settings:
    ggml keeps its search levels for sub-blocks whose `d * sc` is 0.
    """

    SUB_LEN = 32
    PARAM_NAMES = ("d", "dmin", "sc", "m")
    F16_PARAMS = ("d", "dmin")

    def __init__(
        self,
        d: torch.Tensor,
        dmin: torch.Tensor,
        sc: torch.Tensor,
        m: torch.Tensor,
        nmax: int,
        rmin: float,
        nstep: int,
    ):
        assert d.dtype == dmin.dtype == torch.float16, (d.dtype, dmin.dtype)
        assert sc.dtype == m.dtype == torch.uint8, (sc.dtype, m.dtype)
        assert sc.shape == m.shape == (d.shape[0], 8), (sc.shape, m.shape)
        self.d = d
        self.dmin = dmin
        self.sc = sc
        self.m = m
        self.nmax = nmax
        self.rmin = rmin
        self.nstep = nstep

    def _settings(self):
        return {"nmax": self.nmax, "rmin": self.rmin, "nstep": self.nstep}

    @classmethod
    def fit_blocks(cls, x, **settings):
        scheme = cls(
            *ggml_quant_k_scales_mins(
                x, settings["nmax"], settings["rmin"], settings["nstep"]
            ),
            **settings,
        )
        return scheme, scheme.quantize_blocks(x)

    def sub_scales(self):
        return self.d.to(torch.float32)[:, None] * self.sc.to(torch.float32)

    def sub_offsets(self):
        return self.dmin.to(torch.float32)[:, None] * self.m.to(torch.float32)

    def quantize_blocks(self, x):
        """Final ggml levels with the f16 rounded super-block scales."""
        n_blocks = x.shape[0]
        d_sc = self.sub_scales()
        x8 = x.reshape((n_blocks, 8, 32))
        levels = torch.clamp(
            ggml_nearest_int(
                (x8 + self.sub_offsets()[:, :, None]) / d_sc[:, :, None]
            ),
            0,
            self.nmax,
        )
        zero_d = d_sc == 0
        if zero_d.any():
            # ggml keeps `make_qkx2_quants` levels of these sub-blocks
            sub = x8[zero_d]
            _, search_levels, _ = ggml_make_qkx2_quants(
                sub,
                ggml_k_min_sub_weights(sub),
                self.nmax,
                self.rmin,
                0.1,
                self.nstep,
            )
            levels[zero_d] = search_levels.to(levels.dtype)
        return levels.reshape((n_blocks, QK_K)).to(torch.uint8)


class QTensorTractGGMLKMin(QTensorTractGGMLK):
    """Common base of Q4_K and Q5_K: `QScaleGGMLKMin` super-blocks.

    Subclasses define the `make_qkx2_quants` settings (`NMAX`, `RMIN`,
    `NSTEP`) and pack levels after the shared 16 bytes header (f16 `d`,
    f16 `dmin`, 12 bytes of 6bit scales and mins) in `_pack_levels`.
    """

    QSCHEME_CLS = QScaleGGMLKMin
    NMAX: int
    RMIN: float
    NSTEP: int
    # instance attribute: type notation for mypy
    qscheme: QScaleGGMLKMin

    @classmethod
    def _scheme_settings(cls):
        return {"nmax": cls.NMAX, "rmin": cls.RMIN, "nstep": cls.NSTEP}

    @classmethod
    def _pack_levels(cls, levels: torch.Tensor) -> torch.Tensor:
        """Pack u8 (n, 256) levels into the bytes after the header."""
        raise T2NErrorNotImplemented()

    @classmethod
    def _pack_blocks(cls, codes, params, inference_target) -> np.ndarray:
        header = torch.cat(
            [
                ggml_f16_to_bytes(params["d"]),
                ggml_f16_to_bytes(params["dmin"]),
                ggml_pack_scale_min_k4(params["sc"], params["m"]),
            ],
            dim=1,
        )
        return torch.cat([header, cls._pack_levels(codes)], dim=1).numpy()


# formats ##################################################################


# shared bases: block layout, ggml super-block, K-quant with mins
# pylint: disable-next=too-many-ancestors
class QTensorTractQ4_K(QTensorTractGGMLKMin):  # noqa: N801
    """Tract data format it serializes to: Q4_K (ggml `block_q4_K`).

    Each super-block of 256 values is stored as 144 bytes:
    f16 `d`, f16 `dmin`, 12 bytes of eight 6bit scales and mins
    (one per sub-block of 32 values), then 256 4bit levels (4.5bpw).
    Value is `d * scale * level - dmin * min`.

    Quantization ports llama.cpp `quantize_row_q4_K_ref` (no importance
    matrix), dequantization matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports Q4_K weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q4_k.v1"
    FORMAT_NAME = "Q4_K"
    BLOCK_BYTES = 144
    ITEM_TYPE = DatBinHeader.TractCustomTypes.Q4K
    NMAX = 15
    RMIN = -1.0
    NSTEP = 20

    @classmethod
    def _pack_levels(cls, levels):
        # per 64 values: low nibbles from first 32, high from next 32
        n_blocks = levels.shape[0]
        chunks = levels.reshape((n_blocks, 4, 2, 32))
        qs = chunks[:, :, 0, :] | (chunks[:, :, 1, :] << 4)
        return qs.reshape((n_blocks, 128))


def fp_to_tract_q4_k(fp_tensor: torch.Tensor) -> QTensorTractQ4_K:
    """Quantize float tensor to tract supported Q4_K (ggml reference).

    Only use it on embedding weights (see `QTensorTractQ4_K`).

    Raises:
        T2NErrorImpossibleQuantization: invalid dtype or shape, non finite
            values, block scales that overflow f16, or that underflow it
            for weights not negligible against the tensor largest one.
    """
    qtensor = QTensorTractQ4_K.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractQ4_K)
    return qtensor


# IQ2_S ####################################################################

IQ2_S_FORMAT_NAME = "IQ2_S"
# ggml `GROUP_MAX_EPS_IQ2_S`
IQ2_S_GROUP_MAX_EPS = 1e-8
# packed 2bit codes whose digits are at most 2 (43690 is eight 2 digits)
IQ2_S_KMAP_SIZE = 43692
# ggml `iq2s_grid` dequant bytes for levels 0, 1, 2
IQ2_S_CODE_BYTE = (8.0, 25.0, 43.0)
# IQ2_S u8 code: bits 1-0 hold the grid level, bit 2 negates the value
IQ2_S_LEVEL_MASK = 3
IQ2_S_SIGN_SHIFT = 2
# octets processed at once by the off grid neighbour search
IQ2_S_NEIGHBOUR_CHUNK = 1 << 14
# nearest shell padding widths (largest shell holds 70 grid points)
IQ2_S_SHELL_WIDTHS = (1, 2, 4, 8, 16, 32)


def _iq2_s_unpack_levels(codes: torch.Tensor) -> torch.Tensor:
    """Unpack (...) 2bit packed codes into i64 (..., 8) levels."""
    shifts = 2 * torch.arange(8, dtype=torch.int64, device=codes.device)
    return (codes.to(torch.int64)[..., None] >> shifts) & 3


def _iq2_s_pack_levels(levels: torch.Tensor) -> torch.Tensor:
    """Pack (..., 8) levels in [0, 2] into i64 (...) 2bit codes."""
    shifts = 2 * torch.arange(8, dtype=torch.int64, device=levels.device)
    return (levels.to(torch.int64) << shifts).sum(dim=-1)


@lru_cache(maxsize=1)
def _iq2_s_search_cpu() -> T.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    kgrid = torch.tensor(KGRID_2BIT_1024, dtype=torch.int64)
    grid_levels = _iq2_s_unpack_levels(kgrid)
    assert int(grid_levels.max()) <= 2
    grid_odd = 2 * grid_levels + 1
    kmap = torch.full((IQ2_S_KMAP_SIZE,), -1, dtype=torch.int64)
    kmap[kgrid] = torch.arange(1024, dtype=torch.int64)
    # snapped levels are clamped to [0, 2]: only those codes are looked up
    digits = 3 ** torch.arange(8, dtype=torch.int64)
    all_levels = (torch.arange(3**8, dtype=torch.int64)[:, None] // digits) % 3
    all_codes = _iq2_s_pack_levels(all_levels)
    off = kmap[all_codes] < 0
    odd = 2 * all_levels[off] + 1
    # integer squared distances to every grid point, exact
    d2 = torch.zeros((odd.shape[0], 1024), dtype=torch.int64)
    for k in range(8):
        d2 += (odd[:, k, None] - grid_odd[None, :, k]) ** 2
    is_best = d2 == d2.amin(dim=1, keepdim=True)
    max_shell = int(is_best.sum(dim=1).max())
    # shell kept in grid index order (ggml qsort order)
    rows, cols = is_best.nonzero(as_tuple=True)
    slots = is_best.to(torch.int64).cumsum(dim=1)[rows, cols] - 1
    shell = torch.full((is_best.shape[0], max_shell), -1, dtype=torch.int64)
    shell[rows, slots] = cols
    neighbours = torch.full((IQ2_S_KMAP_SIZE, max_shell), -1, dtype=torch.int64)
    neighbours[all_codes[off]] = shell
    shell_len = (neighbours >= 0).sum(dim=1)
    # grid indices fit int16 (cast at use)
    return (
        grid_levels,
        kmap.to(torch.int16),
        neighbours.to(torch.int16),
        shell_len.to(torch.int16),
    )


_IQ2_S_SEARCH_BY_DEVICE: T.Dict[
    str, T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _iq2_s_search(
    device: torch.device,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of tract `build_search` (ggml `iq2_s` init, nearest shell only).

    Returns:
        (grid levels i64 (1024, 8) in [0, 2],
         map i16 (KMAP_SIZE,): packed code to grid index or -1 off grid,
         neighbours i16 (KMAP_SIZE, max_shell): grid indices of the
         nearest shell in index order, padded with -1,
         shell lengths i16 (KMAP_SIZE,): valid neighbours per code),
        on `device`
    """
    key = str(device)
    if key not in _IQ2_S_SEARCH_BY_DEVICE:
        grid_levels, kmap, neighbours, shell_lens = _iq2_s_search_cpu()
        _IQ2_S_SEARCH_BY_DEVICE[key] = (
            grid_levels.to(device),
            kmap.to(device),
            neighbours.to(device),
            shell_lens.to(device),
        )
    return _IQ2_S_SEARCH_BY_DEVICE[key]


def _iq2_s_find_best_neighbour(
    codes: torch.Tensor,
    xval: torch.Tensor,
    waux: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Port of tract `find_best_neighbour` for off grid octets.

    Args:
        codes: i64 (m,) off grid packed codes
        xval: fp32 (m, 8) absolute values
        waux: fp32 (m, 8) weights
        scale: fp32 (m,) trial scale

    Returns:
        levels i64 (m, 8) of the best on grid neighbour
    """
    grid_levels, _, neighbours, shell_lens = _iq2_s_search(xval.device)
    grid_f = (2 * grid_levels + 1).to(torch.float32)
    shell_len = shell_lens[codes]
    out = torch.empty(
        (codes.shape[0], 8), dtype=torch.int64, device=codes.device
    )
    # shells are padded to a few widths to bound the wasted work
    prev_width = 0
    for width in IQ2_S_SHELL_WIDTHS + (neighbours.shape[1],):
        sel_all = torch.nonzero(
            (shell_len > prev_width) & (shell_len <= width)
        ).reshape(-1)
        prev_width = width
        for start in range(0, sel_all.numel(), IQ2_S_NEIGHBOUR_CHUNK):
            sel = sel_all[start : start + IQ2_S_NEIGHBOUR_CHUNK]
            cands = neighbours[codes[sel], :width].to(torch.int64)
            valid = cands >= 0
            cands_ix = torch.where(valid, cands, torch.zeros_like(cands))
            sc = scale[sel, None]
            # gathered once per chunk: (c, 8) values, (c, width, 8) points
            xs = xval[sel]
            ws = waux[sel]
            pts = grid_f[cands_ix]
            # tract order: d2 += (w * diff) * diff, sequentially
            d2 = torch.zeros(
                cands.shape, dtype=torch.float32, device=cands.device
            )
            for k in range(8):
                diff = sc * pts[:, :, k] - xs[:, k, None]
                d2 = d2 + ws[:, k, None] * diff * diff
            # tract strict `d2 < best` scan, first minimum wins on ties
            # (torch argmin keeps the first too, this scan mirrors tract
            # and is faster here); seeding `best` from the first
            # candidate instead of tract `f32::MAX` only differs for
            # inf/NaN distances (weights around 1e9 and above), rejected
            # anyway by the f16 `d` overflow check (from about 1e7)
            best_d2 = d2[:, 0]
            pick = cands[:, 0]
            for j in range(1, cands.shape[1]):
                update = valid[:, j] & (d2[:, j] < best_d2)
                best_d2 = torch.where(update, d2[:, j], best_d2)
                pick = torch.where(update, cands[:, j], pick)
            out[sel] = grid_levels[pick]
    return out


def _iq2_s_snap(
    xval: torch.Tensor,
    waux: torch.Tensor,
    inv_scale: torch.Tensor,
    scale: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Port of tract `snap_octet` vectorized over (m, 8) octets.

    Returns:
        (levels i64 (m, 8) in [0, 2] on grid, on_grid bool (m,) before
         the neighbour search)
    """
    _, kmap, _, _ = _iq2_s_search(xval.device)
    levels = torch.clamp(
        ggml_nearest_int(
            ggml_f32(0.5) * (inv_scale[:, None] * xval - ggml_f32(1.0))
        ),
        0,
        2,
    ).to(torch.int64)
    codes = _iq2_s_pack_levels(levels)
    on_grid = kmap[codes] >= 0
    off = torch.nonzero(~on_grid).reshape(-1)
    if off.numel():
        levels[off] = _iq2_s_find_best_neighbour(
            codes[off], xval[off], waux[off], scale[off]
        )
    return levels, on_grid


def _iq2_s_sums(
    weight: torch.Tensor, xval: torch.Tensor, levels: torch.Tensor
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Group `sumqx` and `sumq2` over (g, 16) values, like tract."""
    q = ggml_f32(2.0) * levels.to(torch.float32) + ggml_f32(1.0)
    sumqx = ggml_seq_sum(weight * xval * q)
    sumq2 = ggml_seq_sum(weight * q * q)
    return sumqx, sumq2


def _iq2_s_quantize_groups(
    xb: torch.Tensor, sigma2: torch.Tensor
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of tract IQ2_S `quantize_group` vectorized over (g, 16) groups.

    Returns:
        (scale fp32 (g,), grid levels i64 (g, 16) in [0, 2],
         negative signs bool (g, 16)); skipped groups (below
         `GROUP_MAX_EPS_IQ2_S`) get 0 everywhere.
    """
    zero = ggml_f32(0.0)
    one = ggml_f32(1.0)
    n_groups = xb.shape[0]
    weight = ggml_f32(0.25) * sigma2[:, None] + xb * xb
    waux = torch.sqrt(weight)
    neg = xb < 0
    xval = torch.where(neg, -xb, xb)
    max_x = xval.amax(dim=1)
    skipped = max_x < ggml_f32(IQ2_S_GROUP_MAX_EPS)
    max_x = torch.where(skipped, one, max_x)
    xval_o = xval.reshape((n_groups * 2, 8))
    waux_o = waux.reshape((n_groups * 2, 8))
    levels = torch.zeros((n_groups, 16), dtype=torch.int64, device=xb.device)
    best = torch.zeros((n_groups,), dtype=torch.float32, device=xb.device)
    scale = max_x / ggml_f32(5.0)
    on_grid = torch.ones((n_groups, 2), dtype=torch.bool, device=xb.device)
    # huge weights overflow sums like in tract (rejected afterwards)
    for step in range(-9, 10):
        inv_scale = (ggml_f32(5.0) + ggml_f32(step) * ggml_f32(0.1)) / max_x
        this_scale = one / inv_scale
        aux, on_aux = _iq2_s_snap(
            xval_o,
            waux_o,
            inv_scale.repeat_interleave(2),
            this_scale.repeat_interleave(2),
        )
        aux = aux.reshape((n_groups, 16))
        sumqx, sumq2 = _iq2_s_sums(weight, xval, aux)
        update = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        new_scale = sumqx / sumq2
        scale = torch.where(update, new_scale, scale)
        best = torch.where(update, new_scale * sumqx, best)
        levels = torch.where(update[:, None], aux, levels)
        on_grid = torch.where(
            update[:, None], on_aux.reshape((n_groups, 2)), on_grid
        )
    # off grid octets are snapped again with the best scale
    refit = (~on_grid).any(dim=1) & (scale > 0)
    redo = torch.nonzero((~on_grid & refit[:, None]).reshape(-1)).reshape(-1)
    if redo.numel():
        scale_o = scale.repeat_interleave(2)[redo]
        levels_o = levels.reshape((n_groups * 2, 8)).clone()
        levels_o[redo], _ = _iq2_s_snap(
            xval_o[redo], waux_o[redo], one / scale_o, scale_o
        )
        levels = levels_o.reshape((n_groups, 16))
        sumqx, sumq2 = _iq2_s_sums(weight, xval, levels)
        scale = torch.where(refit & (sumq2 > 0), sumqx / sumq2, scale)
    flip = scale < 0
    scale = torch.where(flip, -scale, scale)
    neg = neg ^ flip[:, None]
    scale = torch.where(skipped, zero, scale)
    levels = torch.where(skipped[:, None], torch.zeros_like(levels), levels)
    neg = neg & ~skipped[:, None]
    return scale, levels, neg


def ggml_quant_iq2_s(
    x: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IQ2_S super-blocks (ggml `quantize_row_iq2_s_ref`, no imatrix).

    Args:
        x: fp32 (n, 256) super-blocks

    Returns:
        (d f16 (n,), 4bit group scales u8 (n, 16), u8 codes (n, 256),
         see `QScaleGGMLIQ2S`)
    """
    n_blocks = x.shape[0]
    assert x.shape[1] == QK_K, x.shape
    x = x.to(torch.float32)
    sigma2 = ggml_f32(2.0) * ggml_seq_sum(x * x) / ggml_f32(float(QK_K))
    scales, levels, neg = _iq2_s_quantize_groups(
        x.reshape((n_blocks * 16, 16)), sigma2.repeat_interleave(16)
    )
    scales = scales.reshape((n_blocks, 16))
    max_scale = scales.amax(dim=1)
    zero = max_scale == 0
    # nibbles fit against max / 31, stored d is that times 0.9875
    d = torch.where(zero, ggml_f32(1.0), max_scale / ggml_f32(31.0))
    nibbles = torch.clamp(
        ggml_nearest_int(
            ggml_f32(0.5)
            * ((ggml_f32(1.0) / d)[:, None] * scales - ggml_f32(1.0))
        ),
        0,
        15,
    ).to(torch.uint8)
    # f16 overflow to inf is rejected by the calibration checks
    d_f16 = (d * ggml_f32(0.9875)).to(torch.float16)
    codes = (levels | (neg.to(torch.int64) << IQ2_S_SIGN_SHIFT)).reshape(
        (n_blocks, QK_K)
    )
    # an all zero super-block is stored as zeros, like in ggml
    d_f16 = torch.where(zero, torch.zeros_like(d_f16), d_f16)
    nibbles = torch.where(zero[:, None], torch.zeros_like(nibbles), nibbles)
    codes = torch.where(zero[:, None], torch.zeros_like(codes), codes)
    return d_f16, nibbles, codes.to(torch.uint8)


class QScaleGGMLIQ2S(QScaleGGMLK):
    """ggml IQ2_S super-block scheme.

    Per super-block of 256 values: f16 `d` (n,), and per group of 16
    values a 4bit scale `scales` (n, 16). Each u8 code holds the grid
    level of its value in bits 1-0 (0, 1 or 2 for magnitudes 8, 25 and
    43) and its sign in bit 2 (set for negative), so codes are in
    {0, 1, 2, 4, 5, 6}. A code dequantizes to
    `(d * (0.5 + scale) * 0.25) * (magnitude * sign)`.

    Each octet of 8 levels is a point of the ggml `kgrid_2bit_1024` grid
    (its index is looked up when packed). Levels come from a search that
    the parameters alone do not determine: `quantize_blocks` runs the
    deterministic ggml quantizer again and checks it finds these
    parameters.
    """

    SUB_LEN = 16
    PARAM_NAMES = ("d", "scales")

    def __init__(self, d: torch.Tensor, scales: torch.Tensor):
        assert d.dtype == torch.float16, d.dtype
        assert scales.dtype == torch.uint8, scales.dtype
        assert scales.shape == (d.shape[0], 16), (scales.shape, d.shape)
        self.d = d
        self.scales = scales

    @classmethod
    def fit_blocks(cls, x, **settings):
        d, scales, codes = ggml_quant_iq2_s(x)
        return cls(d, scales), codes

    def quantize_blocks(self, x):
        """Codes of the ggml quantizer, checked against these parameters.

        Raises:
            T2NErrorMisuse: `x` quantizes to other parameters.
        """
        d, scales, codes = ggml_quant_iq2_s(x)
        if not (
            torch.equal(d.view(torch.int16), self.d.view(torch.int16))
            and torch.equal(scales, self.scales)
        ):
            raise T2NErrorMisuse(
                f"{IQ2_S_FORMAT_NAME} scheme parameters were not calibrated "
                "on this tensor"
            )
        return codes

    def sub_scales(self):
        """F32 group scales `d * (0.5 + scale) * 0.25` like tract."""
        return (
            self.d.to(torch.float32)[:, None]
            * (ggml_f32(0.5) + self.scales.to(torch.float32))
            * ggml_f32(0.25)
        )

    def levels(self, codes):
        """F32 signed grid magnitudes (multiplying by +-1 is exact)."""
        code_bytes = torch.tensor(
            IQ2_S_CODE_BYTE, dtype=torch.float32, device=codes.device
        )
        mags = code_bytes[(codes & IQ2_S_LEVEL_MASK).to(torch.int64)]
        return torch.where((codes >> IQ2_S_SIGN_SHIFT) != 0, -mags, mags)


class QTensorTractIQ2_S(QTensorTractGGMLK):  # noqa: N801
    """Tract data format it serializes to: IQ2_S (ggml `block_iq2_s`).

    Each super-block of 256 values is stored as 82 bytes (2.5625bpw):
    f16 `d`, 32 low grid index bytes then 32 sign bytes (one per octet
    of 8 values), 8 bytes of 2bit high grid index parts, then sixteen
    4bit scales (one per group of 16 values).
    Each octet is a point of the ggml 1024 entries `kgrid_2bit_1024`
    grid (magnitudes in {8, 25, 43}), value is
    `d * (0.5 + scale) * 0.25 * magnitude * sign`.

    Its u8 storage holds one code per value (grid level and sign, see
    `QScaleGGMLIQ2S`), grid indices are only looked up when written.

    Quantization ports llama.cpp `quantize_row_iq2_s_ref` without
    importance matrix (grid nearest neighbour search with a weighted
    scale fit, about 2s per 16384 super-blocks on CPU), dequantization
    matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports IQ2_S weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_iq2_s.v1"
    FORMAT_NAME = IQ2_S_FORMAT_NAME
    BLOCK_BYTES = 82
    ITEM_TYPE = DatBinHeader.TractCustomTypes.IQ2S
    QSCHEME_CLS = QScaleGGMLIQ2S
    # instance attribute: type notation for mypy
    qscheme: QScaleGGMLIQ2S

    @classmethod
    def _pack_blocks(cls, codes, params, inference_target) -> np.ndarray:
        n_blocks = codes.shape[0]
        assert (codes >> (IQ2_S_SIGN_SHIFT + 1) == 0).all(), "bad IQ2_S code"
        levels = (codes & IQ2_S_LEVEL_MASK).reshape((n_blocks, 32, 8))
        _, kmap, _, _ = _iq2_s_search(codes.device)
        # levels 3 are off the grid (packed codes above the map, or -1)
        packed = _iq2_s_pack_levels(levels).clamp(max=IQ2_S_KMAP_SIZE - 1)
        grid_index = torch.where(
            (levels <= 2).all(dim=2),
            kmap[packed].to(torch.int64),
            torch.full_like(packed, -1),
        )
        assert (grid_index >= 0).all(), "IQ2_S octet is not on the grid"
        neg = (codes >> IQ2_S_SIGN_SHIFT).to(torch.int64)
        bits = torch.arange(8, dtype=torch.int64)
        signs = (neg.reshape((n_blocks, 32, 8)) << bits).sum(dim=2)
        # two high index bits per octet, four octets per qh byte
        hi = (grid_index >> 8).reshape((n_blocks, 8, 4))
        qh = (hi << (2 * torch.arange(4, dtype=torch.int64))).sum(dim=2)
        nib = params["scales"]
        return torch.cat(
            [
                ggml_f16_to_bytes(params["d"]),
                (grid_index & 0xFF).to(torch.uint8),
                signs.to(torch.uint8),
                qh.to(torch.uint8),
                nib[:, 0::2] | (nib[:, 1::2] << 4),
            ],
            dim=1,
        ).numpy()


def fp_to_tract_iq2_s(fp_tensor: torch.Tensor) -> QTensorTractIQ2_S:
    """Quantize float tensor to tract supported IQ2_S (ggml reference).

    Only use it on embedding weights (see `QTensorTractIQ2_S`).

    Raises:
        T2NErrorImpossibleQuantization: invalid dtype or shape, non finite
            values, a super-block scale `d` that overflows f16, or that
            underflows it for weights not negligible against the tensor
            largest one.
    """
    qtensor = QTensorTractIQ2_S.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractIQ2_S)
    return qtensor
