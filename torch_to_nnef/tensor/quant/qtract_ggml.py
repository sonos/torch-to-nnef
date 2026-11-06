"""tract ggml reference block formats (Q8_0, K-quants, IQ formats).

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

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant.base import QScalePerGroupF16, QScheme
from torch_to_nnef.tensor.quant.qtract import (
    QTensorTractBlockQuant,
    QTensorTractScalePerGroupF16,
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
# ggml `GROUP_MAX_EPS`: groups below it are all zeros
GGML_GROUP_MAX_EPS = 1e-15


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


def ggml_sqrt(arr: torch.Tensor) -> torch.Tensor:
    """Correctly rounded f32 square root (IEEE `sqrtf`), on every platform.

    `torch.sqrt` is not: x86 builds compute it with MKL vector math, accurate
    to about 1 ulp only, and its result depends on the CPU model. The f32
    rounding of the f64 root is fixed up with exact f64 checks: the midpoint
    of two consecutive f32 has at most 25 significant bits, so its f64 square
    is exact (and never equals a f32, so there are no ties).
    """
    x = arr.to(torch.float32)
    device = x.device
    if device.type == "mps":
        # no f64 on mps: fix up on cpu
        x = x.cpu()
    xd = x.to(torch.float64)
    root = torch.sqrt(xd).to(torch.float32)
    up = torch.nextafter(root, torch.full_like(root, float("inf")))
    down = torch.nextafter(root, torch.zeros_like(root))
    root_d = root.to(torch.float64)
    mid_up = (root_d + up.to(torch.float64)) * 0.5
    mid_down = (root_d + down.to(torch.float64)) * 0.5
    out = torch.where(mid_up * mid_up < xd, up, root)
    out = torch.where(mid_down * mid_down > xd, down, out)
    return out.to(device)


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


# Q8_0 ######################################################################


def round_half_away(arr: torch.Tensor) -> torch.Tensor:
    """C `roundf` (and rust `f32::round`): round half away from zero.

    Exact in f32, unlike `floor(|x| + 0.5)` (0.49999997 rounds to 1).
    """
    mag = arr.abs()
    rounded = mag.floor()
    rounded = rounded + ((mag - rounded) >= 0.5).to(rounded.dtype)
    return torch.copysign(rounded, arr)


def _q8_0_codes(x: torch.Tensor, d_f32: torch.Tensor) -> torch.Tensor:
    """U8 codes `roundf(x / d) + 128` of f32 (n, 32) blocks, f32 (n, 1) d."""
    inv_d = torch.where(d_f32 == 0, torch.zeros_like(d_f32), 1.0 / d_f32)
    codes = round_half_away(x * inv_d).clamp(-127, 127) + 128
    return codes.to(torch.uint8)


class QScaleQ8_0(QScalePerGroupF16):  # noqa: N801
    """Q8_0 f16 scale `d` per block of 32 (ggml `block_q8_0`).

    Codes are `q + 128` with `q` in [-127, 127], dequantized in f32 as
    `d * q` (like tract), then casted.

    Like llama.cpp `quantize_row_q8_0_ref`, `q = roundf(x / d_f32)` with
    `d_f32 = absmax / 127` of the block in f32, before its f16 rounding.
    `quantize_as_u8` recomputes `d_f32` from each block of the given
    tensor and uses it whenever it rounds to the stored f16 scale: this is
    the case of `fp_to_tract_q8_0_with_min_max_calibration` (positive
    scales). Otherwise, the block falls back to the stored f16 scale
    itself: scales set by an external calibration (GPTQ, clipping, a
    negative or any other scale than f16(absmax / 127) of the block), or
    a tensor quantized with scales calibrated on other weights.
    """

    def quantize_as_u8(self, fp_tensor):
        scale = self.scale.reshape((-1, 1))
        assert scale.shape[0] * self.group_size == fp_tensor.numel(), (
            fp_tensor.shape,
            scale.shape,
        )
        codes = torch.empty(
            fp_tensor.shape, dtype=torch.uint8, device=fp_tensor.device
        )
        flat_codes = codes.view((-1, self.group_size))
        for start, x in ggml_iter_blocks(fp_tensor, self.group_size):
            end = start + x.shape[0]
            chunk_scale = scale[start:end].to(x.device)
            d_ggml = x.abs().amax(dim=1, keepdim=True) / 127.0
            d_f32 = torch.where(
                d_ggml.to(torch.float16) == chunk_scale,
                d_ggml,
                chunk_scale.to(torch.float32),
            )
            flat_codes[start:end] = _q8_0_codes(x, d_f32)
        return codes

    def dequantize(self, u8_tensor, target_dtype):
        """F32 `d * q` like tract, then cast."""
        scale = self.scale.to(torch.float32).reshape((-1, 1))
        codes = u8_tensor.reshape((-1, self.group_size)).to(
            device=scale.device, dtype=torch.float32
        )
        return (
            ((codes - 128.0) * scale).reshape(u8_tensor.shape).to(target_dtype)
        )


class QTensorTractQ8_0(QTensorTractScalePerGroupF16):  # noqa: N801
    """Tract data format it serializes to: Q8_0 (ggml `block_q8_0`).

    Each group of 32 values is stored as 34 bytes:
    f16 scale `d`, then 32 int8 (8.5bpw). Value is `d * q`.

    Use `fp_to_tract_q8_0_with_min_max_calibration` to construct it:
    quantization matches llama.cpp `quantize_row_q8_0_ref` bytes,
    dequantization matches tract bit exactly. External calibrations
    provide their scales with a `QScaleQ8_0`.

    Warning:
        tract (<= 0.23.10) only supports Q8_0 weights in embedding/gathering
        operations: there is no matmul or convolution kernel for it yet.
        Export requires tract >= 0.23.10; older targets warn and are rejected.

    """

    # v3: Q8_0 is unreleased, states of earlier development versions of
    # this format (v1, v2) are incompatible and rejected
    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q8_0.v3"
    FORMAT_NAME = "Q8_0"
    N_BITS = 8
    BLOCK_BYTES = 34
    RESTRICTED_RANGE = True
    QSCHEME_CLS = QScaleQ8_0
    MIN_TRACT_VERSION = "0.23.10"
    VERSION_GATE_NOTE = "Older tract reads item type 0x3080 as Q8_1."

    @classmethod
    def _item_type(cls, inference_target):
        return DatBinHeader.TractCustomTypes.Q80

    @classmethod
    def _pack_blocks(cls, codes, params, inference_target):
        n_blocks = codes.shape[0]
        b_all = np.empty((n_blocks, cls.BLOCK_BYTES), dtype=np.uint8)
        b_all[:, :2] = (
            params["scale"].numpy().astype("<f2").view(np.uint8).reshape(-1, 2)
        )
        b_all[:, 2:] = (
            (codes.to(torch.int16) - 128).to(torch.int8).numpy().view(np.uint8)
        )
        return b_all


def fp_to_tract_q8_0_with_min_max_calibration(
    fp_tensor: torch.Tensor, percentile: float = 1.0
) -> QTensorTractQ8_0:
    """Min-Max method to quantize float tensor to tract supported Q8_0.

    Block scale is `absmax / 127` (ggml reference). With `percentile < 1`
    each block is first clamped to its `percentile` absolute value.
    Only use it on embedding weights (see `QTensorTractQ8_0`).

    Raises:
        T2NErrorImpossibleQuantization: invalid dtype or shape, non finite
            values, block scales that overflow f16, or that underflow it
            for weights not negligible against the tensor largest one.
    """
    assert 0.0 < percentile <= 1.0, percentile
    if isinstance(fp_tensor, torch.nn.Parameter):
        fp_tensor = fp_tensor.data
    block_len = QTensorTractQ8_0.BLOCK_LEN
    format_name = QTensorTractQ8_0.FORMAT_NAME
    check_block_quant_input(fp_tensor, block_len, format_name)
    with torch.no_grad():
        fp_tensor = fp_tensor.detach()
        n_blocks = fp_tensor.numel() // block_len
        scale = torch.empty(
            (n_blocks, 1), dtype=torch.float16, device=fp_tensor.device
        )
        codes = torch.empty(
            fp_tensor.shape, dtype=torch.uint8, device=fp_tensor.device
        )
        flat_codes = codes.view((-1, block_len))
        underflow = GGMLUnderflowCheck(format_name)
        for start, x in ggml_iter_blocks(fp_tensor, block_len):
            end = start + x.shape[0]
            if percentile < 1.0:
                bound = torch.quantile(x.abs(), percentile, dim=1)[:, None]
                x = torch.maximum(torch.minimum(x, bound), -bound)
            d_f32 = x.abs().amax(dim=1, keepdim=True) / 127.0
            d_f16 = d_f32.to(torch.float16)
            check_f16_scales(d_f16, format_name, scale_name="`d`")
            chunk_codes = _q8_0_codes(x, d_f32)
            underflow.update(
                x, (chunk_codes.to(torch.float32) - 128.0) * d_f16.float()
            )
            scale[start:end] = d_f16
            flat_codes[start:end] = chunk_codes
        underflow.check()
        qscheme = QScaleQ8_0(
            group_size=block_len,
            scale=scale,
            n_bits=QTensorTractQ8_0.N_BITS,
            restricted_range=QTensorTractQ8_0.RESTRICTED_RANGE,
        )
        qtensor = QTensorTractQ8_0._from_codes(
            fp_tensor, codes, qscheme, fp_tensor.dtype
        )
        assert isinstance(qtensor, QTensorTractQ8_0)
        return qtensor


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
    av_x = ggml_sqrt(ggml_seq_sum(sub * sub) / ggml_f32(32.0))
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


# Q6_K ######################################################################


def ggml_first_absmax(
    x: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Signed value of max |x| per row (first one on ties), and that |x|.

    Explicit scan like ggml loops (`argmax` ties differ across platforms).
    """
    signed = torch.zeros_like(x[:, 0])
    amax = torch.zeros_like(x[:, 0])
    for i in range(x.shape[1]):
        ax = x[:, i].abs()
        update = ax > amax
        amax = torch.where(update, ax, amax)
        signed = torch.where(update, x[:, i], signed)
    return signed, amax


def ggml_make_qx_quants(
    x: torch.Tensor, nmax: int
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Port of ggml `make_qx_quants` (`rmse_type` 1) over rows of `x`.

    Each value is weighted by its square.

    Returns:
        (scale (n,), levels u8 (n, group) in [0, 2 * nmax))
    """
    zero = ggml_f32(0.0)
    x = x.to(torch.float32)
    vmax, amax = ggml_first_absmax(x)
    empty = amax < ggml_f32(GGML_GROUP_MAX_EPS)
    w = x * x
    wx = w * x

    def levels_for(iscale):
        return torch.clamp(
            ggml_nearest_int(iscale[:, None] * x), -nmax, nmax - 1
        )

    def fit(lev):
        # ggml order: `(w * x) * l` and `(w * l) * l`
        lev_f = lev.to(torch.float32)
        sumlx = ggml_seq_sum(wx * lev_f)
        suml2 = ggml_seq_sum(w * lev_f * lev_f)
        return sumlx, suml2

    # empty rows divide by 0: their results are discarded below
    levels = levels_for(ggml_f32(-nmax) / vmax)
    sumlx, suml2 = fit(levels)
    scale = torch.where(suml2 != 0, sumlx / suml2, zero)
    best = scale * sumlx
    for step in range(-9, 10):
        if step == 0:
            continue
        iscale = -(ggml_f32(nmax) + ggml_f32(0.1) * ggml_f32(step)) / vmax
        cand = levels_for(iscale)
        sumlx, suml2 = fit(cand)
        update = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        levels = torch.where(update[:, None], cand, levels)
        scale = torch.where(update, sumlx / suml2, scale)
        best = torch.where(update, scale * sumlx, best)
    levels = torch.where(empty[:, None], 0, levels + nmax).to(torch.uint8)
    scale = torch.where(empty, zero, scale)
    return scale, levels


def ggml_quant_q6_k_scales(
    x: torch.Tensor,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """Q6_K super-block parameters (ggml `quantize_row_q6_K_ref`).

    Args:
        x: fp32 (n, 256) super-blocks

    Returns:
        (d f16 (n,), i8 sub-block scales (n, 16))
    """
    zero = ggml_f32(0.0)
    n_blocks = x.shape[0]
    assert x.shape[1] == QK_K, x.shape
    scales, _ = ggml_make_qx_quants(x.reshape((n_blocks * 16, 16)), 32)
    scales = scales.reshape((n_blocks, 16))
    max_scale, max_abs_scale = ggml_first_absmax(scales)
    # ggml writes all zeros super-blocks below this
    valid = max_abs_scale >= ggml_f32(GGML_GROUP_MAX_EPS)
    iscale = torch.where(valid, ggml_f32(-128.0) / max_scale, zero)
    # f16 overflow to inf is rejected by the calibration checks
    d_f16 = torch.where(valid, ggml_f32(1.0) / iscale, zero).to(torch.float16)
    sc = torch.clamp(ggml_nearest_int(iscale[:, None] * scales), max=127)
    sc = torch.where(valid[:, None], sc, 0).to(torch.int8)
    return d_f16, sc


class QScaleGGMLK6(QScaleGGMLK):
    """ggml Q6_K symmetric super-block scheme.

    Per super-block of 256 values: f16 `d` (n,), and per sub-block of
    16 values i8 scales `sc` (n, 16). A code `q` in [0, 63] dequantizes
    to `(d * sc) * (q - 32)`.

    ggml keeps its `make_qx_quants` search levels for sub-blocks whose
    `d * sc` is 0, and zeros for super-blocks without any scale.
    """

    SUB_LEN = 16
    PARAM_NAMES = ("d", "sc")

    def __init__(self, d: torch.Tensor, sc: torch.Tensor):
        assert d.dtype == torch.float16, d.dtype
        assert sc.dtype == torch.int8, sc.dtype
        assert sc.shape == (d.shape[0], 16), (sc.shape, d.shape)
        self.d = d
        self.sc = sc

    @classmethod
    def fit_blocks(cls, x, **settings):
        scheme = cls(*ggml_quant_q6_k_scales(x))
        return scheme, scheme.quantize_blocks(x)

    def sub_scales(self):
        return self.d.to(torch.float32)[:, None] * self.sc.to(torch.float32)

    def levels(self, codes):
        return codes.to(torch.float32) - 32

    def quantize_blocks(self, x):
        """Final ggml levels with the f16 rounded super-block scale."""
        n_blocks = x.shape[0]
        d_sc = self.sub_scales()
        x16 = x.reshape((n_blocks, 16, 16))
        levels = (
            torch.clamp(ggml_nearest_int(x16 / d_sc[:, :, None]), -32, 31) + 32
        )
        zero_d = d_sc == 0
        if zero_d.any():
            # ggml keeps `make_qx_quants` levels of these sub-blocks
            search_scales, search_levels = ggml_make_qx_quants(x16[zero_d], 32)
            levels[zero_d] = search_levels.to(levels.dtype)
            # d == 0 super-blocks were fully searched: those without any
            # scale are written as zeros
            scales = torch.zeros_like(d_sc)
            scales[zero_d] = search_scales
            _, max_abs_scale = ggml_first_absmax(scales)
            empty = (self.d == 0) & (
                max_abs_scale < ggml_f32(GGML_GROUP_MAX_EPS)
            )
            levels[empty] = 0
        return levels.reshape((n_blocks, QK_K)).to(torch.uint8)


class QTensorTractQ6_K(QTensorTractGGMLK):  # noqa: N801
    """Tract data format it serializes to: Q6_K (ggml `block_q6_K`).

    Each super-block of 256 values is stored as 210 bytes:
    128 bytes of low 4 bits and 64 bytes of high 2 bits of 256 6bit
    levels, sixteen i8 scales (one per sub-block of 16 values), then
    f16 `d` (6.5625bpw). It is symmetric: value is
    `d * scale * (level - 32)`.

    Quantization ports llama.cpp `quantize_row_q6_K_ref` (no importance
    matrix), dequantization matches tract bit exactly.

    Warning:
        tract (<= 0.23.10) only supports Q6_K weights in
        embedding/gathering operations: there is no matmul or convolution
        kernel for it yet.

    """

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_q6_k.v1"
    FORMAT_NAME = "Q6_K"
    BLOCK_BYTES = 210
    ITEM_TYPE = DatBinHeader.TractCustomTypes.Q6K
    QSCHEME_CLS = QScaleGGMLK6
    # instance attribute: type notation for mypy
    qscheme: QScaleGGMLK6

    @classmethod
    def _pack_blocks(cls, codes, params, inference_target) -> np.ndarray:
        n_blocks = codes.shape[0]
        # per half of 128: l[k] = levels[i + 32 * k] for i in 0..32
        lv = codes.reshape((n_blocks, 2, 4, 32))
        ql = torch.cat(
            [
                (lv[:, :, 0] & 0xF) | ((lv[:, :, 2] & 0xF) << 4),
                (lv[:, :, 1] & 0xF) | ((lv[:, :, 3] & 0xF) << 4),
            ],
            dim=2,
        )
        qh = (
            (lv[:, :, 0] >> 4)
            | ((lv[:, :, 1] >> 4) << 2)
            | ((lv[:, :, 2] >> 4) << 4)
            | ((lv[:, :, 3] >> 4) << 6)
        )
        return torch.cat(
            [
                ql.reshape((n_blocks, 128)),
                qh.reshape((n_blocks, 64)),
                params["sc"].view(torch.uint8),
                ggml_f16_to_bytes(params["d"]),
            ],
            dim=1,
        ).numpy()


def fp_to_tract_q6_k(fp_tensor: torch.Tensor) -> QTensorTractQ6_K:
    """Quantize float tensor to tract supported Q6_K (ggml reference).

    Only use it on embedding weights (see `QTensorTractQ6_K`).

    Raises:
        T2NErrorImpossibleQuantization: invalid dtype or shape, non finite
            values, a super-block scale `d` that overflows f16, or that
            underflows it for weights not negligible against the tensor
            largest one.
    """
    qtensor = QTensorTractQ6_K.from_fp(fp_tensor)
    assert isinstance(qtensor, QTensorTractQ6_K)
    return qtensor
