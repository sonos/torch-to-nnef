"""Tract raw ggml block formats (K-quants): packing and tract execution."""

import hashlib
from functools import partial

import pytest
import torch
from torch import nn

from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
    quantize_weights_ggml_IQ2_S,
    quantize_weights_ggml_Q4_K,
)
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorMisuse,
)
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant import (
    QTensor,
    QTensorTractIQ2_S,
    QTensorTractQ4_K,
    fp_to_tract_iq2_s,
    fp_to_tract_q4_0_with_min_max_calibration,
    fp_to_tract_q4_k,
)
from torch_to_nnef.tensor.quant._iq2_s_grid import KGRID_2BIT_1024
from torch_to_nnef.tensor.quant.qtract_ggml import (
    _iq2_s_search,
    ggml_nearest_int,
    ggml_seq_sum,
)

from .test_qtensor import check_tensor_in_nnef_archive
from .utils import (
    TRACT_INFERENCES_TO_TESTS_EXACT,
    check_model_io_test,
    check_offload_state_write,
    check_rejects_old_tract,
    old_tract_versions,
    skipif_unsupported_qtensor,
    supported_tract_versions,
    tract_target,
)

pytestmark = skipif_unsupported_qtensor


def _ggml_ramp(length: int, zeros: int) -> torch.Tensor:
    """Ramp of tract `ggml_k` tests over [-2, 2] with leading zeros."""
    return torch.tensor(
        [
            0.0 if i < zeros else (((i * 17 + 3) % 41) - 20.0) / 10.0
            for i in range(length)
        ],
        dtype=torch.float32,
    )


def _f32_bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32)) & 0xFFFFFFFF


def _tract_bytes(qtensor) -> bytes:
    return qtensor._build_binary_dat_content(TractNNEF.latest())


def _check_ggml_reference(fp_to_qtensor, data, expected_hex: str, values):
    """Check packed bytes and dequant f32 bits against llama.cpp ones."""
    qtensor = fp_to_qtensor(data.reshape(1, -1))
    assert _tract_bytes(qtensor).hex() == expected_hex
    deq = qtensor.decompress()
    for ix, bits in values:
        assert _f32_bits(deq[0, ix]) == bits, ix


def _check_ggml_dat_content_layout(tmp_path, qtensor, item_type):
    """Check a ggml block `.dat` is the header + packed ggml blocks."""
    qtensor_cls = type(qtensor)
    qtensor.write_in_file(tmp_path, "weight", TractNNEF.latest())
    with (tmp_path / "weight.dat").open("rb") as fh:
        header = DatBinHeader.from_dat_file(fh)
        content = fh.read()
    n_blocks = qtensor.numel() // qtensor_cls.BLOCK_LEN
    assert header.torch_dtype_or_custom == item_type
    assert header.item_type_vendor == DatBinHeader.TRACT_ITEM_TYPE_VENDOR
    assert header.dims[: header.rank] == list(qtensor.shape)
    assert header.bits_per_item == 2**32 - 1
    assert header.data_size_bytes == len(content)
    assert len(content) == n_blocks * qtensor_cls.BLOCK_BYTES
    assert content == _tract_bytes(qtensor)


def _check_ggml_decompress_matches_reference(fp_to_qtensor, fp_tensor):
    """Codes have the float shape, decompress casts the f32 dequant."""
    qtensor = fp_to_qtensor(fp_tensor)
    assert qtensor.shape == fp_tensor.shape
    assert qtensor.numel() == fp_tensor.numel()
    assert qtensor.decompress_to_u8().shape == fp_tensor.shape
    # calibration codes (kept by `_from_codes`) are the scheme codes
    assert torch.equal(
        qtensor.decompress_to_u8(),
        qtensor.qscheme.quantize_as_u8(fp_tensor.float()),
    )
    # always quantized from f32, then dequantized in f32 and casted
    f32_qtensor = fp_to_qtensor(fp_tensor.float())
    assert torch.equal(
        qtensor.decompress_to_u8(), f32_qtensor.decompress_to_u8()
    )
    deq = qtensor.decompress()
    assert deq.dtype == fp_tensor.dtype
    assert torch.equal(deq, f32_qtensor.decompress().to(fp_tensor.dtype))


def _ggml_lcg_cases(n_values: int, seed: int = 7):
    """Deterministic inputs (no torch RNG) of the llama.cpp fixtures."""
    state = seed
    uni = []
    for _ in range(n_values * 4):
        state = (state * 1103515245 + 12345) % (1 << 31)
        uni.append(state / (1 << 31))
    u = torch.tensor(uni, dtype=torch.float64).reshape(4, n_values)
    normal = (u[0] + u[1] + u[2] - 1.5) * 2.0
    cases = {
        "normal": normal,
        "sparse": torch.where(u[3] < 0.1, normal, torch.zeros_like(normal)),
        "positive": u[2],
        "heavy": normal * normal * normal,
    }
    return {k: v.to(torch.float32) for k, v in cases.items()}


# SHA-256 of llama.cpp `quantize_row_*_ref` bytes (ggml-quants.c built
# with -ffp-contract=off) of `_ggml_lcg_cases(64 * 256)`: unlike the ramp
# vectors, these cover ties of the scale searches
GGML_LLAMA_CPP_SHA256 = {
    "Q4_K": {
        "normal": (
            "298f743eccedd95ff88eb1c633eb2fdd0e11d6797e7cb8a4d893e6eea4739872"
        ),
        "sparse": (
            "2b548c844ca0caf3fbe8de8a255c40779fb961421379e03e3c8bd4ec94fb78a5"
        ),
        "positive": (
            "43c19b60612d4d86d042ef2501aff879cadc900023d37f9166caeafca1ae14e6"
        ),
        "heavy": (
            "5dfcb461fb9367b40858fc2664809605b7a73abedcbe682402ab127ee8424449"
        ),
    },
}


def _check_ggml_llama_cpp_hashes(fp_to_qtensor, format_name: str):
    for name, values in _ggml_lcg_cases(64 * 256).items():
        qtensor = fp_to_qtensor(values.reshape(64, 256))
        digest = hashlib.sha256(_tract_bytes(qtensor)).hexdigest()
        assert digest == GGML_LLAMA_CPP_SHA256[format_name][name], name


def _ggml_invalid_weights(block_len: int):
    """Weights any ggml block format must reject (shape, dtype, values)."""
    return [
        pytest.param(torch.ones(block_len), id="rank1"),
        pytest.param(torch.ones(1, 1, 1, 1, block_len), id="rank5"),
        pytest.param(torch.ones(2, block_len // 2), id="incomplete_block"),
        pytest.param(
            torch.ones(2, block_len, dtype=torch.int32), id="non_floating"
        ),
        pytest.param(torch.empty(0, block_len), id="empty"),
        pytest.param(torch.full((2, block_len), float("nan")), id="nan"),
        pytest.param(torch.full((2, block_len), float("inf")), id="inf"),
    ]


# ramp vectors of tract v0.23.10 linalg/src/frame/block_quant/ggml_k.rs
# tests (llama.cpp quantize_row_q4_K_ref / dequantize_row_q4_K)
Q4_K_GGML_REFS = [
    (
        0,
        "4e1c2028fefefffcfbfefdfeddbdfede51c73e950b72d84fa61c83ea50c72d94fb"
        "62d83fa51c73e950b72d84fa61c83ee950b72d94fa61d83ea50c73e940b62d84fa"
        "61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea"
        "40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d9"
        "4fa61d84ea51b72e95fb62d9",
        [(0, 0xBFD20380), (32, 0xBF31E300), (255, 0x3FB61280)],
    ),
    (
        32,
        "4e1c2028c0fefffcc0fefdfeddbdfede50c030900070d040a01080e050c02090f0"
        "60d030a01070e050b02080f060c030e950b72d94fa61d83ea50c73e940b62d84fa"
        "61c73ea50b72d94fb61c83fa51c762d94fa61d73ea51b72e95fb62c83fa60c73ea"
        "40b72e84fb51c83f950c73d940fb62c83f950c73d940b71d84ea51c83e950c62d9"
        "4fa61d84ea51b72e95fb62d9",
        [(0, 0), (32, 0xBF31E300), (255, 0x3FB61280)],
    ),
    (256, "00" * 144, [(0, 0)]),
]


@pytest.mark.parametrize("zeros,expected_hex,values", Q4_K_GGML_REFS)
def test_q4k_matches_ggml_reference(zeros, expected_hex, values):
    _check_ggml_reference(
        fp_to_tract_q4_k, _ggml_ramp(256, zeros), expected_hex, values
    )


def test_q4k_matches_llama_cpp_on_random_super_blocks():
    _check_ggml_llama_cpp_hashes(fp_to_tract_q4_k, "Q4_K")


def test_q4k_dat_content_layout(tmp_path):
    qtensor = fp_to_tract_q4_k(torch.randn(3, 512))
    assert qtensor.u8_blob.shape == (3, 512)
    _check_ggml_dat_content_layout(
        tmp_path, qtensor, DatBinHeader.TractCustomTypes.Q4K
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q4k_dequant_round_trip_error_bounded(seed):
    # same bound as tract q4_k_roundtrip
    ramp = _ggml_ramp(512, 0).reshape(2, 256)
    qtensor = fp_to_tract_q4_k(ramp)
    assert (qtensor.decompress() - ramp).abs().max() < 0.35
    zeros = fp_to_tract_q4_k(torch.zeros(2, 256))
    assert torch.equal(zeros.decompress(), torch.zeros(2, 256))
    gen = torch.Generator().manual_seed(seed)
    # rows of very different magnitudes
    fp_tensor = torch.randn(16, 1024, generator=gen) * 10 ** torch.linspace(
        -3, 3, 16
    ).reshape(16, 1)
    deq = fp_to_tract_q4_k(fp_tensor).decompress()
    # Bound per super-block, not per sub-block: the 6bit quantized
    # sub-block scales and mins (relative to the super-block d and dmin)
    # can exceed one sub-block step (measured up to 1.06). Per super-block
    # the error stays below 0.81 step (range / 15) over 300 seeds,
    # 2 steps leaves a robust margin.
    sb_fp = fp_tensor.reshape(-1, 256)
    sb_range = sb_fp.max(dim=1).values - sb_fp.min(dim=1).values
    sb_err = (deq - fp_tensor).abs().reshape(-1, 256).max(dim=1).values
    assert (sb_err <= 2 * sb_range / 15).all(), (sb_err / sb_range).max()
    # relative error of a 4.5bpw format (measured below 0.075)
    assert (deq - fp_tensor).norm() / fp_tensor.norm() < 0.1


@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256), (2, 2, 2, 256)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_q4k_decompress_matches_reference(shape, dtype):
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(shape, generator=gen).to(dtype)
    _check_ggml_decompress_matches_reference(fp_to_tract_q4_k, fp_tensor)


@pytest.mark.parametrize(
    "weight",
    _ggml_invalid_weights(256)
    + [
        # f16 overflow of d (positive range) and dmin (negative offset)
        pytest.param(torch.full((2, 256), 1e9), id="d_overflow"),
        pytest.param(torch.full((2, 256), -1e9), id="dmin_overflow"),
    ],
)
def test_q4k_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        fp_to_tract_q4_k(weight)


def test_q4k_keeps_negligible_underflow_blocks_as_zeros(caplog):
    # one row far below f16 scales resolution, others of unit magnitude:
    # like llama.cpp and tract, its blocks quantize to zeros
    fp_tensor = torch.randn(8, 512, generator=torch.Generator().manual_seed(0))
    fp_tensor[3] *= 1e-7
    with caplog.at_level("WARNING"):
        qtensor = fp_to_tract_q4_k(fp_tensor)
    assert "2 nonzero block(s) dequantize to zeros" in caplog.text
    deq = qtensor.decompress()
    assert torch.equal(deq[3], torch.zeros(512))
    assert (deq[:3] != 0).any()


@pytest.mark.parametrize(
    "weight",
    [
        pytest.param(torch.full((2, 256), 1e-10), id="whole_tensor"),
        # erased block magnitude above f16 precision of the tensor max
        pytest.param(
            torch.cat([torch.full((1, 256), 1e-6), torch.full((1, 256), 1e-4)]),
            id="relative",
        ),
    ],
)
def test_q4k_rejects_significant_underflow(weight):
    with pytest.raises(T2NErrorImpossibleQuantization, match="underflow"):
        fp_to_tract_q4_k(weight)


def _accelerator_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.skipif(
    _accelerator_device() is None, reason="needs a cuda or mps device"
)
def test_q4k_keeps_fp_tensor_device():
    device = _accelerator_device()
    fp_tensor = torch.randn(4, 256)
    qtensor = fp_to_tract_q4_k(fp_tensor.to(device))
    assert qtensor.device.type == device.type
    deq = qtensor.decompress()
    assert deq.device.type == device.type
    # same bytes as a cpu quantization
    assert torch.equal(
        qtensor.u8_blob.cpu(), fp_to_tract_q4_k(fp_tensor).u8_blob
    )
    out = nn.functional.embedding(torch.tensor([0, 3], device=device), qtensor)
    assert torch.equal(out.cpu(), deq.cpu()[[0, 3]])


@pytest.mark.parametrize(
    "fp_to_qtensor",
    [fp_to_tract_q4_k, fp_to_tract_q4_0_with_min_max_calibration],
    ids=["q4_k", "q4_0"],
)
def test_qtensor_to_keeps_codes_and_source(fp_to_qtensor):
    fp_tensor = torch.randn(4, 256, generator=torch.Generator().manual_seed(0))
    qtensor = fp_to_qtensor(fp_tensor)
    codes = qtensor.decompress_to_u8().clone()
    deq = qtensor.decompress()
    targets = [torch.float16]
    if _accelerator_device() is not None:
        targets.append(_accelerator_device())
    for target in targets:
        moved = qtensor.to(target)
        assert type(moved) is type(qtensor)
        assert torch.equal(moved.decompress_to_u8().cpu(), codes)
        if isinstance(target, torch.dtype):
            assert moved.decompress().dtype == target
        else:
            assert moved.device.type == target.type
            assert torch.equal(moved.decompress().cpu(), deq)
        # the source qtensor is left untouched
        assert qtensor.device.type == "cpu"
        assert torch.equal(qtensor.decompress(), deq)


def test_q4k_zero_sub_scales_keep_ggml_search_levels():
    # a sub-block far below the others gets a 6bit scale of 0: ggml keeps
    # its `make_qkx2_quants` levels, the scheme must reproduce them
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(2, 256, generator=gen)
    fp_tensor[:, :32] *= 1e-4
    qtensor = fp_to_tract_q4_k(fp_tensor)
    qscheme = qtensor.qscheme
    zero_sub = qscheme.sub_scales() == 0
    assert zero_sub[:, 0].all() and not zero_sub[:, 1:].any()
    codes = qtensor.decompress_to_u8()
    assert codes[:, :32].any()
    assert torch.equal(codes, qscheme.quantize_as_u8(fp_tensor))


def test_ggml_nearest_int_matches_ggml():
    vals = torch.tensor(
        [0.5, 1.5, 2.5, -0.5, -1.5, 0.49999997, 3.7, -3.7, 4194303.0],
        dtype=torch.float32,
    )
    assert torch.equal(ggml_nearest_int(vals), torch.round(vals).int())
    # beyond ggml range: same bits as ggml on every platform (a float to
    # int32 cast of out of range values is platform dependent)
    big = torch.tensor([2.0**23, 1e7], dtype=torch.float32)
    assert ggml_nearest_int(big).tolist() == [-2097152, -1291456]


def test_ggml_seq_sum_is_sequential_f32():
    # each 1 + 2**-24 rounds back to 1 in f32 (a wider accumulator or
    # another order gives 1 + 2**-22)
    vals = torch.tensor([[1.0] + [2.0**-24] * 4], dtype=torch.float32)
    assert ggml_seq_sum(vals).item() == 1.0
    assert ggml_seq_sum(vals.flip(1)).item() == 1.0 + 2.0**-22
    assert vals.double().sum().item() == 1.0 + 2.0**-22


@pytest.mark.parametrize(
    "version", old_tract_versions(QTensorTractQ4_K.MIN_TRACT_VERSION)
)
def test_q4k_rejects_old_tract(tmp_path, caplog, version):
    check_rejects_old_tract(
        tmp_path, caplog, fp_to_tract_q4_k(torch.randn(2, 256)), version
    )


@pytest.mark.parametrize(
    "version", supported_tract_versions(QTensorTractQ4_K.MIN_TRACT_VERSION)
)
def test_q4k_offload_state_write(tmp_path, version):
    check_offload_state_write(
        tmp_path, fp_to_tract_q4_k(torch.randn(4, 512)), tract_target(version)
    )


@pytest.mark.parametrize(
    "inference_target",
    [
        i
        for i in TRACT_INFERENCES_TO_TESTS_EXACT
        if i.version >= QTensorTractQ4_K.MIN_TRACT_VERSION
    ],
)
# f16 CPU embedding (index_select) works on torch 1.x: like Q4_0/Q8_1 tests
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantize_with_tract_q4_k_embedding(inference_target, dtype):
    """Q4_K is only supported by tract on embedding/gathering."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        model = nn.Embedding(6, 512).eval().to(dtype)
        q_tensor = fp_to_tract_q4_k(torch.randn(6, 512).to(dtype))
        assert q_tensor.decompress().dtype == dtype
        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={"weight": {"dtype": DatBinHeader.TractCustomTypes.Q4K}},
            ),
        )


def test_compress_ggml_q4_k_embeddings_only_quantize_embeddings():
    model = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32))
    tied = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 10, bias=False))
    tied[1].weight = tied[0].weight
    registry = dynamic_load_registry(DEFAULT_COMPRESSION_REGISTRY)

    registry["ggml_q4_k_embeddings"](model)
    assert isinstance(model[0].weight, QTensorTractQ4_K)
    assert not isinstance(model[1].weight, QTensor)

    # weight shared with a Linear must stay float (no tract Q4_K matmul)
    registry["ggml_q4_k_embeddings"](tied)
    assert not isinstance(tied[0].weight, QTensor)

    with pytest.raises(T2NErrorMisuse):
        quantize_weights_ggml_Q4_K(
            nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32)),
            to_quantize_module_classes=(nn.Linear, nn.Embedding),
        )
    with pytest.raises(T2NErrorMisuse, match="percentile"):
        quantize_weights_ggml_Q4_K(
            nn.Sequential(nn.Embedding(10, 256)), percentile=0.99
        )


# IQ2_S ####################################################################

# llama.cpp quantize_row_iq2_s_ref / dequantize_row_iq2_s vectors
# (from tract v0.23.10 linalg/src/frame/block_quant/iq2_s.rs tests)
IQ2_S_GGML_REFS = [
    (
        0,
        "a82379f32f67b63064b6416db7ae6d479f7d47932f799d2f67b6308db630d3b741"
        "6d29b5522bb5526aa5566aa5d44aadd44aa9955aa995522bb5522ba5566aa5564a"
        "d3a6699ad6b56d9bdfdcceadeedccbeb",
        [(0, 0xBFB96CC0), (32, 0xBEBF6800), (255, 0x3FAD7640)],
    ),
    (
        32,
        "f42200000000b630d3b6416db7ae6d479f7d47932f799d2f67b630d3b630d3b741"
        "6d00000000b5526aa5566aa5d44aadd44aa9955aa995522bb5522ba5566aa5564a"
        "00b6699ad6b56d9b00dddfbfffedddfc",
        # a skipped group keeps grid index 0: zeros come back as 0x3c5e8000
        [(0, 0x3C5E8000), (32, 0xBEBBBC00), (255, 0x3FA86560)],
    ),
    (256, "00" * 82, [(0, 0), (255, 0)]),
]


@pytest.mark.parametrize("zeros,expected_hex,values", IQ2_S_GGML_REFS)
def test_iq2s_matches_ggml_reference(zeros, expected_hex, values):
    _check_ggml_reference(
        fp_to_tract_iq2_s, _ggml_ramp(256, zeros), expected_hex, values
    )


def test_iq2s_dat_content_layout(tmp_path):
    qtensor = fp_to_tract_iq2_s(torch.randn(3, 512))
    assert qtensor.u8_blob.shape == (3, 512)
    _check_ggml_dat_content_layout(
        tmp_path, qtensor, DatBinHeader.TractCustomTypes.IQ2S
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_iq2s_dequant_round_trip_error_bounded(seed):
    # same bound as tract iq2_s_roundtrip
    ramp = _ggml_ramp(512, 0).reshape((2, 256))
    qtensor = fp_to_tract_iq2_s(ramp)
    assert (qtensor.decompress() - ramp).abs().max() < 1.25
    zeros = fp_to_tract_iq2_s(torch.zeros(2, 256))
    assert torch.equal(zeros.decompress(), torch.zeros(2, 256))
    gen = torch.Generator().manual_seed(seed)
    # rows of very different magnitudes
    fp_tensor = torch.randn(16, 1024, generator=gen) * 10 ** torch.linspace(
        -3, 3, 16
    ).reshape((16, 1))
    deq = fp_to_tract_iq2_s(fp_tensor).decompress()
    # signs are stored exactly, grid magnitudes are never 0
    assert torch.equal(deq.sign(), fp_tensor.sign())
    # relative error of a 2.5625bpw format, per row (measured near 0.26)
    row_err = (deq - fp_tensor).norm(dim=1) / fp_tensor.norm(dim=1)
    assert (row_err < 0.35).all(), row_err.max()


@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256), (2, 2, 2, 256)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_iq2s_decompress_matches_reference(shape, dtype):
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(shape, generator=gen).to(dtype)
    _check_ggml_decompress_matches_reference(fp_to_tract_iq2_s, fp_tensor)


@pytest.mark.parametrize(
    "weight",
    _ggml_invalid_weights(256)
    + [
        # f16 overflow of d
        pytest.param(torch.full((2, 256), 1e9), id="d_overflow"),
        # nonzero super-block whose d rounds to 0 in f16
        pytest.param(torch.full((2, 256), 1e-7), id="underflow"),
        # below the ggml group epsilon: every group is skipped
        pytest.param(torch.full((2, 256), 1e-10), id="below_group_eps"),
    ],
)
def test_iq2s_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        fp_to_tract_iq2_s(weight)


def test_iq2s_codes_hold_grid_level_and_sign():
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(4, 512, generator=gen)
    fp_tensor[0, :256] = 0
    qtensor = fp_to_tract_iq2_s(fp_tensor)
    codes = qtensor.decompress_to_u8()
    assert codes.dtype == torch.uint8
    assert set(codes.unique().tolist()) <= {0, 1, 2, 4, 5, 6}
    # bit 2 is the exact sign, an all zero super-block has zero codes
    assert torch.equal((codes >> 2) == 1, fp_tensor < 0)
    assert not codes[0, :256].any()
    assert qtensor.qscheme.d.shape == (8,)
    assert qtensor.qscheme.scales.shape == (8, 16)
    assert int(qtensor.qscheme.scales.max()) <= 15
    # every octet of grid levels is a grid point
    _, kmap, _ = _iq2_s_search(torch.device("cpu"))
    levels = (codes & 3).reshape((-1, 8)).to(torch.int64)
    packed = (levels << (2 * torch.arange(8))).sum(dim=1)
    assert (kmap[packed] >= 0).all()


def test_iq2s_grid_search_tables():
    grid_levels, kmap, neighbours = _iq2_s_search(torch.device("cpu"))
    assert len(set(KGRID_2BIT_1024)) == 1024
    assert grid_levels.shape == (1024, 8) and int(grid_levels.max()) == 2
    # map is the inverse of the grid
    assert torch.equal(kmap[torch.tensor(KGRID_2BIT_1024)], torch.arange(1024))
    assert int((kmap >= 0).sum()) == 1024
    # off grid codes of levels in [0, 2] have a nonempty nearest shell
    digits = 3 ** torch.arange(8)
    all_levels = (torch.arange(3**8)[:, None] // digits) % 3
    all_codes = (all_levels << (2 * torch.arange(8))).sum(dim=1)
    off = all_codes[kmap[all_codes] < 0]
    assert (neighbours[off, 0] >= 0).all()
    assert (neighbours[kmap >= 0] == -1).all()
    # shells are in grid index order, padded at the end
    shell = neighbours[off]
    valid = shell >= 0
    assert (valid[:, :-1] | ~valid[:, 1:]).all()
    assert ((shell[:, 1:] > shell[:, :-1]) | ~valid[:, 1:]).all()


@pytest.mark.skipif(
    _accelerator_device() is None, reason="needs a cuda or mps device"
)
def test_iq2s_keeps_fp_tensor_device():
    device = _accelerator_device()
    fp_tensor = torch.randn(4, 256)
    qtensor = fp_to_tract_iq2_s(fp_tensor.to(device))
    assert qtensor.device.type == device.type
    assert qtensor.qscheme.d.device.type == device.type
    deq = qtensor.decompress()
    assert deq.device.type == device.type
    # same bytes as a cpu quantization
    assert _tract_bytes(qtensor) == _tract_bytes(fp_to_tract_iq2_s(fp_tensor))
    out = nn.functional.embedding(torch.tensor([0, 3], device=device), qtensor)
    assert torch.equal(out.cpu(), deq.cpu()[[0, 3]])


def test_iq2s_scheme_rejects_other_tensor():
    qtensor = fp_to_tract_iq2_s(torch.randn(2, 256))
    with pytest.raises(T2NErrorMisuse, match="IQ2_S"):
        qtensor.qscheme.quantize_as_u8(torch.randn(2, 256))


def test_iq2s_accepts_zero_groups_in_nonzero_block():
    # skipped groups decode to small nonzero values: no underflow error
    fp_tensor = torch.randn(2, 256)
    fp_tensor[:, :32] = 0
    deq = fp_to_tract_iq2_s(fp_tensor).decompress()
    assert (deq[:, :32] > 0).all()


@pytest.mark.parametrize(
    "version", old_tract_versions(QTensorTractIQ2_S.MIN_TRACT_VERSION)
)
def test_iq2s_rejects_old_tract(tmp_path, caplog, version):
    check_rejects_old_tract(
        tmp_path, caplog, fp_to_tract_iq2_s(torch.randn(2, 256)), version
    )


@pytest.mark.parametrize(
    "version", supported_tract_versions(QTensorTractIQ2_S.MIN_TRACT_VERSION)
)
def test_iq2s_offload_state_write(tmp_path, version):
    check_offload_state_write(
        tmp_path,
        fp_to_tract_iq2_s(torch.randn(4, 512)),
        tract_target(version),
    )


@pytest.mark.parametrize(
    "inference_target",
    [
        i
        for i in TRACT_INFERENCES_TO_TESTS_EXACT
        if i.version >= QTensorTractIQ2_S.MIN_TRACT_VERSION
    ],
)
# f16 CPU embedding (index_select) works on torch 1.x: like Q4_0/Q8_1 tests
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantize_with_tract_iq2_s_embedding(inference_target, dtype):
    """IQ2_S is only supported by tract on embedding/gathering."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        model = nn.Embedding(6, 512).eval().to(dtype)
        q_tensor = fp_to_tract_iq2_s(torch.randn(6, 512).to(dtype))
        assert q_tensor.decompress().dtype == dtype
        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {"dtype": DatBinHeader.TractCustomTypes.IQ2S}
                },
            ),
        )


def test_compress_ggml_iq2_s_embeddings_only_quantize_embeddings():
    model = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32))
    tied = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 10, bias=False))
    tied[1].weight = tied[0].weight
    registry = dynamic_load_registry(DEFAULT_COMPRESSION_REGISTRY)

    registry["ggml_iq2_s_embeddings"](model)
    assert isinstance(model[0].weight, QTensorTractIQ2_S)
    assert not isinstance(model[1].weight, QTensor)

    # weight shared with a Linear must stay float (no tract IQ2_S matmul)
    registry["ggml_iq2_s_embeddings"](tied)
    assert not isinstance(tied[0].weight, QTensor)

    with pytest.raises(T2NErrorMisuse):
        quantize_weights_ggml_IQ2_S(
            nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32)),
            to_quantize_module_classes=(nn.Linear, nn.Embedding),
        )
    with pytest.raises(T2NErrorMisuse, match="percentile"):
        quantize_weights_ggml_IQ2_S(
            nn.Sequential(nn.Embedding(10, 256)), percentile=0.99
        )
