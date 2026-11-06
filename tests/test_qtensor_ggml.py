"""Tract ggml super-block formats (K-quants): packing and tract execution."""

from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
    quantize_weights_ggml_Q4_K,
    quantize_weights_ggml_Q5_K,
)
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorMisuse,
)
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant import (
    QTensor,
    QTensorTractQ4_K,
    QTensorTractQ5_K,
    fp_to_tract_q4_k,
    fp_to_tract_q5_k,
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


def _ggml_ramp(length: int, zeros: int) -> np.ndarray:
    """Ramp of tract `ggml_k` tests over [-2, 2] with leading zeros."""
    return np.array(
        [
            0.0 if i < zeros else (((i * 17 + 3) % 41) - 20.0) / 10.0
            for i in range(length)
        ],
        dtype=np.float32,
    )


def _check_ggml_reference(
    qtensor_cls, data: np.ndarray, expected_hex: str, values
):
    """Check quant bytes and dequant f32 bits against llama.cpp ones."""
    blocks = qtensor_cls.quantize_blocks(data.reshape(1, -1))
    assert blocks.tobytes().hex() == expected_hex
    ref_blocks = np.frombuffer(bytes.fromhex(expected_hex), dtype=np.uint8)
    deq = qtensor_cls.dequantize_blocks(ref_blocks.reshape(1, -1))
    for ix, bits in values:
        assert int(deq[0, ix].view(np.uint32)) == bits, ix


def _check_ggml_dat_content_layout(tmp_path, qtensor, item_type):
    """Check a ggml block `.dat` is the header + raw ggml blocks."""
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
    blocks = np.frombuffer(content, dtype=np.uint8).reshape(
        n_blocks, qtensor_cls.BLOCK_BYTES
    )
    assert np.array_equal(blocks, qtensor.decompress_to_u8().numpy())
    deq = qtensor_cls.dequantize_blocks(blocks).reshape(qtensor.shape)
    assert np.array_equal(deq, qtensor.decompress().float().numpy())


def _check_ggml_decompress_matches_reference(qtensor_cls, fp_tensor):
    """Check decompress is the numpy reference dequant in source dtype."""
    qtensor = qtensor_cls.from_fp(fp_tensor)
    assert qtensor.shape == fp_tensor.shape
    assert qtensor.numel() == fp_tensor.numel()
    n_blocks = fp_tensor.numel() // qtensor_cls.BLOCK_LEN
    u8_blocks = qtensor.decompress_to_u8().numpy()
    assert u8_blocks.shape == (n_blocks, qtensor_cls.BLOCK_BYTES)
    fp32_blocks = fp_tensor.float().numpy().reshape((n_blocks, -1))
    assert np.array_equal(u8_blocks, qtensor_cls.quantize_blocks(fp32_blocks))
    expected = (
        torch.from_numpy(qtensor_cls.dequantize_blocks(u8_blocks))
        .reshape(fp_tensor.shape)
        .to(fp_tensor.dtype)
    )
    deq = qtensor.decompress()
    assert deq.dtype == fp_tensor.dtype
    assert torch.equal(deq, expected)


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


# llama.cpp quantize_row_q4_K_ref / dequantize_row_q4_K vectors
# (from tract v0.23.10 linalg/src/frame/block_quant/ggml_k.rs tests)
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
        QTensorTractQ4_K, _ggml_ramp(256, zeros), expected_hex, values
    )


def test_q4k_dat_content_layout(tmp_path):
    qtensor = fp_to_tract_q4_k(torch.randn(3, 512))
    assert qtensor.u8_blob.shape == (6, 144)
    _check_ggml_dat_content_layout(
        tmp_path, qtensor, DatBinHeader.TractCustomTypes.Q4K
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q4k_dequant_round_trip_error_bounded(seed):
    # same bound as tract q4_k_roundtrip
    ramp = torch.from_numpy(_ggml_ramp(512, 0)).reshape(2, 256)
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
    _check_ggml_decompress_matches_reference(QTensorTractQ4_K, fp_tensor)


@pytest.mark.parametrize(
    "weight",
    _ggml_invalid_weights(256)
    + [
        # f16 overflow of d (positive range) and dmin (negative offset)
        pytest.param(torch.full((2, 256), 1e9), id="d_overflow"),
        pytest.param(torch.full((2, 256), -1e9), id="dmin_overflow"),
        # nonzero super-block whose d and dmin round to 0 in f16
        pytest.param(torch.full((2, 256), 1e-10), id="underflow"),
    ],
)
def test_q4k_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        fp_to_tract_q4_k(weight)


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


# llama.cpp quantize_row_q5_K_ref / dequantize_row_q5_K vectors
# (from tract v0.23.10 linalg/src/frame/block_quant/ggml_k.rs tests)
Q5_K_GGML_REFS = [
    (
        0,
        "41181d28fcfefffdfbfefefefecdfeedcc32d966994c33cd269964b3cc32d9669b"
        "4c33c9669964b3cc3699669bcc33c9b28f5d3a07d4a28f5c2af6d4a17e5c29f6c3"
        "a17e4b19f6c3906e4b18e6c2906ec3917e4b18f5c3906d4b18e5c2906d3a18e5b2"
        "8f5c3a07d5b28f5c2907d4a18ed5a27f5d2af7d4a27f4c2af7c4917e4c19e6c491"
        "6e3b19e6b3906e3b08e6b380f6c3906e4b18e6c3906d3b18e5b3805d3a07d5b28f"
        "5d2a07d4a28f5c2af7d4a1",
        [(0, 0xBFD2C780), (32, 0xBF1366C0), (255, 0x3FA6A520)],
    ),
    (
        32,
        "41181d28c0fefffdc0fefefefecdfeedcc32d866984c32cc269864b2cc32d8669a"
        "4c32c8669864b2cc3698669acc32c8b080503000d0a0805020f0d0a0705020f0c0"
        "a0704010f0c090604010e0c09060c3917e4b18f5c3906d4b18e5c2906d3a18e5b2"
        "8f5c3a07d5b28f5c2907d4a18ed5a27f5d2af7d4a27f4c2af7c4917e4c19e6c491"
        "6e3b19e6b3906e3b08e6b380f6c3906e4b18e6c3906d3b18e5b3805d3a07d5b28f"
        "5d2a07d4a28f5c2af7d4a1",
        [(0, 0), (32, 0xBF1366C0), (255, 0x3FA6A520)],
    ),
    (256, "00" * 176, [(0, 0)]),
]


@pytest.mark.parametrize("zeros,expected_hex,values", Q5_K_GGML_REFS)
def test_q5k_matches_ggml_reference(zeros, expected_hex, values):
    _check_ggml_reference(
        QTensorTractQ5_K, _ggml_ramp(256, zeros), expected_hex, values
    )


def test_q5k_dat_content_layout(tmp_path):
    qtensor = fp_to_tract_q5_k(torch.randn(3, 512))
    assert qtensor.u8_blob.shape == (6, 176)
    _check_ggml_dat_content_layout(
        tmp_path, qtensor, DatBinHeader.TractCustomTypes.Q5K
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q5k_dequant_round_trip_error_bounded(seed):
    # same bound as tract q5_k_roundtrip
    ramp = torch.from_numpy(_ggml_ramp(512, 0)).reshape(2, 256)
    qtensor = fp_to_tract_q5_k(ramp)
    assert (qtensor.decompress() - ramp).abs().max() < 0.2
    zeros = fp_to_tract_q5_k(torch.zeros(2, 256))
    assert torch.equal(zeros.decompress(), torch.zeros(2, 256))
    gen = torch.Generator().manual_seed(seed)
    # rows of very different magnitudes
    fp_tensor = torch.randn(16, 1024, generator=gen) * 10 ** torch.linspace(
        -3, 3, 16
    ).reshape(16, 1)
    deq = fp_to_tract_q5_k(fp_tensor).decompress()
    # Bound per super-block like Q4_K: the 6bit quantized sub-block scales
    # and mins can exceed one sub-block step. Per super-block the error
    # stays below 1.01 step (range / 31) over 300 seeds, 2 steps leaves
    # a robust margin.
    sb_fp = fp_tensor.reshape(-1, 256)
    sb_range = sb_fp.max(dim=1).values - sb_fp.min(dim=1).values
    sb_err = (deq - fp_tensor).abs().reshape(-1, 256).max(dim=1).values
    assert (sb_err <= 2 * sb_range / 31).all(), (sb_err / sb_range).max()
    # relative error of a 5.5bpw format (measured below 0.039),
    # and below the Q4_K one on the same weights
    assert (deq - fp_tensor).norm() / fp_tensor.norm() < 0.05
    q4k_deq = fp_to_tract_q4_k(fp_tensor).decompress()
    assert (deq - fp_tensor).norm() < (q4k_deq - fp_tensor).norm()


@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256), (2, 2, 2, 256)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_q5k_decompress_matches_reference(shape, dtype):
    gen = torch.Generator().manual_seed(0)
    fp_tensor = torch.randn(shape, generator=gen).to(dtype)
    _check_ggml_decompress_matches_reference(QTensorTractQ5_K, fp_tensor)


@pytest.mark.parametrize(
    "weight",
    _ggml_invalid_weights(256)
    + [
        # f16 overflow of d (positive range) and dmin (negative offset)
        pytest.param(torch.full((2, 256), 1e9), id="d_overflow"),
        pytest.param(torch.full((2, 256), -1e9), id="dmin_overflow"),
        # nonzero super-block whose d and dmin round to 0 in f16
        pytest.param(torch.full((2, 256), 1e-10), id="underflow"),
    ],
)
def test_q5k_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        fp_to_tract_q5_k(weight)


def test_q5k_accepts_constant_negative_rows():
    # d rounds to 0 while dmin does not: no underflow, like Q4_K
    weight = torch.full((2, 256), -0.5)
    deq = fp_to_tract_q5_k(weight).decompress()
    assert (deq - weight).abs().max() < 1e-3


@pytest.mark.parametrize(
    "version", old_tract_versions(QTensorTractQ5_K.MIN_TRACT_VERSION)
)
def test_q5k_rejects_old_tract(tmp_path, caplog, version):
    check_rejects_old_tract(
        tmp_path, caplog, fp_to_tract_q5_k(torch.randn(2, 256)), version
    )


@pytest.mark.parametrize(
    "version", supported_tract_versions(QTensorTractQ5_K.MIN_TRACT_VERSION)
)
def test_q5k_offload_state_write(tmp_path, version):
    check_offload_state_write(
        tmp_path, fp_to_tract_q5_k(torch.randn(4, 512)), tract_target(version)
    )


@pytest.mark.parametrize(
    "inference_target",
    [
        i
        for i in TRACT_INFERENCES_TO_TESTS_EXACT
        if i.version >= QTensorTractQ5_K.MIN_TRACT_VERSION
    ],
)
# f16 CPU embedding (index_select) works on torch 1.x: like Q4_0/Q8_1 tests
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantize_with_tract_q5_k_embedding(inference_target, dtype):
    """Q5_K is only supported by tract on embedding/gathering."""
    with torch.no_grad():
        test_input = torch.arange(6)
        test_input[:3] = 3
        model = nn.Embedding(6, 512).eval().to(dtype)
        q_tensor = fp_to_tract_q5_k(torch.randn(6, 512).to(dtype))
        assert q_tensor.decompress().dtype == dtype
        model.weight = nn.Parameter(q_tensor, requires_grad=False)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={"weight": {"dtype": DatBinHeader.TractCustomTypes.Q5K}},
            ),
        )


def test_compress_ggml_q5_k_embeddings_only_quantize_embeddings():
    model = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32))
    tied = nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 10, bias=False))
    tied[1].weight = tied[0].weight
    registry = dynamic_load_registry(DEFAULT_COMPRESSION_REGISTRY)

    registry["ggml_q5_k_embeddings"](model)
    assert isinstance(model[0].weight, QTensorTractQ5_K)
    assert not isinstance(model[1].weight, QTensor)

    # weight shared with a Linear must stay float (no tract Q5_K matmul)
    registry["ggml_q5_k_embeddings"](tied)
    assert not isinstance(tied[0].weight, QTensor)

    with pytest.raises(T2NErrorMisuse):
        quantize_weights_ggml_Q5_K(
            nn.Sequential(nn.Embedding(10, 256), nn.Linear(256, 32)),
            to_quantize_module_classes=(nn.Linear, nn.Embedding),
        )
