"""Tract raw ggml block formats (K-quants): packing and tract execution."""

from functools import partial

import pytest
import torch
from torch import nn

from torch_to_nnef.compress import (
    DEFAULT_COMPRESSION_REGISTRY,
    dynamic_load_registry,
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
    QTensorTractQ4_K,
    fp_to_tract_q4_k,
)
from torch_to_nnef.tensor.quant.qtract_ggml import (
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
    # always quantized from f32, then dequantized in f32 and casted
    f32_qtensor = fp_to_qtensor(fp_tensor.float())
    assert torch.equal(
        qtensor.decompress_to_u8(), f32_qtensor.decompress_to_u8()
    )
    deq = qtensor.decompress()
    assert deq.dtype == fp_tensor.dtype
    assert torch.equal(deq, f32_qtensor.decompress().to(fp_tensor.dtype))


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
        fp_to_tract_q4_k, _ggml_ramp(256, zeros), expected_hex, values
    )


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
