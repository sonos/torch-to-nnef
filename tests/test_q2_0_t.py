"""Ternary packing and execution through tract's NNEF reader."""

from functools import partial

import pytest
import torch
from torch import nn

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant import (
    QTensorTractQ2_0_T,
    ternary_to_tract_q2_0_t,
)
from torch_to_nnef.utils import torch_version

from .test_qtensor import check_tensor_in_nnef_archive
from .utils import (
    TRACT_INFERENCES_TO_TESTS_EXACT,
    check_model_io_test,
    skipif_unsupported_qtensor,
    tract_target,
)

pytestmark = skipif_unsupported_qtensor


@pytest.mark.parametrize("version", ["0.23.11", "0.23.12"])
def test_q20t_binary_layout_and_offload(tmp_path, version):
    target = tract_target(version)
    weight = torch.tensor([-0.5, 0, 0.5, -0.5]).repeat(8).reshape(1, 32)
    weight = torch.cat([weight, torch.zeros_like(weight)])
    qtensor = ternary_to_tract_q2_0_t(weight)
    qtensor.write_in_file(tmp_path, "weight", target)
    data = (tmp_path / "weight.dat").read_bytes()
    # 00 01 10 00, low bits first: 0x24. Zero uses code 01: 0x55.
    assert data[128:] == b"\x00\x38" + b"\x24" * 8 + b"\x00\x00" + b"\x55" * 8
    assert data[48:52] == b"\x20\x30RT"
    with (tmp_path / "weight.dat").open("rb") as fh:
        header = DatBinHeader.from_dat_file(fh)
    assert header.torch_dtype_or_custom == DatBinHeader.TractCustomTypes.Q20T
    assert header.dims[: header.rank] == [2, 32]
    assert header.data_size_bytes == 20
    assert torch.equal(qtensor.decompress(), weight)
    state = qtensor.to_offload_state()
    restored = QTensorTractQ2_0_T.from_offload_state(state)
    assert torch.equal(restored.decompress(), weight)
    QTensorTractQ2_0_T.write_offload_state_in_file(
        state, tmp_path, "offloaded", target
    )
    assert (tmp_path / "offloaded.dat").read_bytes() == data
    chunks = list(
        qtensor._iter_binary_dat_content_chunks(
            qtensor.decompress_to_u8(),
            qtensor.qscheme,
            target,
            chunk_blocks=1,
        )
    )
    assert [len(chunk) for chunk in chunks] == [10, 10]
    assert b"".join(chunks) == data[128:]


@pytest.mark.parametrize("shape", [(2, 128), (2, 4, 32), (2, 2, 2, 32)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_q20t_source_groups_and_scale_rounding(shape, dtype):
    signs = (torch.arange(256) % 3 - 1).reshape(shape).to(dtype)
    scales = (
        torch.tensor([0.12345, 0.75], dtype=dtype)
        .repeat_interleave(128)
        .reshape(shape)
    )
    weight = signs * scales
    qtensor = ternary_to_tract_q2_0_t(weight)
    expected = signs * scales.half().to(dtype)
    assert torch.equal(qtensor.decompress(), expected)
    assert torch.equal(qtensor.decompress_to_u8(), (signs + 1).byte())


@pytest.mark.parametrize(
    "weight",
    [
        torch.randn(2, 32),
        torch.ones(32),
        torch.ones(2, 31),
        torch.ones(2, 32, dtype=torch.int32),
        torch.empty(0, 32),
        torch.full((2, 32), float("nan")),
        torch.full((2, 32), float("inf")),
        torch.full((2, 32), 1e6),
        torch.full((2, 32), 1e-10),
    ],
)
def test_q20t_rejects_invalid_weights(weight):
    with pytest.raises(T2NErrorImpossibleQuantization):
        ternary_to_tract_q2_0_t(weight)


@pytest.mark.parametrize(
    "version", ["0.23.9", "0.23.10-pre", "0.23.10", "0.23.11-pre"]
)
def test_q20t_rejects_old_tract(tmp_path, caplog, version):
    qtensor = ternary_to_tract_q2_0_t(torch.ones(2, 32))
    target = tract_target(version)
    with pytest.raises(
        T2NErrorNotImplemented, match=r"requires tract >= 0\.23\.11"
    ):
        qtensor.write_in_file(tmp_path, "weight", target)
    with pytest.raises(
        T2NErrorNotImplemented, match=r"requires tract >= 0\.23\.11"
    ):
        QTensorTractQ2_0_T.write_offload_state_in_file(
            qtensor.to_offload_state(), tmp_path, "offloaded", target
        )

    assert (
        f"Q2_0_T export requires tract >= 0.23.11; target is {version}. "
        "Upgrade tract before exporting Q2_0_T weights." in caplog.text
    )
    assert not list(tmp_path.glob("*.dat"))


@pytest.mark.parametrize(
    "inference_target",
    [
        target
        for target in TRACT_INFERENCES_TO_TESTS_EXACT
        if target.version >= QTensorTractQ2_0_T.MIN_TRACT_VERSION
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.float16,
            marks=pytest.mark.skipif(
                torch_version() < "2.0.0",
                reason="PyTorch 1.x does not support float16 CPU Linear",
            ),
        ),
    ],
)
@pytest.mark.parametrize("tokens", [1, 5])
def test_q20t_linear_check_io(inference_target, dtype, tokens):
    with torch.no_grad():
        model = nn.Linear(128, 8, bias=False).eval().to(dtype)
        weight = (torch.arange(1024) % 3 - 1).reshape(8, 128).to(dtype) * 0.5
        model.weight = nn.Parameter(
            ternary_to_tract_q2_0_t(weight), requires_grad=False
        )
        test_input = (
            (torch.arange(tokens * 128) % 7 - 3).reshape(tokens, 128).to(dtype)
        )
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
            callback_post_export=partial(
                check_tensor_in_nnef_archive,
                labels={
                    "weight": {"dtype": DatBinHeader.TractCustomTypes.Q20T}
                },
            ),
        )
