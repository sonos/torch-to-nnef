import pytest
import torch

from torch_to_nnef.tensor import (
    Q1Tensor,
    Q2Tensor,
    Q3Tensor,
    Q3TensorInU8,
    Q4Tensor,
    Q8Tensor,
)


def test_pack_unpack_8bits():
    x = (torch.rand(6, 10) * 128).to(torch.uint8)
    packed_8 = Q8Tensor.pack(x)
    unpacked_8 = packed_8.unpack()
    assert (unpacked_8 == x).all()


def test_pack_unpack_4bits():
    x = (torch.rand(6, 10) * 16).to(torch.uint8)
    packed = Q4Tensor.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_4bits_trunk():
    x = (torch.rand(6, 10) * 128).to(torch.uint8).clamp(16)
    with pytest.raises(ValueError):
        Q4Tensor.pack(x)


def test_pack_unpack_4bits_wrong_shape():
    x = (torch.rand(5, 10) * 16).to(torch.uint8)
    with pytest.raises(ValueError):
        Q4Tensor.pack(x)


def test_pack_unpack_2bits():
    x = (torch.rand(8, 10) * 4).to(torch.uint8)
    packed = Q2Tensor.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_2bits_wrong_shape():
    x = (torch.rand(6, 10) * 4).to(torch.uint8)
    with pytest.raises(ValueError):
        Q2Tensor.pack(x)


def test_pack_unpack_2bits_wrong_trunk():
    x = (torch.rand(8, 10) * 5).to(torch.uint8).clamp_min(4)
    with pytest.raises(ValueError):
        Q2Tensor.pack(x)


def test_pack_unpack_1bits():
    x = (torch.rand(8, 10)).to(torch.uint8)
    packed = Q1Tensor.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_1bits_wrong_trunk():
    x = (torch.rand(8, 10) * 5).to(torch.uint8).clamp_min(4)
    with pytest.raises(ValueError):
        Q2Tensor.pack(x)


def test_pack_unpack_1bits_wrong_shape():
    x = (torch.rand(20, 10)).to(torch.uint8)
    with pytest.raises(ValueError):
        Q1Tensor.pack(x)


def test_pack_unpack_3bits():
    x = (torch.rand(10, 5) * 8).to(torch.int32)
    packed = Q3Tensor.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_3bits_wrong_shape():
    x = (torch.rand(9, 10) * 8).to(torch.int32)
    with pytest.raises(ValueError):
        Q3Tensor.pack(x)


def test_pack_unpack_3bits_stored_in_u8():
    x = (torch.rand(6, 5) * 8).to(torch.uint8)
    packed = Q3TensorInU8.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()
