import pytest
import torch

from torch_to_nnef.bitpack import (
    TensorB1,
    TensorB2,
    TensorB3,
    TensorB3InU8,
    TensorB4,
    TensorB8,
)


def test_pack_unpack_8bits():
    x = (torch.rand(6, 10) * 128).to(torch.uint8)
    packed_8 = TensorB8.pack(x)
    unpacked_8 = packed_8.unpack()
    assert (unpacked_8 == x).all()


def test_pack_unpack_4bits():
    x = (torch.rand(6, 10) * 16).to(torch.uint8)
    packed = TensorB4.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_4bits_trunk():
    x = (torch.rand(6, 10) * 128).to(torch.uint8).clamp(16)
    with pytest.raises(ValueError):
        TensorB4.pack(x)


def test_pack_unpack_4bits_wrong_shape():
    x = (torch.rand(5, 10) * 16).to(torch.uint8)
    with pytest.raises(ValueError):
        TensorB4.pack(x)


def test_pack_unpack_2bits():
    x = (torch.rand(8, 10) * 4).to(torch.uint8)
    packed = TensorB2.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_2bits_wrong_shape():
    x = (torch.rand(6, 10) * 4).to(torch.uint8)
    with pytest.raises(ValueError):
        TensorB2.pack(x)


def test_pack_unpack_2bits_wrong_trunk():
    x = (torch.rand(8, 10) * 5).to(torch.uint8).clamp_min(4)
    with pytest.raises(ValueError):
        TensorB2.pack(x)


def test_pack_unpack_1bits():
    x = (torch.rand(8, 10)).to(torch.uint8)
    packed = TensorB1.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_1bits_wrong_trunk():
    x = (torch.rand(8, 10) * 5).to(torch.uint8).clamp_min(4)
    with pytest.raises(ValueError):
        TensorB2.pack(x)


def test_pack_unpack_1bits_wrong_shape():
    x = (torch.rand(20, 10)).to(torch.uint8)
    with pytest.raises(ValueError):
        TensorB1.pack(x)


def test_pack_unpack_3bits():
    x = (torch.rand(10, 5) * 8).to(torch.int32)
    packed = TensorB3.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()


def test_pack_unpack_3bits_wrong_shape():
    x = (torch.rand(9, 10) * 8).to(torch.int32)
    with pytest.raises(ValueError):
        TensorB3.pack(x)


def test_pack_unpack_3bits_stored_in_u8():
    x = (torch.rand(6, 5) * 8).to(torch.uint8)
    packed = TensorB3InU8.pack(x)
    unpacked = packed.unpack()
    assert (unpacked == x).all()
