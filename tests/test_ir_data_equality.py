import numpy as np
import torch

from torch_to_nnef.torch_graph.ir_data import TensorVariable, _data_equal


def _tensor_var(name, data):
    return TensorVariable(
        name=name,
        data=data,
        shape=list(data.shape),
        dtype=data.dtype,
        quant=None,
    )


def test_tensor_variable_equality_compares_base_tensor_values():
    left = _tensor_var("weight", torch.tensor([1.0, 2.0]))
    same = _tensor_var("weight", torch.tensor([1.0, 2.0]))
    different = _tensor_var("weight", torch.tensor([1.0, 3.0]))

    assert left == same
    assert left != different


def test_tensor_variable_equality_handles_meta_tensor_data():
    left = _tensor_var("weight", torch.empty(2, 3, device="meta"))
    same = _tensor_var("weight", torch.empty(2, 3, device="meta"))
    different_shape = _tensor_var("weight", torch.empty(3, 2, device="meta"))

    assert left == same
    assert left != different_shape


def test_data_equal_handles_numpy_arrays_without_ambiguous_truth_value():
    left = np.array([1, 2, 3], dtype=np.int64)
    same = np.array([1, 2, 3], dtype=np.int64)
    different = np.array([1, 2, 4], dtype=np.int64)

    assert _data_equal(left, same)
    assert not _data_equal(left, different)
