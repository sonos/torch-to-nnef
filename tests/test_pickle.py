import pickle

from tests.test_qtensor import skipif_unsupported_qtensor
import torch

from torch_to_nnef.tensor.quant import (
    fp_to_tract_q4_0_with_min_max_calibration,
)
from torch_to_nnef.tensor import NamedTensor


def test_pickle_named_tensor():
    named_tensor = NamedTensor(fp_tensor=torch.rand(2, 5), nnef_name="test_a")
    bits = pickle.dumps(named_tensor)
    restored_named_tensor = pickle.loads(bits)
    assert restored_named_tensor.nnef_name == named_tensor.nnef_name


@skipif_unsupported_qtensor
def test_pickle_qtensor():
    fp_tensor = torch.rand(2, 32)
    q_tensor = fp_to_tract_q4_0_with_min_max_calibration(
        fp_tensor, percentile=1.0
    )
    q_tensor.nnef_name = "test_b"
    bits = pickle.dumps(q_tensor)
    restored_q_tensor = pickle.loads(bits)
    assert isinstance(restored_q_tensor, q_tensor.__class__), restored_q_tensor
    assert restored_q_tensor.nnef_name == q_tensor.nnef_name
