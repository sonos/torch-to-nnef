import pytest

from torch_to_nnef.utils import flatten_tuple_or_list_with_idx

FLATTEN_LIST_IOS = []


def _add_flatten_example(inp, out):
    """micro fn to clarify notation in code"""
    FLATTEN_LIST_IOS.append((inp, out))


_add_flatten_example(inp=[1, 2, 3], out=(((0,), 1), ((1,), 2), ((2,), 3)))
_add_flatten_example(inp=[[1], 2, 3], out=(((0, 0), 1), ((1,), 2), ((2,), 3)))
_add_flatten_example(
    inp=[[1, 2], 2, 3], out=(((0, 0), 1), ((0, 1), 2), ((1,), 2), ((2,), 3))
)
_add_flatten_example(
    inp=[[[1], 2], 2, [3]],
    out=(((0, 0, 0), 1), ((0, 1), 2), ((1,), 2), ((2, 0), 3)),
)


@pytest.mark.parametrize("inputs,outputs", FLATTEN_LIST_IOS)
def test_flatten_tuple_or_list_with_idx(inputs, outputs):
    gen_outs = flatten_tuple_or_list_with_idx(inputs)
    assert gen_outs == outputs
