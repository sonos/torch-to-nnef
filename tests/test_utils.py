import pytest

from torch_to_nnef.utils import flatten_dict_tuple_or_list_with_idx_and_types

FLATTEN_LIST_IOS = []


def _add_flatten_example(inp, out):
    """micro fn to clarify notation in code"""
    FLATTEN_LIST_IOS.append((inp, out))


_add_flatten_example(
    inp=[1, 2, 3],
    out=(((list,), (0,), 1), ((list,), (1,), 2), ((list,), (2,), 3)),
)
_add_flatten_example(
    inp=[[1], 2, 3],
    out=(((list, list), (0, 0), 1), ((list,), (1,), 2), ((list,), (2,), 3)),
)
_add_flatten_example(
    inp=[[1, 2], 2, 3],
    out=(
        ((list, list), (0, 0), 1),
        ((list, list), (0, 1), 2),
        ((list,), (1,), 2),
        ((list,), (2,), 3),
    ),
)
_add_flatten_example(
    inp=[[[1], 2], 2, [3]],
    out=(
        ((list, list, list), (0, 0, 0), 1),
        ((list, list), (0, 1), 2),
        ((list,), (1,), 2),
        ((list, list), (2, 0), 3),
    ),
)

_add_flatten_example(
    inp=[{"a": [1, 2], "b": [[3, 4], [5, 6]]}, {"c": 7}],
    out=(
        ((list, dict, list), (0, "a", 0), 1),
        ((list, dict, list), (0, "a", 1), 2),
        ((list, dict, list, list), (0, "b", 0, 0), 3),
        ((list, dict, list, list), (0, "b", 0, 1), 4),
        ((list, dict, list, list), (0, "b", 1, 0), 5),
        ((list, dict, list, list), (0, "b", 1, 1), 6),
        ((list, dict), (1, "c"), 7),
    ),
)

_add_flatten_example(
    inp=[{"a": 1, "b": 3}],
    out=(
        ((list, dict), (0, "a"), 1),
        ((list, dict), (0, "b"), 3),
    ),
)


@pytest.mark.parametrize("inputs,outputs", FLATTEN_LIST_IOS)
def test_flatten_tuple_or_list_with_idx(inputs, outputs):
    gen_outs = flatten_dict_tuple_or_list_with_idx_and_types(inputs)
    assert gen_outs == outputs
