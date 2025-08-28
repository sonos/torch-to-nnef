"""Wrap model to bypass limitation of torch_to_nnef internals.

ie: Cases where inputs or outputs of a model contains tuples

"""

import logging as log
import typing as T

import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.utils import flatten_dict_tuple_or_list

LOGGER = log.getLogger(__name__)


class WrapStructIO(nn.Module):
    """Once traced it should be nop in final graph."""

    def __init__(self, model: nn.Module, input_infos, output_infos) -> None:
        super().__init__()
        self.model = model
        self.input_infos = input_infos
        self.output_infos = output_infos

    def _insert_fixed_nontraceable_args(self, flat_args):
        flat_args = list(flat_args[:])
        for idx, (_, _, elm) in enumerate(self.input_infos):
            if not isinstance(elm, torch.Tensor):
                flat_args.insert(idx, elm)
        flat_args = tuple(flat_args)
        return flat_args

    def build_inputs(self, flat_args):
        if not self.input_infos:
            return flat_args
        inps = []
        for (types, indexes, _), arg in zip(
            self.input_infos, self._insert_fixed_nontraceable_args(flat_args)
        ):
            cur_struct = inps
            for typ, next_typ, idx in zip(
                types, list(types[1:]) + [None], indexes
            ):
                if typ in (list, tuple):
                    if idx >= len(cur_struct):
                        cur_struct += [None] * (idx + 1 - len(cur_struct))
                    assert idx < len(cur_struct)
                elif typ is dict:
                    cur_struct[idx] = None
                if next_typ is tuple:
                    next_typ = list
                if cur_struct[idx] is None:
                    cur_struct[idx] = (
                        next_typ() if next_typ is not None else arg
                    )
                cur_struct = cur_struct[idx]

        # tupleization happen after structure is built
        # because tuples are immutables
        return self._tupleization(inps)

    def _tupleization(self, inps):
        tup_indexes = set()
        for types, i, _ in self.input_infos:
            for idx, typ in enumerate(types):
                # find each tuple struct indexes
                if typ is tuple:
                    tup_indexes.add(i[:idx])
        tup_indexes = sorted(list(tup_indexes), key=len)

        for idxes in tup_indexes:
            if not idxes:
                continue
            cur_struct = inps
            for idx in idxes[:-1]:
                cur_struct = cur_struct[idx]
            cur_struct[idxes[-1]] = tuple(cur_struct[idxes[-1]])
        inps = tuple(inps)
        return inps

    def flatten_outputs(self, struct_output):
        if not self.output_infos:
            return struct_output

        if (
            len(self.output_infos) == 1
            and len(self.output_infos[0][0]) == 1
            and self.output_infos[0][0][0] is tuple
        ):
            return struct_output

        return [
            o
            for _, _, o in flatten_dict_tuple_or_list(struct_output)
            if isinstance(o, torch.Tensor)
        ]

    def forward(self, *flat_args):
        struct_args = self.build_inputs(flat_args)
        struct_outputs = self.model(*struct_args)
        flat_outputs = self.flatten_outputs(struct_outputs)
        return flat_outputs


def _build_new_names_and_elements(
    original_names: T.Optional[T.List[str]],
    elms: T.Iterable,
    default_element_name_tmpl: str,
):
    """Build names of elements based on containers parents.

    Usecase 1:.
        provide:
            original_names: ['input', "a"]
            elms: [[tensor, tensor, tensor], {"arm": tensor, "head": tensor}]
    Expected output names:
        ["input_0", input_1", "input_2", "a", "head"]

    Usecase 2: (undefined names)
        provide:
            original_names: ['plop']
            elms: [[tensor, tensor, tensor], tensor, tensor]
    Expected output names:
        ["plop_0", plop_1", "plop_2",
          default_element_name_tmpl %ix=1,
          default_element_name_tmpl %ix=2
        ]

    Usecase 3: (dict with prefix)
        provide:
            original_names: ['a', 'dic']
            elms: [tensor, {"arm": tensor, "head": tensor}]
    Expected output names:
        ["a", "dic_arm", "dic_head"]
    """
    if original_names is None:
        original_names = []

    provided_names = original_names[:]
    if len(original_names) != len(elms):
        offset = len(original_names)
        for i in range(len(elms) - offset):
            provided_names.append(default_element_name_tmpl.format(i + offset))
    flat_elms = flatten_dict_tuple_or_list(elms)
    new_names = []
    new_elms = []
    for _, idxes, elm in flat_elms:
        root_idx, *rest_idxes = idxes
        if not isinstance(root_idx, int):
            raise T2NErrorNotImplemented(
                "'_build_new_names_and_elements' do only support iterable "
                "as elements not dict like"
            )

        str_idxes = "_".join(str(_) for _ in rest_idxes)
        root_name = provided_names[root_idx]
        if root_name and str_idxes:
            str_idxes = "_" + str_idxes
        if not isinstance(elm, torch.Tensor):
            ix_str = ""
            for i in idxes:
                val = "'" + i + "'" if isinstance(i, str) else i
                ix_str += f"[{val}]"
            LOGGER.warning(
                "Can only keep trace dynamic for torch.Tensor inputs/outputs  "
                "rest is CONSTANTIZED like: "
                "'%s' value: %s at index: %s "
                "(if its a container we assume no torch.Tensor inside)",
                root_name,
                elm,
                ix_str,
            )
            continue
        new_names.append(root_name + str_idxes if str_idxes else root_name)
        new_elms.append(elm)
    return new_names, new_elms, flat_elms


def has_sub_containers(flat_elms):
    return any(len(t) > 1 for t, _, _ in flat_elms)


def has_non_tensor_elements(flat_elms):
    return any(not isinstance(e, torch.Tensor) for _, _, e in flat_elms)


def may_wrap_model_to_flatten_io(model, args, outs, input_names, output_names):
    flat_args = []
    flat_outs = []
    new_input_names, args, flat_args = _build_new_names_and_elements(
        input_names, args, default_element_name_tmpl="input_{}"
    )
    if new_input_names != input_names:
        LOGGER.warning(
            "Graph inputs have been flattened so NNEF inputs are: %s",
            new_input_names,
        )
        input_names = new_input_names

    new_output_names, _, flat_outs = _build_new_names_and_elements(
        output_names, outs, default_element_name_tmpl="output_{}"
    )
    if new_output_names != output_names:
        LOGGER.warning(
            "Graph outputs have been flattened so NNEF outputs are: %s",
            new_output_names,
        )
        output_names = new_output_names

    if (
        has_sub_containers(flat_args)
        or has_sub_containers(flat_outs)
        or has_non_tensor_elements(flat_args)
        or has_non_tensor_elements(flat_outs)
    ):
        model = WrapStructIO(model, flat_args, flat_outs)
    return model, tuple(args), input_names, output_names
