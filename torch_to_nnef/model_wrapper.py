"""Wrap model to bypass limitation of torch_to_nnef internals.

ie: Cases where inputs or outputs of a model contains:
    tuples, list, dicts, Object.

"""

import logging as log
import typing as T
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.utils import blank_from_init, flatten_dict_tuple_or_list

LOGGER = log.getLogger(__name__)


def insert_fixed_nontraceable_args(flat_args, input_infos):
    """Re-insert non-tensor constant values into the flat args list.

    During flattening, non-tensor elements (ints, bools, …) are recorded in
    *input_infos* but excluded from the dynamic *flat_args*.  This function
    splices them back at the correct positions so that *flat_args* aligns 1-1
    with *input_infos* again.
    """
    flat_args = list(flat_args[:])
    for idx, (_, _, elm) in enumerate(input_infos):
        if not isinstance(elm, torch.Tensor):
            flat_args.insert(idx, elm)
    return tuple(flat_args)


def build_structured_inputs(flat_args, input_infos):
    """Rebuild structured inputs from a flat args sequence.

    Args:
        flat_args: Flat sequence of tensor values (non-tensor constants are
            automatically re-inserted from *input_infos*).
        input_infos: Flattened element descriptors produced by
            :func:`flatten_dict_tuple_or_list` — each entry is
            ``(types, indexes, original_value)``.

    Returns:
        Tuple of structured arguments matching the original model signature.
    """
    if not input_infos:
        return flat_args

    full_args = insert_fixed_nontraceable_args(flat_args, input_infos)
    inps: list = []
    for (types, indexes, _), arg in zip(input_infos, full_args):
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
            if isinstance(idx, str) and hasattr(cur_struct, idx):
                setattr(cur_struct, idx, arg)
                cur_struct = getattr(cur_struct, idx)
                continue
            if cur_struct[idx] is None:
                cur_struct[idx] = (
                    blank_from_init(next_typ)
                    if next_typ is not None
                    else arg
                )
            cur_struct = cur_struct[idx]

    return tupleize_structure(inps, input_infos)


def tupleize_structure(inps, input_infos):
    """Convert mutable lists back to tuples where the original had tuples.

    During reconstruction lists are used because tuples are immutable.
    This pass converts them back based on the type information recorded in
    *input_infos*.
    """
    tup_indexes: set = set()
    for types, i, _ in input_infos:
        for idx, typ in enumerate(types):
            if typ is tuple:
                tup_indexes.add(i[:idx])
    tup_indexes_sorted = sorted(list(tup_indexes), key=len)

    for idxes in tup_indexes_sorted:
        if not idxes:
            continue
        cur_struct = inps
        for idx in idxes[:-1]:
            cur_struct = cur_struct[idx]
        cur_struct[idxes[-1]] = tuple(cur_struct[idxes[-1]])
    return tuple(inps)


def flatten_structured_outputs(struct_output, output_infos):
    """Flatten structured model outputs to a flat list of tensors.

    If the output is already a simple tuple of tensors, it is returned as-is.
    """
    if not output_infos:
        return struct_output

    if (
        len(output_infos) == 1
        and len(output_infos[0][0]) == 1
        and output_infos[0][0][0] is tuple
    ):
        return struct_output

    return [
        o
        for _, _, o in flatten_dict_tuple_or_list(struct_output)
        if isinstance(o, torch.Tensor)
    ]


class WrapStructIO(nn.Module):
    """Once traced it should be nop in final graph."""

    def __init__(self, model: nn.Module, input_infos, output_infos) -> None:
        super().__init__()
        self.model = model
        self.input_infos = input_infos
        self.output_infos = output_infos

    def forward(self, *flat_args):
        struct_args = build_structured_inputs(flat_args, self.input_infos)
        struct_outputs = self.model(*struct_args)
        return flatten_structured_outputs(struct_outputs, self.output_infos)


def build_new_names_and_elements(
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
    new_flat_elms = []
    for _, idxes, elm in flat_elms:
        root_idx, *rest_idxes = idxes
        if not isinstance(root_idx, int):
            raise T2NErrorNotImplemented(
                "'build_new_names_and_elements' do only support iterable "
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
        new_flat_elms.append((_, idxes, elm))

    # Guard against duplicate flat names — they would produce a broken
    # graph (ambiguous IO in NNEF, broken outputs_keep filtering, etc.).
    seen: dict[str, int] = {}
    for n in new_names:
        seen[n] = seen.get(n, 0) + 1
    dupes = sorted(n for n, c in seen.items() if c > 1)
    if dupes:
        raise T2NErrorNotImplemented(
            "Flattening produced duplicate names: "
            + ", ".join(f"'{d}'" for d in dupes)
            + ". This happens when a raw name like 'x_0' collides with "
            "a generated suffix from container 'x'. "
            "Rename the model outputs/inputs to avoid the collision."
        )

    return new_names, new_elms, flat_elms, new_flat_elms


def has_sub_containers(flat_elms):
    return any(len(t) > 1 for t, _, _ in flat_elms)


def has_non_tensor_elements(flat_elms):
    return any(not isinstance(e, torch.Tensor) for _, _, e in flat_elms)


@dataclass()
class UnfoldModelInfo:
    """Hold model input/output structure information."""

    model: nn.Module
    original_inputs: T.Tuple[torch.Tensor]
    original_outputs: T.List[torch.Tensor]
    # what will be exported as io for the final NNEF graph,
    # since container are not supported {
    flat_inputs: T.Tuple[torch.Tensor]
    flat_outputs: T.Tuple[torch.Tensor]
    input_names: T.List[str]
    output_names: T.List[str]

    # }

    @property
    def original_model(self) -> nn.Module:
        if isinstance(self.model, WrapStructIO):
            return self.model.model
        return self.model

    def validate(self):
        assert len(self.input_names) == len(self.flat_inputs), (
            "input names length mismatch:"
            f"{len(self.input_names)} != {len(self.flat_inputs)}"
        )
        assert len(self.output_names) == len(self.flat_outputs), (
            "output names length mismatch:"
            f"{len(self.output_names)} != {len(self.flat_outputs)}. "
            f"with output names: {self.output_names}"
        )

    def write_input_npz(self, filepath: Path, tract_compat: bool = False):
        self._write_tensor_npz(
            names=self.input_names,
            tensors=self.flat_inputs,
            filepath=filepath,
            tract_compat=tract_compat,
        )

    def write_output_npz(self, filepath: Path, tract_compat: bool = False):
        self._write_tensor_npz(
            names=self.output_names,
            tensors=self.flat_outputs,
            filepath=filepath,
            tract_compat=tract_compat,
        )

    def _write_tensor_npz(
        self,
        *,
        names: T.Sequence[str],
        tensors: T.Sequence[torch.Tensor],
        filepath: Path,
        tract_compat: bool = False,
    ) -> None:
        def cast(val: torch.Tensor):
            if val.dtype in (torch.float16, torch.bfloat16):
                val = val.to(torch.float32)
            return val.detach().numpy()

        payload = {
            name: (cast(t) if tract_compat else t)
            for name, t in zip(names, tensors)
        }
        np.savez(filepath, **payload)


def cast_tensor_if_int(inp: T.Any) -> torch.Tensor:
    if isinstance(inp, int):
        return torch.tensor(inp)
    return inp


def unfold_model_io(model, args, outs, input_names, output_names):
    if isinstance(model, WrapStructIO):
        raise T2NErrorNotImplemented(
            "Model is already wrapped with 'WrapStructIO', "
            "double wrapping is not supported."
        )
    if not isinstance(model, nn.Module):
        raise T2NErrorNotImplemented(
            "Only 'nn.Module' model type is supported for unfolding, "
            f"got '{type(model)}'."
        )

    if isinstance(outs, torch.Tensor):
        outs = (outs,)

    new_input_names, args, flat_args, new_flat_args = (
        build_new_names_and_elements(
            input_names, args, default_element_name_tmpl="input_{}"
        )
    )
    if new_input_names != input_names:
        LOGGER.warning(
            "Graph inputs have been flattened so NNEF inputs are: %s",
            new_input_names,
        )
        input_names = new_input_names

    new_output_names, _, flat_outs, new_flat_outs = (
        build_new_names_and_elements(
            output_names, outs, default_element_name_tmpl="output_{}"
        )
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
        model.eval()
    return UnfoldModelInfo(
        model=model,
        original_inputs=tuple(args),
        original_outputs=outs,
        flat_inputs=tuple(cast_tensor_if_int(_[2]) for _ in new_flat_args),
        flat_outputs=tuple(cast_tensor_if_int(_[2]) for _ in new_flat_outs),
        input_names=input_names,
        output_names=output_names,
    )
