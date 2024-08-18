"""Wrap model to bypass limitation of torch_to_nnef internals.

ie: Cases where inputs or outputs of a model contains tuples

"""

from torch import nn

from torch_to_nnef.utils import flatten_dict_tuple_or_list


class WrapStructIO(nn.Module):
    """Once traced it should be nop in final graph"""

    def __init__(self, model: nn.Module, input_infos, output_infos) -> None:
        super().__init__()
        self.model = model
        self.input_infos = input_infos
        self.output_infos = output_infos

    def build_inputs(self, flat_args):
        if not self.input_infos:
            return flat_args
        inps = []
        for (types, indexes, _), arg in zip(self.input_infos, flat_args):
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

        # tupleization happen after structure is built because tuples are immutables
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
            and self.output_infos[0][0][0] == tuple
        ):
            return struct_output

        return [o for _, _, o in flatten_dict_tuple_or_list(struct_output)]

    def forward(self, *flat_args):
        struct_args = self.build_inputs(flat_args)
        struct_outputs = self.model(*struct_args)
        flat_outputs = self.flatten_outputs(struct_outputs)
        return flat_outputs


def _build_new_names_and_elements(original_names, elms):
    flat_elms = flatten_dict_tuple_or_list(elms)

    if len(flat_elms) > len(original_names):
        new_names = []
        new_elms = []
        for _, idxes, elm in flat_elms:
            str_idxes = "_".join(str(_) for _ in idxes[1:])
            root_name = original_names[idxes[0]]
            new_names.append(
                root_name + "_" + str_idxes if str_idxes else root_name
            )
            new_elms.append(elm)
        return new_names, new_elms, flat_elms
    return original_names, elms, flat_elms


def has_sub_containers(flat_elms):
    return any(len(t) > 1 for t, _, _ in flat_elms)


def may_wrap_model_to_flatten_io(model, args, outs, input_names, output_names):
    flat_args = []
    flat_outs = []
    if input_names:
        input_names, args, flat_args = _build_new_names_and_elements(
            input_names, args
        )

    if output_names:
        output_names, _, flat_outs = _build_new_names_and_elements(
            output_names, outs
        )

    if has_sub_containers(flat_args) or has_sub_containers(flat_outs):
        model = WrapStructIO(model, flat_args, flat_outs)
    return model, tuple(args), input_names, output_names
