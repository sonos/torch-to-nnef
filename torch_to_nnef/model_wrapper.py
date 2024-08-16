"""Wrap model to bypass limitation of torch_to_nnef internals.

ie: Cases where inputs or outputs of a model contains tuples

"""

from torch import nn

from torch_to_nnef.utils import flatten_dict_tuple_or_list_with_idx_and_types


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
        built_args = []
        for (types, idxes, _), arg in zip(self.input_infos, flat_args):
            cur_built_args = built_args
            for typ, idx in zip(types[:-1], idxes[:-1]):
                if typ in (tuple, list):
                    if idx == len(cur_built_args):
                        cur_built_args.append(typ())
                        cur_built_args = cur_built_args[-1]
                    elif idx < len(cur_built_args):
                        cur_built_args = cur_built_args[idx]
                    else:
                        raise ValueError()
                else:
                    # TODO: implement correctly
                    raise NotImplementedError(typ, idx, arg)
            assert idxes[-1] == len(cur_built_args)
            cur_built_args.append(arg)
        return built_args

    def flatten_outputs(self, struct_output):
        if not self.output_infos:
            return struct_output
        return [
            o
            for _, _, o in flatten_dict_tuple_or_list_with_idx_and_types(
                struct_output
            )
        ]

    def forward(self, *flat_args):
        struct_input = self.build_inputs(flat_args)
        struct_output = self.model(*struct_input)
        flat_output = self.flatten_outputs(struct_output)
        return flat_output


def may_wrap_model_to_flatten_io(model, args, outs, input_names, output_names):
    has_flattened_args = False
    has_flattened_outs = False
    if input_names:
        flat_args = flatten_dict_tuple_or_list_with_idx_and_types(args)
        if len(flat_args) > len(input_names):
            has_flattened_args = True
            new_input_names = []
            new_args = []
            for _, idxes, arg in flat_args:
                str_idxes = "_".join(str(_) for _ in idxes[1:])
                root_name = input_names[idxes[0]]
                new_input_names.append(
                    root_name + "_" + str_idxes if str_idxes else root_name
                )
                new_args.append(arg)
            input_names = new_input_names
            args = new_args

    if output_names:
        flat_outs = flatten_dict_tuple_or_list_with_idx_and_types(outs)
        if len(flat_outs) > len(output_names):
            has_flattened_outs = True
            new_output_names = []
            for _, idxes, _ in flat_outs:
                str_idxes = "_".join(str(_) for _ in idxes[1:])
                root_name = output_names[idxes[0]]
                new_output_names.append(
                    root_name + "_" + str_idxes if str_idxes else root_name
                )
            output_names = new_output_names
    if has_flattened_args or has_flattened_outs:
        model = WrapStructIO(model, flat_args, flat_outs)
    return model, tuple(args), input_names, output_names
