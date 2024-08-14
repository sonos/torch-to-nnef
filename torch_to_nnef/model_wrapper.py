"""Wrap model to bypass limitation of torch_to_nnef internals.

ie: Cases where inputs or outputs of a model contains tuples

"""

from torch import nn

from torch_to_nnef.utils import flatten_tuple_or_list_with_idx


class WrapStructIO(nn.Module):
    """Once traced it should be nop in final graph"""

    def __init__(self, model: nn.Module, input_indexes, output_indexes) -> None:
        super().__init__()
        self.model = model
        self.input_indexes = input_indexes
        self.output_indexes = output_indexes

    def build_inputs(self, flat_args):
        if not self.input_indexes:
            return flat_args
        built_args = []
        for idxes, arg in zip(self.input_indexes, flat_args):
            cur_built_args = built_args
            for idx in idxes[:-1]:
                if idx == len(cur_built_args):
                    cur_built_args.append([])
                    cur_built_args = cur_built_args[-1]
                elif idx < len(cur_built_args):
                    cur_built_args = cur_built_args[idx]
                else:
                    raise ValueError()
            assert idxes[-1] == len(cur_built_args)
            cur_built_args.append(arg)
        if len(built_args) == 2:
            built_args[1] = tuple(built_args[1])
        return built_args

    def flatten_outputs(self, struct_output):
        if not self.output_indexes:
            return struct_output
        return [o for _, o in flatten_tuple_or_list_with_idx(struct_output)]

    def forward(self, *flat_args):
        struct_input = self.build_inputs(flat_args)
        struct_output = self.model(*struct_input)
        flat_output = self.flatten_outputs(struct_output)
        return flat_output


def may_wrap_model_to_flatten_io(model, args, outs, input_names, output_names):
    new_inputs_idxes = []
    new_outputs_idxes = []
    if input_names:
        flat_args = flatten_tuple_or_list_with_idx(args)
        if len(flat_args) > len(input_names):
            new_input_names = []
            new_args = []
            for idxes, arg in flat_args:
                str_idxes = "_".join(str(_) for _ in idxes[1:])
                root_name = input_names[idxes[0]]
                new_input_names.append(
                    root_name + "_" + str_idxes if str_idxes else root_name
                )
                new_inputs_idxes.append(idxes)
                new_args.append(arg)
            input_names = new_input_names
            args = new_args

    if output_names:
        flat_outs = flatten_tuple_or_list_with_idx(outs)
        if len(flat_outs) > len(output_names):
            new_output_names = []
            for idxes, _ in flat_outs:
                str_idxes = "_".join(str(_) for _ in idxes[1:])
                root_name = output_names[idxes[0]]
                new_output_names.append(
                    root_name + "_" + str_idxes if str_idxes else root_name
                )
                new_outputs_idxes.append(idxes)
            output_names = new_output_names
    if new_inputs_idxes or new_outputs_idxes:
        model = WrapStructIO(model, new_inputs_idxes, new_outputs_idxes)
    return model, tuple(args), input_names, output_names
