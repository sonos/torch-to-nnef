import enum
import re
import string
import typing as T
from collections import Counter, defaultdict

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.torch_graph.ir_data import (
    BlobTorchScriptObject,
    Data,
    FixedTensorList,
    PythonConstant,
    TensorVariable,
    TupleTensors,
)
from torch_to_nnef.torch_graph.ir_op import (
    CacheDataNodeTarget,
    CacheDataToOpsNode,
)
from torch_to_nnef.utils import ReactiveNamedItemDict, flatten_dict

MODULE_SEPARATOR = "_."


class VariableNamingScheme(str, enum.Enum):
    RAW = "raw"
    NATURAL_VERBOSE = "natural_verbose"
    NATURAL_VERBOSE_CAMEL = "natural_verbose_camel"
    NUMERIC = "numeric"

    @classmethod
    def default(cls):
        return cls.NATURAL_VERBOSE


DEFAULT_VARNAME_SCHEME = VariableNamingScheme.default()


def apply_nnef_variable_naming_scheme(
    torch_ir_graph, scheme: VariableNamingScheme = DEFAULT_VARNAME_SCHEME
):
    """Rename availlable data node following a scheme.

    by default the natural_verbose pattern built is as close as possible
    to PyTorch graph context info. This pattern might come as too verbose.

    we propose a more concise numeric pattern that allow easier debug
    when looking at NNEF export correctness.

    """
    if scheme in [vns.value for vns in VariableNamingScheme]:
        if scheme == VariableNamingScheme.RAW:
            return
        torch_ir_graph.data_nodes.avoid_name_collision = True  # safety
        {
            VariableNamingScheme.NATURAL_VERBOSE: rename_natural_verbose,
            VariableNamingScheme.NATURAL_VERBOSE_CAMEL: rename_natural_verbose_camel,  # noqa: E501
            VariableNamingScheme.NUMERIC: rename_compact_numeric,
        }[scheme](torch_ir_graph)
        torch_ir_graph.data_nodes.avoid_name_collision = False
        return

    raise T2NErrorNotImplemented(f"renaming scheme: {scheme}")


def rename_natural_verbose(torch_ir_graph, lower: bool = True) -> None:
    out_data_to_ops_node = CacheDataToOpsNode(
        target=CacheDataNodeTarget.OUTPUTS,
        ops=torch_ir_graph.op_nodes,
    )
    for dn in list(torch_ir_graph.data_nodes.iter_renamable()):
        original_name = dn.name
        new_name = original_name.split("/")[-1]
        if all(c in string.digits for c in new_name):
            producers = out_data_to_ops_node.get(dn)
            if len(producers) == 0:
                if isinstance(dn, TensorVariable):
                    new_name = f"v{new_name}"
                elif isinstance(dn, PythonConstant):
                    new_name = f"c{new_name}"
                elif isinstance(dn, FixedTensorList):
                    new_name = f"l{new_name}"
                else:
                    raise T2NErrorNotImplemented(dn)

        if dn.name[-1] in string.digits:
            producers = out_data_to_ops_node.get(dn)
            if len(producers) > 0:
                kind = producers[0].kind.split("::")[-1]

                new_name = get_data_node_name_with_suffix_auto_inc(
                    torch_ir_graph.data_nodes,
                    original_name,
                    new_name,
                    kind,
                    suffix_only_on_underscore=True,
                )
            else:
                new_name = get_data_node_name_with_suffix_auto_inc(
                    torch_ir_graph.data_nodes,
                    original_name,
                    new_name,
                    suffix="",
                )

        if dn.name != new_name and not torch_ir_graph.data_nodes.get_by_name(
            new_name
        ):
            dn.name = new_name
    if torch_ir_graph.is_root_module:
        remove_useless_digits_from_module_names(torch_ir_graph, lower=lower)


def to_camel_case(
    snake_str: str,
    first_lower_case: bool = True,
    maintain_leading_underscore: bool = False,
):
    res = "".join(x.capitalize() for x in snake_str.split("_"))

    if snake_str.startswith("_") and (
        maintain_leading_underscore or res[0].isdigit()
    ):
        res = f"_{res}"
    if first_lower_case:
        res = res[0].lower() + res[1:]
    return res


def rename_natural_verbose_camel(torch_ir_graph):
    rename_natural_verbose(torch_ir_graph, lower=False)
    if torch_ir_graph.is_root_module:

        def _fmt_node_name(name: str):
            return "_".join(
                to_camel_case(name_chunk)
                for name_chunk in name.split(MODULE_SEPARATOR)
            )

        name_map: T.Dict[str, str] = {}
        for dnode in list(torch_ir_graph.data_nodes.iter_renamable()):
            name_map[dnode.name] = _fmt_node_name(dnode.name)
        ct = Counter(name_map.values())
        for dnode in list(torch_ir_graph.data_nodes.iter_renamable()):
            if ct[name_map[dnode.name]] == 1:
                dnode.name = name_map[dnode.name]


def rename_compact_numeric(torch_ir_graph) -> None:
    count_ref: T.Dict[str, int] = defaultdict(int)
    mapping: T.Dict[str, str] = {}
    prefix_map = {
        TensorVariable: "v",
        PythonConstant: "c",
        BlobTorchScriptObject: "b",
        FixedTensorList: "l",
        TupleTensors: "tt",
        Data: "d",  # not used, avoid static analysis complain
    }
    for dnode in list(torch_ir_graph.data_nodes.iter_renamable()):
        prefix = prefix_map[dnode.__class__]
        if dnode.name in mapping:
            dnode.name = mapping[dnode.name]
            continue
        suffix = count_ref[prefix]
        count_ref[prefix] += 1
        mapping[dnode.name] = prefix + str(suffix)
        dnode.name = mapping[dnode.name]


def replace_last_number(
    name: str,
    suffix: str,
    new_idx: int,
    suffix_only_on_underscore: bool = False,
) -> str:
    idx = -1
    while name[idx] in string.digits:
        idx -= 1
        if abs(idx) > len(name):
            assert len(suffix) > 0
            return f"{suffix}{new_idx}"

    trunced_name = name if idx == -1 else name[: idx + 1]
    if suffix and trunced_name.endswith(suffix):
        trunced_name = trunced_name[: -len(suffix)]
    if suffix and trunced_name[:-1].endswith(suffix):
        trunced_name = trunced_name[: -len(suffix) - 1]
    if (
        suffix_only_on_underscore
        and len(trunced_name)
        and trunced_name[-1] != "_"
        and (len(trunced_name) < 2 or trunced_name[-2] != "_")
    ):
        suffix = ""
    return f"{trunced_name}{suffix}{new_idx}"


def get_data_node_name_with_suffix_auto_inc(
    data_nodes: ReactiveNamedItemDict,
    original_name: str,
    refined_name: str,
    suffix="",
    suffix_only_on_underscore: bool = False,
) -> str:
    idx = 0
    new_name = replace_last_number(
        refined_name, suffix, idx, suffix_only_on_underscore
    )
    if original_name == new_name:
        return original_name
    while True:
        colliding_dn = data_nodes.get_by_name(new_name)
        if colliding_dn is None:
            break
        idx += 1
        new_name = replace_last_number(
            refined_name, suffix, idx, suffix_only_on_underscore
        )

    return new_name


def rm_digits_suffix(name: str):
    return re.sub(r"([0-9]+)?$", "", name)


def remove_useless_digits_from_module_names(torch_mod_ir_graph, lower: bool):
    """Cleanup final namings in graph.

    - Remove useless digits from module names
      for example:
        '_20__post_attention_layernorm_4__weight_expanded_1__weight'
        would become
        '_20__post_attention_layernorm__weight_expanded__weight'
        if there is no naming collision with this simlification

    """
    module_separator = MODULE_SEPARATOR
    data_node_names = list(
        i.name for i in torch_mod_ir_graph.data_nodes.iter_renamable()
    )
    assert len(data_node_names) == len(set(data_node_names))
    name_tree: T.Dict[str, T.Any] = {}
    for data_node_name in data_node_names:
        current_sub_tree = name_tree
        chunks = data_node_name.split(module_separator)
        for idx, c in enumerate(chunks):
            if c not in current_sub_tree:
                current_sub_tree[c] = (
                    data_node_name if len(chunks) - 1 == idx else {}
                )
            current_sub_tree = current_sub_tree[c]

    to_explore = [name_tree]
    while len(to_explore) > 0:
        current_sub_tree = to_explore.pop()
        keys = list(current_sub_tree.keys())
        stacked_keys = defaultdict(list)
        for key in keys:
            stacked_keys[re.sub(r"(\.[0-9]+)?$", "", key)].append(key)
        for simplified_key, original_keys in stacked_keys.items():
            if len(original_keys) == 1 and original_keys[0] != simplified_key:
                current_sub_tree[simplified_key] = current_sub_tree[
                    original_keys[0]
                ]
                del current_sub_tree[original_keys[0]]
        for next_sub_tree in current_sub_tree.values():
            if isinstance(next_sub_tree, dict):
                to_explore.append(next_sub_tree)
    remapping_table = flatten_dict(name_tree, sep=module_separator)
    for new_name, original in remapping_table.items():
        if new_name == original:
            continue
        orignal_data_node = torch_mod_ir_graph.data_nodes.get_by_name(original)
        if lower:
            new_name = new_name.lower()
        orignal_data_node.name = new_name


def rename_variable_by_incr(
    name: str,
    named_item_containers: T.List[ReactiveNamedItemDict],
    start_index: int = 1,
) -> str:
    striped_name = rm_digits_suffix(name)
    assert len(striped_name.strip()) > 0, striped_name
    idx = start_index
    proposed_name = f"{striped_name}{idx}"
    while any(
        c.get_by_name(proposed_name) is not None for c in named_item_containers
    ):
        idx += 1
        proposed_name = f"{striped_name}{idx}"
    assert proposed_name != name
    return proposed_name
