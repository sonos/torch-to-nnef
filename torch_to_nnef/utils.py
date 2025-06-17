from collections import defaultdict
import contextlib
import functools
import logging
import os
import typing as T
from abc import ABC
from collections.abc import MutableMapping
from functools import total_ordering

import torch
from torch import _C

from torch_to_nnef.exceptions import (
    DataNodeValueError,
    InconsistentTensorError,
    TorchToNNEFNotImplementedError,
)

LOGGER = logging.getLogger(__name__)

C = T.TypeVar("C")


def cache(func: T.Callable[..., C]) -> C:
    """LRU cache helper that avoid pylint complains"""
    return functools.lru_cache()(func)  # type: ignore


def fullname(o) -> str:
    """Full class name with module path from an object"""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


@contextlib.contextmanager
def cd(path):
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dedup_list(lst: T.List[T.Any]) -> T.List[T.Any]:
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def flatten_dict_tuple_or_list(
    obj,
    collected_types: T.Optional[T.List[T.Type]] = None,
    collected_idxes: T.Optional[T.List[int]] = None,
    current_idx: int = 0,
) -> T.Tuple[
    T.Tuple[T.Tuple[T.Type, ...], T.Tuple[T.Union[int, str], ...], T.Any], ...
]:
    """Flatten dict/list/tuple recursively, return types, indexes and values

    Flatten in depth first search order

    Args:
        obj: dict/tuple/list or anything else (structure can be arbitrary deep)
            this contains N number of element non dict/list/tuple

        collected_types: do not set
        collected_idxes: do not set
        current_idx: do not set

    Return:
        tuple of N tuples each containing a tuple of:
            types, indexes and the element

    Example:
        If initial obj=[{"a": 1, "b": 3}]
        it will output:
            (
                ((list, dict), (0, "a"), 1),
                ((list, dict), (0, "b"), 3),
            )
    """
    if collected_idxes is None:
        collected_idxes = []
    else:
        collected_idxes = collected_idxes[:]

    if collected_types is None:
        collected_types = []
    else:
        collected_types = collected_types[:]

    collected_types.append(type(obj))
    if isinstance(obj, (tuple, list)):
        collected_idxes.append(current_idx)
        current_idx += 1
        if not obj:
            return ()
        if isinstance(obj[0], (tuple, list, dict)):
            return flatten_dict_tuple_or_list(
                obj[0], collected_types, collected_idxes, 0
            ) + flatten_dict_tuple_or_list(
                obj[1:],
                collected_types[:-1],
                collected_idxes[:-1],
                current_idx,
            )
        return (
            (tuple(collected_types), tuple(collected_idxes), obj[0]),
        ) + flatten_dict_tuple_or_list(
            obj[1:], collected_types[:-1], collected_idxes[:-1], current_idx
        )
    if isinstance(obj, dict):
        res = []  # type: ignore
        for k, v in obj.items():
            if isinstance(v, (tuple, list, dict)):
                res += flatten_dict_tuple_or_list(
                    v, collected_types, collected_idxes + [k], 0
                )
            else:
                res += [
                    (tuple(collected_types), tuple(collected_idxes + [k]), v)
                ]
        return tuple(res)
    return ()


@contextlib.contextmanager
def init_empty_weights(include_buffers: T.Optional[bool] = None):
    """Borrowed from `accelerate`
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from  import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    include_buffers = include_buffers or False
    with init_on_device(
        torch.device("meta"), include_buffers=include_buffers
    ) as f:
        yield f


@contextlib.contextmanager
def init_on_device(
    device: torch.device, include_buffers: T.Optional[bool] = None
):
    """Borrowed from `accelerate`
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    include_buffers = include_buffers or False

    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


@total_ordering
class SemanticVersion:
    """Helper to check a version is higher than another"""

    TAGS = ["major", "minor", "patch"]

    def __init__(self, **kwargs):
        for t in self.TAGS:
            assert isinstance(kwargs[t], int), kwargs[t]
            assert kwargs[t] >= 0, kwargs[t]

        self.version = {t: kwargs[t] for t in self.TAGS}

    @classmethod
    def from_str(cls, version_str, sep="."):
        version_chunks = version_str.strip().split(sep)
        if "-" in version_chunks[-1]:
            version_chunks[-1] = version_chunks[-1].split("-")[0]
        vtags = list(map(int, version_chunks))
        assert len(vtags) == len(cls.TAGS)
        return cls(**dict(zip(cls.TAGS, vtags)))

    def __eq__(self, other: object):
        if isinstance(other, str):
            other = SemanticVersion.from_str(other)
        assert isinstance(other, SemanticVersion), other
        return all(self.version[t] == other.version[t] for t in self.TAGS)

    def __lt__(self, other: object):
        if isinstance(other, str):
            other = SemanticVersion.from_str(other)
        assert isinstance(other, SemanticVersion), other
        for t in self.TAGS:
            if self.version[t] < other.version[t]:
                return True
            if self.version[t] > other.version[t]:
                return False
        return False

    def to_str(self):
        return ".".join(str(self.version[t]) for t in self.TAGS)

    def __repr__(self) -> str:
        return f"<Version {self.to_str()}>"


def torch_version() -> SemanticVersion:
    """Semantic version for torch"""
    return SemanticVersion.from_str(torch.__version__.split("+")[0])


def select_ctx_disable_torch_fn():
    if hasattr(_C, "DisableTorchFunctionSubclass"):  # post torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunctionSubclass()
    elif hasattr(_C, "DisableTorchFunction"):  # pre torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunction()
    else:
        raise TorchToNNEFNotImplementedError(
            f"How to disable torch function in torch=={torch_version()}"
        )
    return ctx_disable_torch_fn


class NamedItem(ABC):
    # must implement name attribute
    name: str

    def register_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()
        if listener in self._name_hooks:
            raise DataNodeValueError("Already registered  listener !")
        self._name_hooks.add(listener)

    def detach_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()
        self._name_hooks.remove(listener)

    def __setattr__(self, attr_name, attr_value):
        if attr_name == "name" and hasattr(self, "_name_hooks"):
            for name_hook in self._name_hooks:
                name_hook(self.name, attr_value)
        super().__setattr__(attr_name, attr_value)


def get_parent_module_and_param_name(
    model: torch.nn.Module, full_name: str
) -> T.Tuple[torch.nn.Module, str]:
    ref_mod = model
    chunked_names = full_name.split(".")
    for mod_name in chunked_names[:-1]:
        ref_mod = getattr(ref_mod, mod_name)
    return ref_mod, chunked_names[-1]


class ParametersModelUpdater:
    """Helper to update parameters of a model cleanly"""

    def __init__(
        self,
        model: torch.nn.Module,
        add_parameter_if_unset=True,
    ):
        self.name_to_id = {}
        id_to_names = defaultdict(set)
        mod_name_to_param_names = defaultdict(list)
        for param_name, param in model.named_parameters(remove_duplicate=False):
            self.name_to_id[param_name] = id(param)
            id_to_names[id(param)].add(param_name)
            mod_name, _ = self.split_param_name(param_name)
            mod_name_to_param_names[mod_name].append(param_name)
        self.id_to_names = {k: frozenset(v) for k, v in id_to_names.items()}
        self.name_to_parent_module = {}
        for mod_name, mod in model.named_modules():
            for p_name in mod_name_to_param_names[mod_name]:
                self.name_to_parent_module[p_name] = mod
        self.model = model
        self.add_parameter_if_unset = add_parameter_if_unset

    @staticmethod
    def split_param_name(name: str) -> T.Tuple[str, str]:
        if "." in name:
            mod_name, param_local_name = name.rsplit(".", 1)
        else:
            mod_name = ""
            param_local_name = name
        return mod_name, param_local_name

    def check_same_dtype(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if tensor0.dtype != tensor1.dtype:
            raise InconsistentTensorError(
                f"not same dtype between {tensor0.dtype} and {tensor1.dtype}"
            )

    def check_same_device(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if tensor0.device != tensor1.device:
            raise InconsistentTensorError(
                f"not same device between {tensor0.device} and {tensor1.device}"
            )

    def check_same_shape(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if tensor0.shape != tensor1.shape:
            raise InconsistentTensorError(
                f"not same shape between {tensor0.shape} and {tensor1.shape}"
            )

    def check_all(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        self.check_same_device(tensor0, tensor1)
        self.check_same_dtype(tensor0, tensor1)
        self.check_same_shape(tensor0, tensor1)

    def maybe_parameterize(self, new_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(new_tensor, torch.nn.Parameter):
            if self.add_parameter_if_unset:
                new_tensor = torch.nn.Parameter(new_tensor, requires_grad=False)
            else:
                raise ValueError(f"{new_tensor} must be a parameter")
        return new_tensor

    def update_by_ref(
        self,
        ref: torch.nn.Parameter,
        new_tensor: torch.Tensor,
        enforce_same_shape_dtype_device: bool = True,
    ):
        new_tensor = self.maybe_parameterize(new_tensor)
        if enforce_same_shape_dtype_device:
            self.check_all(ref, new_tensor)
        need_grad = False
        if ref.requires_grad:
            need_grad = True
            ref.requires_grad = False
        ref.set_(new_tensor)
        if need_grad and not ref.requires_grad:
            ref.requires_grad = True

    def update_by_name(
        self,
        name: str,
        new_tensor: torch.Tensor,
        tie_replacements: bool = True,
        enforce_same_shape_dtype_device: bool = True,
    ):
        new_tensor = self.maybe_parameterize(new_tensor)
        mod = self.name_to_parent_module[name]
        _, p_local_name = self.split_param_name(name)
        old = mod._parameters[p_local_name]
        if enforce_same_shape_dtype_device:
            self.check_all(old, new_tensor)
        if tie_replacements:
            need_grad = False
            if old.requires_grad:
                need_grad = True
                old.requires_grad = False
            if (
                old.device == new_tensor.device
                and old.dtype == new_tensor.dtype
            ):
                old.set_(new_tensor)
            else:
                # slower but allow device or dtype change
                for n in self.id_to_names[id(old)]:
                    self.name_to_parent_module[n]._parameters[
                        self.split_param_name(n)[1]
                    ] = new_tensor
            if need_grad and not old.requires_grad:
                old.requires_grad = True
        else:
            mod._parameters[p_local_name] = new_tensor
            if (
                old.requires_grad
                and not mod._parameters[p_local_name].requires_grad
            ):
                mod._parameters[p_local_name].requires_grad = True


class ReactiveNamedItemDict:
    """Named items ordered Dict data structure

    Ensure that 'NO' 2 items are inserted with same 'name' attribute
    and maintains fast name update and with some additive colision
    protections.

    Warning! only aimed at NamedItem subclass.

    Expose a 'list' like interface. (with limited index access)

    """

    def __init__(self, items: T.Optional[T.Iterable[NamedItem]] = None):
        self._map: T.Dict[str, T.Any] = {}
        self._last_inserted_item: T.Optional[NamedItem] = None
        if items is not None:
            self.__add__(items)
        self.avoid_name_collision = False
        self._protected_names: T.Set[str] = set()

    @classmethod
    def from_list(cls, items) -> "ReactiveNamedItemDict":
        if not items:
            return cls()
        return cls() + items

    def _change_name_hook(self, old_name: str, new_name: str):
        """maintain sync between data structure and name changes in items"""
        if old_name in self._protected_names:
            raise DataNodeValueError(
                f"Not allowed to alter protected_name: {old_name}"
            )
        if new_name in self._protected_names:
            raise DataNodeValueError(
                f"Not allowed to alter protected_name: {new_name}"
            )
        if new_name == old_name:
            return
        if new_name in self._map:
            msg = f"node with name:{new_name} overwritten in {self}"
            LOGGER.debug(msg)
            if self.avoid_name_collision:
                raise DataNodeValueError(msg)
        self._map[new_name] = self._map[old_name]
        del self._map[old_name]

    def detach_listener_name_change_for_item(self, item):
        self._map[item.name].detach_listener_name_change(self._change_name_hook)

    def remove(
        self,
        item: NamedItem,
        raise_exception_if_not_found: bool = True,
        raise_exception_if_protected_name: bool = True,
    ):
        if item.name not in self._map:
            msg = f"item '{item.name}' requested for deletion. Not Found !"
            if raise_exception_if_not_found:
                raise DataNodeValueError(msg)
            LOGGER.debug(msg)
            return
        if (
            item.name in self._protected_names
            and not raise_exception_if_protected_name
        ):
            raise DataNodeValueError(f"Not authorized to remove: '{item.name}'")
        self.detach_listener_name_change_for_item(item)
        del self._map[item.name]

    def get_by_name(self, name: str, default: T.Any = None):
        return self._map.get(name, default)

    def contains(self, item: NamedItem, strict: bool = False):
        name_exists = item.name in self._map
        if name_exists and strict:
            return self._map[item.name] == item
        return name_exists

    def append(self, item: NamedItem):
        """Append item to ordered set

        WARNING: This is crucial that all added items use this
        function as it set the hook to listen to name changes
        """
        if item.name in self._map:
            raise DataNodeValueError(
                f"`{item.name}` already exist in container:"
                f" {self._map[item.name]}, "
                f"but tried to add an item: {item} with same name."
            )
        item.register_listener_name_change(self._change_name_hook)
        self._map[item.name] = item
        self._last_inserted_item = item

    def protect_item_names(self, names: T.Iterable[str]):
        self._protected_names.update(set(names))

    def is_empty(self):
        return not bool(self._map)

    def __getitem__(self, index: T.Any):
        if index == -1:
            if self._last_inserted_item is None:
                raise ValueError("No last value found")
            return self._last_inserted_item
        if isinstance(index, (slice, int)):
            return list(self._map.values())[index]
        raise NotImplementedError(index)

    def iter_renamable(self):
        for item_name, item in self._map.items():
            if item_name in self._protected_names:
                continue
            yield item

    def __add__(self, items):
        for item in items:
            self.append(item)
        return self

    def __setitem__(self, index: T.Any, value: NamedItem):
        raise NotImplementedError(
            "Assigning a specific index is not supported to date"
        )

    def __iter__(self):
        yield from self._map.values()

    def __len__(self):
        return len(self._map)

    def __repr__(self):
        names = "\n\t".join(f"'{k}'" for k in self._map)
        protected = ""
        if self._protected_names:
            pnames = ",\n".join(f"\t'{k}'" for k in self._protected_names)
            protected = f"\nprotected_names=[\n{pnames}]\n"
        return f"<ReactiveNamedItemDict ({len(self._map)}) stored_names=[{names}] {protected}>"
