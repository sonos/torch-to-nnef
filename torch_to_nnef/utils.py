import contextlib
import functools
import importlib
import inspect
import logging
import os
import re
import typing as T
from abc import ABC
from collections.abc import MutableMapping
from enum import Enum
from functools import lru_cache, total_ordering, wraps

import torch
from torch import _C

from torch_to_nnef.exceptions import (
    T2NErrorDataNodeValue,
    T2NErrorMisuse,
    T2NErrorNotImplemented,
)

LOGGER = logging.getLogger(__name__)

C = T.TypeVar("C")


def cache(func: T.Callable[..., C]) -> C:
    """LRU cache helper that avoid pylint complains."""
    return functools.lru_cache()(func)  # type: ignore


def ensure_tuple_io(value: T.Any) -> T.Tuple[T.Any, ...]:
    """Normalize inputs/outputs into a tuple.

    Behavior:
    - If already a tuple, return as-is.
    - If a list or other finite sequence, return tuple(value).
    - If a single Tensor, number, bool, or mapping-like (has items and
      __getitem__), wrap into a 1-tuple.
    - Otherwise, return (value,).
    """
    if isinstance(value, tuple):
        return value
    # Preserve common containers
    if isinstance(value, list):
        return tuple(value)
    # Torch tensor or primitive
    if isinstance(value, (torch.Tensor, int, float, bool)):
        return (value,)
    # Dicts are preserved as single entries; other custom containers are not
    # treated specially to avoid surprising wrapping heuristics.
    if isinstance(value, dict):
        return (value,)
    return (value,)


def fullname(o) -> str:
    """Full class name with module path from an object."""
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


@contextlib.contextmanager
def cd(path):
    """Context manager for changing the current working directory."""
    old_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_path)


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    """Flatten a nested dictionary."""
    items: T.List[T.Tuple[str, T.Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dedup_list(lst: T.List[T.Any]) -> T.List[T.Any]:
    """Remove duplicates from list while preserving order."""
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def normalize_cli_list_option(
    values: T.Optional[T.Iterable[T.Any]],
) -> T.Optional[T.List[str]]:
    """Normalize repeated/CSV CLI options into a list of unique strings.

    Accepts values from argparse patterns like `action="append"` and also
    tolerates a single string. Splits on commas, strips whitespace, removes
    empty entries, and de-duplicates while preserving order. Returns None if
    the input is falsy.
    """
    if not values:
        return None
    # Coerce a single string to an iterable interface
    if isinstance(values, str):
        values = [values]
    flat: T.List[str] = []
    for item in values:
        text = item if isinstance(item, str) else str(item)
        flat.extend(part.strip() for part in text.split(",") if part.strip())
    # Deduplicate preserving order
    seen: set[str] = set()
    out: T.List[str] = []
    for s in flat:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def flatten_dict_tuple_or_list(
    obj: T.Any,
    collected_types: T.Optional[T.List[T.Type]] = None,
    collected_idxes: T.Optional[T.List[int]] = None,
    current_idx: int = 0,
) -> T.Tuple[
    T.Tuple[T.Tuple[T.Type, ...], T.Tuple[T.Union[int, str], ...], T.Any], ...
]:
    """Flatten dict/list/tuple recursively, return types, indexes and values.

    Flatten happen in depth first search order

    Args:
        obj: dict/tuple/list or anything else (structure can be arbitrary deep)
            this contains N number of element non dict/list/tuple
        collected_types: do not set
        collected_idxes: do not set
        current_idx: do not set

    Returns:
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
    collected_idxes = [] if collected_idxes is None else collected_idxes[:]
    collected_types = [] if collected_types is None else collected_types[:]

    def is_flattenable_obj(obj):
        return not isinstance(obj, torch.Tensor) and (
            hasattr(obj, "__getitem__") or hasattr(obj, "__dict__")
        )

    def is_supported_container(obj):
        return isinstance(obj, (tuple, list, dict))

    def is_flattenable(obj):
        return is_flattenable_obj(obj) or is_supported_container(obj)

    collected_types.append(type(obj))
    if isinstance(obj, (tuple, list)):
        collected_idxes.append(current_idx)
        current_idx += 1
        if not obj:
            return ()
        if is_flattenable(obj[0]):
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
    if is_flattenable_obj(obj):
        if hasattr(obj, "__dict__"):
            obj = obj.__dict__
        res = []  # type: ignore
        for k, v in obj.items():
            if is_flattenable(v):
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
def init_empty_weights(
    include_buffers: T.Optional[bool] = None,
) -> T.Iterator[None]:
    """A context manager under which models init with meta device.

    Borrowed from `accelerate`

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model.
    Useful when just initializing the model would blow the available RAM.


    Args:
        include_buffers:
            Whether or not to also put all buffers on the meta device
            while initializing.

    Returns:
        (None) Just a context manager

    Example:
    ```python
    import torch.nn as nn
    from  import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and
    # without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights.
    As such you can't do something like `model.to(some_device)` with it.
    To load weights inside an empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param
    for [`load_checkpoint_and_dispatch`], otherwise dispatch is not called.

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
) -> T.Iterator[None]:
    """Context manager under which models are init on the specified device.

    Borrowed from `accelerate`

    Args:
        device:
            Device to initialize all parameters on.
        include_buffers:
            Whether or not to also put all buffers on the meta device
            while initializing.

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
        # pylint: disable-next=possibly-used-before-assignment
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    tensor_constructors_to_patch: T.Dict[str, T.Callable] = {}
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch:
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
    """SemVer 2.0 compatible version class.

    Supports:
        1.2.3
        1.2.3-alpha
        1.2.3-alpha.1
        1.2.3-rc.1
        1.2.3+build.5  (build metadata ignored in ordering)

    Allows symmetric comparison with strings:
        "1.2.0" < SemanticVersion.from_str("1.3.0")

    Example:
        >>> version = SemanticVersion.from_str("1.2.13")
        >>> "1.2.12" < version < "1.2.14"
        True
        >>> "1.3.12" < version
        False
        >>> version == "1.2.13"
        True
    """

    __slots__ = (
        "major",
        "minor",
        "patch",
        "prerelease",
        "build",
        "_cmp_key",
    )

    _SEMVER_RE = re.compile(
        r"""
        ^
        (?P<major>0|[1-9]\d*)\.
        (?P<minor>0|[1-9]\d*)\.
        (?P<patch>0|[1-9]\d*)
        (?:-(?P<prerelease>[0-9A-Za-z.-]+))?
        (?:\+(?P<build>[0-9A-Za-z.-]+))?
        $
        """,
        re.VERBOSE,
    )

    def __init__(self, major, minor, patch, prerelease=None, build=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.prerelease = tuple(prerelease or ())
        self.build = build

        # Precompute comparison key (critical for performance)
        self._cmp_key = self._build_cmp_key()

    @classmethod
    def from_str(cls, version: str) -> "SemanticVersion":
        m = cls._SEMVER_RE.match(version.strip())
        if not m:
            raise ValueError(f"Invalid semantic version: {version}")

        major = int(m.group("major"))
        minor = int(m.group("minor"))
        patch = int(m.group("patch"))

        prerelease_raw = m.group("prerelease")
        if prerelease_raw:
            parsed = []
            for part in prerelease_raw.split("."):
                if part.isdigit():
                    parsed.append(int(part))
                else:
                    parsed.append(part)
            prerelease = tuple(parsed)
        else:
            prerelease = ()

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=m.group("build"),
        )

    def _build_cmp_key(self):
        """Build a tuple usable for correct SemVer precedence comparison."""
        # Core version
        core = (self.major, self.minor, self.patch)

        # Pre-release ordering:
        # No prerelease > prerelease
        if not self.prerelease:
            pre_key = (1,)  # higher than any prerelease
        else:
            normalized = []
            for identifier in self.prerelease:
                if isinstance(identifier, int):
                    normalized.append((0, identifier))
                else:
                    normalized.append((1, identifier))
            pre_key = (0, tuple(normalized))

        return core + (pre_key,)

    def _coerce_other(self, other):
        if isinstance(other, str):
            return SemanticVersion.from_str(other)
        if isinstance(other, SemanticVersion):
            return other
        raise TypeError(f"Cannot compare SemanticVersion with {type(other)}")

    def __eq__(self, other):
        other = self._coerce_other(other)
        return self._cmp_key == other._cmp_key

    def __lt__(self, other):
        other = self._coerce_other(other)
        return self._cmp_key < other._cmp_key

    def to_str(self) -> str:
        """Return canonical SemVer string."""
        return str(self)

    def __str__(self):
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += "-" + ".".join(str(p) for p in self.prerelease)
        if self.build:
            base += f"+{self.build}"
        return base

    def __repr__(self):
        return f"<SemanticVersion {self}>"


def torch_version() -> SemanticVersion:
    """Semantic version for torch."""
    return SemanticVersion.from_str(torch.__version__.split("+")[0])


def select_ctx_disable_torch_fn():
    if hasattr(_C, "DisableTorchFunctionSubclass"):  # post torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunctionSubclass()
    elif hasattr(_C, "DisableTorchFunction"):  # pre torch 2.0.0
        ctx_disable_torch_fn = _C.DisableTorchFunction()
    else:
        raise T2NErrorNotImplemented(
            f"How to disable torch function in torch=={torch_version()}"
        )
    return ctx_disable_torch_fn


def get_parent_module_and_param_name(
    model: torch.nn.Module, full_name: str
) -> T.Tuple[torch.nn.Module, str]:
    ref_mod = model
    chunked_names = full_name.split(".")
    for mod_name in chunked_names[:-1]:
        ref_mod = getattr(ref_mod, mod_name)
    return ref_mod, chunked_names[-1]


@lru_cache(10)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


class NamedItem(ABC):
    # must implement name attribute
    name: str

    def register_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()  # pylint: disable=attribute-defined-outside-init
        if listener in self._name_hooks:
            raise T2NErrorDataNodeValue("Already registered  listener !")
        self._name_hooks.add(listener)

    def detach_listener_name_change(self, listener):
        if not hasattr(self, "_name_hooks"):
            self._name_hooks = set()  # pylint: disable=attribute-defined-outside-init
        self._name_hooks.remove(listener)

    def __setattr__(self, attr_name, attr_value):
        if attr_name == "name" and hasattr(self, "_name_hooks"):
            for name_hook in self._name_hooks:
                name_hook(self.name, attr_value)
        super().__setattr__(attr_name, attr_value)


class ReactiveNamedItemDict:
    """Named items ordered Dict data structure.

    Ensure that 'NO' 2 items are inserted with same 'name' attribute
    and maintains fast name update and with some additive colision
    protections.

    Warning! only aimed at NamedItem subclass.

    Expose a 'list' like interface. (with limited index access)

    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class DummyItem(NamedItem):
        ...     name: str
        ...
        >>> namespace = ReactiveNamedItemDict()
        >>> item = DummyItem("hello")
        >>> for i in "abc":
        ...     namespace.append(DummyItem(i))
        >>> namespace.append(item)
        >>> try:
        ...     namespace.append(DummyItem("a"))
        ...     assert False
        ... except T2NErrorDataNodeValue:
        ...     pass
        >>> item.name = "world"
        >>> namespace.append(DummyItem("hello"))

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
        """Maintain sync between data structure and name changes in items."""
        if old_name in self._protected_names:
            raise T2NErrorDataNodeValue(
                f"Not allowed to alter protected_name: {old_name}"
            )
        if new_name in self._protected_names:
            raise T2NErrorDataNodeValue(
                f"Not allowed to alter protected_name: {new_name}"
            )
        if new_name == old_name:
            return
        if new_name in self._map:
            msg = f"node with name:{new_name} overwritten in {self}"
            LOGGER.debug(msg)
            if self.avoid_name_collision:
                raise T2NErrorDataNodeValue(msg)
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
                raise T2NErrorDataNodeValue(msg)
            LOGGER.debug(msg)
            return
        if (
            item.name in self._protected_names
            and not raise_exception_if_protected_name
        ):
            raise T2NErrorDataNodeValue(
                f"Not authorized to remove: '{item.name}'"
            )
        self.detach_listener_name_change_for_item(item)
        del self._map[item.name]

    def get_by_name(self, name: str, default: T.Any = None):
        return self._map.get(name, default)

    def contains(self, item: NamedItem, strict: bool = False):
        name_exists = item.name in self._map
        if name_exists and strict:
            # Strict mode means the exact same object instance must be stored.
            # Using identity avoids fragile __eq__ implementations (e.g.,
            # tensors with NaNs that are not equal to themselves).
            return self._map[item.name] is item
        return name_exists

    def append(self, item: NamedItem):
        """Append item to ordered set.

        WARNING: This is crucial that all added items use this
        function as it set the hook to listen to name changes
        """
        if item.name in self._map:
            raise T2NErrorDataNodeValue(
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
                raise T2NErrorMisuse("No last value found")
            return self._last_inserted_item
        if isinstance(index, (slice, int)):
            return list(self._map.values())[index]
        raise T2NErrorNotImplemented(index)

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
        raise T2NErrorNotImplemented(
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
        return (
            f"<ReactiveNamedItemDict ({len(self._map)}) "
            f"stored_names=[{names}] {protected}>"
        )


class T2NExtra(str, Enum):
    """Special extra names used in T2N framework."""

    LLM_TRACT = "llm-tract"
    LLM_TRACT_ACCELERATE = "llm-tract-accelerate"
    SAFETENSORS = "safetensors"
    PEFT = "peft"
    NEMO_TRACT = "nemo-tract"


@lru_cache(maxsize=None)
def _import_module_cached(module: str) -> T.Any:
    return importlib.import_module(module)


def require_extra(*, extra: T2NExtra, module: str) -> T.Any:
    """Lazily import an optional dependency and raise a error if missing."""
    extra = T2NExtra(extra)
    try:
        return _import_module_cached(module)
    except ImportError as exp:
        raise T2NErrorMisuse(
            f"This feature requires torch_to_nnef[{extra.value}] extra "
            f"(missing optional dependency '{module}')"
        ) from exp


F = T.TypeVar("F", bound=T.Callable[..., T.Any])


class _Injected:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<InjectedDependency>"


INJECTED = _Injected()
Injected = type(INJECTED)


def require_extra_decorator(
    *,
    extra: T2NExtra,
    module: str,
    kw: T.Union[str, None] = None,
) -> T.Callable[[F], F]:
    inject_name = kw or module.split(".", 1)[0]

    def decorator(fn: F) -> F:
        sig = inspect.signature(fn)
        param = sig.parameters.get(inject_name)

        # ---- Decoration-time validation ----
        if param is None:
            raise TypeError(
                f"{fn.__qualname__} must declare keyword-only parameter "
                f"'{inject_name}' to receive injected dependency"
            )

        if param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"Injected parameter '{inject_name}' in {fn.__qualname__} "
                f"must be keyword-only"
            )

        if param.default is inspect.Parameter.empty:
            raise TypeError(
                f"Injected parameter '{inject_name}' in {fn.__qualname__} "
                f"must default to INJECTED"
            )

        if param.default is not INJECTED:
            raise TypeError(
                f"Injected parameter '{inject_name}' in {fn.__qualname__} "
                f"must default to INJECTED (not {param.default!r})"
            )

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # ---- Call-time validation ----
            value = kwargs.get(inject_name, INJECTED)

            if value is not INJECTED:
                raise TypeError(
                    f"Argument '{inject_name}' is injected by @requires_extra "
                    "and must not be provided by the caller"
                )

            kwargs[inject_name] = require_extra(
                extra=extra,
                module=module,
            )
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def blank_from_init(cls):
    """Create a blank instance of cls by bypassing __init__."""
    # Bypass __init__
    obj = cls.__new__(cls)

    sig = inspect.signature(cls.__init__)

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Skip *args and **kwargs (no attribute name to bind)
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        setattr(obj, name, None)

    return obj


def check_torch_ecosystem():
    """Check that torch, torchaudio and torchvision versions are compatible.

    This is a common source of runtime errors, so we proactively check and raise
    a clear error message with instructions if we detect a mismatch.

    (avoid cryptic symbol not found errors that can occur missmatched versions)

    """
    torch_mm = SemanticVersion.from_str(torch.__version__)

    for name in ("torchaudio", "torchvision"):
        try:
            mod = __import__(name)
        except ModuleNotFoundError:
            continue

        mod_version = SemanticVersion.from_str(mod.__version__)
        compatible = True
        hint = None
        if name == "torchaudio":
            # torchaudio follows torch major.minor
            compatible = (
                mod_version.major == torch_mm.major
                and mod_version.minor == torch_mm.minor
            )
            hint = (
                f"pip install torch=={torch_mm.major}.{torch_mm.minor}.* "
                f"torchaudio=={torch_mm.major}.{torch_mm.minor}.*"
            )
        elif name == "torchvision":
            # torchvision 0.(15 + torch_minor).x pairs with torch 2.torch_minor.x
            # Examples: torch 2.6.x <-> torchvision 0.21.x; 2.9.x <-> 0.24.x
            expected_major = 0
            expected_minor = 15 + torch_mm.minor
            compatible = (
                mod_version.major == expected_major
                and mod_version.minor == expected_minor
            )
            hint = (
                f"pip install torch=={torch_mm.major}.{torch_mm.minor}.* "
                f"torchvision=={expected_major}.{expected_minor}.*"
            )
        if not compatible:
            raise T2NErrorMisuse(
                f"{name} ({mod.__version__}) is incompatible with "
                f"torch ({torch.__version__}). Install matching versions, e.g.:\n  {hint}"
            )
