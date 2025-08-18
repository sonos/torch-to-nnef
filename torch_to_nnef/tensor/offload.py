"""OffLoad Tensor.

Tensor subclass to work around memories limit on various devices
by offloading on disk or on a different 'memory' than final one.

It hold an internal memory storage (permanent) and a temporary
    instantiation at each operation accessing it on targeted device.

## HuggingFace 'accelerate' difference

This is different than HuggingFace 'accelerate' that would
spread once the layout of your network accross the different
devices available, but preventing to move data to other device afterward.

Indeed we use the torch "Tensor" API instead of the torch.device("meta")
allowing to hold more informations such as the final targeted device (and other stuff).

This avoid us to have any need for the Hooking system done in accelerate,
and skip need to align data flow graph by pre&post casting.

In short it is transparent for end-user that can use those like read-only
device movable tensors (mutation support could be envisioned if needed).

"""

import gc
import json
import logging
import os
import tempfile
import typing as T
import warnings
from pathlib import Path

import torch
from torch import nn
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.exceptions import T2NErrorMissUse
from torch_to_nnef.tensor.opaque import OpaqueTensor
from torch_to_nnef.utils import select_ctx_disable_torch_fn, torch_version

AUTO_DEVICE_MAP_KEY = "t2n_auto"
ON_DISK_DEVICE_MAP_KEY = "t2n_offload_disk"

MODEL_NAME = "pytorch_model"
WEIGHTS_NAME = f"{MODEL_NAME}.bin"
SAFE_MODEL_NAME = "model"
SAFE_WEIGHTS_NAME = f"{SAFE_MODEL_NAME}.safetensors"


TDEVICE = T.Union[int, str, torch.device]

LOGGER = logging.getLogger(__name__)


class OffloadedTensor(OpaqueTensor):
    """Tensor subclass that maintains data on disk

    It hold an virtual internal memory storage (permanent)
    and a temporary instantiation at each operation accessing it on targeted device.

    Warning:
        we recommend to version of PyTorch > 1.12 for best compatibility.
    """

    @staticmethod
    def __new__(
        cls,
        elem,
        device,
        *args,
        offload_dir: Path,
        **kwargs,
    ):
        if torch_version() < "1.12.0":
            warnings.warn(
                "OffloadedTensor expect PyTorch aten ops support 'meta' "
                "device tensors (which is very limited before 1.12)",
                stacklevel=2,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            mk_sub_kwargs = {}
            if torch_version() >= "1.13.0":
                mk_sub_kwargs = {
                    "dispatch_device": True,
                    "device_for_backend_keys": device,
                }
            self = torch.Tensor._make_subclass(
                cls, elem, elem.requires_grad, **mk_sub_kwargs
            )
        if torch_version() >= "2.4.0":
            torch._C._set_warn_deprecated_on_mutable_data_ptr(self)
        assert elem.device.type == "meta", elem.device.type
        device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        # normalize device.
        if device.type in ["cuda", "xpu"]:
            # pylint: disable-next=import-outside-toplevel
            from torch._subclasses.fake_tensor import init_gpu_context

            init_gpu_context(device)
        devs = []
        if torch_version() > "2.4.0":
            devs = [torch._C._get_privateuse1_backend_name()]
        if (
            device.type
            in [
                "cuda",
                "hpu",
                "xpu",
                "mps",
            ]
            + devs
            and device.index is None
        ):
            if (
                device.type != "mps"
                and getattr(torch, device.type).is_initialized()
            ):
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        self.target_device = device
        return self

    def __init__(
        self,
        elem,
        device,
        offload_dir: Path,
        name: str,
        offloaded_tensor_type: T.Type[torch.Tensor],
        force_gc_collect: bool = False,
    ):
        self.elem = elem
        self.target_device = torch.device(device)
        self._name = name
        self.offload_dir = offload_dir
        self.offloaded_tensor_type = offloaded_tensor_type
        self.force_gc_collect = force_gc_collect

    @property
    def dtype(self) -> torch.dtype:
        return self.elem.dtype

    def numel(self) -> int:
        return self.elem.numel()

    @property
    def device(self) -> torch.device:
        return self.target_device

    @property
    def offload_path(self):
        return self._offload_path(self.offload_dir, self._name, self.elem.dtype)

    @staticmethod
    def _offload_path(offload_dir: Path, name: str, dtype: torch.dtype) -> Path:
        return offload_dir / f"{name}_{dtype}.pt"

    @classmethod
    def from_original_tensor(
        cls,
        tensor: torch.Tensor,
        name: str,
        offload_dir: T.Optional[Path] = None,
        suffix_log_msg: str = "",
    ):
        """Take an torch.Tensor or torch_to_nnef.tensor.OpaqueTensor and offload it to disk

        Args:
            tensor:
                the torch.Tensor or torch_to_nnef.tensor.OpaqueTensor to dump on disk
            name:
                the name of the tensor that will be used to create the filename store on disk
            offload_dir:
                The directory where this file will be stored (temporarly)
        """
        if offload_dir is None:
            if not hasattr(cls, "tmp_basedir"):
                cls.tmp_basedir = Path(
                    tempfile.mkdtemp(prefix="t2n_offload_disk")
                )
            offload_dir = cls.tmp_basedir
        cls._save(tensor, offload_dir, name)
        off_tensor = cls(
            torch.zeros(tensor.shape, dtype=tensor.dtype, device="meta"),
            tensor.device,
            offload_dir=offload_dir,
            name=name,
            offloaded_tensor_type=type(tensor),
        )
        LOGGER.info(
            "Offloaded param (kept on-disk): '%s' %s", name, suffix_log_msg
        )
        return off_tensor

    def to(self, *args, **kwargs):
        """OffloadedTensor support changing the target device when in memory"""
        if len(args) > 1:
            kwargs.update(zip(["device", "dtype"], args))
        else:
            # unfortunately arg order is not guarantied
            # in torch .to ...
            try:
                torch.device(args[0])
                kwargs["device"] = args[0]
            except TypeError:
                kwargs["dtype"] = args[0]

        if kwargs.get("dtype") is not None:
            dtype = kwargs["dtype"]
            assert isinstance(dtype, torch.dtype)
            if dtype != self.elem.dtype:
                tensor = self.reload().to(dtype)
                self._save(
                    tensor,
                    self.offload_dir,
                    self._name,
                )
                self.offload_path.unlink()
                self.elem = self.elem.to(dtype)
                self.__dict__["dtype"] = dtype
                LOGGER.info(
                    "[casted to %s] offload tensor '%s'", dtype, self._name
                )
        if kwargs.get("device") is not None:
            self.target_device = torch.device(kwargs["device"])
        return self

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        return self

    def update_values(self, values: torch.Tensor):
        """replace offloaded tensor by new 'values' tensor


        Args:
            values:
                The tensor that will replace it on disk
                assertion are made to ensure same shape, dtype
                as prior
        """
        assert self.elem.dtype == values.dtype
        assert self.elem.shape == values.shape
        assert self._offload_path(
            self.offload_dir, self._name, values.dtype
        ).exists()
        OffloadedTensor._save(values, self.offload_dir, self._name)
        LOGGER.debug("updated values: '%s'", self._name)

    @classmethod
    def _save(cls, tensor, offload_dir, name):
        return torch.save(
            tensor, cls._offload_path(offload_dir, name, tensor.dtype)
        )

    def reload(self):
        if issubclass(self.offloaded_tensor_type, OpaqueTensor):
            load_kwargs = {}
            if torch_version() >= "1.13.0":
                load_kwargs["weights_only"] = False
            return torch.load(self.offload_path, **load_kwargs).to(
                self.target_device
            )
        return torch.load(self.offload_path).to(self.target_device)

    def _to_base_tensor(self) -> torch.Tensor:
        return self.reload()

    def __repr__(self, *, tensor_contents=None):
        return (
            f"<OffloadedTensor name='{self._name}', "
            f"device='{self.target_device}', "
            f"offload_dir='{self.offload_dir}', "
            f"offloaded_tensor_type='{self.offloaded_tensor_type.__name__}' >"
        )

    def __del__(self):
        """Remove parameter from disk."""
        if self.offload_path.exists():
            self.offload_path.unlink()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.
        we modify it so it's always reference torch.Tensor.
        """

        if kwargs is None:
            kwargs = {}

        with select_ctx_disable_torch_fn():
            skip_expansion = func in get_default_nowrap_functions().union(
                {cls.__repr__}
            ) or any(
                _ in str(func)
                for _ in ["'__get__'", "'__set__'", "Tensor.__reduce_ex__"]
            )
            if not skip_expansion:
                new_args = []
                for a in args:
                    while isinstance(a, OpaqueTensor):
                        a = a.to_base_tensor()
                    new_args.append(a)
                args = new_args
                new_kwargs = {}
                for k, v in kwargs.items():
                    while isinstance(v, OpaqueTensor):
                        v = v.to_base_tensor()
                    new_kwargs[k] = v
                kwargs = new_kwargs
            ret = func(*args, **kwargs)
            if isinstance(args[0], OpaqueTensor) and args[0].force_gc_collect:
                gc.collect()
            if skip_expansion:
                return ret
            # important modification
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)


def safe_load_file(
    filename: T.Union[str, os.PathLike],
    device: TDEVICE = "cpu",
    offload_dir: T.Optional[Path] = None,
    apply_offload: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.
        offload_dir: Path
            location where tensor with device disk will be offloaded
        apply_offload:
            if offload is applyied or left to cpu

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    # pylint: disable-next=import-outside-toplevel
    import safetensors

    result = {}

    with safetensors.safe_open(
        filename, framework="pt", device=maybe_extract_target_device(device)
    ) as f:
        for k in f:
            v = f.get_tensor(k)
            if apply_offload:
                v = maybe_load_offload_tensor(v, device, k, offload_dir)
            result[k] = v
    return result


def maybe_extract_target_device(device: TDEVICE) -> TDEVICE:
    if isinstance(device, str):
        real_device = device.split("_")[-1]
        if "disk" in real_device:
            real_device = "cpu"
        return real_device
    return device


def maybe_load_offload_tensor(
    value: torch.Tensor,
    device: TDEVICE,
    original_tensor_name: str,
    offload_dir: T.Optional[Path],
) -> torch.Tensor:
    if (
        isinstance(device, str)
        and device.startswith("disk")
        and not isinstance(value, OffloadedTensor)
    ):
        offloaded_value = OffloadedTensor.from_original_tensor(
            value, original_tensor_name, offload_dir
        ).to(maybe_extract_target_device(device))
        return offloaded_value
    return value


def load_state_dict(
    checkpoint_file,
    device_map=None,
    offload_dir: T.Optional[Path] = None,
    apply_offload: bool = False,
):
    """
    Load a checkpoint from a given file. If the checkpoint is in the safetensors format and a device map is passed, the
    weights can be fast-loaded directly on the GPU.

    Args:
        checkpoint_file (`str`): The path to the checkpoint to load.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        offload_dir: Path *optional*
            Offload directory to store tensors
        apply_offload: bool
            if activated it will offload each loaded tensor as soon as possible
            (we disable it in most case to allow set_module_tensor_to_device dtype casting in memory directly)
    """

    if not checkpoint_file.name.endswith(".safetensors"):
        return torch.load(checkpoint_file, map_location=torch.device("cpu"))
    # pylint: disable-next=import-outside-toplevel
    import safetensors

    with safetensors.safe_open(checkpoint_file, framework="pt") as f:
        metadata = f.metadata()
        weight_names = f.keys()

    if metadata is None:
        LOGGER.warning(
            "The safetensors archive passed at %s does not contain metadata. "
            "Make sure to save your model with the `save_pretrained` method. Defaulting to 'pt' metadata.",
            checkpoint_file,
            stacklevel=2,
        )
        metadata = {"format": "pt"}

    if metadata.get("format") not in ["pt", "tf", "flax"]:
        raise T2NErrorMissUse(
            f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
            "you save your model with the `save_pretrained` method."
        )
    if metadata["format"] != "pt":
        raise T2NErrorMissUse(
            f"The checkpoint passed was saved with {metadata['format']}, we need a the pt format."
        )
    if device_map is None:
        return safe_load_file(
            checkpoint_file,
            offload_dir=offload_dir,
            apply_offload=apply_offload,
        )

    # if we only have one device we can load everything directly
    if len(set(device_map.values())) == 1:
        device = list(device_map.values())[0]
        target_device = device

        return safe_load_file(
            checkpoint_file,
            device=target_device,
            offload_dir=offload_dir,
            apply_offload=apply_offload,
        )

    devices = list(set(device_map.values()) - {"disk"})
    # cpu device should always exist as fallback option
    if "cpu" not in devices:
        devices.append("cpu")

    # For each device, get the weights that go there
    device_weights: T.Dict[str, T.List[str]] = {
        device: [] for device in devices
    }
    for module_name, device in device_map.items():
        if device in devices:
            device_weights[device].extend(
                [
                    k
                    for k in weight_names
                    if k == module_name or k.startswith(module_name + ".")
                ]
            )

    # all weights that haven't defined a device should be loaded on CPU
    device_weights["cpu"].extend(
        [k for k in weight_names if k not in sum(device_weights.values(), [])]
    )
    tensors = {}
    for device in devices:
        target_device = device
        with safetensors.safe_open(
            checkpoint_file, framework="pt", device=target_device
        ) as f:
            for key in device_weights[device]:
                v = f.get_tensor(key)
                if apply_offload:
                    v = maybe_load_offload_tensor(
                        v,
                        device,
                        key,
                        offload_dir=offload_dir,
                    )
                tensors[key] = v

    return tensors


# pylint: disable-next=too-many-branches,too-many-statements
def t2n_load_checkpoint_and_dispatch(
    model: nn.Module,
    checkpoint: Path,
    device_map: T.Optional[
        T.Union[str, T.Dict[str, T.Union[str, int, torch.device]]]
    ],
    offload_dir: Path,
    strict: bool = False,
    offload_at_load_state_dict: bool = False,
):
    """
    offload_at_load_state_dict:
        Allow to offload as soon as possible
        this may be benefical in some rare case where
        partitioned safetensors file are too big for RAM
        else it's better to offload after
        dtype cast in set_module_tensor_to_device
    """
    if isinstance(device_map, str):
        if device_map == AUTO_DEVICE_MAP_KEY:
            # pylint: disable-next=import-outside-toplevel
            import accelerate

            device_map = accelerate.infer_auto_device_map(model)
        elif device_map == ON_DISK_DEVICE_MAP_KEY:
            device_map = {"": "disk_cpu"}

    checkpoint_files = None
    index_filename = None
    if checkpoint.is_file():
        if str(checkpoint).endswith(".json"):
            index_filename = checkpoint
        else:
            checkpoint_files = [checkpoint]
    elif checkpoint.is_dir():
        # check if the whole state dict is present
        potential_state_bin = [
            f for f in checkpoint.iterdir() if f.name == WEIGHTS_NAME
        ]
        potential_state_safetensor = [
            f for f in checkpoint.iterdir() if f.name == SAFE_WEIGHTS_NAME
        ]
        if len(potential_state_bin) == 1:
            checkpoint_files = [checkpoint / potential_state_bin[0]]
        elif len(potential_state_safetensor) == 1:
            checkpoint_files = [checkpoint / potential_state_safetensor[0]]
        else:
            # otherwise check for sharded checkpoints
            potential_index = [
                f
                for f in checkpoint.iterdir()
                if f.name.endswith(".index.json")
            ]
            if len(potential_index) == 0:
                raise T2NErrorMissUse(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                )
            if len(potential_index) != 1:
                raise T2NErrorMissUse(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
            index_filename = checkpoint / potential_index[0]
    else:
        raise T2NErrorMissUse(
            "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
            f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
        )
    if index_filename is not None:
        checkpoint_folder = index_filename.parent
        with index_filename.open() as f:
            index = json.loads(f.read())

        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        checkpoint_files = [checkpoint_folder / f for f in checkpoint_files]
    unexpected_keys = set()
    model_keys = set(model.state_dict().keys())
    assert checkpoint_files is not None
    for checkpoint_file in checkpoint_files:
        loaded_checkpoint = load_state_dict(
            checkpoint_file,
            device_map=device_map,
            offload_dir=offload_dir,
            apply_offload=offload_at_load_state_dict,
        )
        if device_map is None:
            model.load_state_dict(loaded_checkpoint, strict=strict)
            unexpected_keys.update(set(loaded_checkpoint.keys()) - model_keys)
        else:
            for param_name, param in loaded_checkpoint.items():
                # skip SCB parameter (for 8-bit serialization)
                if "SCB" in param_name:
                    LOGGER.error("unsupported SCB param")
                    continue

                if param_name not in model_keys:
                    unexpected_keys.add(param_name)
                    if not strict:
                        continue  # Skip loading this parameter.

                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    raise T2NErrorMissUse(
                        f"{param_name} doesn't have any device set."
                    )
                param_device = device_map[module_name]
                set_module_tensor_to_device(
                    model,
                    param_name,
                    param_device,
                    value=param,
                    dtype=None,
                    offload_dir=offload_dir,
                )
        # Force Python to clean up.
        del loaded_checkpoint
        gc.collect()
    if not strict and len(unexpected_keys) > 0:
        LOGGER.warning(
            "Some weights of the model checkpoint at %s were not used when"
            " initializing %s: %s. "
            "This may or may not be an issue - make sure that the checkpoint "
            "does not have unnecessary parameters, or that the model definition "
            "correctly corresponds to the checkpoint.",
            checkpoint,
            model.__class__.__name__,
            unexpected_keys,
            stacklevel=2,
        )


def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: TDEVICE,
    value: T.Optional[torch.Tensor] = None,
    dtype: T.Optional[T.Union[str, torch.dtype]] = None,
    offload_dir: T.Optional[Path] = None,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
    """
    # Recurse if needed
    original_tensor_name = tensor_name
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise T2NErrorMissUse(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if (
        tensor_name not in module._parameters
        and tensor_name not in module._buffers
    ):
        raise T2NErrorMissUse(
            f"{module} does not have a parameter or a buffer named {tensor_name}."
        )
    old_value = getattr(module, tensor_name)

    param = module._parameters.get(tensor_name)
    param_cls = type(param)

    if value is None:
        raise T2NErrorMissUse("Missing value")
    # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
    # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
    if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
        raise T2NErrorMissUse(
            f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" '
            f"(which has shape {old_value.shape}), this looks incorrect."
        )

    # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
    if dtype is None:
        value = value.to(old_value.dtype)
    elif not str(value.dtype).startswith(
        ("torch.uint", "torch.int", "torch.bool")
    ):
        value = value.to(dtype)

    new_value = maybe_load_offload_tensor(
        value, device, original_tensor_name, offload_dir
    )
    module._parameters[tensor_name] = nn.Parameter(new_value)
