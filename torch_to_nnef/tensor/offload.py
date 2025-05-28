"""OffLoad Tensor.

Tensor subclass to work around memories limit on various devices
by offloading on disk or on a different 'memory' than final one.

It hold an internal memory storage (permanent) and a temporary
    instantiation at each operation accessing it on targeted device.

## HuggingFace 'accelerate' difference

This is different than HuggingFace 'accelerate' that would
spread once the layout of your network accross the different
devices available, but preventing to move data to other device afterward.

Indeed we use the torch "FakeTensor" API instead of the torch.device("meta")
allowing to hold more informations such as the final targeted device (and other stuff).

This avoid us to have any need for the Hooking system inside accelerate,
and skip need to align data flow graph by pre&post casting.

As a drawback it may be slower than `accelerate` in some senarios.

But help making it 'more' robust to user changes.

"""

import gc
import json
import warnings
from pathlib import Path
import typing as T
import logging
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from torch import nn
from torch._tensor import _convert
from torch.jit import TracerWarning
from torch.overrides import get_default_nowrap_functions
from torch_to_nnef.tensor.opaque import OpaqueTensor
from torch_to_nnef.utils import select_ctx_disable_torch_fn
import torch


AUTO_DEVICE_MAP_KEY = "t2n_auto"
ON_DISK_DEVICE_MAP_KEY = "t2n_offload_disk"

MODEL_NAME = "pytorch_model"
WEIGHTS_NAME = f"{MODEL_NAME}.bin"
SAFE_MODEL_NAME = "model"
SAFE_WEIGHTS_NAME = f"{SAFE_MODEL_NAME}.safetensors"


LOGGER = logging.getLogger(__name__)


class OffloadedTensor(OpaqueTensor):
    @staticmethod
    def __new__(
        cls,
        elem,
        device,
        *args,
        offload_dir: Path,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=TracerWarning)
            self = torch.Tensor._make_subclass(
                cls,
                elem,
                elem.requires_grad,
                dispatch_device=True,
                device_for_backend_keys=device,
            )
        torch._C._set_warn_deprecated_on_mutable_data_ptr(self)
        assert elem.device.type == "meta", elem.device.type
        device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        # normalize device.
        if device.type in ["cuda", "xpu"]:
            from torch._subclasses.fake_tensor import init_gpu_context

            init_gpu_context(device)
        if (
            device.type
            in [
                "cuda",
                "hpu",
                "xpu",
                "mps",
                torch._C._get_privateuse1_backend_name(),
            ]
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

    def __init__(self, elem, device, offload_dir: Path, name: str):
        self.elem = elem
        self.target_device = torch.device(device)
        self._name = name
        self.offload_dir = offload_dir

    @property
    def device(self) -> torch.device:
        return self.target_device

    @property
    def offload_path(self):
        return self._offload_path(self.offload_dir, self._name)

    @staticmethod
    def _offload_path(offload_dir: Path, name: str) -> Path:
        return offload_dir / f"{name}.pt"

    @classmethod
    def from_real_tensor(
        cls,
        tensor: torch.Tensor,
        name: str,
        offload_dir: T.Optional[Path] = None,
    ):
        if offload_dir is None:
            if not hasattr(cls, "tmp_basedir"):
                cls.tmp_basedir = Path(
                    tempfile.mkdtemp(prefix="t2n_offload_disk")
                )
            offload_dir = cls.tmp_basedir
        torch.save(tensor, cls._offload_path(offload_dir, name))
        off_tensor = cls(
            torch.zeros(tensor.shape, dtype=tensor.dtype, device="meta"),
            tensor.device,
            offload_dir=offload_dir,
            name=name,
        )
        LOGGER.info(f"Offloaded param (kept on-disk): '{name}'")
        return off_tensor

    def reload(self):
        tensor = torch.load(self.offload_path).to(self.target_device)
        return tensor

    def _to_base_tensor(self) -> torch.Tensor:
        return self.reload()

    def __repr__(self, *, tensor_contents=None):
        return (
            f"<OffloadedTensor name='{self._name}', "
            f"device='{self.target_device}', "
            f"offload_dir='{self.offload_dir}'>"
        )

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

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.tensor import NamedTensor

        if not all(
            issubclass(cls, t) or issubclass(NamedTensor, t) for t in types
        ):
            return NotImplemented

        with select_ctx_disable_torch_fn():
            skip_expansion = func in get_default_nowrap_functions().union(
                {cls.__repr__}
            ) or any(
                _ in str(func)
                for _ in ["'__get__'", "'__set__'", "Tensor.__reduce_ex__"]
            )
            if not skip_expansion:
                args = [
                    a.to_base_tensor() if isinstance(a, cls) else a
                    for a in args
                ]
                kwargs = {
                    k: v.to_base_tensor() if isinstance(v, cls) else v
                    for k, v in kwargs.items()
                }
            ret = func(*args, **kwargs)
            if skip_expansion:
                return ret
            # important modification
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)


def load_state_dict(checkpoint_file, device_map=None):
    """
    Load a checkpoint from a given file. If the checkpoint is in the safetensors format and a device map is passed, the
    weights can be fast-loaded directly on the GPU.

    Args:
        checkpoint_file (`str`): The path to the checkpoint to load.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
    """
    if checkpoint_file.name.endswith(".safetensors"):
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
            weight_names = f.keys()

        if metadata is None:
            LOGGER.warning(
                f"The safetensors archive passed at {checkpoint_file} does not contain metadata. "
                "Make sure to save your model with the `save_pretrained` method. Defaulting to 'pt' metadata."
            )
            metadata = {"format": "pt"}

        if metadata.get("format") not in ["pt", "tf", "flax"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        elif metadata["format"] != "pt":
            raise ValueError(
                f"The checkpoint passed was saved with {metadata['format']}, we need a the pt format."
            )
        if device_map is None:
            return safe_load_file(checkpoint_file)
        else:
            # if we only have one device we can load everything directly
            if len(set(device_map.values())) == 1:
                device = list(device_map.values())[0]
                target_device = device

                if isinstance(target_device, str) and target_device.startswith(
                    "disk"
                ):
                    target_device = target_device.split("disk_")[-1]
                return safe_load_file(checkpoint_file, device=target_device)

            devices = list(set(device_map.values()) - {"disk"})
            # cpu device should always exist as fallback option
            if "cpu" not in devices:
                devices.append("cpu")

            # For each device, get the weights that go there
            device_weights = {device: [] for device in devices}
            for module_name, device in device_map.items():
                if device in devices:
                    device_weights[device].extend(
                        [
                            k
                            for k in weight_names
                            if k == module_name
                            or k.startswith(module_name + ".")
                        ]
                    )

            # all weights that haven't defined a device should be loaded on CPU
            device_weights["cpu"].extend(
                [
                    k
                    for k in weight_names
                    if k not in sum(device_weights.values(), [])
                ]
            )
            tensors = {}
            for device in devices:
                target_device = device
                with safe_open(
                    checkpoint_file, framework="pt", device=target_device
                ) as f:
                    for key in device_weights[device]:
                        # TODO: apply OffloadedTensor here directly (to lazy load .pt)
                        tensors[key] = f.get_tensor(key)

            return tensors
    else:
        return torch.load(checkpoint_file, map_location=torch.device("cpu"))


def t2n_load_checkpoint_and_dispatch(
    model: nn.Module,
    checkpoint: Path,
    device_map: T.Union[str, T.Dict[str, T.Union[str, int, torch.device]]],
    offload_dir: Path,
    strict: bool = False,
):
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
                raise ValueError(
                    f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
                )
            elif len(potential_index) == 1:
                index_filename = checkpoint / potential_index[0]
            else:
                raise ValueError(
                    f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
                )
    else:
        raise ValueError(
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
            checkpoint_file, device_map=device_map
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
                    raise ValueError(
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
            f"Some weights of the model checkpoint at {checkpoint} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}. "
            "This may or may not be an issue - make sure that the checkpoint "
            "does not have unnecessary parameters, or that the model definition "
            "correctly corresponds to the checkpoint."
        )


def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: T.Union[int, str, torch.device],
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
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if (
        tensor_name not in module._parameters
        and tensor_name not in module._buffers
    ):
        raise ValueError(
            f"{module} does not have a parameter or a buffer named {tensor_name}."
        )
    old_value = getattr(module, tensor_name)

    param = (
        module._parameters[tensor_name]
        if tensor_name in module._parameters
        else None
    )
    param_cls = type(param)

    if value is None:
        raise ValueError("Missing value")
    # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
    # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
    if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
        raise ValueError(
            f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" '
            f"(which has shape {old_value.shape}), this looks incorrect."
        )

    if dtype is None:
        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
        value = value.to(old_value.dtype)
    elif not str(value.dtype).startswith(
        ("torch.uint", "torch.int", "torch.bool")
    ):
        value = value.to(dtype)
    # check to same tensors
    if device.startswith("disk"):
        real_device = device.split("_")[-1]
        if "disk" in real_device:
            real_device = "cpu"
        offloaded_value = OffloadedTensor.from_real_tensor(
            value, original_tensor_name, offload_dir
        )
        new_value = torch.nn.Parameter(
            offloaded_value, requires_grad=old_value.requires_grad
        ).to(real_device)
    else:
        new_value = value
    module._parameters[tensor_name] = new_value
