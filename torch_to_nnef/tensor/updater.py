import enum
import typing as T
from collections import defaultdict

import torch

from torch_to_nnef.exceptions import T2NErrorInconsistentTensor, T2NErrorMisuse
from torch_to_nnef.tensor.named import NamedTensor
from torch_to_nnef.tensor.opaque import OpaqueTensor, OpaqueTensorRef
from torch_to_nnef.tensor.utils import get_named_buffers, get_named_parameters
from torch_to_nnef.utils import torch_version


class TensorHoldKind(str, enum.Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"
    UNREGISTRED = "unregistred"


class ModTensorUpdater:
    """Helper to update parameter/buffer/unregistred tensor of a model cleanly.

    Cleanly means without breaking shared reference between Tensors.

    An example is the shared reference on transformers between
    first input_ids embedding and last linear layer projection weights.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        add_parameter_if_unset: bool = True,
        add_buffers: bool = False,
        add_unregistred_tensor: bool = False,
        disable_requires_grad: bool = False,
    ):
        """Init ModTensorUpdater.

        Args:
            model:
                nn.Module model that will have tensors updated with this class

            add_parameter_if_unset:
                if you add a tensor where there is not yet a torch.nn.Parameters
                in the model it will add it

            add_buffers:
                Scope all nn.Buffer PyTorch object of the model
                to be 'updatable'

            add_unregistred_tensor:
                Scope all tensor PyTorch object of the model not referenced in
                nn.Parameters & nn.Buffer

            disable_requires_grad:
                If set it force tensors replaced to be with no 'requires_grad'
                at update time
        """
        self.name_to_id = {}
        self.id_to_kind = {}
        self._disabled_requires_grad_names = []
        id_to_names = defaultdict(set)
        mod_name_to_tensor_names = defaultdict(list)
        for param_name, param in get_named_parameters(
            model, remove_duplicate=False
        ):
            id_tensor = id(param)
            if disable_requires_grad and param.requires_grad:
                param.requires_grad = False
                self._disabled_requires_grad_names.append(param_name)
            self.name_to_id[param_name] = id(param)
            id_to_names[id_tensor].add(param_name)
            mod_name, _ = self.split_param_name(param_name)
            mod_name_to_tensor_names[mod_name].append(param_name)
            self.id_to_kind[id_tensor] = TensorHoldKind.PARAMETER

        if add_buffers:
            for buffer_name, buffer in get_named_buffers(
                model, remove_duplicate=False
            ):
                id_tensor = id(buffer)
                self.name_to_id[buffer_name] = id(buffer)
                id_to_names[id_tensor].add(buffer_name)
                mod_name, _ = self.split_param_name(buffer_name)
                mod_name_to_tensor_names[mod_name].append(buffer_name)
                self.id_to_kind[id_tensor] = TensorHoldKind.BUFFER

        self.id_to_names = {k: frozenset(v) for k, v in id_to_names.items()}
        self.name_to_parent_module = {}
        self.id_to_modules = defaultdict(list)
        for mod_name, mod in model.named_modules():
            if add_unregistred_tensor:
                for attr_name, attr_val in mod.__dict__.items():
                    id_tensor = id(attr_val)
                    if id_tensor in self.id_to_names:
                        continue
                    if not isinstance(attr_val, torch.Tensor):
                        continue
                    # we need to capture every thing that is a tensor
                    full_name = attr_name
                    if mod_name:
                        full_name = f"{mod_name}.{attr_name}"
                    self.name_to_id[full_name] = id(attr_val)
                    id_to_names[id_tensor].add(full_name)
                    mod_name_to_tensor_names[mod_name].append(full_name)
                    self.id_to_kind[id_tensor] = TensorHoldKind.UNREGISTRED

            for t_name in mod_name_to_tensor_names[mod_name]:
                t_local_name = self.split_param_name(t_name)[1]
                if t_local_name in mod._parameters:
                    tensor = mod._parameters[t_local_name]
                elif t_local_name in mod._buffers:
                    tensor = mod._buffers[t_local_name]
                else:
                    tensor = getattr(mod, t_local_name)
                self.id_to_modules[id(tensor)].append(mod)
                self.name_to_parent_module[t_name] = mod

        self.model = model
        self.add_parameter_if_unset = add_parameter_if_unset

    def restore_require_grad(self):
        for name in self._disabled_requires_grad_names:
            mod = self.name_to_parent_module[name]
            local_name = self.split_param_name(name)[1]
            mod._parameters[local_name].requires_grad = True

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
            raise T2NErrorInconsistentTensor(
                f"not same dtype between {tensor0.dtype} and {tensor1.dtype}"
            )

    def check_same_device(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if tensor0.device != tensor1.device:
            raise T2NErrorInconsistentTensor(
                f"not same device between {tensor0.device} and {tensor1.device}"
            )

    def check_same_shape(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        if tensor0.shape != tensor1.shape:
            raise T2NErrorInconsistentTensor(
                f"not same shape between {tensor0.shape} and {tensor1.shape}"
            )

    def check_consistency(self, tensor0: torch.Tensor, tensor1: torch.Tensor):
        self.check_same_device(tensor0, tensor1)
        self.check_same_dtype(tensor0, tensor1)
        self.check_same_shape(tensor0, tensor1)

    def maybe_parameterize(
        self, ref: torch.Tensor, new_tensor: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(new_tensor, torch.nn.Parameter):
            kind = self.id_to_kind[id(ref)]
            if kind == TensorHoldKind.PARAMETER:
                if self.add_parameter_if_unset and torch_version() >= "2.0.0":
                    new_tensor = torch.nn.Parameter(
                        new_tensor, requires_grad=False
                    )
            elif kind == TensorHoldKind.BUFFER:
                if (
                    torch_version() >= "2.5.0"
                    and not isinstance(new_tensor, torch.nn.parameter.Buffer)
                    and isinstance(ref, torch.nn.parameter.Buffer)
                ):
                    new_tensor = torch.nn.parameter.Buffer(
                        new_tensor, persistent=ref.persistent
                    )
            else:
                raise T2NErrorMisuse(f"{new_tensor} must be a parameter")
        return new_tensor

    def _set_tensor(self, old, new):
        need_grad = False
        if (
            isinstance(old, torch.nn.Parameter)
            and not isinstance(new, torch.nn.Parameter)
            and not self.add_parameter_if_unset
        ):
            raise T2NErrorMisuse("new tensor setted should be a parameter")
        if old.requires_grad:
            need_grad = True
            old.requires_grad = False
        if (
            old.device == new.device
            and old.dtype == new.dtype
            and not isinstance(
                new, (NamedTensor, OpaqueTensor, OpaqueTensorRef)
            )
        ):
            old.set_(new)
        else:
            # slower but allow device or dtype change
            for n in self.id_to_names[id(old)]:
                t_local_name = self.split_param_name(n)[1]
                mod = self.name_to_parent_module[n]
                if t_local_name in mod._parameters:
                    mod._parameters[t_local_name] = new
                else:
                    mod._buffers[t_local_name] = new
            # update references {
            self._update_reference(old, new)
            # }
        if need_grad and not getattr(old, "requires_grad", False):
            old.requires_grad = True

    def _update_reference(self, ref, new_tensor):
        # update references {
        self.id_to_names[id(new_tensor)] = self.id_to_names[id(ref)]
        del self.id_to_names[id(ref)]
        self.id_to_modules[id(new_tensor)] = self.id_to_modules[id(ref)]
        del self.id_to_modules[id(ref)]
        self.id_to_kind[id(new_tensor)] = self.id_to_kind[id(ref)]
        del self.id_to_kind[id(ref)]
        # }
        #

    def get_by_name(self, name: str) -> torch.Tensor:
        """Get tensor based on it's  reference name."""
        mod = self.name_to_parent_module[name]
        _, p_local_name = self.split_param_name(name)
        return getattr(mod, p_local_name)

    def update_by_ref(
        self,
        ref: torch.nn.Parameter,
        new_tensor: torch.Tensor,
        enforce_tensor_consistency: bool = True,
    ) -> torch.Tensor:
        """Update tensor based on it's  reference object."""
        new_tensor = self.maybe_parameterize(ref, new_tensor)
        if enforce_tensor_consistency:
            self.check_consistency(ref, new_tensor)
        self._set_tensor(ref, new_tensor)
        return new_tensor

    def update_by_name(
        self,
        name: str,
        new_tensor: torch.Tensor,
        tie_replacements: bool = True,
        enforce_tensor_consistency: bool = True,
    ) -> torch.Tensor:
        """Update tensor based on it's  reference name."""
        mod = self.name_to_parent_module[name]
        _, p_local_name = self.split_param_name(name)
        ref = getattr(mod, p_local_name)
        new_tensor = self.maybe_parameterize(ref, new_tensor)
        if enforce_tensor_consistency:
            self.check_consistency(ref, new_tensor)
        if tie_replacements:
            self._set_tensor(ref, new_tensor)
        else:
            if p_local_name in mod._parameters:
                mod._parameters[p_local_name] = new_tensor
                if (
                    ref.requires_grad
                    and not mod._parameters[p_local_name].requires_grad
                ):
                    mod._parameters[p_local_name].requires_grad = True
            elif p_local_name in mod._buffers:
                mod._buffers[p_local_name] = new_tensor
            else:
                setattr(mod._buffers, p_local_name, new_tensor)
            self._update_reference(ref, new_tensor)
        return new_tensor

    @property
    def available_names(self):
        return [_ for fs in self.id_to_names.values() for _ in fs]
