import typing as T
from collections import defaultdict

import torch

from torch_to_nnef.exceptions import InconsistentTensorError
from torch_to_nnef.tensor.named import NamedTensor
from torch_to_nnef.tensor.opaque import OpaqueTensor, OpaqueTensorRef


class ParametersUpdater:
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
        self.id_to_modules = defaultdict(list)
        for mod_name, mod in model.named_modules():
            for p_name in mod_name_to_param_names[mod_name]:
                self.id_to_modules[
                    id(mod._parameters[self.split_param_name(p_name)[1]])
                ].append(mod)
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

    def _set_param(self, old, new):
        need_grad = False
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
                self.name_to_parent_module[n]._parameters[
                    self.split_param_name(n)[1]
                ] = new
            # update references {
            self.id_to_names[id(new)] = self.id_to_names[id(old)]
            del self.id_to_names[id(old)]
            self.id_to_modules[id(new)] = self.id_to_modules[id(old)]
            del self.id_to_modules[id(old)]
            # }

        if need_grad and not old.requires_grad:
            old.requires_grad = True

    def update_by_ref(
        self,
        ref: torch.nn.Parameter,
        new_tensor: torch.Tensor,
        enforce_same_shape_dtype_device: bool = True,
    ) -> torch.Tensor:
        new_tensor = self.maybe_parameterize(new_tensor)
        if enforce_same_shape_dtype_device:
            self.check_all(ref, new_tensor)
        self._set_param(ref, new_tensor)
        return new_tensor

    def update_by_name(
        self,
        name: str,
        new_tensor: torch.Tensor,
        tie_replacements: bool = True,
        enforce_same_shape_dtype_device: bool = True,
    ) -> torch.Tensor:
        new_tensor = self.maybe_parameterize(new_tensor)
        mod = self.name_to_parent_module[name]
        _, p_local_name = self.split_param_name(name)
        old = mod._parameters[p_local_name]
        if enforce_same_shape_dtype_device:
            self.check_all(old, new_tensor)
        if tie_replacements:
            self._set_param(old, new_tensor)
        else:
            mod._parameters[p_local_name] = new_tensor
            if (
                old.requires_grad
                and not mod._parameters[p_local_name].requires_grad
            ):
                mod._parameters[p_local_name].requires_grad = True
        return new_tensor
