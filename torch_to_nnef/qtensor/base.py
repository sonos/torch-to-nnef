import abc
import logging
import typing as T
from pathlib import Path

import torch
from torch import _C
from torch._tensor import _convert
from torch.overrides import get_default_nowrap_functions

from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError

LOGGER = logging.getLogger(__name__)


class QScheme(abc.ABC):
    @abc.abstractmethod
    def quantize_as_torch(self, fp_tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def dequantize(self, u8_tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def clone_with_scale_factor(self, scale_factor):
        raise NotImplementedError()


class QScalePerGroupF16(QScheme):
    """Tract aligned

    using negative scales

    """

    def __init__(
        self,
        group_size: int,
        scale: torch.Tensor,
        n_bits: int,
    ):
        scale = scale.to(torch.float16)
        # assert (scale != 0).all(), scale
        self.group_size: int = group_size
        self.scale = scale
        self.n_bits = n_bits  # needed for bit-shift before packing

    def quantize_as_torch(self, fp_tensor):
        raise TorchToNNEFNotImplementedError(
            "native torch does not suport per chunk"
        )

    def quantize_as_u8(self, fp_tensor):
        assert (
            len(fp_tensor.shape) == 2 and fp_tensor.shape[1] == self.group_size
        )
        recip_scale = torch.where(self.scale == 0, self.scale, 1.0 / self.scale)
        fu8_tensor_per_group = (
            ((fp_tensor * recip_scale) + (2**self.n_bits + 1) / 2)
            .floor()
            .clamp(0, 2**self.n_bits - 1)
        )
        return fu8_tensor_per_group.to(torch.uint8)

    def dequantize(self, u8_tensor):
        u8_tensor_per_group = u8_tensor.flatten().reshape(-1, self.group_size)
        offset = 2**self.n_bits / 2
        fp_tensor_per_group = u8_tensor_per_group.to(torch.float16) - offset
        fp_tensor_per_group *= self.scale
        return fp_tensor_per_group.reshape(u8_tensor.shape)

    def clone_with_scale_factor(self, scale_factor):
        return self.__class__(
            scale=(self.scale / scale_factor).to(dtype=self.scale.dtype),
            group_size=self.group_size,
            n_bits=self.n_bits,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(group_size={self.group_size}, "
            f"scale={self.scale})"
        )


class QTensor(torch.Tensor):
    """Common interface for all Compressed storage"""

    @staticmethod
    def __new__(
        cls,
        u8_values_tensor,
        qscheme,
        dequant_to_dtype,
        *args,
        **kwargs,
    ):
        return super().__new__(cls, u8_values_tensor, *args, **kwargs)

    def __init__(
        self,
        u8_values_tensor: torch.Tensor,
        qscheme: QScheme,
        dequant_to_dtype=torch.float32,
    ):
        super().__init__()
        self.u8_values_tensor = u8_values_tensor
        self.qscheme = qscheme
        self.dequant_to_dtype = dequant_to_dtype
        self.requires_grad = False

    def to_torch_float_tensor(self):
        return self.qscheme.dequantize(self.u8_values_tensor).to(
            self.dequant_to_dtype
        )

    def clone(self, *args, **kwargs):
        return self.__class__(
            super().clone(*args, **kwargs),
            self.qscheme,
            self.dequant_to_dtype,
        )

    def to(self, *args, **kwargs):
        new_dtype = kwargs.get("dtype") or args[0] if args else None
        if new_dtype is None:
            return self
        new_obj = self.__class__(
            self.u8_values_tensor,
            self.qscheme,
            new_dtype,
        )
        new_obj.requires_grad = False
        return new_obj

    def detach(self):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("QTensor does not support detach")
        return self

    def requires_grad_(self, requires_grad):
        # need overwrite since nn.Paramater use it at __new__
        LOGGER.debug("QTensor does not support requires_grad")
        return self

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

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with _C.DisableTorchFunctionSubclass():
            new_args = args
            new_kwargs = kwargs
            if func not in get_default_nowrap_functions().union(
                {cls.__repr__}
            ) and "'__set__'" not in str(
                func
            ):  # class should not be expanded for setattr
                new_args = [
                    QTensorRef(a.to_torch_float_tensor(), a)
                    if isinstance(a, cls)
                    else a
                    for a in args
                ]
                new_kwargs = {
                    k: QTensorRef(v.to_torch_float_tensor(), v)
                    if isinstance(v, cls)
                    else v
                    for k, v in kwargs.items()
                }

            ret = func(*new_args, **new_kwargs)
            if func in get_default_nowrap_functions():
                return ret
            # important modification
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)

    @property
    def data(self):
        """very important to keep access to all special attr of QTensor"""
        return self

    def write_in_file(self, dirpath: T.Union[str, Path], label: str):
        """Called at NNEF write time.

        Each specific inference engine format should implement
        the file dump prefered.

        """
        raise TorchToNNEFNotImplementedError()


class QTensorRef(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        fp_tensor,
        q_tensor,
        *args,
        **kwargs,
    ):
        return super().__new__(cls, fp_tensor, *args, **kwargs)

    def __init__(
        self,
        fp_tensor: torch.Tensor,
        q_tensor: QTensor,
    ):
        super().__init__()
        self.q_tensor = q_tensor

    @property
    def data(self):
        return self

    def clone(self, *args, **kwargs):
        return self.__class__(
            super().clone(*args, **kwargs),
            self.q_tensor,
        )

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        # need overwrite since nn.Paramater use it at __new__
        return self

    def requires_grad_(self, requires_grad):
        # need overwrite since nn.Paramater use it at __new__
        return self

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

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with _C.DisableTorchFunctionSubclass():
            new_args = [a.clone() if isinstance(a, cls) else a for a in args]
            new_kwargs = {
                k: v.clone() if isinstance(v, cls) else v
                for k, v in kwargs.items()
            }
            ret = func(*new_args, **new_kwargs)
            if func in get_default_nowrap_functions():
                return ret
            # important modification
            # do not propagate this qtype
            return _convert(ret, torch.Tensor)


def if_qtensor_in_params_set_as_ref(model):
    """Move QTensor into QTensorRef which expand tensor

    This is applied at export time of torch_to_nnef
    Just before doing any tracing.

    """
    for named_p, param in model.named_parameters():
        if not isinstance(param, QTensor):
            continue
        ref_mod = model
        chunked_names = named_p.split(".")
        for mod_name in chunked_names[:-1]:
            ref_mod = getattr(ref_mod, mod_name)

        setattr(
            ref_mod,
            chunked_names[-1],
            torch.nn.Parameter(
                QTensorRef(param.to_torch_float_tensor(), param),
                requires_grad=False,
            ),
        )


def qscale_per_group_f16_min_max_calibration(
    fp_tensor,
    n_bits: int,
    group_size: int,
    percentile: float = 1.0,
) -> T.Tuple["QScalePerGroupF16", torch.Tensor]:
    """Build QScalePerGroupF16 and calibrate requested float tensor.

    Return:
        Tuple(
            QScalePerGroupF16 qscheme,
            torch.Tensor[uint8]
        )
    """
    assert 0.0 < percentile <= 1.0
    volume = 1
    for fp_dim in fp_tensor.shape:
        volume *= fp_dim
    if volume % group_size != 0:
        raise ValueError(
            f"tensor provided volume: {volume} but group size are {group_size} "
            "incomplete groups aren't supported."
        )
    fp_tensor_per_group = fp_tensor.flatten().reshape(-1, group_size)

    # we use full-range symmetric
    # like torch, ONNX, but oposed to restricted range from
    # TensorFlow, NVIDIA TensorRT and Intel DNNL
    scale = torch.quantile(fp_tensor_per_group.abs(), percentile, dim=1) / (
        -(2**n_bits) / 2
    )

    assert scale.shape[0] == fp_tensor_per_group.shape[0]
    qshape = [scale.shape[0]] + [1]
    qscheme = QScalePerGroupF16(
        group_size=group_size,
        scale=scale.reshape(qshape),
        n_bits=n_bits,
    )
    return (
        qscheme,
        qscheme.quantize_as_u8(fp_tensor_per_group).reshape(fp_tensor.shape),
    )
