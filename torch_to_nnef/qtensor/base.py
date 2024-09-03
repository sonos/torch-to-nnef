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
    def dequantize(self, u8_tensor, target_dtype):
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

    @staticmethod
    def reshape_tensor_per_group(fp_tensor, group_size: int):
        return fp_tensor.flatten().reshape(-1, group_size)

    def quantize_as_u8(self, fp_tensor):
        fp_tensor_per_group = self.reshape_tensor_per_group(
            fp_tensor, self.group_size
        )
        assert (
            len(fp_tensor_per_group.shape) == 2
            and fp_tensor_per_group.shape[1] == self.group_size
        )
        recip_scale = torch.where(self.scale == 0, self.scale, 1.0 / self.scale)
        fu8_tensor_per_group = (
            ((fp_tensor_per_group * recip_scale) + (2**self.n_bits + 1) / 2)
            .floor()
            .clamp(0, 2**self.n_bits - 1)
        )
        return fu8_tensor_per_group.to(torch.uint8).reshape(fp_tensor.shape)

    def dequantize(self, u8_tensor, target_dtype):
        u8_tensor_per_group = self.reshape_tensor_per_group(
            u8_tensor, self.group_size
        )
        offset = 2**self.n_bits / 2
        fp_tensor_per_group = (u8_tensor_per_group - offset).to(target_dtype)
        fp_tensor_per_group *= self.scale.to(target_dtype)
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


class U8Compressor:
    """Abstract class to add u8 compression methods

    This can be used to
    > Apply bitpack elements bellow 8bit
    > Apply classic compression algorithm

    Warning !! .shape of u8_tensor compressed
                must be same as
               .shape once decompressed
    """

    @abc.abstractmethod
    def compress(self, u8_tensor) -> torch.Tensor:
        """
        Args:
            u8_tensor:  tensor to be compressed with dtype torch.uint8
        Return:
            compressed tensor with dtype torch.uint8
        """

    @abc.abstractmethod
    def decompress(self, u8_tensor) -> torch.Tensor:
        """
        Args:
            u8_tensor:  compressed tensor with dtype torch.uint8
        Return:
            tensor decompressed with dtype torch.uint8
        """


class QTensor(torch.Tensor):
    """Common interface for all Compressed storage"""

    @staticmethod
    def __new__(
        cls,
        fp_tensor,
        qscheme,
        *args,
        dequant_to_dtype=torch.float32,
        u8_compressors: T.Optional[T.List[U8Compressor]] = None,
        **kwargs,
    ):
        u8_blob = qscheme.quantize_as_u8(fp_tensor)
        for u8_compressor in u8_compressors or []:
            u8_blob = u8_compressor.compress(u8_blob)
        # we apply all quant/compress prior to __new__
        # because it is the operation that define tensor
        obj = super().__new__(cls, u8_blob, *args, **kwargs)
        # contrary to usual practice we assign
        obj.u8_blob = u8_blob
        obj.qscheme = qscheme
        obj.u8_compressors = u8_compressors or []
        obj.dequant_to_dtype = dequant_to_dtype
        obj.requires_grad = False
        return obj

    def decompress_to_u8(self):
        decompress_u8 = self.u8_blob
        for u8_compressor in reversed(self.u8_compressors):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        return decompress_u8

    def to_torch_float_tensor(self):
        return self.qscheme.dequantize(
            self.decompress_to_u8(), target_dtype=self.dequant_to_dtype
        )

    def clone(self, *args, **kwargs):
        return self.__class__(
            super().clone(*args, **kwargs),
            qscheme=self.qscheme,
            dequant_to_dtype=self.dequant_to_dtype,
        )

    def to(self, *args, **kwargs):
        new_dtype = kwargs.get("dtype") or args[0] if args else None
        if new_dtype is None:
            return self
        new_obj = self.__class__(
            self.u8_blob,
            qscheme=self.qscheme,
            dequant_to_dtype=new_dtype,
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
                    a.to_torch_float_tensor() if isinstance(a, cls) else a
                    for a in args
                ]
                new_kwargs = {
                    k: v.to_torch_float_tensor() if isinstance(v, cls) else v
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


def qscale_per_group_f16_min_max_calibration(
    fp_tensor,
    n_bits: int,
    group_size: int,
    percentile: float = 1.0,
) -> T.Tuple["QScalePerGroupF16"]:
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
    fp_tensor_per_group = QScalePerGroupF16.reshape_tensor_per_group(
        fp_tensor, group_size
    )

    # we use full-range symmetric
    # like torch, ONNX, but oposed to restricted range from
    # TensorFlow, NVIDIA TensorRT and Intel DNNL
    scale = torch.quantile(fp_tensor_per_group.abs(), percentile, dim=1) / (
        -(2**n_bits) / 2
    )

    assert scale.shape[0] == fp_tensor_per_group.shape[0]
    qshape = [scale.shape[0]] + [1]
    return QScalePerGroupF16(
        group_size=group_size,
        scale=scale.reshape(qshape),
        n_bits=n_bits,
    )


def apply_qtensor_in_params_set_as_ref(model: torch.nn.Module):
    """Transform QTensor Parameters into QTensorRef

    This is applied at export time of `torch_to_nnef`
    Just before doing any tracing

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
