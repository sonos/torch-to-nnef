import typing as T
from enum import Enum

import torch
from nnef_tools.model import Operation as NOperation

from torch_to_nnef.exceptions import (
    StrictNNEFSpecError,
    TorchToNNEFNotImplementedError,
)
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor import bitpack
from torch_to_nnef.qtensor.base import (
    QScheme,
    QTensor,
    QZPScalePerChannel,
    QZPScalePerGroup,
    QZPScaleScalar,
)


class PackingLayout(str, Enum):
    TILED = "tiled"
    CONTIGUOUS = "contiguous"


class PackingStrategy:
    def __init__(
        self,
        bit_width: int,
        layout: T.Optional[PackingLayout] = PackingLayout.TILED,
    ):
        assert bit_width in [1, 2, 3, 4, 8]
        assert isinstance(layout, PackingLayout)
        if layout == PackingLayout.CONTIGUOUS:
            raise NotImplementedError("packing contiguous not yet implemented")
        self.bit_width = bit_width
        self.layout = layout

    def pack(self, uint8_tensor: torch.Tensor):
        if self.layout == PackingLayout.CONTIGUOUS:
            raise NotImplementedError("packing contiguous not yet implemented")
        return {
            1: bitpack.TensorB1,
            2: bitpack.TensorB2,
            3: bitpack.TensorB3,
            4: bitpack.TensorB4,
            8: bitpack.TensorB8,
        }[self.bit_width].pack(uint8_tensor)


class TargetDType:
    """targetted dtype

    This structure is needed to allow facade between
    """

    def __init__(self, torch_dtype, qscheme: T.Optional[QScheme] = None):
        self.torch_dtype = torch_dtype
        if qscheme is not None:
            # because in tract
            # this will only be part of
            assert isinstance(
                qscheme, QZPScaleScalar
            ), "other qscheme can not leaks into rest of graph"
        self.qscheme = qscheme

    def to_torch_tensor(self, fp_tensor):
        if self.qscheme is None:
            return fp_tensor.to(self.torch_dtype)
        return self.qscheme.quantize_as_torch(fp_tensor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(torch_dtype={self.torch_dtype}, qscheme={self.qscheme})"


class QTensorSepParamsWithPack(QTensor):
    """Quantized Tensor

    This format is usefull to store directly in RAM as torch tensor as there
    is very few manipulations to restore float tensor whatever PyTorch backend
    this make it faster than the others format against all hardware achitecture
    while saving RAM significantly without dedicated C/CPP code.

    Pro:
        1. lot of variants supported (all group size / per chan ...)

    Limitation:
        1. Your initial float tensor first dimension need to be divisible by
        number of element packable in a byte
        2. No implementation in tract

    """

    def __init__(
        self,
        packed_tensor: bitpack.BitPackedTensor,
        qscheme: T.Optional[QScheme],
        target_dtype: TargetDType,
    ):
        super().__init__()
        assert isinstance(qscheme, QScheme)
        assert isinstance(packed_tensor, bitpack.BitPackedTensor)
        self.packed_torch_tensor = packed_tensor
        self.qscheme = qscheme
        self.target_dtype = target_dtype

    def to_torch_tensor(self) -> torch.Tensor:
        u8_tensor = self.packed_torch_tensor.unpack()
        if self.qscheme:
            fp_tensor = self.qscheme.dequantize(u8_tensor)
        else:
            fp_tensor = u8_tensor
        return self.target_dtype.to_torch_tensor(fp_tensor)

    def detach(self):
        return self

    def requires_grad_(self, *args, **kwargs):
        return self

    def clone(self):
        raise NotImplementedError()

    def to(self, *args, **kwargs):
        raise NotImplementedError()

    def cast_into_n_bits(self, n_bits: int):
        current_n_bits_per_elm = self.packed_torch_tensor.n_bits()
        if n_bits == current_n_bits_per_elm:
            return self
        tt = self.packed_torch_tensor.unpack()
        # tmin, tmax = tt.min(), tt.max()
        # max_upper_bound = self.packed_torch_tensor.max_val()
        # assert tmax == max_upper_bound
        # assert tmin == 0
        scale_factor = 2**current_n_bits_per_elm / 2**n_bits
        new_u8_tensor = (tt.to(torch.float32) / scale_factor).to(torch.uint8)
        return QTensorSepParamsWithPack(
            packed_tensor=PackingStrategy(bit_width=n_bits).pack(new_u8_tensor),
            qscheme=self.qscheme.clone_with_scale_factor(scale_factor),
            target_dtype=self.target_dtype,
        )

    @classmethod
    def from_torch_qtensor(
        cls,
        tensor,
        packing_strategy: T.Optional[PackingStrategy] = None,
        target_dtype=None,
    ) -> "QTensor":
        assert tensor.is_quantized
        if tensor.dtype in [torch.quint4x2, torch.quint2x4]:
            # TODO: handle torch.quint4x2 and torch.quint2x4
            raise TorchToNNEFNotImplementedError(tensor.dtype)
        if packing_strategy is None:
            # default
            packing_strategy = PackingStrategy(bit_width=8)
        assert isinstance(packing_strategy, PackingStrategy)
        torch_qscheme = tensor.qscheme()
        qscheme: T.Optional[QScheme] = None
        itensor = tensor.int_repr()
        offset_zp = 0
        if itensor.dtype == torch.int8:
            itensor = (itensor.to(torch.int16) + 128).to(torch.uint8)
            offset_zp += 128

        if torch_qscheme == torch.per_channel_affine:
            qscale = tensor.q_per_channel_scales()
            qzerop = tensor.q_per_channel_zero_points() + offset_zp
            dim = tensor.q_per_channel_axis()

            # reshapes to allow torch jit works
            qshape = [1] * len(tensor.shape)
            qshape[dim] = qscale.shape[0]
            qscale = qscale.reshape(qshape)
            qzerop = qzerop.reshape(qshape)

            qscheme = QZPScalePerChannel(qzerop, qscale, dim=dim)
        elif torch_qscheme == torch.per_tensor_affine:
            qscale = tensor.q_scale()
            qzerop = tensor.q_zero_point() + offset_zp
            qscheme = QZPScaleScalar(qzerop, qscale)
        else:
            raise TorchToNNEFNotImplementedError(
                f"not supported quantization scheme {qscheme}"
            )

        if target_dtype is None:
            target_dtype = TargetDType(tensor.dtype)
        return cls(
            packing_strategy.pack(itensor),
            qscheme=qscheme,
            target_dtype=target_dtype,
        )

    def forward(self):
        return self.to_torch_tensor()

    def __repr__(self) -> str:
        try:
            return f"{self.__class__.__name__}({self.packed_torch_tensor}, {self.qscheme})"
        except AttributeError:
            return f"{self.__class__.__name__}(?)"


# export logic {


def _build_nnef_zp_scale_tensors(g, name_to_tensor, node, qtensor):
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.op.primitive import base

    torch_zero_point = qtensor.qscheme.zero_point
    nnef_zero_point_tensor = base.add_tensor_variable_node_as_nnef_tensor(
        # build imaginary node to fill data correctly
        node=base.TensorVariable(
            name=node.outputs[0].name,
            data=torch_zero_point,
            shape=list(torch_zero_point.shape),
            dtype=torch_zero_point.dtype,
        ),
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix="zero_points",
    )

    torch_scale = qtensor.qscheme.scale
    nnef_scale_tensor = base.add_tensor_variable_node_as_nnef_tensor(
        # build imaginary node to fill data correctly
        node=base.TensorVariable(
            name=node.outputs[0].name,
            data=torch_scale,
            shape=list(torch_scale.shape),
            dtype=torch_scale.dtype,
        ),
        g=g,
        name_to_tensor=name_to_tensor,
        name_suffix="scales",
    )
    return nnef_zero_point_tensor, nnef_scale_tensor


def outer_per_channel_dequantization(
    g,
    nnef_unpacked_tensor,
    nnef_zero_point_tensor,
    nnef_scale_tensor,
    nnef_output_tensor,
    attribs,
    name_to_tensor,
    tract_custom_operator: bool = False,
):
    # pylint: disable-next=import-outside-toplevel
    from torch_to_nnef.op.primitive import base

    if tract_custom_operator:
        # Implementation with dedicated operator
        NOperation(
            g,
            type="tract_core_zpscale_per_channel",
            inputs=(
                nnef_unpacked_tensor,
                nnef_zero_point_tensor,
                nnef_scale_tensor,
            ),
            outputs=nnef_output_tensor,
            attribs=attribs,
        )
    else:
        # Implementation without any special operators
        nnef_sub_zp_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_unpacked_tensor.name,
                data=None,
                shape=list(nnef_unpacked_tensor.shape),
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="sub_zp",
        )
        NOperation(
            g,
            type="sub",
            inputs=(
                nnef_unpacked_tensor,
                nnef_zero_point_tensor,
            ),
            outputs=nnef_sub_zp_tensor,
            attribs=attribs,
        )
        nnef_float_out_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_output_tensor.name,
                data=None,
                shape=list(nnef_output_tensor.shape),
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="out_float",
        )
        NOperation(
            g,
            type="mul",
            inputs=(
                nnef_sub_zp_tensor,
                nnef_scale_tensor,
            ),
            outputs=nnef_float_out_tensor,
            attribs=attribs,
        )
        NOperation(
            g,
            type="tract_core_cast",
            inputs=nnef_float_out_tensor,
            outputs=nnef_output_tensor,
            attribs={"to": attribs["to"]},
        )


def outer_per_group_dequantization(
    g,
    nnef_unpacked_tensor,
    nnef_zero_point_tensor,
    nnef_scale_tensor,
    nnef_output_tensor,
    attribs,
    name_to_tensor,
    tract_custom_operator: bool = False,
):
    if tract_custom_operator:
        # Implementation with dedicated operator
        NOperation(
            g,
            type="tract_core_zpscale_per_chunk",
            inputs=(
                nnef_unpacked_tensor,
                nnef_zero_point_tensor,
                nnef_scale_tensor,
            ),
            outputs=nnef_output_tensor,
            attribs=attribs,
        )
    else:
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.primitive import base

        group_size = attribs.pop("group_size")
        # Implementation without any special operators
        original_shape = nnef_unpacked_tensor.shape
        total_n_elements = 1
        for _ in original_shape:
            total_n_elements *= _
        assert total_n_elements % group_size == 0
        n_groups = int(total_n_elements / group_size)
        per_group_shape = [group_size, n_groups]

        nnef_reshaped_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_unpacked_tensor.name,
                data=None,
                shape=list(per_group_shape),
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="reshaped_per_group",
        )
        NOperation(
            g,
            type="reshape",
            inputs=nnef_unpacked_tensor,
            outputs=nnef_reshaped_tensor,
            attribs={"shape": per_group_shape},
        )
        nnef_sub_zp_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_unpacked_tensor.name,
                data=None,
                shape=list(per_group_shape),
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="sub_zp",
        )
        NOperation(
            g,
            type="sub",
            inputs=(
                nnef_reshaped_tensor,
                nnef_zero_point_tensor,
            ),
            outputs=nnef_sub_zp_tensor,
            attribs=attribs,
        )
        nnef_float_out_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_output_tensor.name,
                data=None,
                shape=list(per_group_shape),
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="out_float",
        )
        NOperation(
            g,
            type="mul",
            inputs=(
                nnef_sub_zp_tensor,
                nnef_scale_tensor,
            ),
            outputs=nnef_float_out_tensor,
            attribs=attribs,
        )
        nnef_fp_per_group_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=nnef_output_tensor.name,
                data=None,
                shape=per_group_shape,
                dtype=torch.float32,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
            name_suffix="_fp_shaped_per_group",
        )
        NOperation(
            g,
            type="tract_core_cast",
            inputs=nnef_float_out_tensor,
            outputs=nnef_fp_per_group_tensor,
            attribs={"to": attribs["to"]},
        )

        NOperation(
            g,
            type="reshape",
            inputs=nnef_fp_per_group_tensor,
            outputs=nnef_output_tensor,
            attribs={"shape": original_shape},
        )


class QTensorSepParamsWithPackExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorSepParamsWithPack

    def convert_to_nnef(
        self,
        g,
        node,
        name_to_tensor,
        null_ref,
        torch_graph,
        nnef_spec_strict: bool,
        **kwargs,
    ):
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.dtypes import TORCH_DTYPE_TO_TRACT_STR

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op import quantized

        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef.op.primitive import base

        if nnef_spec_strict:
            raise StrictNNEFSpecError(
                "Impossible to export QTensor with NNEF spec compliance activated"
            )

        qtensor = node.op_ref

        is_8bit = qtensor.packed_torch_tensor.n_bits() == 8
        packed_tensor = qtensor.packed_torch_tensor.raw_tensor
        nnef_packed_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            name_suffix="raw_bit_packed" if not is_8bit else "u8_unpacked",
            # build imaginary node to fill data correctly
            node=base.TensorVariable(
                name=node.outputs[0].name,
                data=packed_tensor,
                shape=list(packed_tensor.shape),
                dtype=packed_tensor.dtype,
            ),
            g=g,
            name_to_tensor=name_to_tensor,
        )

        if qtensor.packed_torch_tensor.n_bits() == 8:
            nnef_unpacked_tensor = nnef_packed_tensor
        else:
            nnef_unpacked_tensor = base.add_tensor_variable_node_as_nnef_tensor(
                name_suffix="_u8_unpacked",
                # build imaginary node to fill data correctly
                node=base.TensorVariable(
                    name=node.outputs[0].name,
                    data=None,
                    shape=list(qtensor.packed_torch_tensor.shape),
                    dtype=packed_tensor.dtype,
                ),
                g=g,
                name_to_tensor=name_to_tensor,
            )

            NOperation(
                g,
                type="tract_core_dyn_bit_unpack",
                inputs=nnef_packed_tensor,
                outputs=nnef_unpacked_tensor,
                attribs={
                    "bit_width": qtensor.packed_torch_tensor.n_bits(),
                    "layout": "tiled",
                },
            )

        real_output = qtensor.to_torch_tensor()
        if real_output.is_quantized:
            nnef_output_tensor = quantized.add_quantized_tensor_to_ngraph(
                g,
                node=base.TensorVariable(
                    name=node.outputs[0].name,
                    data=None,
                    shape=list(real_output.shape),
                    dtype=real_output.dtype,
                ),
                qtensor=real_output,
                name_to_tensor=name_to_tensor,
                tensor_name=None,
            )
        else:
            nnef_output_tensor = base.add_tensor_variable_node_as_nnef_tensor(
                # build imaginary node to fill data correctly
                node=base.TensorVariable(
                    name=node.outputs[0].name,
                    data=None,
                    shape=list(real_output.shape),
                    dtype=real_output.dtype,
                ),
                g=g,
                name_to_tensor=name_to_tensor,
            )

        if qtensor.qscheme is not None:
            if isinstance(qtensor.qscheme, QZPScaleScalar):
                torch_quantized_tensor = qtensor.qscheme.quantize_as_torch(
                    real_output
                )
                nnef_quant_applied_tensor = (
                    quantized.add_quantized_tensor_to_ngraph(
                        g,
                        node=base.TensorVariable(
                            name=node.outputs[0].name,
                            data=None,
                            shape=list(torch_quantized_tensor.shape),
                            dtype=torch_quantized_tensor.dtype,
                        ),
                        qtensor=torch_quantized_tensor,
                        name_to_tensor=name_to_tensor,
                        tensor_name="casted_linear_scalar_quantized",
                    )
                )
                NOperation(
                    g,
                    type="tract_core_cast",
                    inputs=nnef_unpacked_tensor,
                    outputs=nnef_quant_applied_tensor,
                    attribs={},
                )
                nnef_unpacked_tensor = nnef_quant_applied_tensor

        attribs: T.Dict[str, T.Any] = {}
        if not real_output.is_quantized:
            attribs = {
                "to": TORCH_DTYPE_TO_TRACT_STR[real_output.dtype],
            }
        if isinstance(
            qtensor.target_dtype.qscheme, QZPScaleScalar
        ) or isinstance(
            qtensor.qscheme,
            QZPScaleScalar,
        ):
            NOperation(
                g,
                type="tract_core_cast",
                inputs=nnef_unpacked_tensor,
                outputs=nnef_output_tensor,
                attribs=attribs,
            )
        else:
            (
                nnef_zero_point_tensor,
                nnef_scale_tensor,
            ) = _build_nnef_zp_scale_tensors(g, name_to_tensor, node, qtensor)
            if isinstance(qtensor.qscheme, QZPScalePerChannel):
                outer_per_channel_dequantization(
                    g,
                    nnef_unpacked_tensor,
                    nnef_zero_point_tensor,
                    nnef_scale_tensor,
                    nnef_output_tensor,
                    attribs,
                    name_to_tensor,
                )
            elif isinstance(qtensor.qscheme, QZPScalePerGroup):
                attribs["group_size"] = int(qtensor.qscheme.group_size)
                outer_per_group_dequantization(
                    g,
                    nnef_unpacked_tensor,
                    nnef_zero_point_tensor,
                    nnef_scale_tensor,
                    nnef_output_tensor,
                    attribs,
                    name_to_tensor,
                )
            else:
                raise NotImplementedError(f"not handled: {qtensor.qscheme}")
        return ["tract_core"]


# }
