import typing as T

import torch
from nnef_tools.model import Operation as NOperation

from torch_to_nnef.exceptions import StrictNNEFSpecError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor import (
    QTensorBasic,
    QTensorGGUF,
    QTensorSepParamsWithPack,
    QZPScalePerChannel,
    QZPScalePerGroup,
    QZPScaleScalar,
)


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


class QTensorBasicExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensorBasic

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
        raise NotImplementedError(
            "QTensorBasic are not meant to be exported."
            " please re-consider your quantization tensor type if you wish to export"
            " with (QTensorSepParamsWithPack | QTensorGGUF)"
        )


try:

    class QTensorGGUFExtractor(ModuleInfoExtractor):
        MODULE_CLASS = QTensorGGUF

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
            """implementation with storage"""
            # pylint: disable-next=import-outside-toplevel
            from torch_to_nnef.dtypes import numpy_dtype_to_tract_str

            # pylint: disable-next=import-outside-toplevel
            from torch_to_nnef.op.primitive import base

            # pylint: disable-next=import-outside-toplevel
            from torch_to_nnef.op.primitive.base import add_nnef_operation

            q_tensor = node.op_ref
            out_node = node.outputs[0]
            nnef_tensor_ref = base.add_tensor_variable_node_as_nnef_tensor(
                g, out_node, name_to_tensor, prevent_variable=True
            )
            nnef_tensor_ref.qtensor = (
                q_tensor  # main assign to allow corect dump
            )
            add_nnef_operation(
                graph=g,
                type="tract_core_gguf_variable",
                inputs=None,
                outputs=nnef_tensor_ref,
                attribs={
                    "gguf_filename": f"{out_node.export_name}.gguf",
                    "gguf_tensor_name": out_node.export_name,
                    "gguf_dtype": q_tensor.gguf_data_type_name,
                    "shape": list(nnef_tensor_ref.shape),
                    "output_datum_type": numpy_dtype_to_tract_str(
                        nnef_tensor_ref.dtype
                    ),
                },
            )
            return ["tract_core"]

except ImportError as exp:
    # feature gate: gguf_dtype
    print(exp)
