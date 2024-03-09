import typing as T

from nnef_tools.model import Operation as NOperation

from torch_to_nnef.exceptions import StrictNNEFSpecError
from torch_to_nnef.op.custom_extractors.base import ModuleInfoExtractor
from torch_to_nnef.qtensor import (
    QTensor,
    QZPScalePerChannel,
    QZPScalePerChunk,
    QZPScaleScalar,
)


class QTensorExtractor(ModuleInfoExtractor):
    MODULE_CLASS = QTensor

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

        packed_tensor = qtensor.packed_torch_tensor.raw_tensor
        nnef_packed_tensor = base.add_tensor_variable_node_as_nnef_tensor(
            name_suffix="raw_bit_packed",
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
        if (
            isinstance(qtensor.target_dtype.qscheme, QZPScaleScalar)
            or qtensor.target_dtype.qscheme is None
        ):
            NOperation(
                g,
                type="tract_core_cast",
                inputs=nnef_unpacked_tensor,
                outputs=nnef_output_tensor,
                attribs=attribs,
            )
        else:
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
            if isinstance(qtensor.qscheme, QZPScalePerChannel):
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
            elif isinstance(qtensor.qscheme, QZPScalePerChunk):
                attribs["chunk_size"] = int(qtensor.qscheme.chunk_size)
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
                raise NotImplementedError(f"not handled: {qtensor.qscheme}")
        return ["tract_core"]
