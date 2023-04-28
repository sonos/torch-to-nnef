from torch_to_nnef.exceptions import TorchToNNEFNotImplementedError
from torch_to_nnef.op.primitive.base import (
    add_single_output_op,
    add_tensor_variable_node_as_nnef_tensor,
    cast_and_add_nnef_operation,
    get_or_add_tensor_variable_in_nnef,
    pick_rank,
    weight_bias_and_output_tensor,
)


def batch_norm(g, node, name_to_tensor, null_ref, **kwargs):
    """

    nnef inputs:
        input: tensor<scalar>
        mean: tensor<scalar>
        variance: tensor<scalar>
        offset: tensor<scalar>
        scale: tensor<scalar>
        epsilon: scalar

    nnef op:
        output = offset + scale * (input - mean) / sqrt(variance + epsilon);
    """
    (
        input_node,
        weight_node,
        bias_node,
        running_mean_node,
        running_var_node,
        _,  # training
        _,  # momentum
        eps_node,
        _,  # cudnn_enabled
    ) = node.inputs

    # expand in stored variables export to avoid unsqueeze guessing in graph {
    params_nodes = [weight_node, running_mean_node, running_var_node]
    if bias_node.data is not None:
        params_nodes.append(bias_node)
    for param_node in params_nodes:
        param_node.data = param_node.data.unsqueeze(0)
        param_node.shape = list(param_node.data.shape)
        for _ in range(input_node.rank - param_node.rank):
            param_node.data = param_node.data.unsqueeze(-1)
            param_node.shape = list(param_node.data.shape)
    # }

    weight_ref, bias_ref, output_tensor = weight_bias_and_output_tensor(
        g,
        node,
        weight_node,
        bias_node,
        name_to_tensor,
        null_ref,
    )
    running_mean_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="running_mean",
        node=running_mean_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    running_var_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="running_var",
        node=running_var_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    cast_and_add_nnef_operation(
        name_to_tensor=name_to_tensor,
        graph=g,
        type="batch_normalization",
        name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            running_mean_ref,
            running_var_ref,
            bias_ref,
            weight_ref,
        ),
        outputs=output_tensor,
        attribs={"epsilon": eps_node.data},
    )


def norm(g, node, name_to_tensor, **kwargs):
    """
    NOTE this is only the normed vector
    """
    input_node, p_node, axes_node, keep_dim_node = node.inputs
    if p_node.data not in [1, 2]:
        raise TorchToNNEFNotImplementedError(
            "norm with p only supported for 1 and 2"
        )

    custom_fragment_name = f"norm_p{p_node.data}"
    out = add_single_output_op(
        g,
        node,
        name_to_tensor,
        custom_fragment_name,
        inputs=get_or_add_tensor_variable_in_nnef(
            g, input_node, name_to_tensor
        ),
        attrs={"axes": [pick_rank(input_node, dim) for dim in axes_node.data]},
        output_tensor_name_suffix="_norm" if not keep_dim_node.data else "",
    )
    if not keep_dim_node.data:
        add_single_output_op(
            g,
            node,
            name_to_tensor,
            "squeeze",
            inputs=out,
            attrs={
                "axes": [pick_rank(input_node, dim) for dim in axes_node.data]
            },
            pass_quantization_params=True,
        )
    return [custom_fragment_name]


def layer_norm(g, node, name_to_tensor, null_ref, **kwargs):
    (
        input_tensor_node,
        normalized_shape_node,
        weight_node,
        bias_node,
        eps_node,
        elementwise_affine_node,
    ) = node.inputs

    mean_axes = sorted(
        input_tensor_node.rank - r - 1
        for r, _ in enumerate(normalized_shape_node.data)
    )
    has_affine = elementwise_affine_node.data and not (
        # check affine as any use
        (bias_node.data == 0).all().tolist()
        and (weight_node.data == 1).all().tolist()
    )
    inputs = [input_tensor_node]
    op_name = "layer_norm"
    if has_affine:
        op_name = "layer_norm_with_affine"
        inputs += [weight_node, bias_node]
    add_single_output_op(
        g,
        node,
        name_to_tensor,
        nnef_op_type=op_name,
        inputs=[
            get_or_add_tensor_variable_in_nnef(g, _, name_to_tensor)
            if _
            else null_ref
            for _ in inputs
        ],
        attrs={"mean_axes": mean_axes, "eps": eps_node.data},
    )

    return [op_name]


def group_norm(g, node, name_to_tensor, **kwargs):
    """
    It is a special case of NNEF batch_normalization
    with variance and mean being tensor
    """
    (
        input_node,
        n_groups_node,
        scale_node,
        offset_node,
        eps_node,
        _,  # is_affine_node
    ) = node.inputs
    for nd in [offset_node, scale_node]:
        for _ in range(input_node.rank - nd.rank - 1):
            nd.data = nd.data.unsqueeze(-1)
        nd.shape = list(nd.data.shape)

    offset_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="offset",
        node=offset_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )
    scale_ref = add_tensor_variable_node_as_nnef_tensor(
        name_suffix="scale",
        node=scale_node,
        g=g,
        name_to_tensor=name_to_tensor,
    )

    # x.reshape(3, 1* 2* 2).mean_or_std(dim=1).repeat(2, 1).t().reshape(6)
    add_single_output_op(
        g=g,
        name_to_tensor=name_to_tensor,
        node=node,
        nnef_op_type="group_norm",
        # name=f"{node.outputs[0].export_name}_op",
        inputs=(
            get_or_add_tensor_variable_in_nnef(g, input_node, name_to_tensor),
            offset_ref,
            scale_ref,
        ),
        attrs={
            "epsilon": eps_node.data,
            "num_groups": n_groups_node.data,
            "batch_size": input_node.shape[0],
            "num_channels": input_node.shape[1],
        },
    )
    return ["group_norm"]
