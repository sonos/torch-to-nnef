from pathlib import Path

import numpy as np
import torch
from nnef_tools.io.nnef.writer import Writer as NNEFWriter
from torch.onnx import TrainingMode
from torch.onnx.utils import (
    _decide_input_format,
    _validate_dynamic_axes,
    select_model_mode_for_export,
)

from . import __version__
from .nnef_graph import GraphExtractor


fragments = {
    "silu": """
fragment silu( input: tensor<scalar> ) -> ( output : tensor<scalar> )
{
  output = input * sigmoid(input);
}
""".strip(),
    "selu": """
fragment selu(
  x: tensor<scalar>,
  alpha: scalar = 1.67326319,
  lambda: scalar = 1.05070102 )
-> ( y: tensor<scalar> )
{
    y = lambda * select(x < 0.0, alpha * (exp(x) - 1.0), x);
}
""".strip(),
    "accurate_gelu": """
fragment gelu( x: tensor<scalar> ) -> ( y: tensor<scalar> )
{
    y = 0.5 * x * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)));
}
""".strip(),
}


def export_model_to_nnef(
    model,
    args,
    base_path: Path,
    input_names,
    output_names,
    dynamic_axes=None,
    verbose=True,
):
    with select_model_mode_for_export(model, TrainingMode.EVAL):
        args = _decide_input_format(model, args)
        if dynamic_axes is None:
            dynamic_axes = {}
        _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)
        if isinstance(args, (torch.Tensor, int, float, bool)):
            args = (args,)

        graph_extractor = GraphExtractor(model, args)
        if verbose:
            graph_extractor._torch_graph_helper.printall()

        nnef_graph = graph_extractor.parse(
            input_names,
            output_names,
        )

        NNEFWriter(
            compression=1,
            fragments=fragments,
            generate_custom_fragments=True,
            version_custom_fragments=__version__,
        )(nnef_graph, str(base_path))
