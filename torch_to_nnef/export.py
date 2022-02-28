from pathlib import Path

import torch
from nnef_tools.io.nnef.writer import Writer as NNEFWriter
from torch.onnx import TrainingMode
from torch.onnx.utils import (
    _decide_input_format,
    _validate_dynamic_axes,
    select_model_mode_for_export,
)

from .nnef_graph import GraphExtractor


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
        nnef_graph = GraphExtractor(model, args).parse()
        NNEFWriter(compression=1)(nnef_graph, str(base_path))
