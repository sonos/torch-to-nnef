"""Script to generate export variant of quantized GGUF format"""

import argparse
from pathlib import Path

import torch
from ggml import ffi, lib
from torch import nn

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.qtensor import QTensorGGUF, replace_nn_ops

# from ggml.utils import init, numpy, copy
# import numpy as np
# from math import pi, cos, sin, ceil
#
# import matplotlib.pyplot as plt


QUANTS = [
    typ
    for typ in range(lib.GGML_TYPE_COUNT)
    if lib.ggml_is_quantized(typ)
    # Apparently not supported
    and typ
    not in [
        # not supported {
        lib.GGML_TYPE_IQ1_M,
        lib.GGML_TYPE_IQ1_S,
        lib.GGML_TYPE_Q8_K,
        lib.GGML_TYPE_Q8_1,
        lib.GGML_TYPE_Q8_K,
        # }
        #
        # need specific .map , .neighbours, .grids to be assigned in static {
        # see ggml-quants.c
        # // ================================ IQ2 quantization =============================================
        lib.GGML_TYPE_IQ2_S,
        lib.GGML_TYPE_IQ2_XS,
        lib.GGML_TYPE_IQ2_XXS,
        lib.GGML_TYPE_IQ3_S,
        lib.GGML_TYPE_IQ3_XXS,
        # }
    ]
]


def get_name(q_type):
    name = lib.ggml_type_name(q_type)
    return ffi.string(name).decode("utf-8") if name else "?"


QUANTS.sort(key=get_name)


def linear():
    return nn.Linear(5, 2)


def parser_cli():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="dir to dump tract unit-tests for GGUF formats",
    )

    return parser.parse_args()


def main():
    args = parser_cli()
    dirpath = Path(args.dir)
    assert dirpath.exists()
    for q_type in QUANTS:
        lin = linear()
        slug = f"tract_gguf_linear_{q_type}"
        q_weight = QTensorGGUF(lin.weight, q_type)
        qlin = replace_nn_ops(lin, q_weight)
        args = torch.arange(10).reshape(2, 5).float()
        export_model_to_nnef(
            qlin,
            args,  # args pushed with *args in forward of module
            file_path_export=dirpath / f"{slug}.nnef.tgz",
            input_names=["input_0"],
            output_names=["output_0"],
            dynamic_axes=None,
            compression_level=0,
            nnef_spec_strict=False,
        )


if __name__ == "__main__":
    main()
