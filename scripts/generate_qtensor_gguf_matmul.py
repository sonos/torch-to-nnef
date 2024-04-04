"""Script to generate export variant of quantized GGUF format"""

import argparse
from pathlib import Path

import numpy as np
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
    return nn.Linear(256, 512, bias=False)


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
    with torch.no_grad():
        inps = torch.arange(256).reshape(1, 256).float() - 128.0
        lin = linear()
        fp_outs = lin(inps)
        slug = "full_precision"
        export_model_to_nnef(
            lin,
            inps,  # inps pushed with *inps in forward of module
            file_path_export=dirpath / f"{slug}.nnef.tgz",
            input_names=["input_0"],
            output_names=["output_0"],
            dynamic_axes=None,
            compression_level=0,
            nnef_spec_strict=False,
        )
        np.savez(
            dirpath / f"{slug}.npz",
            **{
                "input_0": inps.detach().numpy(),
                "output_0": fp_outs.detach().numpy(),
            },
        )
        for q_type in QUANTS:
            q_weight = QTensorGGUF(lin.weight, q_type)
            slug = f"tract_gguf_linear_{q_weight.gguf_data_type_name}"
            qlin = replace_nn_ops(lin, q_weight)
            q_outs = qlin(inps)
            export_model_to_nnef(
                qlin,
                inps,
                file_path_export=dirpath / f"{slug}.nnef.tgz",
                input_names=["input_0"],
                output_names=["output_0"],
                dynamic_axes=None,
                compression_level=0,
                nnef_spec_strict=False,
            )
            print()
            print(slug)
            diff = fp_outs - q_outs
            print("diff l2", np.linalg.norm(diff, 2))
            print("diff linf", np.linalg.norm(diff, np.inf))
            np.savez(
                dirpath / f"{slug}_io.npz",
                **{
                    "input_0": inps.detach().numpy(),
                    "output_0": q_outs.detach().numpy(),
                },
            )


if __name__ == "__main__":
    main()
