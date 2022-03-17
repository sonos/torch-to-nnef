""" Tools to manipulate tract programatically """

import subprocess
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn


def tract_convert_onnx_to_nnef(onnx_path, io_npz_path, nnef_path):
    subprocess.check_call(
        f'tract {onnx_path} --input-bundle {io_npz_path}  dump --nnef {nnef_path}',
        shell=True,
    )


def tract_assert_io(nnef_path: Path, io_npz_path: Path):
    cmd = f"tract {nnef_path} --input-bundle {io_npz_path} -O run --assert-output-bundle {io_npz_path}"
    try:
        subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def special_quantize_io(x, model, is_input):
    if is_input:
        # return model.quantize()x
        return model.quant(x).int_repr()
    return model.quant(x).int_repr()


def nop(x, *args, **kwargs):
    return x


def build_io(model, test_input, io_npz_path=None):
    tup_inputs = test_input if isinstance(test_input, tuple) else (test_input,)
    input_names = [f"input_{idx}" for idx, _ in enumerate(tup_inputs)]
    test_output = model(*tup_inputs)
    # We do not handle complex outputs except tensor or list of tensor {
    if isinstance(test_output, torch.Tensor):
        test_output = [test_output]
    test_output = list(test_output)
    if len(test_output) == 2 and isinstance(
        test_output[1], tuple
    ):  # LSTM special case
        test_output[1] = test_output[1][0]
    # }

    output_names = [f"output_{idx}" for idx, _ in enumerate(test_output)]

    if io_npz_path is not None:
        if isinstance(model, torch.quantization.QuantWrapper):
            fn = partial(special_quantize_io, model=model)
        else:
            fn = nop
        kwargs = {
            key: fn(input_arg.detach(), is_input=True).numpy()
            for key, input_arg in zip(input_names, tup_inputs)
        }
        kwargs.update(
            {
                key: fn(output_arg.detach(), is_input=True).numpy()
                for key, output_arg in zip(output_names, test_output)
            }
        )
        np.savez(io_npz_path, **kwargs)
    return input_names, output_names


def pytorch_to_onnx_to_tract_to_nnef(
    model: nn.Module, test_input, nnef_path, onnx_path=None, io_npz_path=None
):
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = onnx_path or (Path(tmpdir) / "model.onnx")
        io_npz_path = io_npz_path or (Path(tmpdir) / "io.npz")
        input_names, output_names = build_io(model, test_input, io_npz_path)
        torch.onnx.export(
            model,
            test_input,
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
        )
        tract_convert_onnx_to_nnef(
            onnx_path,
            io_npz_path,
            nnef_path=nnef_path,
        )


def debug_dumper_pytorch_to_onnx_to_nnef(
    model: nn.Module, test_input, target_folder: Path
):
    assert not target_folder.exists()
    target_folder.mkdir()
    onnx_path = target_folder / "model.onnx"
    nnef_path = target_folder / "tract_onnx_converted_model.nnef"
    io_npz_path = target_folder / "io.npz"
    pytorch_to_onnx_to_tract_to_nnef(
        model,
        test_input,
        nnef_path,
        onnx_path=onnx_path,
        io_npz_path=io_npz_path,
    )
    subprocess.check_output(
        f"cd {target_folder} && tar -xvf {nnef_path}", shell=True
    )
