""" Tools to manipulate tract programatically """

import subprocess
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn


class OnnxExportError(RuntimeError):
    pass


class TractOnnxToNNEFError(RuntimeError):
    pass


class IOPytorchTractNotISOError(ValueError):
    pass


def tract_convert_onnx_to_nnef(onnx_path, io_npz_path, nnef_path):
    subprocess.check_call(
        f'tract {onnx_path} --input-bundle {io_npz_path}  dump --nnef {nnef_path}',
        shell=True,
        stderr=subprocess.STDOUT,
    )


def tract_assert_io(nnef_path: Path, io_npz_path: Path, raise_exception=True):
    cmd = f"tract {nnef_path} --input-bundle {io_npz_path} -O run --assert-output-bundle {io_npz_path}"
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        _, err = proc.communicate()
        if err and "ERROR" in err.decode("utf8"):
            if raise_exception:
                raise IOPytorchTractNotISOError(err.decode("utf8"))
            return False
    return True


def special_quantize_io(x, model, is_input):
    if is_input:
        return model.quant(x).int_repr()
    return x.int_repr()


def nop(x, *args, **kwargs):
    return x


def _unfold_outputs(test_outputs):
    if isinstance(test_outputs, torch.Tensor):
        test_outputs = [test_outputs]
    test_outputs = list(test_outputs)

    unfolded_outputs = []
    for out in test_outputs:
        if isinstance(out, torch.Tensor):
            unfolded_outputs.append(out)
        elif isinstance(out, (list, tuple)):
            for sub_out in out:
                unfolded_outputs.append(sub_out)
    return unfolded_outputs


def build_io(model, test_input, io_npz_path=None):
    tup_inputs = test_input if isinstance(test_input, tuple) else (test_input,)
    input_names = [f"input_{idx}" for idx, _ in enumerate(tup_inputs)]
    test_outputs = _unfold_outputs(model(*tup_inputs))
    output_names = [f"output_{idx}" for idx, _ in enumerate(test_outputs)]

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
                key: fn(output_arg.detach(), is_input=False).numpy()
                for key, output_arg in zip(output_names, test_outputs)
            }
        )
        np.savez(io_npz_path, **kwargs)
    return input_names, output_names


def pytorch_to_onnx_to_tract_to_nnef(
    model: nn.Module,
    test_input,
    nnef_path,
    onnx_path=None,
    io_npz_path=None,
    raise_export_error: bool = True,
) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = onnx_path or (Path(tmpdir) / "model.onnx")
        io_npz_path = io_npz_path or (Path(tmpdir) / "io.npz")
        input_names, output_names = build_io(model, test_input, io_npz_path)
        try:
            torch.onnx.export(
                model,
                test_input,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
            )
        # parametrized failure exception emission
        # pylint: disable-next=broad-except
        except Exception as exp:
            if raise_export_error:
                raise OnnxExportError(exp.args) from exp
            print("ONNX export error")
            return False
        try:
            tract_convert_onnx_to_nnef(
                onnx_path,
                io_npz_path,
                nnef_path=nnef_path,
            )
        # parametrized failure exception emission
        # pylint: disable-next=broad-except
        except Exception as exp:
            if raise_export_error:
                raise TractOnnxToNNEFError(exp.args) from exp
            print("tract ONNX->NNEF export error")
            return False
        return True


def debug_dumper_pytorch_to_onnx_to_nnef(
    model: nn.Module,
    test_input,
    target_folder: Path,
    raise_export_error: bool = True,
) -> bool:
    assert not target_folder.exists()
    target_folder.mkdir()
    onnx_path = target_folder / "model.onnx"
    nnef_path = target_folder / "tract_onnx_converted_model.nnef"
    io_npz_path = target_folder / "io.npz"
    sucessfull_export = pytorch_to_onnx_to_tract_to_nnef(
        model,
        test_input,
        nnef_path,
        onnx_path=onnx_path,
        io_npz_path=io_npz_path,
        raise_export_error=raise_export_error,
    )
    if not sucessfull_export:
        return False
    subprocess.check_output(
        f"cd {target_folder} && tar -xvf {nnef_path}", shell=True
    )
    return True
