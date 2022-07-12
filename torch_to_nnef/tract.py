""" Tools to manipulate tract programatically


NOTE: interaction are done with *Nix tty system in mind, no support for Window

"""

import logging
import os
import subprocess
import tempfile
import typing as T
from functools import partial
from pathlib import Path

import nnef
import numpy as np
import torch
from torch import nn
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import select_model_mode_for_export  # type: ignore

from torch_to_nnef.collect_env import dump_environment_versions

TRACT_PATH = os.environ.get("TRACT_PATH", "tract")

LOGGER = logging.getLogger(__name__)


class OnnxExportError(RuntimeError):
    pass


class TractOnnxToNNEFError(RuntimeError):
    pass


class IOPytorchTractNotISOError(ValueError):
    pass


def tract_convert_onnx_to_nnef(onnx_path, io_npz_path, nnef_path):
    subprocess.check_call(
        (
            f"{TRACT_PATH} {onnx_path} --input-bundle {io_npz_path} "
            f"--nnef-tract-core --nnef-tract-pulse dump --nnef {nnef_path}"
        ),
        shell=True,
        stderr=subprocess.STDOUT,
    )


def tract_assert_io(
    nnef_path: Path,
    io_npz_path: Path,
    raise_exception=True,
):
    cmd = (
        f"{TRACT_PATH} {nnef_path} --input-bundle {io_npz_path} "
        f"--nnef-tract-core --nnef-tract-pulse "
        + f"-vvv -O run --assert-output-bundle {io_npz_path}"
    )
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        _, err = proc.communicate()
        if err:
            serr = err.decode("utf8")
            if any(_ in serr for _ in ["RUST_BACKTRACE", "ERROR"]):
                if raise_exception:
                    raise IOPytorchTractNotISOError(serr)
                return False
            LOGGER.debug(err)
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


def build_io(
    model, test_input, io_npz_path=None, input_names=None, output_names=None
):
    tup_inputs = test_input if isinstance(test_input, tuple) else (test_input,)
    if input_names is None:
        input_names = [f"input_{idx}" for idx, _ in enumerate(tup_inputs)]

    with select_model_mode_for_export(model, TrainingMode.EVAL):
        test_outputs = _unfold_outputs(model(*tup_inputs))

    if output_names is None:
        output_names = [f"output_{idx}" for idx, _ in enumerate(test_outputs)]

    assert len(input_names) == len(tup_inputs)
    assert len(output_names) == len(test_outputs)

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
            LOGGER.warning(f"ONNX export error: {exp}")
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
            LOGGER.warning(f"tract ONNX->NNEF export error: {exp}")
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


def all_close_map_weights(weight_map_file_paths: T.Dict[Path, Path]):
    for wpath, owpath in weight_map_file_paths.items():
        with wpath.open("rb") as fh:
            with owpath.open("rb") as fh_o:
                arr = nnef.read_tensor(fh)
                oarr = nnef.read_tensor(fh_o)
                assert np.allclose(arr, oarr), f"{wpath} vs {owpath}"


def assert_io_and_debug_bundle(
    model: nn.Module,
    test_input,
    nnef_file_path: Path,
    io_npz_path: T.Optional[Path] = None,
    debug_bundle_path: T.Optional[Path] = None,
    input_names: T.Optional[T.List[str]] = None,
    output_names: T.Optional[T.List[str]] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if io_npz_path is None:
                io_npz_path = Path(tmpdir) / "io.npz"
                build_io(
                    model,
                    test_input,
                    io_npz_path=io_npz_path,
                    input_names=input_names,
                    output_names=output_names,
                )
            assert nnef_file_path.exists()
            assert io_npz_path.exists()
            LOGGER.info("Start checking IO is ISO between tract and Pytorch")
            tract_assert_io(
                nnef_file_path,
                io_npz_path,
                raise_exception=True,
            )
            LOGGER.info(
                f"IO bit match between tract and Pytorch for {nnef_file_path}"
            )
        except IOPytorchTractNotISOError as exp:
            if debug_bundle_path is None:
                raise exp
            nnef_file_path = nnef_file_path.absolute()
            no_suffix_debug_bundle_path = debug_bundle_path.with_suffix(
                ""
            ).absolute()
            no_suffix_debug_bundle_path.mkdir(parents=True)
            subprocess.check_output(
                f"cd {no_suffix_debug_bundle_path} && "
                f"cp {nnef_file_path} {no_suffix_debug_bundle_path}/model.nnef.tgz && "
                f"tar -xvzf {nnef_file_path} && "
                f"cp {io_npz_path} {no_suffix_debug_bundle_path}/io.npz",
                shell=True,
            )
            dump_environment_versions(no_suffix_debug_bundle_path)

            debug_dumper_pytorch_to_onnx_to_nnef(
                model,
                test_input,
                target_folder=no_suffix_debug_bundle_path / "tract",
                raise_export_error=False,
            )
            if any(
                extension in debug_bundle_path.suffix
                for extension in ["tgz", "tar.gz"]
            ):
                subprocess.check_output(
                    f"cd {no_suffix_debug_bundle_path.parent} && "
                    f"tar -cvzf {debug_bundle_path.absolute()} {no_suffix_debug_bundle_path.name} && cd - && "
                    # rm acceptable since dir created ensured empty before use
                    f"rm -r {no_suffix_debug_bundle_path}",
                    shell=True,
                )
            LOGGER.info(f"debug bundle built at {debug_bundle_path}")

            exp.args = tuple(
                [f"test with model: {model}\n" + exp.args[0]]
                + list(exp.args)[1:]
            )
            raise exp
