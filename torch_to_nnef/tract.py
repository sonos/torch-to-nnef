""" Tools to manipulate tract programatically


NOTE: interaction are done with *Nix tty system in mind, no support for Window

"""

import logging
import os
import subprocess
import tempfile
import typing as T
from pathlib import Path

import nnef
import numpy as np
import torch
from torch import nn
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import select_model_mode_for_export  # type: ignore

from torch_to_nnef.collect_env import dump_environment_versions
from torch_to_nnef.exceptions import (
    IOPytorchTractNotISOError,
    OnnxExportError,
    TractError,
    TractOnnxToNNEFError,
)
from torch_to_nnef.utils import SemanticVersion

TRACT_PATH = os.environ.get("TRACT_PATH", "tract")

LOGGER = logging.getLogger(__name__)


def tract_version() -> SemanticVersion:
    return SemanticVersion.from_str(
        subprocess.check_output(
            f"{TRACT_PATH} --version".split(" "),
            stderr=subprocess.STDOUT,
        )
        .decode("utf8")
        .split(" ")[1]
    )


def tract_version_lower_than(version: str) -> bool:
    """In case tract is not installed on  machine return default"""
    try:
        return tract_version() < SemanticVersion.from_str(version)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def tract_version_greater_than(version: str, inclusive: bool = False) -> bool:
    """In case tract is not installed on  machine return default"""
    try:
        if inclusive:
            return tract_version() >= SemanticVersion.from_str(version)
        return tract_version() > SemanticVersion.from_str(version)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def tract_convert_onnx_to_nnef(onnx_path, io_npz_path, nnef_path):
    return subprocess.check_output(
        (
            f"{TRACT_PATH} {onnx_path}"
            f"--nnef-tract-core --nnef-tract-pulse "
            "dump "
            f"--input-from-bundle {io_npz_path} "
            f"--nnef {nnef_path} "
        ),
        shell=True,
        stderr=subprocess.STDOUT,
    )


def tract_assert_io(
    nnef_path: Path,
    io_npz_path: Path,
    raise_exception=True,
):
    extra_param = (
        "--nnef-tract-extra "
        if tract_version_greater_than("0.20.20", inclusive=True)
        else ""
    )
    cmd = (
        f"{TRACT_PATH} {nnef_path} "
        "--nnef-tract-core --nnef-tract-pulse "
        f"{extra_param} -O "
    )
    if tract_version_lower_than("0.18.0"):
        cmd += (
            f"--input-bundle {io_npz_path} "
            # NOTE: resolution of streaming pre 0.18 not handled
            "run "
            f"--assert-output-bundle {io_npz_path}"
        )
    else:
        cmd += (
            f"--input-facts-from-bundle {io_npz_path} "
            "run "
            f"--input-from-bundle {io_npz_path} "
            f"--assert-output-bundle {io_npz_path}"
        )
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        _, err = proc.communicate()
        if err:
            serr = err.decode("utf8")
            if raise_exception:
                if any(_ in serr for _ in ["RUST_BACKTRACE", "ERROR"]):
                    raise IOPytorchTractNotISOError(serr)
                # NOTE: tract up to at least 0.20.7 stderr info and trace messages
                # we filter those to check if any other messages remain
                err_filtered = ""
                for serrline in serr.split("\n"):
                    if any(_ in serrline for _ in ["Ignore unknown extension"]):
                        continue

                    if all(  # NOTE: discuss with @kali about migration
                        _ in serrline
                        for _ in [
                            "tract_pulse_streaming_symbol",
                            "deprecated",
                            "WARN",
                        ]
                    ):
                        continue

                    err_filtered += f"{serrline}\n".strip()
                if len(err_filtered) > 0:
                    raise TractError(err_filtered)
                return True
            LOGGER.debug(serr)
            return False
    return True


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

    assert len(input_names) == len(
        tup_inputs
    ), f"{len(input_names)} != {len(tup_inputs)}"
    assert len(output_names) == len(test_outputs)

    if io_npz_path is not None:
        kwargs = {
            key: input_arg.detach().numpy()
            for key, input_arg in zip(input_names, tup_inputs)
        }
        kwargs.update(
            {
                key: output_arg.detach().numpy()
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
) -> T.Tuple[bool, str]:
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
                opset_version=17,
            )
        # parametrized failure exception emission
        # pylint: disable-next=broad-except
        except Exception as exp:
            if raise_export_error:
                raise OnnxExportError(exp.args) from exp
            LOGGER.warning(f"ONNX export error: {exp}")
            return False, str(exp.args)
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
            error_msg = str(exp.args[-1])
            if isinstance(exp, subprocess.CalledProcessError):
                error_msg = exp.output.decode("utf8")
            LOGGER.warning(f"tract ONNX->NNEF export error: {error_msg}")
            return False, error_msg
        return True, ""


def debug_dumper_pytorch_to_onnx_to_nnef(
    model: nn.Module,
    test_input,
    target_folder: Path,
    raise_export_error: bool = True,
) -> bool:
    assert not target_folder.exists()
    target_folder.mkdir()
    onnx_path = target_folder.parent / "model_exported_by_torch.onnx"
    nnef_path = target_folder / "onnx_converted_by_tract_model.nnef.tgz"
    io_npz_path = target_folder / "io.npz"
    sucessfull_export, error_msg = pytorch_to_onnx_to_tract_to_nnef(
        model,
        test_input,
        nnef_path,
        onnx_path=onnx_path,
        io_npz_path=io_npz_path,
        raise_export_error=raise_export_error,
    )
    if error_msg:
        with (target_folder / "tract_convert_error.log").open(
            "w", encoding="utf8"
        ) as fh:
            fh.write(error_msg)
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

            idx = 0
            free_debug_bundle_path = no_suffix_debug_bundle_path
            while True:
                try:
                    free_debug_bundle_path.mkdir(parents=True)
                    no_suffix_debug_bundle_path = free_debug_bundle_path
                    break
                except FileExistsError:
                    free_debug_bundle_path = free_debug_bundle_path.parent / (
                        no_suffix_debug_bundle_path.name + "_" + str(idx)
                    )
                    idx += 1
            no_suffix_debug_bundle_torch_to_nnef_path = (
                no_suffix_debug_bundle_path / "torch_to_nnef"
            )
            no_suffix_debug_bundle_torch_to_nnef_path.mkdir(parents=True)
            with (
                no_suffix_debug_bundle_torch_to_nnef_path / "io_iso_error.log"
            ).open("w", encoding="utf8") as fh:
                fh.write(exp.args[0])
            subprocess.check_output(
                f""
                f"cd {no_suffix_debug_bundle_torch_to_nnef_path} && "
                f"cp {nnef_file_path} {no_suffix_debug_bundle_torch_to_nnef_path}/model.nnef.tgz && "
                f"tar -xvzf {nnef_file_path} && "
                f"cp {io_npz_path} {no_suffix_debug_bundle_torch_to_nnef_path}/io.npz",
                shell=True,
            )
            dump_environment_versions(no_suffix_debug_bundle_path)

            debug_dumper_pytorch_to_onnx_to_nnef(
                model,
                test_input,
                target_folder=no_suffix_debug_bundle_path
                / "tract_onnx_converted_model",
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
