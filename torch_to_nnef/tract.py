"""Tools to manipulate tract programatically


NOTE: interaction are done with *Nix tty system in mind, no support for Window

"""

import logging
import platform
import subprocess
import sys
import tempfile
import typing as T
from functools import cached_property
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

DEFAULT_CACHE_DIR = Path.home() / ".tract"

LOGGER = logging.getLogger(__name__)


class TractCli:
    """tract calls from CLI

    Why not use python package provided since few release of tract ?

    - we do not want to be coupled with a python lib as we declare
      version requested in API
      because this would lead to the need for an auto package download/import then rollback
      (since original environement may use another version)

    """

    def __init__(self, tract_path: Path):
        self.tract_path = tract_path
        assert self.tract_path.exists()

    @classmethod
    def download(cls, version: SemanticVersion) -> "TractCli":
        return cls(TractBinaryDownloader(version).tract_filepath)

    @cached_property
    def version(self) -> SemanticVersion:
        return SemanticVersion.from_str(
            subprocess.check_output(
                f"{self.tract_path} --version".split(" "),
                stderr=subprocess.STDOUT,
            )
            .decode("utf8")
            .split(" ")[1]
        )

    def convert_onnx_to_nnef(self, onnx_path, io_npz_path, nnef_path):
        return subprocess.check_output(
            (
                f"{self.tract_path} {onnx_path} "
                f"--nnef-tract-core --nnef-tract-pulse "
                "dump "
                f"--input-from-bundle {io_npz_path} "
                f"--nnef {nnef_path} "
            ),
            shell=True,
            stderr=subprocess.STDOUT,
        )

    def assert_io(
        self,
        nnef_path: Path,
        io_npz_path: Path,
        raise_exception=True,
    ):
        extra_param = "--nnef-tract-extra " if "0.20.20" <= self.version else ""
        cmd = (
            f"{self.tract_path} {nnef_path} "
            "--nnef-tract-core --nnef-tract-pulse "
            f"{extra_param} -O "
        )
        if self.version < "0.18.0":
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
                        print(cmd)
                        raise IOPytorchTractNotISOError(serr)
                    # NOTE: tract up to at least 0.20.7 stderr info and trace messages
                    # we filter those to check if any other messages remain
                    err_filtered = ""
                    for serrline in serr.split("\n"):
                        if any(
                            _ in serrline for _ in ["Ignore unknown extension"]
                        ):
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

                        if all(  # NOTE: discuss with @kali about migration
                            _ in serrline
                            for _ in [
                                "Flattening the shape will be deprecated.",
                                "Reshape",
                                "WARN",
                            ]
                        ):
                            continue

                        err_filtered += f"{serrline}\n".strip()
                    if len(err_filtered) > 0:
                        raise TractError(cmd, err_filtered)
                    return True
                LOGGER.debug(serr)
                return False
        return True


class TractBinaryDownloader:
    """Tract Downloader.

    NOTE: Current version assume you are using hardware officialy supported by
    tract with pre-built binaries.
    """

    def __init__(self, version: SemanticVersion, auto_download: bool = True):
        self.version = version.to_str()
        DEFAULT_CACHE_DIR.mkdir(exist_ok=True)
        self.extract_dir = DEFAULT_CACHE_DIR / self.version
        if not self.tract_filepath.exists() and auto_download:
            self.dl_tract()

    @property
    def arch(self):
        machine = platform.machine()
        if sys.platform in ["linux", "linux2"]:
            # linux ARM
            if machine == "x86_64":
                return "x86_64-unknown-linux-musl"
            if machine == "aarch64":
                return "tract-aarch64-unknown-linux-musl"
            raise NotImplementedError(
                f"No binary prebuild for machine: {machine}"
            )
            # missing: tract-armv7-unknown-linux-musleabihf-0.20.5.tgz ?
        if sys.platform == "darwin":
            # OS X
            if machine == "x86_64":
                return "tract-x86_64-apple-darwin"
            if machine == "aarch64":
                return "aarch64-apple-darwin"
            raise NotImplementedError(
                f"No binary prebuild for machine: {machine}"
            )
        if sys.platform == "win32":
            # Windows...
            raise NotImplementedError("No binary prebuild for Windows OS")
        raise NotImplementedError(f"No binary prebuild for {sys.platform}")

    @property
    def archive_name(self):
        return f"tract-{self.arch}-{self.version}"

    @property
    def binary_url(self):
        return f"https://github.com/sonos/tract/releases/download/{self.version}/{self.archive_name}.tgz"

    @property
    def tract_filepath(self) -> Path:
        return self.extract_dir / "tract"

    def dl_tract(self):
        self.extract_dir.mkdir()
        subprocess.check_output(
            f"""
        cd {self.extract_dir} && \
        wget --quiet "{self.binary_url}" && \
        tar -xvzf {self.archive_name}.tgz && \
        rm {self.archive_name}.tgz && \
        mv {self.archive_name}/tract ./
        """,
            shell=True,
        )


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
    tract_cli: TractCli,
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
            tract_cli.convert_onnx_to_nnef(
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
    tract_cli: TractCli,
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
        tract_cli=tract_cli,
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
    tract_cli: TractCli,
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
            assert nnef_file_path.exists(), nnef_file_path
            assert io_npz_path.exists()
            LOGGER.info("Start checking IO is ISO between tract and Pytorch")
            tract_cli.assert_io(
                nnef_file_path,
                io_npz_path,
                raise_exception=True,
            )
            LOGGER.info(
                f"IO bit match between tract and Pytorch for {nnef_file_path}"
            )
        except (IOPytorchTractNotISOError, TractError) as exp:
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
                tract_cli=tract_cli,
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
