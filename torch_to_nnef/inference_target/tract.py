"""Tools to manipulate tract programatically


NOTE: interaction are done with *Nix tty system in mind, no support for Window

"""

import enum
import gc
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import typing as T
import urllib.request
from datetime import datetime
from functools import cached_property
from pathlib import Path

import nnef
import numpy as np
import torch
from nnef_tools.model import Graph as NGraph
from torch import nn
from torch.onnx import TrainingMode  # type: ignore
from torch.onnx.utils import (
    _validate_dynamic_axes,  # type: ignore
    select_model_mode_for_export,  # type: ignore
)

from torch_to_nnef.collect_env import (
    dump_environment_versions,
    get_hostname,
    get_uname,
    get_user,
    python_version,
)
from torch_to_nnef.exceptions import (
    DynamicShapeValue,
    IOPytorchTractNotISOError,
    OnnxExportError,
    TorchToNNEFNotImplementedError,
    TractError,
    TractOnnxToNNEFError,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.utils import SemanticVersion, cd, dedup_list, torch_version

T2N_CHECK_IO_RAISE_EXCEPTION = "T2N_CHECK_IO_RAISE_EXCEPTION"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "svc" / "tract"
LOGGER = logging.getLogger(__name__)


class TractFeatureFlag(str, enum.Enum):
    DEFAULT = "default"
    COMPLEX = "complex"


class TractCheckTolerance(str, enum.Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    CLOSE = "close"
    SUPER = "super"


class TractNNEF(InferenceTarget):
    OFFICIAL_SUPPORTED_VERSIONS = [
        SemanticVersion.from_str(version)
        for version in [
            "0.21.7",  # 0.21.8 introduce regression in some kernels for now
            "0.20.22",
        ]
    ]

    @classmethod
    def latest(cls):
        return cls(cls.OFFICIAL_SUPPORTED_VERSIONS[0])

    def __init__(
        self,
        version: T.Union[str, SemanticVersion],
        feature_flags: T.Optional[T.Set[TractFeatureFlag]] = None,
        check_io: bool = True,
        dynamic_axes: T.Optional[T.Dict[str, T.Dict[int, str]]] = None,
        specific_tract_binary_path: T.Optional[Path] = None,
        check_io_tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE,
        specific_properties: T.Optional[T.Dict[str, str]] = None,
        dump_identity_properties: bool = True,
        force_attention_inner_in_f32: bool = False,
        force_linear_accumulation_in_f32: bool = False,
        force_norm_in_f32: bool = False,
    ):
        super().__init__(version, check_io)
        self.feature_flags = feature_flags or set()
        self.dynamic_axes = dynamic_axes or {}
        self.check_io_tolerance = check_io_tolerance
        self.specific_properties = specific_properties
        self.force_attention_inner_in_f32 = force_attention_inner_in_f32
        self.force_linear_accumulation_in_f32 = force_linear_accumulation_in_f32
        self.force_norm_in_f32 = force_norm_in_f32
        self.dump_identity_properties = dump_identity_properties
        if self.feature_flags:
            LOGGER.info(f"use tract features flags: {self.feature_flags}")

        if specific_tract_binary_path is None:
            if self.feature_flags:
                raise TorchToNNEFNotImplementedError(
                    "feature_flags need specific_tract_binary_path provided"
                )
            tract_cli = TractCli.download(self.version)
            # we can not check easily feature flags compat so it's left
        else:
            tract_cli = TractCli(specific_tract_binary_path)
        LOGGER.info(f"use tract:{tract_cli.tract_path.absolute()}")
        self.tract_cli = tract_cli
        assert tract_cli.version == self.version

    def specific_fragments(self, model: nn.Module) -> T.Dict[str, str]:
        """Optional custom fragments to pass"""
        from torch_to_nnef import __version__

        items = {
            "tract_target_version": self.version.to_str(),
            "torch_to_nnef_version": __version__,
            "torch_version": torch_version().to_str(),
        }

        try:
            import transformers

            items["transformers_version"] = transformers.__version__
        except ImportError:
            pass

        if self.dump_identity_properties:
            items["os"] = get_uname()
            items["hostname"] = get_hostname()
            items["user"] = get_user()

        items["py_version"] = python_version()
        items["export_date"] = str(datetime.now())

        from torch_to_nnef.model_wrapper import WrapStructIO

        if isinstance(model, WrapStructIO):
            model = model.model
        items["exported_py_class"] = model.__class__.__name__
        if sys.argv:
            items["export_cmd"] = " ".join(sys.argv)
        if self.specific_properties is not None:
            items.update(self.specific_properties)

        properties = ",\n".join(
            [f'    ("{k}", "{v}")' for k, v in items.items()]
        )
        return {
            "tract_core_properties": (
                "fragment tract_core_properties(\n"
                ") -> (properties: (string, tensor<scalar>)[])\n"
                "{\n"
                f"  properties = [\n{properties}\n  ];\n"
                "}\n\n"
            )
        }

    @property
    def has_dynamic_axes(self) -> bool:
        return bool(self.dynamic_axes)

    def pre_trace(
        self,
        model: nn.Module,
        input_names: T.Optional[T.List[str]],
        output_names: T.Optional[T.List[str]],
    ):
        _validate_dynamic_axes(
            self.dynamic_axes, model, input_names, output_names
        )

    def post_trace(self, nnef_graph, active_custom_extensions):
        if self.dynamic_axes is not None:
            custom_extensions = apply_dynamic_shape_in_nnef(
                self.dynamic_axes, nnef_graph, self.version
            )
            active_custom_extensions += custom_extensions

    def post_export(
        self,
        model: nn.Module,
        nnef_graph: NGraph,
        args: T.List[T.Any],
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        if self.check_io:
            # CHECK input and output are different
            input_names = [str(t.name) for t in nnef_graph.inputs]
            output_names = [str(t.name) for t in nnef_graph.outputs]
            _output_names = set(output_names)
            _input_names = set(input_names)
            if len(_output_names.difference(_input_names)) == 0:
                raise TractError(
                    "Tract does not support input passed as output without any transform: "
                    f"outputs={_output_names} inputs={_input_names}"
                )
            if debug_bundle_path is None:
                assert_io(
                    model,
                    args,
                    exported_filepath,
                    input_names=input_names,
                    output_names=output_names,
                    tract_cli=self.tract_cli,
                    check_tolerance=self.check_io_tolerance,
                )
            else:
                assert_io_and_debug_bundle(
                    model,
                    args,
                    exported_filepath,
                    debug_bundle_path=debug_bundle_path,
                    input_names=input_names,
                    output_names=output_names,
                    tract_cli=self.tract_cli,
                    check_tolerance=self.check_io_tolerance,
                )


def apply_dynamic_shape_in_nnef(dynamic_axes, nnef_graph, tract_version):
    custom_extensions = []
    for node_name, named_dims in dynamic_axes.items():
        found_name = False
        for inp_tensor in nnef_graph.inputs:
            if inp_tensor.name == node_name:
                found_name = True
                # LOGGER.debug()
                assert len(inp_tensor.producers) == 1
                external_op = inp_tensor.producers[0]
                assert external_op.type in [
                    "external",
                    "tract_core_external",
                ], external_op.type
                for axis, axis_name in named_dims.items():
                    if len(axis_name) != 1:
                        raise DynamicShapeValue(
                            "axis_name in dynamic_axes must "
                            "be of length 1 to follow tract convention "
                            f"but was given '{axis_name}' "
                            f"in dynamic_axes={dynamic_axes}"
                        )
                    shape = external_op.attribs["shape"]
                    if len(shape) - 1 < abs(axis):
                        raise DynamicShapeValue(
                            f"axis of '{node_name}' in dynamic_axes "
                            f"must be within rank size: {len(shape)} but "
                            f"provided {axis}."
                        )

                    if axis < 0:  # set as positive axis for comparison
                        axis = len(shape) - axis

                    external_op.attribs["shape"] = [
                        (
                            nnef.Identifier(str(axis_name))
                            if idx == axis
                            else dim_size
                        )
                        for idx, dim_size in enumerate(shape)
                    ]
                    if tract_version < "0.18.2":
                        custom_extensions.append(
                            "tract_pulse_streaming_symbol")
                    else:
                        custom_extensions.append(f"tract_symbol {axis_name}")
                break
        if not found_name:
            if any(
                node_name == out_tensor.name
                for out_tensor in nnef_graph.outputs
            ):
                LOGGER.warning(
                    "useless to set output dynamic axes "
                    "since not interpreted by inference engines"
                )
            raise DynamicShapeValue(
                f"Requested dynamic_axes on input named: '{node_name}', "
                f"is not in graph inputs: {nnef_graph.inputs}"
            )

    LOGGER.debug("applied dynamic axes in NNEF")
    return dedup_list(custom_extensions)


def log_io_check_call_err(cmd_shell: str, serr: str):
    LOGGER.error(f"check_io call: {cmd_shell}")
    for errline in tract_err_filter(serr).split("\n"):
        if errline.strip():
            LOGGER.error(f"> {errline}")


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
            [
                self.tract_path,
                str(onnx_path),
                "--nnef-tract-core",
                "--nnef-tract-pulse",
                "dump",
                "--input-from-bundle",
                str(io_npz_path),
                "--nnef",
                str(nnef_path),
            ],
            stderr=subprocess.STDOUT,
        )

    def run(self, args):
        cmd_ = [
            self.tract_path,
        ] + args
        return subprocess.check_call(cmd_)

    def assert_io(
        self,
        nnef_path: Path,
        io_npz_path: Path,
        raise_exception=True,
        check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
    ):
        extra_param = (
            ["--nnef-tract-extra"] if "0.20.20" <= self.version else []
        )
        cmd_ = (
            [
                self.tract_path,
                nnef_path,
                "--nnef-tract-core",
                "--nnef-tract-pulse",
            ]
            + extra_param
            + ["-O"]
        )

        if self.version < "0.18.0":
            cmd_ += [
                "--input-bundle",
                io_npz_path,
                # NOTE: resolution of streaming pre 0.18 not handled
                "run",
                "--assert-output-bundle",
                io_npz_path,
            ]
        else:
            cmd_ += [
                "--input-facts-from-bundle",
                io_npz_path,
                "run",
                "--input-from-bundle",
                io_npz_path,
                "--assert-output-bundle",
                io_npz_path,
            ]
        cmd_ += ["--allow-float-casts"]
        if self.version >= "0.21.7":
            cmd_ += ["--approx", check_tolerance.value]
        cmd = [str(c) for c in cmd_]
        cmd_shell = " ".join(_ for _ in cmd)
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            _, err = proc.communicate()
            if err:
                serr = err.decode("utf8")
                if raise_exception:
                    if any(_ in serr for _ in ["RUST_BACKTRACE", "ERROR"]):
                        log_io_check_call_err(cmd_shell, serr)
                        raise IOPytorchTractNotISOError(serr)
                    # NOTE: tract up to at least 0.20.7 stderr info and trace messages
                    # we filter those to check if any other messages remain
                    err_filtered = tract_err_filter(serr)
                    if len(err_filtered) > 0:
                        raise TractError(cmd_shell, err_filtered)
                    return True
                log_io_check_call_err(cmd_shell, serr)
                return False
        return True


def tract_err_filter(serr: str) -> str:
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

        if all(  # NOTE: discuss with @kali about migration
            _ in serrline
            for _ in [
                "Flattening the shape will be deprecated.",
                "Reshape",
                "WARN",
            ]
        ):
            continue

        serrline = serrline.strip()
        if serrline:
            err_filtered += f"{serrline}\n"
    return err_filtered.strip()


class TractBinaryDownloader:
    """Tract Downloader.

    NOTE: Current version assume you are using hardware officialy supported by
    tract with pre-built binaries.
    """

    def __init__(self, version: SemanticVersion, auto_download: bool = True):
        self.version = version.to_str()
        DEFAULT_CACHE_DIR.mkdir(exist_ok=True, parents=True)
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
            if machine in ["arm64", "aarch64"]:
                return "aarch64-unknown-linux-musl"
            raise NotImplementedError(
                f"No binary prebuild for machine: {machine}"
            )
            # missing: tract-armv7-unknown-linux-musleabihf-0.20.5.tgz ?
        if sys.platform == "darwin":
            # OS X
            if machine == "x86_64":
                return "x86_64-apple-darwin"
            if machine in ["arm64", "aarch64"]:
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
        self.extract_dir.mkdir(exist_ok=True)
        with cd(self.extract_dir):
            archive_path = self.extract_dir / self.archive_name
            archive_gz_path = archive_path.with_suffix(".tgz")
            try:
                urllib.request.urlretrieve(self.binary_url, archive_gz_path)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(
                    f"Error downloading tract at URL {self.binary_url}"
                ) from exc
            subprocess.check_output(["tar", "-xzf", str(archive_gz_path)])
            shutil.move(archive_path / "tract", self.extract_dir)
            shutil.rmtree(archive_path)
            archive_gz_path.unlink()


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

    assert len(input_names) == len(tup_inputs), (
        f"{len(input_names)} != {len(tup_inputs)}"
    )
    assert len(output_names) == len(test_outputs)

    if io_npz_path is not None:

        def cast(val):
            if val.dtype in [torch.float16, torch.bfloat16]:
                val = val.to(torch.float32)  # tract --allow-float-casts
            val = val.detach().numpy()
            return val

        kwargs = {
            key: cast(input_arg)
            for key, input_arg in zip(input_names, tup_inputs)
        }
        kwargs.update(
            {
                key: cast(output_arg)
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
    with cd(target_folder):
        subprocess.check_output(["tar", "-xf", str(nnef_path)])
    return True


def all_close_map_weights(weight_map_file_paths: T.Dict[Path, Path]):
    for wpath, owpath in weight_map_file_paths.items():
        with wpath.open("rb") as fh:
            with owpath.open("rb") as fh_o:
                arr = nnef.read_tensor(fh)
                oarr = nnef.read_tensor(fh_o)
                assert np.allclose(arr, oarr), f"{wpath} vs {owpath}"


def assert_io(
    model: nn.Module,
    test_input,
    nnef_file_path: Path,
    tract_cli: TractCli,
    io_npz_path: T.Optional[Path] = None,
    input_names: T.Optional[T.List[str]] = None,
    output_names: T.Optional[T.List[str]] = None,
    check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
):
    """simple assertion without debug bundle.

    With addition of gc of model once output is generated.

    """
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

            del model
            del test_input
            gc.collect()

            assert nnef_file_path.exists(), nnef_file_path
            assert io_npz_path.exists()
            LOGGER.info("Start checking IO is ISO between tract and PyTorch")
            raise_exception = bool(
                int(os.environ.get(T2N_CHECK_IO_RAISE_EXCEPTION, 1))
            )
            if tract_cli.assert_io(
                nnef_file_path,
                io_npz_path,
                raise_exception=raise_exception,
                check_tolerance=check_tolerance,
            ):
                LOGGER.info(
                    f"IO bit match between tract and PyTorch for {nnef_file_path}"
                )
        except (IOPytorchTractNotISOError, TractError) as exp:
            raise exp


def assert_io_and_debug_bundle(
    model: nn.Module,
    test_input,
    nnef_file_path: Path,
    tract_cli: TractCli,
    io_npz_path: T.Optional[Path] = None,
    debug_bundle_path: T.Optional[Path] = None,
    input_names: T.Optional[T.List[str]] = None,
    output_names: T.Optional[T.List[str]] = None,
    check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
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
            LOGGER.info("Start checking IO is ISO between tract and PyTorch")
            raise_exception = bool(
                int(os.environ.get(T2N_CHECK_IO_RAISE_EXCEPTION, 1))
            )
            tract_cli.assert_io(
                nnef_file_path,
                io_npz_path,
                raise_exception=raise_exception,
                check_tolerance=check_tolerance,
            )
            LOGGER.info(
                f"IO bit match between tract and PyTorch for {nnef_file_path}"
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
            with cd(no_suffix_debug_bundle_torch_to_nnef_path):
                shutil.copy(
                    nnef_file_path,
                    no_suffix_debug_bundle_torch_to_nnef_path
                    / "model.nnef.tgz",
                )
                subprocess.check_output(["tar", "-xzf", str(nnef_file_path)])
                if io_npz_path:
                    shutil.copy(
                        io_npz_path,
                        no_suffix_debug_bundle_torch_to_nnef_path / "io.npz",
                    )
            dump_environment_versions(
                no_suffix_debug_bundle_path, tract_cli.tract_path
            )

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
                with no_suffix_debug_bundle_path.parent:
                    subprocess.check_output(
                        [
                            "tar",
                            "-cvf",
                            str(debug_bundle_path.absolute()),
                            str(no_suffix_debug_bundle_path.name),
                        ]
                    )
                # rm acceptable since dir created ensured empty before use
                shutil.rmtree(no_suffix_debug_bundle_path)
            LOGGER.info(f"debug bundle built at {debug_bundle_path}")

            exp.args = tuple(
                [f"test with model: {model}\n" + exp.args[0]]
                + list(exp.args)[1:]
            )
            raise exp
