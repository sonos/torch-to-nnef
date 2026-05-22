"""Tools to manipulate tract programatically.

NOTE: interaction are done with *Nix tty system in mind, no support for Windows

"""

import enum
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import typing as T
import urllib.request
from copy import deepcopy
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
    T2NErrorDynamicShapeValue,
    T2NErrorInvalidArgument,
    T2NErrorIOPytorchTractNotISO,
    T2NErrorNotImplemented,
    T2NErrorOnnxExport,
    T2NErrorTract,
    T2NErrorTractDownload,
    T2NErrorTractOnnxToNNEF,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.model_wrapper import (
    UnfoldModelInfo,
    WrapStructIO,
    unfold_model_io,
)
from torch_to_nnef.utils import SemanticVersion, cd, dedup_list, torch_version

T2N_CHECK_IO_RAISE_EXCEPTION = "T2N_CHECK_IO_RAISE_EXCEPTION"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "svc" / "tract"
LOGGER = logging.getLogger(__name__)


class TractFeatureFlag(str, enum.Enum):
    DEFAULT = "default"
    COMPLEX = "complex"


class TractCheckTolerance(str, enum.Enum):
    """Level of tolerated difference between output values of PyTorch and tract.

    (those are defined in tract)

    """

    EXACT = "exact"
    APPROXIMATE = "approximate"
    CLOSE = "close"
    VERY = "very"
    SUPER = "super"
    ULTRA = "ultra"


class TractNNEF(InferenceTarget):
    """Tract NNEF inference target."""

    OFFICIAL_SUPPORTED_VERSIONS = [
        SemanticVersion.from_str(version)
        for version in [
            "0.22.1",
            "0.21.15",
        ]
    ]

    def with_dynamic_axes(
        self, dynamic_axes: T.Dict[str, T.Dict[int, str]]
    ) -> "TractNNEF":
        new_instance = deepcopy(self)
        new_instance.dynamic_axes = dynamic_axes
        return new_instance

    def with_check_io_tolerance(
        self, check_io_tolerance: TractCheckTolerance
    ) -> "TractNNEF":
        new_instance = deepcopy(self)
        new_instance.check_io_tolerance = check_io_tolerance
        return new_instance

    def with_specific_properties(
        self, specific_properties: T.Dict[str, str]
    ) -> "TractNNEF":
        new_instance = deepcopy(self)
        new_instance.specific_properties = specific_properties
        return new_instance

    @classmethod
    def latest_version(cls) -> "SemanticVersion":
        return cls.OFFICIAL_SUPPORTED_VERSIONS[0]

    @classmethod
    def latest(cls) -> "TractNNEF":
        return cls(cls.latest_version())

    def __init__(
        self,
        version: T.Union[str, SemanticVersion],
        feature_flags: T.Optional[T.Set[TractFeatureFlag]] = None,
        check_io: bool = True,
        dynamic_axes: T.Optional[T.Dict[str, T.Dict[int, str]]] = None,
        specific_tract_binary_path: T.Optional[Path] = None,
        check_io_tolerance: TractCheckTolerance = TractCheckTolerance.APPROXIMATE,  # noqa: E501
        specific_properties: T.Optional[T.Dict[str, str]] = None,
        dump_identity_properties: bool = True,
        force_attention_inner_in_f32: bool = False,
        force_linear_accumulation_in_f32: bool = False,
        force_norm_in_f32: bool = False,
        reify_sdpa_operator: T.Optional[bool] = None,
        upsample_with_debox: bool = False,
    ):
        """Init.

        Args:
            version:
                tract version targeted for export
            feature_flags:
                set of possibly added feature flags from tract
                (for example complex numbers)
            check_io:
                check between tract cli and Pytorch original model that given
                provided input, output is similar
            dynamic_axes:
                Optional specification of dynamic dimension
                By default the exported model will have the shapes of all input
                and output tensors set to exactly match those given in args.
                To specify axes of tensors as dynamic
                (i.e. known only at runtime)
                set dynamic_axes to a dict with schema:
                    KEY (str): an input or output name. Each name must also
                        be provided in input_names or output_names.
                    VALUE (dict or list): If a dict, keys are axis indices
                        and values are axis names. If a list, each element is
                        an axis index.
            specific_tract_binary_path:
                filepath of tract cli in case of custom non released
                version of tract (for testing purpose)
            check_io_tolerance:
                TractCheckTolerance level of difference tolerance between
                original output values and those generated by tract
                (those are defined tract levels)
            specific_properties:
                custom tract_properties you wish to add inside NNEF asset
                (will be parsed by tract as metadata)
            dump_identity_properties:
                add tract_properties relative to user identity
                (host, username, OS...), helpfull for debug
            force_attention_inner_in_f32:
                    control if attention should be forced as f32 inside
                    (even if inputs are all f16), usefull for unstable networks
                    like qwen2.5
            force_linear_accumulation_in_f32:
                usefull for f16 models to ensure that output of f16.
                f16 matmul become f32 accumulators.
            force_norm_in_f32:
                ensure that all normalization layers are in f32
                whatever the original PyTorch modeling.
            reify_sdpa_operator:
                (Optional) enable the conversion of scaled_dot_product_attention
                as a tract operator (intead of a NNEF fragment), default false
                until tract v0.22.0 included then true, except if specified.
                Experimental feature.
            upsample_with_debox:
                use debox upsample operator instead of deconvolution.
                This should be faster.
                (if tract version support it).
                Experimental feature.
        """
        super().__init__(version, check_io)
        if (
            check_io_tolerance != TractCheckTolerance.APPROXIMATE
            and self.version < "0.21.7"
        ):
            LOGGER.warning(
                "check_io_tolerance='%s' can NOT be applied "
                "on tract version prior 0.21.7 (please use newer version)",
                check_io_tolerance,
            )
        if (
            check_io_tolerance
            in [TractCheckTolerance.VERY, TractCheckTolerance.ULTRA]
            and self.version == "0.21.7"
        ):
            LOGGER.warning(
                "tract version 0.21.7 have not check_io_tolerance='%s' "
                "falling-back to 'super' (use newer version to solve this)",
                check_io_tolerance,
            )
            check_io_tolerance = TractCheckTolerance.SUPER

        self.feature_flags = feature_flags or set()
        self.dynamic_axes = dynamic_axes or {}
        self.check_io_tolerance = check_io_tolerance
        self.specific_properties = specific_properties
        self.force_attention_inner_in_f32 = force_attention_inner_in_f32
        self.force_linear_accumulation_in_f32 = force_linear_accumulation_in_f32
        self.force_norm_in_f32 = force_norm_in_f32
        if reify_sdpa_operator is None:
            reify_sdpa_operator = self.version > "0.22.0"
        self.reify_sdpa_operator = reify_sdpa_operator
        self.upsample_with_debox = upsample_with_debox
        self.dump_identity_properties = dump_identity_properties
        if self.feature_flags:
            LOGGER.info("use tract features flags: %s", self.feature_flags)

        if specific_tract_binary_path is None:
            if self.feature_flags:
                raise T2NErrorNotImplemented(
                    "feature_flags need specific_tract_binary_path provided"
                )
            tract_cli = TractCli.download(self.version)
            # we can not check easily feature flags compat so it's left
        else:
            tract_cli = TractCli(specific_tract_binary_path)
        LOGGER.info("use tract: %s", tract_cli.tract_path.absolute())
        self.tract_cli = tract_cli
        assert tract_cli.version == self.version

    def specific_fragments(self, model: nn.Module) -> T.Dict[str, str]:
        """Optional custom fragments to pass."""
        # pylint: disable-next=import-outside-toplevel
        from torch_to_nnef import __version__

        items = {
            "tract_target_version": self.version.to_str(),
            "torch_to_nnef_version": __version__,
            "torch_version": torch_version().to_str(),
        }

        try:
            # pylint: disable-next=import-outside-toplevel
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

        if isinstance(model, WrapStructIO):
            model = model.model
        items["exported_py_class"] = model.__class__.__name__
        if sys.argv:
            items["export_cmd"] = " ".join(sys.argv)
        if self.specific_properties is not None:
            items.update(self.specific_properties)

        def fmt(obj):
            """Minimal safety fmt."""
            return (
                str(obj)
                .replace("\n", " ")
                .replace('"', "'")
                .replace("\\", " ")
                .strip()
            )

        properties = ",\n".join(
            [f'    ("{fmt(k)}", "{fmt(v)}")' for k, v in items.items()]
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
        """Check dynamic_axes are correctly formated."""
        _validate_dynamic_axes(
            self.dynamic_axes, model, input_names, output_names
        )

    def post_trace(self, nnef_graph, active_custom_extensions):
        """Add dynamic axes in the NNEF graph."""
        if self.dynamic_axes is not None:
            custom_extensions = apply_dynamic_shape_in_nnef(
                self.dynamic_axes, nnef_graph, self.version
            )
            active_custom_extensions += custom_extensions

    def post_export(
        self,
        model_info: UnfoldModelInfo,
        nnef_graph: NGraph,
        exported_filepath: Path,
        debug_bundle_path: T.Optional[Path] = None,
    ):
        """Perform check io and build debug bundle if fail."""
        if self.check_io:
            # CHECK input and output are different
            input_names = [str(t.name) for t in nnef_graph.inputs]
            output_names = [str(t.name) for t in nnef_graph.outputs]
            del nnef_graph
            _output_names = set(output_names)
            _input_names = set(input_names)
            if len(_output_names.difference(_input_names)) == 0:
                raise T2NErrorTract(
                    "Tract does not support input passed as output without "
                    "any transform: "
                    f"outputs={_output_names} inputs={_input_names}"
                )
            with tempfile.TemporaryDirectory() as tmpdir:
                input_bundle = Path(tmpdir) / "inputs.npz"
                output_bundle = Path(tmpdir) / "outputs.npz"
                model_info.write_input_npz(input_bundle, tract_compat=True)
                model_info.write_output_npz(output_bundle, tract_compat=True)
                if debug_bundle_path is None:
                    assert_io(
                        nnef_file_path=exported_filepath,
                        tract_cli=self.tract_cli,
                        input_bundle_path=input_bundle,
                        output_bundle_path=output_bundle,
                        check_tolerance=self.check_io_tolerance,
                    )
                else:
                    assert_io_and_debug_bundle(
                        model_info,
                        exported_filepath,
                        debug_bundle_path=debug_bundle_path,
                        tract_cli=self.tract_cli,
                        input_bundle_path=input_bundle,
                        output_bundle_path=output_bundle,
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
                    if len(axis_name) != 1 and tract_version < "0.19.0":
                        raise T2NErrorDynamicShapeValue(
                            "axis_name in dynamic_axes must "
                            "be of length 1 to follow tract convention "
                            f"but was given '{axis_name}' "
                            f"in dynamic_axes={dynamic_axes}"
                        )
                    shape = external_op.attribs["shape"]
                    if len(shape) - 1 < abs(axis):
                        raise T2NErrorDynamicShapeValue(
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
                        custom_extensions.append("tract_pulse_streaming_symbol")
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
            LOGGER.warning(
                "dynamic_axes references input '%s' which was pruned "
                "during tracing (not in graph inputs: %s) — skipping",
                node_name,
                nnef_graph.inputs,
            )
            continue

    LOGGER.debug("applied dynamic axes in NNEF")
    return dedup_list(custom_extensions)


def log_io_check_call_err(cmd_shell: str, serr: str):
    LOGGER.error("check_io call: %s", cmd_shell)
    for errline in tract_err_filter(serr).split("\n"):
        if errline.strip():
            LOGGER.error("> %s", errline)


class TractCli:
    """tract calls from CLI.

    Why not use python package provided since few release of tract ?

    - we do not want to be coupled with a python lib as we declare
      version requested in API
      because this would lead to the need for an auto package
      download/import then rollback
      (since original environement may use another version)

    """

    def __init__(self, tract_path: Path):
        self.tract_path = tract_path
        assert self.tract_path.exists()

    @classmethod
    def download(cls, version: SemanticVersion) -> "TractCli":
        """Download tract requested version in cache directory."""
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

    def convert_onnx_to_nnef(self, onnx_path, input_bundle_path, nnef_path):
        return subprocess.check_output(
            [
                self.tract_path,
                str(onnx_path),
                "--nnef-tract-core",
                "--nnef-tract-pulse",
                "dump",
                "--input-from-bundle",
                str(input_bundle_path),
                "--nnef",
                str(nnef_path),
            ],
            stderr=subprocess.STDOUT,
        )

    def run(self, args, quiet=False):
        cmd_ = [
            self.tract_path,
        ] + args
        kwargs = {}
        if quiet:
            # pylint: disable-next=consider-using-with
            with open(os.devnull, "wb") as fh:
                kwargs["stdout"] = fh
                return subprocess.call(cmd_, **kwargs)
        return subprocess.call(cmd_, **kwargs)

    def _run_cmd_prefix(self, nnef_path: Path) -> T.List:
        """Common prefix shared by `assert_io_cmd_str` and `run_io_cmd_str`.

        Stops just before the ``run`` subcommand and the input/output flags.
        """
        extra_param = []
        if self.version >= "0.20.20":
            extra_param.append("--nnef-tract-extra")
        if self.version >= "0.22.0":
            extra_param.append("--nnef-tract-transformers")
        return (
            [
                self.tract_path,
                nnef_path,
                "--nnef-tract-core",
                "--nnef-tract-pulse",
            ]
            + extra_param
            + ["-O"]
        )

    def assert_io_cmd_str(
        self,
        nnef_path: Path,
        input_bundle_path: Path,
        output_bundle_path: Path,
        check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
    ):
        """Assert a NNEF asset has outputs within tolerance bound with tract."""
        cmd_ = self._run_cmd_prefix(nnef_path)
        if self.version < "0.18.0":
            cmd_ += [
                "--input-bundle",
                input_bundle_path,
                # NOTE: resolution of streaming pre 0.18 not handled
                "run",
                "--assert-output-bundle",
                output_bundle_path,
            ]
        else:
            cmd_ += [
                "run",
                "--input-from-bundle",
                input_bundle_path,
                "--assert-output-bundle",
                output_bundle_path,
            ]
        cmd_ += ["--allow-float-casts"]
        if self.version >= "0.21.7":
            cmd_ += ["--approx", check_tolerance.value]
        return [str(c) for c in cmd_]

    def run_save_outputs_cmd_str(
        self,
        nnef_path: Path,
        input_bundle_path: Path,
        output_bundle_path: Path,
    ) -> T.List[str]:
        """Run a NNEF asset and save tract's outputs to an NPZ bundle.

        Returns the command line as a list of strings, suitable for
        `subprocess.run`. Used by the proptest comparator to do the
        Python-side, NaN-aware comparison that tract's strict
        `--assert-output-bundle` cannot express.
        """
        cmd_ = self._run_cmd_prefix(nnef_path) + [
            "run",
            "--input-from-bundle",
            input_bundle_path,
            "--save-outputs-npz",
            output_bundle_path,
            "--allow-float-casts",
        ]
        return [str(c) for c in cmd_]

    def assert_io(
        self,
        nnef_path: Path,
        input_bundle_path: Path,
        output_bundle_path: Path,
        raise_exception=True,
        check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
    ):
        cmd = self.assert_io_cmd_str(
            nnef_path=nnef_path,
            input_bundle_path=input_bundle_path,
            output_bundle_path=output_bundle_path,
            check_tolerance=check_tolerance,
        )
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
                        raise T2NErrorIOPytorchTractNotISO(serr)
                    # NOTE: tract up to at least 0.20.7 stderr info
                    # and trace messages
                    # we filter those to check if any other messages remain
                    err_filtered = tract_err_filter(serr)
                    if len(err_filtered) > 0:
                        raise T2NErrorTract(cmd_shell, err_filtered)
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


def _extract_tar_archive(archive_path: Path) -> None:
    """Extract a tar archive, detecting gzip by suffix.

    Uses `tar -xf` for plain `.tar` and `tar -xzf` for `.tgz`/`.tar.gz`.
    """
    path_str = str(archive_path)
    gz = path_str.endswith((".tgz", ".tar.gz"))
    cmd = ["tar", "-xzf" if gz else "-xf", path_str]
    subprocess.check_output(cmd)


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
        """Current OS architecture name needed to download tract cli asset."""
        machine = platform.machine()
        if sys.platform in ["linux", "linux2"]:
            # linux ARM
            if machine == "x86_64":
                return "x86_64-unknown-linux-musl"
            if machine in ["arm64", "aarch64"]:
                return "aarch64-unknown-linux-musl"

            raise T2NErrorNotImplemented(
                f"No binary prebuild for machine: {machine}"
            )
            # missing: tract-armv7-unknown-linux-musleabihf-0.20.5.tgz ?
        if sys.platform == "darwin":
            # OS X
            if machine == "x86_64":
                return "x86_64-apple-darwin"
            if machine in ["arm64", "aarch64"]:
                return "aarch64-apple-darwin"
            raise T2NErrorNotImplemented(
                f"No binary prebuild for machine: {machine}"
            )
        if sys.platform == "win32":
            # Windows...
            raise T2NErrorNotImplemented("No binary prebuild for Windows OS")

        raise T2NErrorNotImplemented(f"No binary prebuild for {sys.platform}")

    @property
    def archive_name(self):
        return f"tract-{self.arch}-{self.version}"

    def _binary_url(self, tag: str):
        return (
            "https://github.com/sonos/tract/releases/download/"
            f"{tag}/{self.archive_name}.tgz"
        )

    @property
    def binary_url(self):
        return self._binary_url(str(self.version))

    @property
    def tract_filepath(self) -> Path:
        return self.extract_dir / "tract"

    def dl_tract(self):
        """Download tract requested version in cache directory."""
        self.extract_dir.mkdir(exist_ok=True)
        with cd(self.extract_dir):
            archive_path = self.extract_dir / self.archive_name
            archive_gz_path = archive_path.with_suffix(".tgz")
            # Try without then with "v" prefix -- tract tags are inconsistent.
            url = self.binary_url
            try:
                urllib.request.urlretrieve(url, archive_gz_path)
            except urllib.error.HTTPError:
                url = self._binary_url(f"v{self.version}")
                try:
                    urllib.request.urlretrieve(url, archive_gz_path)
                except urllib.error.HTTPError as exc:
                    raise T2NErrorTractDownload(
                        f"Error downloading tract at URL {self.binary_url}"
                        f" (also tried {url})"
                    ) from exc
            # Tract binary release is always a gzipped tarball.
            subprocess.check_output(["tar", "-xzf", str(archive_gz_path)])
            shutil.move(archive_path / "tract", self.extract_dir)
            shutil.rmtree(archive_path)
            archive_gz_path.unlink()


def build_io(
    model,
    test_input,
    input_bundle_path=None,
    output_bundle_path=None,
    input_names=None,
    output_names=None,
):
    if isinstance(test_input, torch.Tensor):
        test_input = (test_input,)
    with (
        select_model_mode_for_export(model, TrainingMode.EVAL),
        torch.no_grad(),
        torch.inference_mode(),
    ):
        try:
            test_outputs = model(*test_input)
        except (RuntimeError, ValueError, TypeError, AttributeError) as exp:
            # Map eager failures stemming from torch.library custom ops to a
            # T2NErrorInvalidArgument so tests validate shape errors uniformly.
            tb = exp.__traceback__
            saw_custom_op = False
            while tb is not None:
                fname = tb.tb_frame.f_code.co_filename
                if ("torch/_library/custom_ops.py" in fname) or (
                    "torch/_ops.py" in fname
                ):
                    saw_custom_op = True
                    break
                tb = tb.tb_next
            if saw_custom_op:
                raise T2NErrorInvalidArgument(str(exp)) from exp
            # Otherwise, preserve the original exception type (e.g., f16
            # LayerNorm not implemented) to match test expectations.
            raise
    model_info = unfold_model_io(
        model, test_input, test_outputs, input_names, output_names
    )

    model_info.validate()

    # Prefer separate input/output bundles
    if input_bundle_path is not None:
        model_info.write_input_npz(
            filepath=input_bundle_path, tract_compat=True
        )
    if output_bundle_path is not None:
        model_info.write_output_npz(
            filepath=output_bundle_path, tract_compat=True
        )
    return model_info.input_names, model_info.output_names


def pytorch_to_onnx_to_tract_to_nnef(
    model_info,
    nnef_path,
    tract_cli: TractCli,
    onnx_path=None,
    input_bundle_path=None,
    raise_export_error: bool = True,
) -> T.Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = onnx_path or (Path(tmpdir) / "model.onnx")
        input_bundle_path = input_bundle_path or (Path(tmpdir) / "inputs.npz")
        try:
            torch.onnx.export(
                model_info.model,
                model_info.flat_inputs,
                str(onnx_path),
                input_names=model_info.input_names,
                output_names=model_info.output_names,
                opset_version=17,
            )
        # parametrized failure exception emission
        except (RuntimeError, ValueError, TypeError) as exp:
            if raise_export_error:
                raise T2NErrorOnnxExport(exp.args) from exp
            LOGGER.warning("ONNX export error: %s", exp)
            return False, str(exp.args)
        try:
            tract_cli.convert_onnx_to_nnef(
                onnx_path,
                input_bundle_path,
                nnef_path=nnef_path,
            )
        # parametrized failure exception emission
        except (
            subprocess.CalledProcessError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exp:
            if raise_export_error:
                raise T2NErrorTractOnnxToNNEF(exp.args) from exp
            error_msg = str(exp.args[-1])
            if isinstance(exp, subprocess.CalledProcessError):
                error_msg = exp.output.decode("utf8")
            LOGGER.warning("tract ONNX->NNEF export error: %s", error_msg)
            return False, error_msg
        return True, ""


def debug_dumper_pytorch_to_onnx_to_nnef(
    model_info: UnfoldModelInfo,
    target_folder: Path,
    input_bundle_path: Path,
    tract_cli: TractCli,
    raise_export_error: bool = True,
) -> bool:
    """Try to export the model with ONNX and convert the ONNX to NNEF via tract.

    Used in debug bundle build
    (if it works, that's give a valuable reference, to debug T2N)
    """
    assert not target_folder.exists()
    target_folder.mkdir()
    onnx_path = target_folder.parent / "model_exported_by_torch.onnx"
    nnef_path = target_folder / "onnx_converted_by_tract_model.nnef.tgz"
    sucessfull_export, error_msg = pytorch_to_onnx_to_tract_to_nnef(
        model_info,
        nnef_path,
        onnx_path=onnx_path,
        input_bundle_path=input_bundle_path,
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
        with wpath.open("rb") as fh, owpath.open("rb") as fh_o:
            arr = nnef.read_tensor(fh)
            oarr = nnef.read_tensor(fh_o)
            assert np.allclose(arr, oarr), f"{wpath} vs {owpath}"


def assert_io(
    nnef_file_path: Path,
    tract_cli: TractCli,
    input_bundle_path: Path,
    output_bundle_path: Path,
    check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
):
    """Simple assertion without debug bundle.

    With addition of gc of model once output is generated.

    """
    assert nnef_file_path.exists(), nnef_file_path
    assert input_bundle_path.exists()
    assert output_bundle_path.exists()
    LOGGER.info("Start checking IO is ISO between tract and PyTorch")
    raise_exception = bool(int(os.environ.get(T2N_CHECK_IO_RAISE_EXCEPTION, 1)))
    if tract_cli.assert_io(
        nnef_file_path,
        input_bundle_path,
        output_bundle_path,
        raise_exception=raise_exception,
        check_tolerance=check_tolerance,
    ):
        LOGGER.info(
            "IO bit match between tract and PyTorch for %s", nnef_file_path
        )


def assert_io_and_debug_bundle(
    model_info: UnfoldModelInfo,
    nnef_file_path: Path,
    tract_cli: TractCli,
    input_bundle_path: Path,
    output_bundle_path: Path,
    debug_bundle_path: T.Optional[Path] = None,
    check_tolerance: TractCheckTolerance = TractCheckTolerance.EXACT,
):
    """Core check to ensure tract give same output as PyTorch within bounds."""
    assert nnef_file_path.exists(), nnef_file_path
    assert input_bundle_path.exists()
    assert output_bundle_path.exists()
    try:
        LOGGER.info("Start checking IO is ISO between tract and PyTorch")
        raise_exception = bool(
            int(os.environ.get(T2N_CHECK_IO_RAISE_EXCEPTION, 1))
        )
        tract_cli.assert_io(
            nnef_file_path,
            input_bundle_path,
            output_bundle_path,
            raise_exception=raise_exception,
            check_tolerance=check_tolerance,
        )
        LOGGER.info(
            "IO bit match between tract and PyTorch for %s", nnef_file_path
        )
    except (T2NErrorIOPytorchTractNotISO, T2NErrorTract) as exp:
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
            # Use a filename that matches the original archive type
            is_gz = str(nnef_file_path).endswith((".tgz", ".tar.gz"))
            model_archive_name = "model.nnef.tgz" if is_gz else "model.nnef.tar"
            shutil.copy(
                nnef_file_path,
                no_suffix_debug_bundle_torch_to_nnef_path / model_archive_name,
            )
            _extract_tar_archive(nnef_file_path)
            shutil.copy(
                input_bundle_path,
                no_suffix_debug_bundle_torch_to_nnef_path / "inputs.npz",
            )
            shutil.copy(
                output_bundle_path,
                no_suffix_debug_bundle_torch_to_nnef_path / "outputs.npz",
            )
        dump_environment_versions(
            no_suffix_debug_bundle_path, tract_cli.tract_path
        )

        debug_dumper_pytorch_to_onnx_to_nnef(
            model_info,
            target_folder=no_suffix_debug_bundle_path
            / "tract_onnx_converted_model",
            input_bundle_path=input_bundle_path,
            raise_export_error=False,
            tract_cli=tract_cli,
        )
        run_sh_path = no_suffix_debug_bundle_torch_to_nnef_path / "run.sh"
        with run_sh_path.open("w") as fh:
            cmd = tract_cli.assert_io_cmd_str(
                nnef_path=Path(
                    "./model.nnef.tgz" if is_gz else "./model.nnef.tar"
                ),
                input_bundle_path=Path("./inputs.npz"),
                output_bundle_path=Path("./outputs.npz"),
                check_tolerance=check_tolerance,
            )
            fh.write("${1:-%s} " % cmd[0])
            fh.write(" ".join(cmd[1:]))
        subprocess.check_call(["chmod", "+x", run_sh_path])
        if any(
            extension in debug_bundle_path.suffix
            for extension in ["tgz", "tar.gz"]
        ):
            with cd(no_suffix_debug_bundle_path.parent):
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
        LOGGER.info("debug bundle built at %s", debug_bundle_path)

        exp.args = tuple(
            [f"test with model: {model_info.model}\n" + exp.args[0]]
            + list(exp.args)[1:]
        )
        raise exp
