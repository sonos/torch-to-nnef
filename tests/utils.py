"""Make training and any ops involving random reproducible."""

import inspect
import os
import random
import shutil
import tempfile
import typing as T
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch as Torch
from torch.nn.utils.weight_norm import WeightNorm

from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import (
    InferenceTarget,
    KhronosNNEF,
    TractNNEF,
)
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    build_io,
)
from torch_to_nnef.log import log
from torch_to_nnef.tensor.opaque import NEW_OPAQUE_TRACING_STRATEGY
from torch_to_nnef.tensor.quant.base import (
    QTENSOR_UNSUPPORTED,
    QTENSOR_UNSUPPORTED_MSG,
)
from torch_to_nnef.torch_graph.ir_naming import DEFAULT_VARNAME_SCHEME
from torch_to_nnef.utils import SemanticVersion, torch_version

TRACT_INFERENCES_TO_TESTS_APPROX = [
    # we maintain last 3 majors of tract
    TractNNEF(v)
    for v in TractNNEF.OFFICIAL_SUPPORTED_VERSIONS
]

# Options to easily test new tract versions
if "T2N_TEST_TRACT_PATH" in os.environ:
    _tract_cli_path = Path(os.environ["T2N_TEST_TRACT_PATH"])
    assert _tract_cli_path.exists(), _tract_cli_path
    _tract_cli = TractCli(_tract_cli_path)
    _tract_inf = TractNNEF(
        _tract_cli.version, specific_tract_binary_path=_tract_cli_path
    )
    TRACT_INFERENCES_TO_TESTS_APPROX = [_tract_inf]
elif "T2N_TEST_TRACT_VERSION" in os.environ:
    _tract_inf = TractNNEF(os.environ["T2N_TEST_TRACT_VERSION"])
    TRACT_INFERENCES_TO_TESTS_APPROX = [_tract_inf]
elif "T2N_TEST_SKIP_TRACT" in os.environ and bool(
    os.environ["T2N_TEST_SKIP_TRACT"]
):
    TRACT_INFERENCES_TO_TESTS_APPROX = []


TRACT_INFERENCES_TO_TESTS_EXACT = deepcopy(TRACT_INFERENCES_TO_TESTS_APPROX)
for _ in TRACT_INFERENCES_TO_TESTS_EXACT:
    _.check_io_tolerance = TractCheckTolerance.EXACT

INFERENCE_TARGETS_TO_TESTS = TRACT_INFERENCES_TO_TESTS_APPROX + [
    KhronosNNEF(v) for v in KhronosNNEF.OFFICIAL_SUPPORTED_VERSIONS
]


DUMP_DIRPATH = os.environ.get("DUMP_DIRPATH", "")
IS_DEBUG = bool(int(os.environ.get("DEBUG", False)))


def change_dynamic_axes(it, dynamic_axes):
    assert isinstance(it, TractNNEF)
    new_it = deepcopy(it)
    new_it.dynamic_axes = dynamic_axes
    return new_it


class TestSuiteInferenceExactnessBuilder:
    # https://docs.pytest.org/en/stable/example/pythoncollection.html#customizing-test-collection
    __test__ = False

    def __init__(self, inference_targets: T.List[InferenceTarget]):
        self.test_samples = []
        self.inference_targets = inference_targets

    def generate_test_name(self, data, module):
        data_fmt = ""
        if isinstance(data, Torch.Tensor):
            data_fmt = f"{data.dtype}{list(data.shape)}"
        else:
            for d in data:
                if hasattr(d, "dtype"):
                    data_fmt += f"{d.dtype}{list(d.shape)}, "
                else:
                    data_fmt += str(d)
        if len(str(module)) > 100:
            module = str(module.__class__.__name__) + "__" + str(module)[:100]
        test_name = f"{module}({data_fmt})"
        return test_name

    def reset(self):
        self.test_samples = []

    def add(
        self,
        inputs,
        model,
        test_name=None,
        inference_conditions=None,
        inference_modifier=None,
    ):
        test_name = test_name or self.generate_test_name(inputs, model)
        for it in self.inference_targets:
            if inference_conditions is None or inference_conditions(it):
                new_it = (
                    it if inference_modifier is None else inference_modifier(it)
                )
                self.test_samples.append(
                    (f"{it}__{test_name}", inputs, model, new_it)
                )

    @property
    def ids(self):
        return [_[0] for _ in self.test_samples]

    def __repr__(self):
        return (
            "<TestSuiteInferenceExactnessBuilder "
            f"len({len(self.test_samples)})>"
        )


def set_seed(seed=0, cudnn=False, torch=True):
    if cudnn and Torch.cuda.is_available():
        Torch.backends.cudnn.deterministic = True
        Torch.backends.cudnn.benchmark = False
    if torch:
        Torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ModelWrapper(Torch.nn.Module):
    def __init__(self, model: Torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def check_model_io_test(
    model: Torch.nn.Module,
    test_input,
    inference_target,
    input_names=None,
    output_names=None,
    nnef_variable_naming_scheme=DEFAULT_VARNAME_SCHEME,
    custom_extensions=None,
    callback_post_export=None,
    unit_test_naming=None,
    allow_same_io_names=False,
    check_io_names_qte_match=True,
):
    unittest_slug = datetime.now().strftime("%Y_%m_%d")
    if unit_test_naming:
        unittest_slug = f"{unittest_slug}_{unit_test_naming}"
    else:
        for _ in range(5):  # up to 5 deep fn call stacking
            caller_fn_name = inspect.stack()[_ + 1][3]
            if caller_fn_name.startswith("test_"):
                unittest_slug = f"{unittest_slug}_{caller_fn_name}"
                break
    dbg_base_dir = Path(DUMP_DIRPATH) if DUMP_DIRPATH else Path.cwd()
    dbg_base_dir.mkdir(exist_ok=True, parents=True)
    dbg_path = dbg_base_dir / "failed_tests" / unittest_slug

    idx = 0
    while dbg_path.exists():
        idx += 1
        dbg_path = dbg_base_dir / "failed_tests" / f"{unittest_slug}_{idx}"

    idx = 0
    dump_test_tz_path = dbg_base_dir / f"{unittest_slug}.nnef.tgz"
    while dump_test_tz_path.exists():
        idx += 1
        dump_test_tz_path = dbg_base_dir / f"{unittest_slug}_{idx}.nnef.tgz"

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        inputs_npz_path = Path(tmpdir) / "inputs.npz"
        outputs_npz_path = Path(tmpdir) / "outputs.npz"

        if not hasattr(model, "eval"):
            model = ModelWrapper(model)

        model = model.eval()

        input_names, output_names = build_io(
            model,
            test_input,
            input_bundle_path=inputs_npz_path,
            output_bundle_path=outputs_npz_path,
            input_names=input_names,
            output_names=output_names,
        )
        # For Khronos checks, export as .tgz to
        # satisfy nnef-tools decompression.
        # Otherwise, keep deterministic tar (0)
        # for other targets.
        comp_level = 1 if isinstance(inference_target, KhronosNNEF) else 0
        exported_path = export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            compression_level=comp_level,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            debug_bundle_path=(dbg_path if IS_DEBUG else None),
            inference_target=inference_target,
            nnef_variable_naming_scheme=nnef_variable_naming_scheme,
            custom_extensions=custom_extensions,
            allow_same_io_names=True,
            check_io_names_qte_match=check_io_names_qte_match,
        )
        export_path = exported_path
        if DUMP_DIRPATH:
            shutil.copy(
                export_path,
                dump_test_tz_path,
            )
            shutil.copy(
                inputs_npz_path, dump_test_tz_path.with_suffix(".inputs.npz")
            )
            shutil.copy(
                outputs_npz_path, dump_test_tz_path.with_suffix(".outputs.npz")
            )
        if callback_post_export is not None:
            callback_post_export(inference_target, export_path)


def remove_weight_norm(module):
    module_list = list(module.children())
    if len(module_list) == 0:
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
    else:
        for mod in module_list:
            remove_weight_norm(mod)


def id_tests(test_fixtures):
    test_names = []
    for data, module in test_fixtures:
        data_fmt = ""
        if isinstance(data, Torch.Tensor):
            data_fmt = f"{data.dtype}{list(data.shape)}"
        else:
            for d in data:
                if hasattr(d, "dtype"):
                    data_fmt += f"{d.dtype}{list(d.shape)}, "
                else:
                    data_fmt += str(d)
        if len(str(module)) > 100:
            module = str(module.__class__.__name__) + "__" + str(module)[:100]
        test_name = f"{module}({data_fmt})"
        test_names.append(test_name)
    return test_names


skipif_unsupported_qtensor = pytest.mark.skipif(
    QTENSOR_UNSUPPORTED, reason=QTENSOR_UNSUPPORTED_MSG
)


skipif_unsupported_tensor_updater = pytest.mark.skipif(
    condition=torch_version() < "2.0.0",
    reason="torch version need to be >= 2.0 to use ModTensorUpdater",
)


skipif_limited_offload_support = pytest.mark.skipif(
    condition=torch_version() < "1.12.0",
    reason="torch version need to be >= 1.12.0 to use OffloadedTensor",
)


# Keyed on the implementation flag itself (not a hardcoded version) so the
# gate can never drift from torch_to_nnef.tensor.opaque. Tests that assert the
# meta placeholder behaviour (weights stay opaque, meta-arithmetic errors)
# only hold under this strategy; on older torch the ref materializes eagerly.
skipif_no_meta_opaque_tracing = pytest.mark.skipif(
    condition=not NEW_OPAQUE_TRACING_STRATEGY,
    reason="opaque meta-tracing strategy requires torch >= 2.4",
)


def combine_conditions(conds: T.List[T.Callable[[InferenceTarget], bool]]):
    """Combine a list of inference target conditions callback.

    Into a callback combining all of them.

    """
    return lambda i: all(cond(i) for cond in conds)


def transformers_tract_export_test_condition(
    inference_target: InferenceTarget,
) -> bool:
    """Condition to skip tests based on transformers and tract versions.

    Some versions of transformers have ops not supported in old tract versions.
    """
    return isinstance(inference_target, TractNNEF) and (
        inference_target.version > "0.21.5"
    )


def cond_tract_gt_0_22_0(inference_target: InferenceTarget) -> bool:
    """Condition to enable tests only for tract versions > 0.22.1."""
    return isinstance(inference_target, TractNNEF) and (
        inference_target.version > "0.22.1"
    )


# tract block quantized QTensor shared checks ##############################


def tract_target(version: str) -> TractNNEF:
    """Latest tract target relabeled with `version` (prereleases allowed)."""
    target = deepcopy(TractNNEF.latest())
    target.version = SemanticVersion.from_str(version)
    return target


def tract_versions_just_below(min_version: str) -> T.List[str]:
    """Version strings just below `min_version`, to test version gates.

    The previous release (previous patch, or previous minor when the patch
    is 0) and two prereleases of `min_version` itself, which SemVer sorts
    before it. For "0.23.10": ["0.23.9", "0.23.10-pre", "0.23.10-rc.1"].
    """
    major, minor, patch = (int(v) for v in min_version.split("."))
    versions = [f"{min_version}-pre", f"{min_version}-rc.1"]
    if patch > 0:
        versions.insert(0, f"{major}.{minor}.{patch - 1}")
    elif minor > 0:
        versions.insert(0, f"{major}.{minor - 1}.0")
    return versions


def tract_versions_just_at_or_above(min_version: str) -> T.List[str]:
    """Version strings at or just above `min_version`, to test version gates.

    `min_version` itself, with build metadata (ignored by SemVer ordering),
    a prerelease of the next patch and the next patch. For "0.23.10":
    ["0.23.10", "0.23.10+build.1", "0.23.11-pre", "0.23.11"].
    """
    major, minor, patch = (int(v) for v in min_version.split("."))
    next_patch = f"{major}.{minor}.{patch + 1}"
    return [
        min_version,
        f"{min_version}+build.1",
        f"{next_patch}-pre",
        next_patch,
    ]


def check_rejects_old_tract(tmp_path, caplog, qtensor, version: str):
    """Both write paths warn with the upgrade hint and write no data."""
    qtensor_cls = type(qtensor)
    target = tract_target(version)
    match = (
        f"{qtensor_cls.FORMAT_NAME} export requires tract >= "
        + qtensor_cls.MIN_TRACT_VERSION.replace(".", r"\.")
    )
    with caplog.at_level("WARNING"):
        with pytest.raises(T2NErrorNotImplemented, match=match):
            qtensor.write_in_file(tmp_path, "weight", target)
        with pytest.raises(T2NErrorNotImplemented, match=match):
            qtensor_cls.write_offload_state_in_file(
                qtensor.to_offload_state(), tmp_path, "offloaded", target
            )
    hint = (
        f"{qtensor_cls.FORMAT_NAME} export requires tract >= "
        f"{qtensor_cls.MIN_TRACT_VERSION}; target is {target.version}. "
        f"Upgrade tract before exporting {qtensor_cls.FORMAT_NAME} weights."
    )
    assert caplog.text.count(hint) == 2, caplog.text
    assert not list(Path(tmp_path).glob("*.dat"))


def check_offload_state_write(tmp_path, qtensor, target: TractNNEF):
    """Offload state reload and direct write match the in-memory qtensor."""
    qtensor_cls = type(qtensor)
    tmp_path = Path(tmp_path)
    ref_dir = tmp_path / "ref"
    out_dir = tmp_path / "out"
    reload_dir = tmp_path / "reload"
    for dirpath in [ref_dir, out_dir, reload_dir]:
        dirpath.mkdir()
    qtensor.write_in_file(ref_dir, "weight", target)
    expected = (ref_dir / "weight.dat").read_bytes()
    state = qtensor.to_offload_state()
    assert qtensor_cls.is_offload_state(state)
    qtensor_cls.write_offload_state_in_file(
        state, out_dir, "weight", inference_target=target
    )
    assert (out_dir / "weight.dat").read_bytes() == expected
    reloaded = qtensor_cls.from_offload_state(state)
    assert type(reloaded) is qtensor_cls
    assert Torch.equal(reloaded.decompress(), qtensor.decompress())
    reloaded.write_in_file(reload_dir, "weight", target)
    assert (reload_dir / "weight.dat").read_bytes() == expected
    # same label written twice with same content is accepted
    qtensor.write_in_file(ref_dir, "weight", target)
