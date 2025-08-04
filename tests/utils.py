"""Make training and any ops involving random reproducible"""

import os
import random
import shutil
import tempfile
import typing as T
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import inspect

import pytest
import numpy as np
import torch as Torch
from torch.nn.utils.weight_norm import WeightNorm

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import (
    InferenceTarget,
    KhronosNNEF,
    TractNNEF,
)
from torch_to_nnef.tensor.quant.base import (
    QTENSOR_UNSUPPORTED,
    QTENSOR_UNSUPPORTED_MSG,
)
from torch_to_nnef.inference_target.tract import (
    TractCheckTolerance,
    TractCli,
    build_io,
)
from torch_to_nnef.log import log
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.utils import torch_version

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
        return f"<TestSuiteInferenceExactnessBuilder len({len(self.test_samples)})>"


def set_seed(seed=0, cudnn=False, torch=True):
    if cudnn and Torch.cuda.is_available():
        Torch.backends.cudnn.deterministic = True
        Torch.backends.cudnn.benchmark = False
    if torch:
        Torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_model_io_test(
    model: Torch.nn.Module,
    test_input,
    inference_target,
    input_names=None,
    output_names=None,
    nnef_variable_naming_scheme=VariableNamingScheme.default(),
    custom_extensions=None,
    callback=None,
    unit_test_naming=None,
):
    dump_dirpath = os.environ.get("DUMP_DIRPATH", "")
    is_debug = bool(int(os.environ.get("DEBUG", False)))

    unittest_slug = datetime.now().strftime("%Y_%m_%d")
    if unit_test_naming:
        unittest_slug = f"{unittest_slug}_{unit_test_naming}"
    else:
        for _ in range(5):  # up to 5 deep fn call stacking
            caller_fn_name = inspect.stack()[_ + 1][3]
            if caller_fn_name.startswith("test_"):
                unittest_slug = f"{unittest_slug}_{caller_fn_name}"
                break
    dbg_base_dir = Path(dump_dirpath) if dump_dirpath else Path.cwd()
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
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        input_names, output_names = build_io(
            model,
            test_input,
            io_npz_path=io_npz_path,
            input_names=input_names,
            output_names=output_names,
        )
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            debug_bundle_path=(dbg_path if is_debug else None),
            inference_target=inference_target,
            nnef_variable_naming_scheme=nnef_variable_naming_scheme,
            custom_extensions=custom_extensions,
        )
        export_path = export_path.with_suffix(".nnef.tgz")
        if dump_dirpath:
            shutil.copy(
                export_path,
                dump_test_tz_path,
            )
            shutil.copy(
                io_npz_path,
                dump_test_tz_path.with_suffix(".io.npz"),
            )
        if callback is not None:
            callback(inference_target, export_path)


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


def skipif_unsupported_qtensor(f):
    @pytest.mark.skipif(
        condition=QTENSOR_UNSUPPORTED,
        reason=QTENSOR_UNSUPPORTED_MSG,
    )
    def wrap(*args, **kargs):
        return f(*args, **kargs)

    return wrap


def skipif_unsupported_tensor_updater(f):
    @pytest.mark.skipif(
        condition=torch_version() < "2.0.0",
        reason="torch version need to be >= 2.0 to use ModTensorUpdater",
    )
    def wrap(*args, **kargs):
        return f(*args, **kargs)

    return wrap
