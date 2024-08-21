"""Make training and any ops involving random reproducible"""

import os
import random
import shutil
import tempfile
import typing as T
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as Torch
from torch.nn.utils.weight_norm import WeightNorm

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.inference_target import InferenceTarget, TractNNEF
from torch_to_nnef.log import log
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
from torch_to_nnef.tract import build_io

TRACT_INFERENCES_TO_TESTS = [
    # we maintain last 3 majors of tract
    TractNNEF(version=TractNNEF.LATEST_KNOWN_STABLE_VERSION),
    TractNNEF(version="0.20.22"),
    TractNNEF(version="0.19.16"),
]

INFERENCE_TARGETS_TO_TESTS = TRACT_INFERENCES_TO_TESTS + [
    # TODO: add KhronosNNEF 1.0.5
]


class TestSuiteInferenceExactnessBuilder:
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
    renaming_scheme=VariableNamingScheme.default(),
):
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
            debug_bundle_path=(
                Path.cwd()
                / "failed_tests"
                / datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            )
            if os.environ.get("DEBUG", False)
            else None,
            inference_target=inference_target,
            renaming_scheme=renaming_scheme,
        )
        dump_filepath = os.environ.get("DUMP_FILEPATH", False)
        if dump_filepath:
            shutil.copy(export_path.with_suffix(".nnef.tgz"), dump_filepath)


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
