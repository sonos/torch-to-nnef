""" Make training and any ops involving random reproducible """

import os
import random
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as Torch
from torch.nn.utils.weight_norm import WeightNorm

from torch_to_nnef.export import export_model_to_nnef
from torch_to_nnef.log import log
from torch_to_nnef.tract import build_io


def set_seed(seed=0, cudnn=False, torch=True):
    if cudnn and Torch.cuda.is_available():
        Torch.backends.cudnn.deterministic = True
        Torch.backends.cudnn.benchmark = False
    if torch:
        Torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _test_check_model_io(model: Torch.nn.Module, test_input, dynamic_axes=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.nnef"
        io_npz_path = Path(tmpdir) / "io.npz"

        model = model.eval()

        input_names, output_names = build_io(
            model, test_input, io_npz_path=io_npz_path
        )
        export_model_to_nnef(
            model=model,
            args=test_input,
            file_path_export=export_path,
            input_names=input_names,
            output_names=output_names,
            log_level=log.INFO,
            check_same_io_as_tract=True,
            debug_bundle_path=(
                Path.cwd()
                / "failed_tests"
                / datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
            )
            if os.environ.get("DEBUG", False)
            else None,
            dynamic_axes=dynamic_axes,
        )


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
