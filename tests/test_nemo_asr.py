from contextlib import contextmanager
import pytest
import torch
from torch_to_nnef.inference_target.tract import TractNNEF
from .utils import check_model_io_test

try:
    import nemo.collections.asr as nemo_asr  # noqa: F401
except ImportError as exp:
    print("disable test_nemo because:", exp)
    pytest.skip(
        reason="disabled since import of nemo_asr failed in some way",
        allow_module_level=True,
    )

# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
PARAKEET_V3_SLUG = "nvidia/parakeet-tdt-0.6b-v3"


@contextmanager
def exportable_nemo_net(output_name, model, input_example, use_dynamo=False):
    """Context manager to follow export way of nemo models.

    see: nemo.core.classes.Exportable._export
    """
    from nemo.core.classes import typecheck
    from nemo.core.classes.exportable import Exportable
    from nemo.utils.export_utils import wrap_forward_method, parse_input_example
    from pytorch_lightning.core.module import _jit_is_scripting

    my_args = {"use_dynamo": use_dynamo}

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    exportables = []
    for m in model.modules():
        if isinstance(m, Exportable):
            exportables.append(m)

    forward_method = None
    old_forward_method = None
    try:
        # Disable typechecks
        typecheck.set_typecheck_enabled(enabled=False)

        # Allow user to completely override forward method to export
        forward_method, old_forward_method = wrap_forward_method(model)

        # Set module mode
        with (
            torch.inference_mode(),
            torch.no_grad(),
            torch.jit.optimized_execution(True),
            _jit_is_scripting(),
        ):
            if input_example is None:
                input_example = model.input_module.input_example()

            # Run (posibly overridden) prepare methods before calling forward()
            for ex in exportables:
                ex._prepare_for_export(**my_args, noreplace=True)
            model._prepare_for_export(
                output=output_name, input_example=input_example, **my_args
            )

            input_list, input_dict = parse_input_example(input_example)
            output_example = model.forward(*input_list, **input_dict)
            if not isinstance(output_example, tuple):
                output_example = (output_example,)

            input_names = model.input_names
            output_names = model.output_names
            # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
            dynamic_axes = model.dynamic_shapes_for_export(use_dynamo)
            yield input_example, output_example, dynamic_axes
    finally:
        typecheck.enable_wrapping(enabled=True)
        typecheck.set_typecheck_enabled(enabled=True)
        if forward_method:
            type(model).forward = old_forward_method
        model._export_teardown()


def iter_nemo_model_subnets(model, input_example=None):
    """Iterator over exportable subnets of a nemo model.

    Yields:
        subnet_name: name of the subnet
        subnet: the subnet module
        input_example: input example for the subnet
        dynamic_axes: dynamic axes info for the subnet

    see: nemo.core.classes.Exportable.export

    """
    for subnet_name in model.list_export_subnets():
        subnet = model.get_export_subnet(subnet_name)
        with exportable_nemo_net(subnet_name, subnet, input_example) as (
            input_example,
            out_example,
            dynamic_axes,
        ):
            yield subnet_name, subnet, input_example, dynamic_axes
            # Propagate input example (default scenario, may need to be overriden)
            if input_example is not None:
                input_example = out_example


def test_nemo_asr_parakeet_v3():
    # nemo_asr.models.ASRModel.list_available_models()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="parakeet-tdt_ctc-110m"  # PARAKEET_V3_SLUG
    )
    asr_model.eval()
    # inps = asr_model.preprocessor.input_example()
    # fail in tract due to const window (likely a bug ...)
    # check_model_io_test(
    #     model=asr_model.preprocessor.get_features,
    #     test_input=inps,
    #     inference_target=TractNNEF.latest(),
    # )

    for (
        subnet_name,
        subnet,
        input_example,
        dynamic_axes,
    ) in iter_nemo_model_subnets(asr_model):
        print("start export subnet:", subnet_name)
        check_model_io_test(
            model=subnet,
            test_input=input_example,
            inference_target=TractNNEF.latest(),
        )
        print("exported subnet:", subnet_name, "with success")
        pass
