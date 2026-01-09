import pytest
from torch_to_nnef.utils import SemanticVersion


from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    check_model_io_test,
    cond_tract_gt_0_22_0,
)

try:
    import nemo
    import nemo.collections.asr as nemo_asr  # noqa: F401
    from torch_to_nnef.nemo_tract import (
        PARAKEET_V3_SLUG,
        iter_export_params_for_generic_nemo_asr_model,
    )
except ImportError as exp:
    print("disable test_nemo because:", exp)
    pytest.skip(
        reason="disabled since import of nemo_asr failed in some way",
        allow_module_level=True,
    )


def check_export_asr_model(model_slug, skip_preprocessor=False):
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    if (
        not cond_tract_gt_0_22_0(inference_target)
        and SemanticVersion.from_str(nemo.__version__) > "2.1.0"
    ):
        pytest.skip(
            "skip test for tract>0.22.0 && nemo>2.1"
            "since tract needs fix & features"
        )
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_slug)

    for export_params in iter_export_params_for_generic_nemo_asr_model(
        asr_model, inference_target, skip_preprocessor=skip_preprocessor
    ):
        print(f"testing export of: {model_slug}: {export_params.name}")
        check_model_io_test(
            model=export_params.model,
            test_input=export_params.test_input,
            inference_target=export_params.inference_target,
            input_names=export_params.input_names,
            output_names=export_params.output_names,
            custom_extensions=export_params.custom_extensions,
            allow_same_io_names=export_params.allow_same_io_names,
        )


def test_nemo_asr_parakeet_v3():
    check_export_asr_model(PARAKEET_V3_SLUG)
