import pytest

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


def test_nemo_asr_parakeet_v3():
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=PARAKEET_V3_SLUG
    )
    pass
