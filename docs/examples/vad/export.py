"""Simple export script of MarbleNet VAD"""

from pathlib import Path
import copy

from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
import torch

from torch_to_nnef import export_model_to_nnef, TractNNEF


# sample rate, Hz
SAMPLE_RATE = 16000

vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
    "vad_multilingual_marblenet"
)


cfg = copy.deepcopy(vad_model._cfg)
print(OmegaConf.to_yaml(cfg))

vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)

# Set model to inference mode
vad_model.eval()
vad_model = vad_model.to(vad_model.device)

vad_model.preprocessor.featurizer.pad_to = 0  # in streaming this is important
# since we don't know when it will finish
# tract being strict about symbol interpretation (S dimension)

file_path_export = Path("vad_marblenet.nnef.tgz")
export_model_to_nnef(
    model=vad_model,  # any nn.Module
    args=(
        torch.rand(1, 512),
        torch.tensor([512]),
    ),  # list of model arguments (here simply an example of tensor image)
    file_path_export=file_path_export,  # filepath to dump NNEF archive
    inference_target=TractNNEF(  # inference engine to target
        version=TractNNEF.latest_version(),  # tract version (to ensure compatible operators)
        check_io=True,  # default False (tract binary will be installed on the machine on fly)
        dynamic_axes={"input_signal": {0: "B", 1: "S"}, "input_len": {0: "B"}},
    ),
    input_names=["input_signal", "input_len"],
    output_names=["output"],
    debug_bundle_path=Path(
        "./debug.tgz"
    ),  # create a debug bundle in case model export work
    custom_extensions=[
        "tract_assert S > 1",
        "tract_assert B > 0",
    ],
)
