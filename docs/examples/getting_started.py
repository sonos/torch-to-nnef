from pathlib import Path
import torch
from torchvision import models as vision_mdl
from torch_to_nnef import export_model_to_nnef, TractNNEF

my_image_model = vision_mdl.vit_b_16(pretrained=False)

input_data_sample = torch.rand(1, 3, 224, 224)
file_path_export = Path("vit_b_16.nnef.tgz")
export_model_to_nnef(
    model=my_image_model,  # any nn.Module
    args=input_data_sample,  # list of model arguments (here simply an example of tensor image)
    file_path_export=file_path_export,  # filepath to dump NNEF archive
    inference_target=TractNNEF(  # inference engine to target
        version="0.21.13",  # tract version (to ensure compatible operators)
        check_io=True,  # default False (tract binary will be installed on the machine on fly)
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path(
        "./debug.tgz"
    ),  # create a debug bundle in case model export work
    # but NNEF fail in tract (either due to load error or precision mismatch)
)
print(f"exported {file_path_export.absolute()}")
