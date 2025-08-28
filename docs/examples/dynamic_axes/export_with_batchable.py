from pathlib import Path

import torch
from torchvision import models as vision_mdl
from torchvision.io import read_image

from torch_to_nnef import TractNNEF, export_model_to_nnef

my_image_model = vision_mdl.vit_b_16(pretrained=True)

img = read_image("./Grace_Hopper.jpg")
classification_task = vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1
input_data_sample = classification_task.transforms()(img.unsqueeze(0))
file_path_export = Path("vit_b_16_batchable.nnef.tgz")
export_model_to_nnef(
    model=my_image_model,  # any nn.Module
    args=input_data_sample,  # list of model arguments
    # (here simply an example of tensor image)
    file_path_export=file_path_export,  # filepath to dump NNEF archive
    inference_target=TractNNEF(  # inference engine to target
        version=TractNNEF.latest_version(),  # tract version
        # (to ensure compatible operators)
        check_io=True,  # default False
        # (tract binary will be installed on the machine on fly)
        dynamic_axes={"input": {0: "B"}},
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path(
        "./debug.tgz"
    ),  # create a debug bundle in case model export work
    # but NNEF fail in tract (either due to load error or precision mismatch)
)
with torch.no_grad():
    predicted_index = my_image_model(input_data_sample).argmax(1).tolist()[0]
    print(
        "class id:",
        predicted_index,
        "label: ",
        classification_task.meta["categories"][predicted_index],
    )
print(f"exported {file_path_export.absolute()}")
