"""Same logic as VIT but with efficientnet and batch dynamic_axes"""

from pathlib import Path
import torch
from torchvision import models as vision_mdl
from torchvision.io import read_image
from torch_to_nnef import export_model_to_nnef, TractNNEF

base_model = vision_mdl.efficientnet_b0
weights = vision_mdl.EfficientNet_B0_Weights

my_image_model = base_model(pretrained=True)

img = read_image("./Grace_Hopper.jpg")
classification_task = weights.IMAGENET1K_V1
input_data_sample = classification_task.transforms()(img.unsqueeze(0))
file_path_export = Path("efficientnet_b0_batchable.nnef.tgz")
export_model_to_nnef(
    model=my_image_model,  # any nn.Module
    args=input_data_sample,  # list of model arguments (here simply an example of tensor image)
    file_path_export=file_path_export,  # filepath to dump NNEF archive
    inference_target=TractNNEF(  # inference engine to target
        version="0.21.13",  # tract version (to ensure compatible operators)
        check_io=True,  # default False (tract binary will be installed on the machine on fly)
        dynamic_axes={"inp": {0: "B"}},
    ),
    input_names=["inp"],
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
with Path("./classes.txt").open("w", encoding="utf8") as fh:
    for c in weights.IMAGENET1K_V1.meta["categories"]:
        fh.write(f"{c}\n")
print(f"exported {file_path_export.absolute()}")
