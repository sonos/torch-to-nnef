import numpy as np
import tract
from torchvision import models as vision_mdl
from torchvision.io import read_image

# let's read our example image again with torch vision and transform it
# in numpy feature matrix
img = read_image("./Grace_Hopper.jpg")
classification_task = vision_mdl.ViT_B_16_Weights.IMAGENET1K_V1
input_data_sample = classification_task.transforms()(img.unsqueeze(0)).numpy()

model = (
    tract.nnef()
    .with_tract_core()
    .model_for_path("./vit_b_16.nnef.tgz")
    .into_optimized()
    .into_runnable()
)

result = model.run([input_data_sample])
confidences = result[0].to_numpy()
predicted_index = np.argmax(confidences)
print(
    "class id:",
    predicted_index,
    "label: ",
    classification_task.meta["categories"][predicted_index],
)
