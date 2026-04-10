from copy import deepcopy
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter, NMSModel
from ultralytics.nn.tasks import (
    DetectionModel,
    SegmentationModel,
)
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.patches import arange_patch

import torch_to_nnef

tract_target = torch_to_nnef.TractNNEF.latest()
model = YOLO(
    "yolo11n-pose.pt"
)  # load a pretrained model (recommended for training)

PREFIX_DEFAULT = colorstr("NNEF:")


def global_export_nnef(self, prefix=PREFIX_DEFAULT):
    """Export YOLO model to NNEF format."""
    LOGGER.info(
        f"\n{prefix} starting export with NNEF: "
        f"{torch_to_nnef.__version__} targetting "
        f"tract: {tract_target.version}..."
    )
    f = str(self.file.with_suffix(".nnef.tgz"))
    output_names = (
        ["output0", "output1"]
        if isinstance(self.model, SegmentationModel)
        else ["output0"]
    )
    dynamic = self.args.dynamic
    if dynamic:
        dynamic = {
            "images": {0: "batch", 2: "height", 3: "width"}
        }  # shape(1,3,640,640)
        if isinstance(self.model, SegmentationModel):
            dynamic["output0"] = {
                0: "batch",
                2: "anchors",
            }  # shape(1, 116, 8400)
            dynamic["output1"] = {
                0: "batch",
                2: "mask_height",
                3: "mask_width",
            }  # shape(1,32,160,160)
        elif isinstance(self.model, DetectionModel):
            dynamic["output0"] = {
                0: "batch",
                2: "anchors",
            }  # shape(1, 84, 8400)
        if self.args.nms:  # only batch size is dynamic with NMS
            dynamic["output0"].pop(2)

    with arange_patch(self.args):
        export_nnef(
            NMSModel(self.model, self.args) if self.args.nms else self.model,
            self.im,
            f,
            input_names=["images"],
            output_names=output_names,
            dynamic=dynamic or None,
        )
    return f, None


def export_nnef(model, im, filepath, input_names, output_names, dynamic):
    inference_target = deepcopy(tract_target)
    inference_target.dynamic_axes = dynamic or {}
    start_time = perf_counter()
    torch_to_nnef.export_model_to_nnef(
        model=model,
        args=im,
        file_path_export=filepath,
        inference_target=inference_target,
        input_names=input_names,
        output_names=output_names,
    )
    end_time = perf_counter()
    LOGGER.info(
        f"{colorstr('NNEF:')} export success "
        f"in {end_time - start_time:.3f}s, saved as {filepath}"
    )


# check onnx export works
model.export(format="onnx")

path = Path("./Grace_Hopper.jpg")
if path.exists():
    start_time = perf_counter()
    res = model.track(path)
    end_time = perf_counter()
    LOGGER.info("==================")
    LOGGER.info(
        f"Tracking completed in {end_time - start_time:.3f}s "
        "(with ultralytics .track for same image)"
    )
    LOGGER.info("==================")

Exporter.export_onnx = global_export_nnef

model.export(format="onnx")
