from pathlib import Path

import torch
from torch import nn

from torch_to_nnef import TractNNEF, export_model_to_nnef


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.cnn1 = nn.Conv1d(10, 10, 3)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv1d(10, 1, 3)
        self.relu2 = nn.ReLU()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.dequant(x)
        return x


torch.backends.quantized.engine = "qnnpack"
m = Model()
m.eval()
m.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
mf = torch.ao.quantization.fuse_modules(
    m, [["cnn1", "relu1"], ["cnn2", "relu2"]]
)
mp = torch.ao.quantization.prepare(mf)
input_fp32 = torch.randn(1, 10, 15)
mp(input_fp32)
model_int8 = torch.ao.quantization.convert(mp)
res = model_int8(input_fp32)
file_path_export = Path("model_q8_ptq.nnef.tgz")
export_model_to_nnef(
    model=model_int8,
    args=input_fp32,
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=True,
    ),
    input_names=["input"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
