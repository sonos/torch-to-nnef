from pathlib import Path
import torch
import torchaudio

from torch_to_nnef import export_model_to_nnef, TractNNEF


class CustomDeepSpeech(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = torch.nn.Sequential(
            torch.nn.BatchNorm1d(64),
            torch.nn.Conv1d(64, 128, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=5),
            torch.nn.ReLU(),
        )
        self.maxpool = torch.nn.MaxPool1d(2)
        self.deepspeech = torchaudio.models.DeepSpeech(128, n_hidden=256)

    def forward(self, x):
        x = x.permute([0, 2, 1])
        x = self.pre(x)
        x = self.maxpool(x)
        x = x.permute([0, 2, 1])
        x = x.unsqueeze(1)
        return self.deepspeech(x)


file_path_export = Path("custom_deepspeech.nnef.tgz")
custom_deepspeech = CustomDeepSpeech()
input = torch.rand(7, 100, 64)
export_model_to_nnef(
    model=custom_deepspeech,
    args=input,
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        dynamic_axes={
            "melbank": {0: "B", 1: "S"},
        },
        version=TractNNEF.latest_version(),
        check_io=True,
    ),
    input_names=["melbank"],
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
    custom_extensions=[
        "tract_assert S >= 1",
        "tract_assert B >= 1",
    ],
)
