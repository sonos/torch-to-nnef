from transformers import AlbertModel, AlbertTokenizer
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF


tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer(
    ["Hello, I am happy", "and also I am blond"], return_tensors="pt"
)
albert_model = AlbertModel.from_pretrained("albert-base-v2")

file_path_export = Path("albert_v2_dyn.nnef.tgz")
input_names = ["input_ids", "attention_mask", "token_type_ids"]
export_model_to_nnef(
    model=albert_model,
    args=[inputs[k] for k in input_names],
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        dynamic_axes={
            "input_ids": {0: "B", 1: "S"},
            "attention_mask": {0: "B", 1: "S"},
            "token_type_ids": {0: "B", 1: "S"},
        },
        version=TractNNEF.latest_version(),
        check_io=True,
    ),
    input_names=input_names,
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
    custom_extensions=[
        "tract_assert S >= 1",
        "tract_assert S <= 32000",
    ],
)
