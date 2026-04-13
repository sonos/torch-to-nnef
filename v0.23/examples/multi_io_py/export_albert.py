from pathlib import Path

from transformers import AlbertModel, AlbertTokenizer

from torch_to_nnef import TractNNEF, export_model_to_nnef

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer("Hello, I am happy", return_tensors="pt")
albert_model = AlbertModel.from_pretrained("albert-base-v2")

file_path_export = Path("albert_v2.nnef.tgz")
input_names = ["input_ids", "attention_mask", "token_type_ids"]
export_model_to_nnef(
    model=albert_model,
    args=[inputs[k] for k in input_names],
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version=TractNNEF.latest_version(),
        check_io=True,
    ),
    input_names=input_names,
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
