import torch
from model import MyModel

from torch_to_nnef import TractNNEF, export_model_to_nnef


def main():
    model = MyModel().eval()
    x = torch.randn(2, 3)

    # Export to a local archive; use compression=0 for a .tar.
    export_path = "my_relu.nnef.tgz"
    export_model_to_nnef(
        model,
        args=(x,),
        file_path_export=export_path,
        inference_target=TractNNEF.latest(),
        compression_level=0,
        input_names=["x"],
        output_names=["y"],
        load_extra_op_modules=["t2n_custom.handlers"],
    )
    print(f"Exported: {export_path}")


if __name__ == "__main__":
    main()
