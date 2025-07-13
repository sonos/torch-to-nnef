# 3. Model with multiple inputs or outputs

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export multi io neural network
    2. :octicons-cross-reference-24: data type limitations of the NNEF representation

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 5 min to read this page

Some neural network model requires more than 1 input or output.
In that case we support no less that classical ONNX export.

### How to do ?

To exemplify our issue simply let's try to export a classical transformer
model called Albert from the transformer library.

First let's create a dir and install few dependencies:

```bash title="setup"
mkdir multi_io_py
cd multi_io_py
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==2.7.0 transformers==4.53.2 sentencepiece==0.2.0 torch_to_nnef
touch export_albert.py
```

We are now ready to start and to this purpose we will need:

```python title="load model and input sample in 'export_albert.py'"

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer("Hello, I am happy", return_tensors="pt")
albert_model = AlbertModel.from_pretrained("albert-base-v2")
```

What would happen if we try to export it up front with the API we saw in getting started ?

Let's try together:

```python title="wrong approach"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("albert_v2.nnef.tgz")
export_model_to_nnef(
    model=albert_model,
    args=inputs.values(),
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version="0.21.13",
        check_io=True,
    ),
    input_names=inputs.keys(),
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
```

```python title="proper model wrapping"
class ALBERTModelWrapper(torch.nn.Module):
    def __init__(self, transformer_model: torch.nn.Module):
        super().__init__()
        self.transformer_model = transformer_model

    def forward(self, *args):
        outputs = self.transformer_model(*args)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

wrapped_model = ALBERTModelWrapper(albert_model)
```

### Limitations

#### Dynamic number of input or output is not supported out of the box

Given your network have a variable number of input or outputs
with the same shape you can envision to wrap your network inside
a `torch.nn.Module` that will concatenate those into a single tensor
of variable size.

If the shapes are of varying size preventing the direct concatenation,
you can pad your tensors before and add another tensor responsible to keep
track of your padding values.

While suboptimal in RAM and compute, it should allow to express any possible
network in that respect.

#### Python Object and primitives

All python object and primitive are ignored during the export step that trace the
graph. That means that if you wish to keep part of those parameters dynamic at runtime you need
to wrap your network inside a `torch.nn.Module`, that: expose each tensor primitive to be assigned
inside those object in the `forward` function.
