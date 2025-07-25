# 3. Model with multiple inputs or outputs

!!! abstract "Goals"

    At the end of this tutorial you will know:

    1. :material-toolbox: How to export multi io neural network
    2. :octicons-cross-reference-24: data type limitations of the NNEF representation

!!! example "Prerequisite"

    - [ ] PyTorch and Python basics
    - [ ] 10 min to read this page

A lot neural network model requires more than 1 input or output.
In that case we support no less that classical ONNX export.

## How to export ?

To exemplify this,  let's simply try to export a classical Language
model called **Albert** (from ['*ALBERT: A lite BERT for self-supervised
learning of language representations*',  2020](https://arxiv.org/pdf/1909.11942)) with the [`transformers`](https://github.com/huggingface/transformers) library.

First let's create a dir and install few dependencies:

```bash title="setup"
mkdir multi_io_py
cd multi_io_py
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch==2.7.0 \
    transformers==4.53.2 \
    sentencepiece==0.2.0 \
    torch_to_nnef
touch export_albert.py
```

We are now ready to start, load the model and prepare and
input sample:

```python title="load model and input sample in ('export_albert.py' part 1)"

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
inputs = tokenizer("Hello, I am happy", return_tensors="pt")
albert_model = AlbertModel.from_pretrained("albert-base-v2")
```

### Using basic export API

What would happen if we used the same call as previously in the [getting started tutorial](./1_getting_started.md) ?

Let's not forget that `inputs` generated from the tokenizer is a Python object  [`BatchEncoding`](https://github.com/huggingface/transformers/blob/a1ad9197c5756858e9014a0e01fe5fb1791efdf2/src/transformers/tokenization_utils_base.py#L192). It contains the tensors that we will use in the `forward` pass of this network in the following attributes:
`input_ids`, `attention_mask`, `token_type_ids`.
So we need to add those in `args` and `input_names` of export API.

Let's try together:

```python title="simple approach ('export_albert.py' part 2)"
from pathlib import Path
from torch_to_nnef import export_model_to_nnef, TractNNEF

file_path_export = Path("albert_v2.nnef.tgz")
input_names = [
    'input_ids', 'attention_mask', 'token_type_ids'
]
export_model_to_nnef(
    model=albert_model,
    # here we can not simply write
    # args=inputs.values()
    # because order of values
    # is different than .forward parameters !
    args=[inputs[k] for k in input_names],
    file_path_export=file_path_export,
    inference_target=TractNNEF(
        version="0.21.13",
        check_io=True,
    ),
    input_names=input_names,
    output_names=["output"],
    debug_bundle_path=Path("./debug.tgz"),
)
```

!!! warning "Warning"

    This export is for demonstration of multi inputs outputs only ! The [dynamic dimensions](./4_streaming_dimension.md) specification is missing which create a
    limited sub-optimal exported NNEF model.

If you run this script you should get a model very close to the
definition in transformer library with the *graph.nnef* signature that look like this:

```nnef title="nnef graph signature"
graph network(
    input_ids,
    token_type_ids,
    attention_mask
) -> (output_last_hidden_state, output_pooler_output)
```

#### Wait but what did just happen to the outputs ?

- This transformer model return in Python a special object:  [`BaseModelOutputWithPooling`](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/modeling_outputs.py#L71-L99)
- `torch_to_nnef` export is called with the `output_names` specified with 1 element named `output`

##### So how come we get 2 outputs ?

It turns out `torch_to_nnef` try hard to make sense of inputs and outputs provided.

In this case, the output object have been partially filled because of the inputs provided
to the model. In the upper snippet, we did not add parameters for the `AlbertModel.forward` method: `output_attentions` or `output_hidden_states`: to `True`. So all the graph
traced and exported use the control-flow not collecting those outputs.

!!! warning "Warning"
    This is one of the key limitation NNEF export, since it is based on internal Graph representation
    in **PyTorch** it doesn't really know more that PyTorch. All the control-flow existing Python side are unknown.
    This is why selecting correctly your input so that the correct trace and outputs end up being exported is very
    important.

    That's also why conditional sub-model execution is not embededable directly to
    NNEF (think Mixture Of Experts by example). But fear not we have solutions for that.

Ok now that's a bit clearer 😮‍💨, but why output names differ from those in Python modeling ?

Well we requested the first output object to be named `output` so all it's '`Map`' content
that have tensors value have been prefixed like this `output_{internal_object_key}`.

## IO Specification

A bit lost about what is and is not possible to export as Input/Outputs ?

Input(s) provided in export parameter: `args` can be:

- a simple `torch.Tensor`
- a list of **supported elements**

Those **supported elements** being:

- a `torch.Tensor`
- a dict of `torch.Tensor`
- a list or a tuple of `torch.Tensor`
- An object that mimic a dict by implementing [`__getitem__`](https://docs.python.org/3/reference/datamodel.html#object.__getitem__) magic function
- Containers (dict, list, tuples, mimic object) can themselves embed sub-containers that
    contains `torch.Tensor`

**Important:**

- Python Primitives (boolean, integer, float, string values) are passed but the trace and export will 'constantize' them so they will not be variable anymore in NNEF.
- List, Tuple and dict are as well fixed in length and keys possible at export time.

!!! info "Variable Python primitive in NNEF"

    To work-around Python primitives constantization you can
    transform those into `torch.Tensor`. This will only work on
    primitive that does not change the control-flow.

Outputs have the same object flexibility.

Also, if some names are not provided in `input_names` and `output_names` they will be automatically generated with following template `input_{}` and `output_{}` where the content of the brackets depends on indexes and keys.

### Selection inputs and outputs to export

Ok that's nice, we should now start to better understand what's is possible to do with simple `torch_to_nnef` export call.

What about if you want something that only export the `last_hidden_states` ?

In that case you can simply wrap the model into a new `nn.Module` like so:

```python title="basic model wrapping"
import torch

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

You can now export this wrapped model with the same API call we specified upper.
The same logic would apply if you wish to ignore some `inputs` to can set those in
the `__init__` and reference them in the `forward` pass.

A concrete example of this is available in the `torch_to_nnef` codebase with regard
to LLM wrappers that need to handle the KV-cache properly: [here](https://github.com/sonos/torch-to-nnef/tree/main/torch_to_nnef/llm_tract/models/base.py).

### Working around limitations

#### Dynamic number of input or output is not supported out of the box

Given your network have a variable number of input or outputs
with the same shape you can envision to wrap your network inside
a `torch.nn.Module` that will concatenate those into a single tensor
of variable size.

If the shapes are of varying size preventing the direct concatenation,
you can pad your tensors before and add another tensor responsible to keep
track of your padding values.

This again will only work if the underlying PyTorch IR graph is not altered
by the different inputs.

While suboptimal in RAM and compute, it should allow to express any possible
network in that respect.
