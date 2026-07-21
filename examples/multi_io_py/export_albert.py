from pathlib import Path

import torch
import transformers
from transformers import AlbertModel, AlbertTokenizer

from torch_to_nnef import TractNNEF, export_model_to_nnef
from torch_to_nnef.utils import SemanticVersion


def apply_transformers_trace_compat():
    """Make the transformers>=5.5 mask path survive torch.jit.trace.

    Under trace, ``inputs_embeds.shape[1]`` becomes a 0-dim tensor, which
    ``sdpa_mask`` mistakes for a deprecated ``cache_position`` arg and crashes
    on ``q_length.shape[0]``. Re-express that scalar as the 1-D position tensor
    the back-compat branch expects so the mask builds under trace.

    Scoped to 5.5.x: that back-compat branch is documented for removal in 5.6,
    where rewriting the scalar into a 1-D tensor would instead break the trace.
    """
    version = SemanticVersion.from_str(transformers.__version__)
    if not "5.5.0" <= version < "5.6.0":
        return
    import transformers.masking_utils as mu  # pylint: disable=import-outside-toplevel

    orig_sdpa_mask = mu.sdpa_mask

    def fix(q_length):
        if isinstance(q_length, torch.Tensor) and q_length.dim() == 0:
            return torch.arange(q_length, device=q_length.device)
        return q_length

    def traceable_sdpa_mask(*args, **kwargs):
        if len(args) > 1:
            args = list(args)
            args[1] = fix(args[1])
        if "q_length" in kwargs:
            kwargs["q_length"] = fix(kwargs["q_length"])
        return orig_sdpa_mask(*args, **kwargs)

    # eager_mask resolves sdpa_mask through the module globals, i.e. this same
    # attribute, so one assignment covers both entry points.
    mu.sdpa_mask = traceable_sdpa_mask


apply_transformers_trace_compat()

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
# transformers 5.x no longer returns token_type_ids by default; request it
# explicitly so the export below keeps its three-input signature on 4.x and 5.x.
inputs = tokenizer(
    "Hello, I am happy", return_tensors="pt", return_token_type_ids=True
)
# transformers 5.x defaults to a fused SDPA attention path that feeds a
# non-float attn_mask into scaled_dot_product_attention, which fails during the
# export forward. Eager attention decomposes into core ops that export cleanly.
model_kwargs = {}
if SemanticVersion.from_str(transformers.__version__) >= "5.0.0":
    model_kwargs["attn_implementation"] = "eager"
albert_model = AlbertModel.from_pretrained("albert-base-v2", **model_kwargs)

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
