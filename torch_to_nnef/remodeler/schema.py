"""Provider-agnostic schema constants for shape-config.

These keys define the nested YAML/JSON structure shared by providers that
support the remodeler boundary plan (collapse, bind, renamed_symbols, etc.).
"""

# Top-level subnet mapping keys
SHAPE_KEY_INPUTS = "inputs"
SHAPE_KEY_OUTPUTS = "outputs"
SHAPE_KEY_RENAMED = "renamed_symbols"
SHAPE_KEY_OUTPUTS_KEEP = "outputs_keep"

# Per-input fields
INPUT_FIELD_ORIGINAL_SHAPE = "original_shape"
INPUT_FIELD_COLLAPSE_DIMS = "collapse_dims"
# Canonical bind key in user config (descriptive)
INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE = "bind_scalar_to_dim_size"
INPUT_FIELD_EVAL_SYMBOLS = "eval_symbols"

# Per-subnet fields
SHAPE_KEY_EXTENSIONS = "extensions"

# Per-output fields
OUTPUT_FIELD_COLLAPSE_DIMS = "collapse_dims"

__all__ = [
    "SHAPE_KEY_INPUTS",
    "SHAPE_KEY_OUTPUTS",
    "SHAPE_KEY_RENAMED",
    "SHAPE_KEY_OUTPUTS_KEEP",
    "INPUT_FIELD_ORIGINAL_SHAPE",
    "INPUT_FIELD_COLLAPSE_DIMS",
    "INPUT_FIELD_BIND_SCALAR_TO_DIM_SIZE",
    "INPUT_FIELD_EVAL_SYMBOLS",
    "SHAPE_KEY_EXTENSIONS",
    "OUTPUT_FIELD_COLLAPSE_DIMS",
]
