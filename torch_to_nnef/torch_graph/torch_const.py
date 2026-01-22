"""Some Core PyTorch core naming, used to trace/parse their JIT."""

PRIM_STARTID = "prim::"
CALL_KIND = "prim::CallMethod"
CONSTANT_KIND = "prim::Constant"
DICTCONSTRUCT_KIND = "prim::DictConstruct"
GETATTR_KIND = "prim::GetAttr"
LISTCONSTRUCT_KIND = "prim::ListConstruct"
LISTUNPACK_KIND = "prim::ListUnpack"
NUMTOTENSOR_KIND = "prim::NumToTensor"
PARAM_KIND = "prim::Param"
TUPLECONSTRUCT_KIND = "prim::TupleConstruct"
TUPLEUNPACK_KIND = "prim::TupleUnpack"

ATEN_STARTID = "aten::"
ATEN_ALIAS = "aten::alias"
ATEN_ARANGE = "aten::arange"
ATEN_BADDMM = "aten::baddbmm"
ATEN_CLONE = "aten::clone"
ATEN_CONTIGUOUS_KIND = "aten::contiguous"
ATEN_CUMSUM = "aten::cumsum"
ATEN_EINSUM = "aten::einsum"
ATEN_EMPTY = "aten::empty"
ATEN_EMPTY_LIKE = "aten::empty_like"
ATEN_EMBEDDING = "aten::embedding"
ATEN_LINEAR = "aten::linear"
ATEN_MATMUL = "aten::matmul"
ATEN_FULL = "aten::full"
ATEN_FULL_LIKE = "aten::full_like"
ATEN_GELU = "aten::gelu"
ATEN_INT = "aten::Int"
ATEN_LINALG_NORM = "aten::linalg_norm"
ATEN_LINALG_VECTOR_NORM = "aten::linalg_vector_norm"
ATEN_MASKED_FILL = "aten::masked_fill"
ATEN_MASKED_FILL_ = "aten::masked_fill_"
ATEN_NEW_ONES = "aten::new_ones"
ATEN_ONES_LIKE = "aten::ones_like"
ATEN_PROD = "aten::prod"
ATEN_REPEAT_INTERLEAVE = "aten::repeat_interleave"
ATEN_RELU = "aten::relu"
ATEN_SILU = "aten::silu"
ATEN_SCALARIMPLICIT = "aten::ScalarImplicit"
ATEN_SCALAR_TENSOR = "aten::scalar_tensor"
ATEN_SCALED_DOT_PRODUCT_ATTENTION = "aten::scaled_dot_product_attention"
ATEN_SELECT = "aten::select"
ATEN_SIZE_KIND = "aten::size"
ATEN_TO = "aten::to"
ATEN_TO_COPY = "aten::_to_copy"
ATEN_VIEW_KIND = "aten::view"
ATEN_WHERE = "aten::where"
ATEN_ZEROS = "aten::zeros"
ATEN_NEW_ZEROS = "aten::new_zeros"
ATEN_ZERO_LIKE = "aten::zeros_like"
ATEN_NEW_EMPTY = "aten::new_empty"


CLASSTYPE_KIND = "ClassType"
TUPLETYPE_KIND = "TupleType"
LISTTYPE_KIND = "ListType"
NONETYPE_KIND = "NoneType"
INTTYPE_KIND = "IntType"
NUMBERTYPE_KIND = "NumberType"  # This type represents a Python number
DICTTYPE_KIND = "DictType"
# Subtype hierarchy for Number Types (NumberType as the base type):
# IntType <: NumberType
# FloatType <: NumberType
# ComplexType <:NumberType

MODULE_PATH_ATEN = "TORCH_INTERNAL_ATEN"
MODULE_PATH_QUANTIZED = "TORCH_INTERNAL_QUANTIZED"
SPECIAL_ATEN_REMAP_PYTORCH = {"__and__": "bitwise_and", "__or__": "bitwise_or"}

WIRED_CUSTOM_LSTM = "wired_custom::LSTM"

MAP_TO_NOP = [
    NUMTOTENSOR_KIND,
    LISTCONSTRUCT_KIND,
    ATEN_SCALARIMPLICIT,
    ATEN_ALIAS,
]
MAP_TO_TENSOR_FN = [ATEN_CONTIGUOUS_KIND, ATEN_VIEW_KIND]
