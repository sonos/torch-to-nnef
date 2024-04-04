""" Regroup all Exceptions that can happen in torch_to_nnef.

rational:
    Can catch all package related error with an except TorchToNNEFError
    same apply for tract errors with except TractError

    This library is embedded in others codebase:
    Errors should be specific to be catchable so no builtin Exception
    should be raised
"""


class TorchToNNEFError(Exception):
    """Generic error that all errors in this lib inherit"""


class TorchToNNEFNotImplementedError(NotImplementedError, TorchToNNEFError):
    pass


class DynamicShapeValue(ValueError, TorchToNNEFError):
    pass


class BitPackingError(ValueError, TorchToNNEFError):
    pass


class FragmentFileError(KeyError, TorchToNNEFError):
    pass


# strict NNEF spec related
class StrictNNEFSpecError(TorchToNNEFError):
    pass


class IRError(TorchToNNEFError):
    pass


class TorchGraphExtraction(TorchToNNEFError):
    pass


# Torch are linked to torch_graph module errors
class TorchJitTraceFailed(RuntimeError, TorchGraphExtraction):
    pass


class TorchUnableToTraceData(ValueError, TorchGraphExtraction):
    pass


class TorchOpTranslatedDifferently(ValueError, TorchGraphExtraction):
    pass


class TorchNotFoundDataNode(ValueError, TorchGraphExtraction):
    pass


class TorchNotFoundOp(ValueError, TorchGraphExtraction):
    pass


class TorchCheckError(ValueError, TorchGraphExtraction):
    pass


# custom related
class NotFoundModuleExtractor(KeyError, TorchGraphExtraction):
    pass


# tract related
class TractError(TorchToNNEFError):
    pass


class OnnxExportError(RuntimeError, TractError):
    pass


class TractOnnxToNNEFError(RuntimeError, TractError):
    pass


class IOPytorchTractNotISOError(ValueError, TractError):
    pass
