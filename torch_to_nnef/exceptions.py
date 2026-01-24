"""Regroup all Exceptions that can happen in torch_to_nnef.

rational:
    Can catch all package related error with an except T2NError
    same apply for tract errors with except T2NErrorTract

    This library is embedded in others codebase:
    Errors should be specific to be catchable so no builtin Exception
    should be raised
"""


class T2NError(Exception):
    """Generic error that all errors in this lib inherit."""


class T2NErrorImport(T2NError):
    """Import error that all errors in this lib inherit."""


class T2NErrorInvalidArgument(ValueError, T2NError):
    """specification of torch_to_nnef export not respected."""


class T2NErrorNotFoundFile(ValueError, T2NError):
    """missing exit for file path."""


class T2NErrorRuntime(RuntimeError, T2NError):
    pass


class T2NErrorNotImplemented(NotImplementedError, T2NError):
    pass


class T2NErrorMisuse(ValueError, T2NError):
    pass


class T2NErrorTestFailed(ValueError, T2NError):
    pass


class T2NErrorImpossibleQuantization(NotImplementedError, T2NError):
    pass


class T2NErrorIoQuantity(ValueError, T2NError):
    pass


class T2NErrorDataNodeValue(ValueError, T2NError):
    pass


class T2NErrorInconsistentTensor(ValueError, T2NError):
    pass


class T2NErrorConsistency(ValueError, T2NError):
    pass


class T2NErrorKhronosNNEFModuleLoad(ValueError, T2NError):
    pass


class T2NErrorKhronosInterpreterDiffValue(ValueError, T2NError):
    pass


class T2NErrorDynamicShapeValue(ValueError, T2NError):
    pass


class T2NErrorBitPacking(ValueError, T2NError):
    pass


class T2NErrorFragmentFile(KeyError, T2NError):
    pass


# strict NNEF spec related
class T2NErrorStrictNNEFSpec(T2NError):
    pass


class T2NErrorIR(T2NError):
    pass


class T2NErrorTorchGraphExtraction(T2NError):
    pass


# Torch are linked to torch_graph module errors
class T2NErrorTorchJitTraceFailed(RuntimeError, T2NErrorTorchGraphExtraction):
    pass


class T2NErrorTorchUnableToTraceData(ValueError, T2NErrorTorchGraphExtraction):
    pass


class T2NErrorTorchOpTranslatedDifferently(
    ValueError, T2NErrorTorchGraphExtraction
):
    pass


class T2NErrorTorchNotFoundDataNode(ValueError, T2NErrorTorchGraphExtraction):
    pass


class T2NErrorTorchNotFoundOp(ValueError, T2NErrorTorchGraphExtraction):
    pass


class T2NErrorTorchCheck(ValueError, T2NErrorTorchGraphExtraction):
    pass


# custom related
class T2NErrorNotFoundModuleExtractor(KeyError, T2NErrorTorchGraphExtraction):
    pass


# tract related
class T2NErrorTract(T2NError):
    pass


class T2NErrorTractDownload(T2NErrorTract):
    pass


class T2NErrorOnnxExport(RuntimeError, T2NErrorTract):
    pass


class T2NErrorTractOnnxToNNEF(RuntimeError, T2NErrorTract):
    pass


class T2NErrorIOPytorchTractNotISO(ValueError, T2NErrorTract):
    pass


class T2NErrorIRDataConsistency(Exception):
    pass
