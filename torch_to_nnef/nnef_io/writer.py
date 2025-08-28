"""Code borrowed from `nnef_tools` Khronos group package.

original module fullname `nnef_tools.io.nnef.writer`

This module is adapted with following goals:

- 1. Handling special Tract quantization variables storage with custom .dat
  data storage format
- 2. in `torch_to_nnef` transformation to numpy array of torch
tensor is postponed to just before serialization. this avoid COPY to stay
in memory (
    so the 'nnef.Graph' and data hold tensor of different kind
    than initially intended by Khronos group developpers
). This is crucial to export large models.

Also some minimal adaptation like code style have been done be pythonic.
(force utf8 encoding, avoid builtin redefinition, use fstring, simple expr ...)

"""

import logging
import os
import shutil
import tempfile
import typing as T

import nnef
import numpy as np
import torch
from nnef_tools.io.nnef.helpers import tgz_compress
from nnef_tools.model import Tensor
from nnef_tools.utils.types import as_str, from_numpy

from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.khronos import KhronosNNEF
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.tensor.offload import OffloadedTensor

LOGGER = logging.getLogger(__name__)

_DtypeFromNumpy = {
    np.float16: "scalar",
    np.float32: "scalar",
    np.float64: "scalar",
    np.int8: "integer",
    np.uint8: "integer",
    np.int16: "integer",
    np.uint16: "integer",
    np.int32: "integer",
    np.uint32: "integer",
    np.int64: "integer",
    np.uint64: "integer",
    np.bool_: "logical",
}


_DtypeFromPyType = {
    str: "string",
    float: "scalar",
    int: "integer",
    bool: "logical",
    None: "dtype",
}


def maybe_torch_to_np(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().numpy()
    return tensor


def _nnef_dtype(dtype):
    return (
        _DtypeFromNumpy[dtype.type if isinstance(dtype, np.dtype) else dtype]
        if dtype is not None
        else None
    )


def _print(
    graph, file, extensions, fragments, version_custom_ops, annotate_shapes
):
    assert graph.is_sorted(), "graph must be topologically sorted"
    assert all(
        tensor.name is not None
        or (tensor.producer is None and tensor.data is not None)
        for tensor in graph.tensors
    ), "all tensors must have names"
    assert all(
        all(s is not None for s in op.attribs["shape"])
        for op in graph.operations
        if op.type == "external"
    ), "external ops must not contain undefined shapes"

    print(nnef.format_version((1, 0)), file=file)
    if len(extensions):
        print(file=file)
        print(nnef.format_extensions(extensions), file=file)
    if fragments:
        print(file=file)
        print(fragments, file=file)
    print(file=file)

    graph_name = as_str(graph.name) if graph.name is not None else "G"
    graph_inputs = [as_str(item.name) for item in graph.inputs]
    graph_outputs = [as_str(item.name) for item in graph.outputs]

    graph_str = (
        f"graph {graph_name}({', '.join(graph_inputs)}) -> "
        f"({', '.join(graph_outputs)})"
    )
    print(graph_str, file=file)
    print("{", file=file)

    versions = {}
    for op in graph.operations:
        assert all(isinstance(item, Tensor) for item in op.outputs)

        inputs = (
            (
                (
                    from_numpy(maybe_torch_to_np(item.data))
                    if item.producer is None
                    else nnef.Identifier(as_str(item.name))
                )
                if isinstance(item, Tensor)
                else item
            )
            for item in op.inputs
        )
        inputs = (
            tuple(inputs) if isinstance(op.inputs, tuple) else (list(inputs),)
        )

        outputs = (nnef.Identifier(as_str(item.name)) for item in op.outputs)
        outputs = (
            tuple(outputs)
            if isinstance(op.outputs, tuple)
            else (list(outputs),)
        )

        attribs = {as_str(key): value for key, value in op.attribs.items()}

        name = (
            _next_version(op.type, versions)
            if op.type not in nnef.StandardOperations and version_custom_ops
            else op.type
        )

        dtype = attribs.get("dtype")
        if dtype is not None:
            dtype = _nnef_dtype(dtype)
            del attribs["dtype"]

        for key, value in attribs.items():
            if isinstance(value, (type, np.dtype)):
                attribs[key] = _nnef_dtype(value)

        invocation = nnef.format_invocation(
            name=name,
            dtype=dtype,
            attribs=attribs,
            inputs=inputs,
            outputs=outputs,
        )
        annotation = (
            "    # "
            + ", ".join(
                _nnef_dtype(output.dtype) + str(output.shape)
                for output in op.outputs
            )
            if annotate_shapes
            else ""
        )

        print(f"    {invocation};{annotation}", file=file)

    print("}", file=file)


def write_nnef_tensor(array, filename, quantized):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        nnef.write_tensor(file=file, tensor=array, quantized=quantized)


def write_tensor_quantization_infos(tensor, file):
    assert tensor.quant is not None
    op_name = tensor.quant["op-name"]
    attribs = ", ".join(
        f"{k} = {_printable_value(v)}"
        for k, v in tensor.quant.items()
        if k != "op-name" and v is not None
    )
    if attribs:
        print(
            f'"{tensor.name}": {op_name}({attribs});',
            file=file,
        )


def _write_quantization(graph, file):
    for tensor in graph.tensors:
        if tensor.quant:
            write_tensor_quantization_infos(tensor, file)


def _printable_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _next_version(name, versions):
    version = versions.get(name, 0) + 1
    versions[name] = version
    return f"{name}_v{version}"


def _generate_custom_fragments(graph, fragments, version):
    versions = {} if version else None
    return "\n".join(
        _generate_fragment(op, versions)
        for op in graph.operations
        if op.type not in nnef.StandardOperations and op.type not in fragments
    )


def _generate_fragment(op, versions):
    attribs = {
        key: _make_attrib_type(value) for key, value in op.attribs.items()
    }
    inputs = [_make_tensor_type(value) for value in op.inputs]
    outputs = [_make_tensor_type(value) for value in op.outputs]
    dtype = _nnef_dtype(op.attribs.get("dtype"))
    name = _next_version(op.type, versions) if versions is not None else op.type

    return (
        "fragment "
        + _fragment_signature(name, dtype, attribs, inputs, outputs)
        + ";"
    )


def _fragment_signature(name, dtype, attribs, inputs, outputs):
    txt = name
    if dtype is not None:
        txt += "<" + dtype + ">"
    txt += "( "
    txt += _types_str([f"_I{i + 1}" for i in range(len(inputs))], inputs, True)
    if len(inputs) and len(attribs):
        txt += ", "
    txt += _types_str(attribs.keys(), attribs.values(), False)
    txt += " ) -> ( "
    txt += _types_str(
        [f"_O{i + 1}" for i in range(len(outputs))], outputs, True
    )
    txt += " )"
    return txt


def _make_attrib_type(value):
    repeated = False
    if isinstance(value, list):
        if len(value) == 0:
            return None, False
        tp = type(value[0])
        if not all(isinstance(v, tp) for v in value):
            return None, False
        repeated = True
        value = value[0]

    if not isinstance(value, (float, int, bool, str)):
        return None, False

    return _DtypeFromPyType[type(value)], repeated


def _make_tensor_type(value):
    repeated = False
    if isinstance(value, list):
        if len(value) == 0:
            return None, False
        dtype = value[0].dtype
        if not all(v.dtype == dtype for v in value):
            return None, False
        repeated = True
        value = value[0]

    return _nnef_dtype(value.dtype), repeated


def _types_str(names, items, tensor):
    return ", ".join(
        name
        + ": "
        + (f"tensor<{type}>" if tensor else type)
        + ("[]" if repeated else "")
        for name, (type, repeated) in zip(names, items)
    )


class Writer:
    def __init__(
        self,
        compression=None,
        extensions=None,
        fragments=None,
        generate_custom_fragments=False,
        version_custom_fragments=True,
        annotate_shapes=False,
        inference_target: T.Optional[InferenceTarget] = None,
    ):
        if inference_target is None:
            inference_target = KhronosNNEF.latest()
        self._compression = compression
        self._extensions = extensions or []
        self._fragments = fragments or {}
        self._generate_custom_fragments = generate_custom_fragments
        self._version_custom_fragments = version_custom_fragments
        self._annotate_shapes = annotate_shapes
        self._inference_target = inference_target

    def _write_tensors_from_operators(self, graph, folder):
        for op in graph.operations:
            if op.type == "variable":
                if op.attribs.pop("custom_datatype", "") == "quant_tensor":
                    qtensor = op.output.qtensor
                    while isinstance(qtensor, OffloadedTensor):
                        qtensor = qtensor.to_base_tensor()
                    label = op.attribs["label"]
                    qtensor.write_in_file(folder, label, self._inference_target)
                    LOGGER.info("written qtensor: '%s'", label)
                else:
                    filename = op.attribs["label"] + ".dat"
                    write_nnef_tensor(
                        np.asarray(
                            maybe_torch_to_np(op.output.data), order="C"
                        ),
                        os.path.join(folder, filename),
                        quantized=bool(op.output.quant),
                    )

    def __call__(self, graph, path):
        LOGGER.info("start writting NNEF graph into: '%s'", path)
        folder = None
        try:
            if self._compression is not None:
                folder = tempfile.mkdtemp(prefix="nnef_")
            else:
                folder = path
                if not os.path.exists(folder):
                    os.makedirs(folder)

            self._write_tensors_from_operators(graph, folder)

            fragments = "".join(text for _, text in self._fragments.items())
            if self._generate_custom_fragments:
                customs = _generate_custom_fragments(
                    graph,
                    fragments=self._fragments,
                    version=self._version_custom_fragments,
                )
                if fragments and customs:
                    fragments += "\n"
                fragments += customs

            if len(fragments) and not isinstance(
                self._inference_target, TractNNEF
            ):
                if "KHR_enable_fragment_definitions" not in self._extensions:
                    self._extensions.append("KHR_enable_fragment_definitions")
                if "KHR_enable_operator_expressions" not in self._extensions:
                    self._extensions.append("KHR_enable_operator_expressions")

            graph_filename = os.path.join(folder, "graph.nnef")
            with open(graph_filename, "w", encoding="utf8") as file:
                _print(
                    graph,
                    file,
                    extensions=self._extensions,
                    fragments=fragments,
                    version_custom_ops=self._generate_custom_fragments
                    and self._version_custom_fragments,
                    annotate_shapes=self._annotate_shapes,
                )

            if any(tensor.quant for tensor in graph.tensors):
                quant_filename = os.path.join(folder, "graph.quant")
                with open(quant_filename, "w", encoding="utf8") as file:
                    _write_quantization(graph, file)
        finally:
            if self._compression is not None and folder:
                tgz_compress(
                    folder, path + ".tgz", compression_level=self._compression
                )
                shutil.rmtree(folder)
        LOGGER.info("finished writting NNEF graph into: %s", path)

    @staticmethod
    def _used_operators(graph, dependencies):
        used = {op.type for op in graph.operations}
        count = len(used)
        changed = True
        while changed:
            for key, deps in dependencies.items():
                if key in used:
                    used.update(deps)

            changed = len(used) > count
            count = len(used)

        return used
