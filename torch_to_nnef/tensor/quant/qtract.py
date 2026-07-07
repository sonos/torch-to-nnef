import filecmp
import operator
import platform
import tempfile
import typing as T
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target.base import InferenceTarget
from torch_to_nnef.inference_target.tract import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant.base import (
    QScalePerGroupF16,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)


class QTensorTract(QTensor):
    """All QTensorTract implementations."""


class QTensorTractScaleOnly(QTensorTract):
    """Tract data format it serializes to: Q4_0."""

    OFFLOAD_STATE_TAG = "torch_to_nnef.qtensor_tract_scale_only.v1"
    qscheme: QScalePerGroupF16  # type notation for mypy

    def __init__(
        self, *args, specific_machine: T.Optional[str] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.qscheme, QScalePerGroupF16), self.qscheme
        # tract limited support of packing
        assert self.qscheme.n_bits == 4, self.qscheme.n_bits
        self.decompressed_shape = self.decompress_to_u8().shape
        assert len(self.decompressed_shape) in [2, 3, 4], (
            self.decompressed_shape
        )
        assert reduce(operator.mul, self.decompressed_shape[1:]) % 32 == 0, (
            self.decompressed_shape
        )
        self.specific_machine = specific_machine

    def decompress(self):
        """Tract dequantization depends on hardware.

        Typically dequantization happen with ops in f16
        on ARM and f32 (scale directly casted) on others
        so we overwrite the function to be consistant
        with tract.

        """
        assert self.qscheme is not None
        decompress_u8 = self.u8_blob
        for u8_compressor in reversed(self.u8_compressors):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        if (self.specific_machine and "arm" in self.specific_machine) or (
            self.specific_machine is None and "arm" in platform.machine()
        ):
            return self.qscheme.dequantize(
                decompress_u8, target_dtype=torch.float16
            ).to(self.dequant_to_dtype)
        return self.qscheme.dequantize(
            decompress_u8, target_dtype=self.dequant_to_dtype
        )

    @classmethod
    def is_offload_state(cls, state) -> bool:
        return (
            isinstance(state, dict)
            and state.get("__t2n_qtensor_state__") == cls.OFFLOAD_STATE_TAG
        )

    def to_offload_state(self) -> T.Dict[str, T.Any]:
        return {
            "__t2n_qtensor_state__": self.OFFLOAD_STATE_TAG,
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "u8_blob": self.u8_blob,
            "u8_compressors": self.u8_compressors,
            "qscheme": self.qscheme,
            "dequant_to_dtype": self.dequant_to_dtype,
            "nnef_name": self.nnef_name,
            "decompressed_shape": tuple(self.decompressed_shape),
            "specific_machine": self.specific_machine,
        }

    @classmethod
    def from_offload_state(cls, state) -> "QTensorTractScaleOnly":
        assert cls.is_offload_state(state), state
        fp_placeholder = torch.empty(state["shape"], dtype=state["dtype"])
        qtensor = cls.__new__(
            cls,
            fp_placeholder,
            state["qscheme"],
            dequant_to_dtype=state["dequant_to_dtype"],
        )
        qtensor.u8_compressors = state["u8_compressors"]
        qtensor.u8_blob = state["u8_blob"]
        qtensor.qscheme = state["qscheme"]
        qtensor.dequant_to_dtype = state["dequant_to_dtype"]
        qtensor.nnef_name = state["nnef_name"]
        qtensor.requires_grad = False
        qtensor.decompressed_shape = state["decompressed_shape"]
        qtensor.specific_machine = state["specific_machine"]
        return qtensor

    @staticmethod
    def _build_binary_dat_header_for_shape(
        decompressed_shape, post_tract_21_11: bool = False
    ) -> bytes:
        if post_tract_21_11:
            item_type = DatBinHeader.TractCustomTypes.Q40
        else:
            item_type = DatBinHeader.TractCustomTypes.Q40_LEGACY
        return DatBinHeader.build_tract_qtensor(
            item_type, decompressed_shape
        ).to_bytes()

    def _build_binary_dat_header(self, post_tract_21_11: bool = False) -> bytes:
        return self._build_binary_dat_header_for_shape(
            self.decompressed_shape, post_tract_21_11
        )

    @staticmethod
    def _decompress_u8_blob(u8_blob, u8_compressors):
        decompress_u8 = u8_blob
        for u8_compressor in reversed(u8_compressors or []):
            decompress_u8 = u8_compressor.decompress(decompress_u8)
        return decompress_u8

    @staticmethod
    def _pack_binary_dat_values(
        tensor_flat: torch.Tensor, post_tract_21_11: bool
    ) -> np.ndarray:
        if post_tract_21_11:
            tensor_per_group = tensor_flat.reshape(-1, 2, 16)
            return (
                tensor_per_group[:, 0, :] | (tensor_per_group[:, 1, :] << 4)
            ).numpy()
        tensor_per_group = tensor_flat.reshape(-1, 16, 2)
        return (
            tensor_per_group[:, :, 0] | (tensor_per_group[:, :, 1] << 4)
        ).numpy()

    @classmethod
    def _iter_binary_dat_content_chunks(
        cls,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        post_tract_21_11: bool = False,
        chunk_groups: int = 1 << 20,
    ):
        n_bytes_per_group = 18
        tensor_flat = u8_blob.flatten()
        assert tensor_flat.numel() % 32 == 0, tensor_flat.shape
        group_count = tensor_flat.numel() // 32
        scale_bytes = (
            scale.reshape(group_count, 1)
            .contiguous()
            .numpy()
            .view(np.uint8)
            .reshape(group_count, 2)
        )

        for start in range(0, group_count, chunk_groups):
            end = min(start + chunk_groups, group_count)
            chunk_flat = tensor_flat[start * 32 : end * 32]
            b_vals = cls._pack_binary_dat_values(
                chunk_flat, post_tract_21_11
            ).view(np.uint8)
            b_all = np.empty((end - start, n_bytes_per_group), dtype=np.uint8)
            b_all[:, :2] = scale_bytes[start:end]
            b_all[:, 2:] = b_vals.reshape(end - start, 16)
            b_arr = b_all.tobytes()
            assert len(b_arr) % n_bytes_per_group == 0
            yield b_arr

    @classmethod
    def _build_binary_dat_content_from_parts(
        cls,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        post_tract_21_11: bool = False,
    ) -> bytes:
        return b"".join(
            cls._iter_binary_dat_content_chunks(
                u8_blob,
                scale,
                post_tract_21_11=post_tract_21_11,
            )
        )

    def _build_binary_dat_content(
        self, post_tract_21_11: bool = False
    ) -> bytes:
        return self._build_binary_dat_content_from_parts(
            self.decompress_to_u8(),
            self.qscheme.scale,
            post_tract_21_11=post_tract_21_11,
        )

    @classmethod
    def _write_binary_dat_content(cls, fh, u8_blob, scale, post_tract_21_11):
        for chunk in cls._iter_binary_dat_content_chunks(
            u8_blob, scale, post_tract_21_11=post_tract_21_11
        ):
            fh.write(chunk)

    @classmethod
    def _write_binary_dat(
        cls, path: Path, bin_header: bytes, bin_content: bytes
    ):
        with path.open("wb") as fh:
            fh.write(bin_header)
            fh.write(bin_content)

    @classmethod
    def _write_binary_dat_from_parts(
        cls,
        path: Path,
        bin_header: bytes,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        post_tract_21_11: bool,
    ):
        with path.open("wb") as fh:
            fh.write(bin_header)
            cls._write_binary_dat_content(fh, u8_blob, scale, post_tract_21_11)

    @classmethod
    def write_offload_state_in_file(
        cls,
        state,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: T.Optional[InferenceTarget] = None,
    ):
        assert cls.is_offload_state(state), state
        if inference_target is None:
            inference_target = TractNNEF.latest()
        path = Path(dirpath) / f"{label}.dat"
        if path.exists():
            with tempfile.TemporaryDirectory() as _td:
                td = Path(_td)
                cls.write_offload_state_in_file(
                    state, td, label, inference_target=inference_target
                )
                new_path = td / f"{label}.dat"
                assert new_path.exists(), new_path
                if filecmp.cmp(path, new_path):
                    return
            raise T2NErrorNotImplemented(
                "At least 2 variables in the NNEF graph, "
                f"share same Parameters: '{label}' but they try "
                "to use different data-type (likely quantization format). "
                "This variable collision as no resolution strategy yet."
            )
        assert isinstance(inference_target, TractNNEF), inference_target
        post_tract_21_11 = inference_target.version >= "0.21.11"
        bin_header = cls._build_binary_dat_header_for_shape(
            state["decompressed_shape"], post_tract_21_11
        )
        cls._write_binary_dat_from_parts(
            path,
            bin_header,
            cls._decompress_u8_blob(state["u8_blob"], state["u8_compressors"]),
            state["qscheme"].scale,
            post_tract_21_11,
        )

    def write_in_file(
        self,
        dirpath: T.Union[str, Path],
        label: str,
        inference_target: T.Optional[InferenceTarget] = None,
    ):
        if inference_target is None:
            inference_target = TractNNEF.latest()
        path = Path(dirpath) / f"{label}.dat"
        if path.exists():
            # already created a variable dump with that name.
            # check we would produce identical serialized data
            with tempfile.TemporaryDirectory() as _td:
                td = Path(_td)
                self.write_in_file(td, label, inference_target=inference_target)
                new_path = td / f"{label}.dat"
                assert new_path.exists(), new_path
                if filecmp.cmp(path, new_path):
                    return
            raise T2NErrorNotImplemented(
                "At least 2 variables in the NNEF graph, "
                f"share same Parameters: '{label}' but they try "
                "to use different data-type (likely quantization format). "
                "This variable collision as no resolution strategy yet."
            )
        assert not path.exists(), path
        assert isinstance(inference_target, TractNNEF), inference_target
        post_tract_21_11 = inference_target.version >= "0.21.11"
        bin_header = self._build_binary_dat_header(post_tract_21_11)
        self._write_binary_dat_from_parts(
            path,
            bin_header,
            self.decompress_to_u8(),
            self.qscheme.scale,
            post_tract_21_11,
        )


def fp_to_tract_q4_0_with_min_max_calibration(
    fp_tensor, percentile: float = 1.0
) -> QTensorTractScaleOnly:
    """Min-Max method to quantize float tensor to tract supported Q4_0."""
    q4_group_size = 32
    if isinstance(fp_tensor, torch.nn.Parameter):
        fp_tensor = fp_tensor.data

    if len(fp_tensor.shape) not in [2, 3, 4]:
        raise T2NErrorImpossibleQuantization(
            "tract Q4_0 does only support weight "
            f"of shape 2d, 3d or 4d but found {fp_tensor.shape}"
        )

    multiple_axis = 1
    if (
        reduce(operator.mul, fp_tensor.shape[multiple_axis:]) % q4_group_size
        != 0
    ):
        raise T2NErrorImpossibleQuantization(
            f"tract Q4_0 does only support weight with dim={multiple_axis} "
            f"divisible by {q4_group_size} but "
            f"found {fp_tensor.shape[multiple_axis:]}"
        )
    with torch.no_grad():
        q_scheme = qscale_per_group_f16_min_max_calibration(
            fp_tensor, n_bits=4, group_size=q4_group_size, percentile=percentile
        )
        return QTensorTractScaleOnly(
            fp_tensor,
            qscheme=q_scheme,
            dequant_to_dtype=fp_tensor.dtype,
        )
