import os
import struct
import typing as T
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from nnef.binary import _fromfile, _numpy_dtype_make

from torch_to_nnef.dtypes import NUMPY_TO_TORCH_DTYPE
from torch_to_nnef.exceptions import T2NErrorMisuse


class DatBinHeader:
    """DatBinHeader.

    Parse and serialize .dat NNEF binary header.

    This class handles parsing and serializing the binary header for
    NNEF tensor files.
    """

    MAX_TENSOR_RANK = 8  # Maximum supported tensor rank

    TRACT_ITEM_TYPE_VENDOR = struct.pack(
        "h", struct.unpack("B", b"T")[0] << 8 | struct.unpack("B", b"R")[0]
    )  # Vendor code for tract custom types (hex 5254)

    class TractCustomTypes(str, Enum):
        """TractCustomTypes.

        Custom tract quantisation types used in NNEF headers.

        Values are the little endian hex of the tract `.dat` item type.

        Note:
            tract 0.23.10 reassigned item type `0x3080` from `Q8_1` to
            `Q8_0`. Both meanings share the header value "8030", so
            `Q81_LEGACY` is an Enum alias of `Q80`: both names stay
            usable but are the same member, and decoding a "8030" header
            returns `Q80`. The header alone can not tell them apart: it
            depends on the targeted tract version (or on the payload
            size: 34 bytes per block of 32 for `Q8_0`, 36 for `Q8_1`).
        """

        Q20T = "2030"
        Q40 = "4030"
        Q40_LEGACY = "4020"
        Q80 = "8030"  # 0x3080: tract >= 0.23.10
        Q81 = "8130"  # 0x3081: tract >= 0.23.10
        # alias of Q80 (same value), 0x3080: Q8_1 for tract < 0.23.10
        Q81_LEGACY = "8030"
        Q4K = "4031"  # 0x3140: tract >= 0.23.10
        Q5K = "5031"  # 0x3150: tract >= 0.23.10
        Q6K = "6031"  # 0x3160: tract >= 0.23.10

    def __init__(
        self,
        data_size_bytes: int,
        rank: int,
        dims: T.List[int],
        item_type: str,
        item_type_vendor: bytes,
        version_maj: int = 1,
        version_min: int = 0,
        item_type_params_deprecated: T.Optional[T.List[int]] = None,
        padding: T.Optional[T.List[int]] = None,
        bits_per_item: int = 32,
        magic: str = "4eef",
    ):
        """DatBinHeader initialization.

        Args:
            data_size_bytes (int): Size of the data section in bytes.
            rank (int): Rank of the tensor.
            dims (List[int]): Dimensions of the tensor.
            item_type (str): Hex string representing the item type.
            item_type_vendor (bytes): Vendor specific code for the item type.
            version_maj (int, optional): Major version number.
            version_min (int, optional): Minor version number.
            item_type_params_deprecated (Optional[List[int]], optional):
                Deprecated item type parameters.
            padding (Optional[List[int]], optional): Padding values.
            bits_per_item (int, optional): Bits per item.
            magic (str, optional): Magic string for the header.
        """
        assert isinstance(magic, str) and len(magic) == 4
        self.magic = magic
        assert isinstance(version_maj, int)
        self.version_maj = version_maj
        assert isinstance(version_min, int)
        self.version_min = version_min
        assert isinstance(data_size_bytes, int)
        self.data_size_bytes = data_size_bytes
        assert isinstance(rank, int)
        self.rank = rank
        assert isinstance(dims, list) and all(isinstance(d, int) for d in dims)
        self.dims = dims
        assert isinstance(bits_per_item, int)
        self.bits_per_item = bits_per_item
        assert isinstance(item_type, str) and len(item_type) == 4
        self.item_type = item_type
        assert (
            isinstance(item_type_vendor, bytes) and len(item_type_vendor) == 2
        )
        self.item_type_vendor = item_type_vendor

        assert item_type_params_deprecated is None or (
            len(item_type_params_deprecated) == 32
            and all(isinstance(i, int) for i in item_type_params_deprecated)
        )
        self.item_type_params_deprecated = item_type_params_deprecated

        assert padding is None or (
            len(padding) == 11 and all(isinstance(i, int) for i in padding)
        )
        self.padding = padding

    @classmethod
    def build_tract_qtensor(
        cls,
        item_type: TractCustomTypes,
        shape: T.Tuple[int, ...],
        *,
        data_size_bytes: int,
    ) -> "DatBinHeader":
        """Build a binary header for tract custom types.

        Args:
            item_type (TractCustomTypes): Tract custom quantisation type.
            shape (Tuple[int]): Shape of the tensor.
            data_size_bytes (int): serialized data size, computed by each
                qtensor format: an item type alone does not always tell it
                (like "8030": Q8_0 or legacy Q8_1).

        Returns:
            DatBinHeader: Constructed header instance.
        """
        return DatBinHeader(
            data_size_bytes=data_size_bytes,
            bits_per_item=2**32 - 1,
            rank=len(shape),
            dims=list(shape),
            item_type_vendor=cls.TRACT_ITEM_TYPE_VENDOR,
            item_type=item_type,
        )

    def to_bytes(self):
        """Serialize the binary header into bytes.

        Returns:
            bytes: Serialized header data.
        """
        b_arr = bytearray(b"")
        # magic: [u8; 2],
        b_arr.extend(bytes.fromhex(self.magic))
        # version_maj: u8,
        b_arr.extend(struct.pack("B", self.version_maj))
        # version_min: u8,
        b_arr.extend(struct.pack("B", self.version_min))
        # data_size_bytes: u32,
        b_arr.extend(struct.pack("I", self.data_size_bytes))
        # rank: u32
        b_arr.extend(struct.pack("I", self.rank))
        # dims: [u32; 8],
        sh = list(self.dims)
        sh += [0] * (8 - len(sh))
        b_arr.extend(struct.pack("8I", *sh))
        b_arr.extend(struct.pack("I", self.bits_per_item))
        # item_type: u16,
        b_arr.extend(bytes.fromhex(self.item_type))
        # item_type_vendor: u16,
        b_arr.extend(self.item_type_vendor)
        # item_type_params_deprecated: [u8; 32],
        item_type_params_deprecated = self.item_type_params_deprecated or (
            [0] * 32
        )
        if len(item_type_params_deprecated) != 32:
            raise T2NErrorMisuse(
                "item_type_params_deprecated must be array of 32 int"
            )
        b_arr.extend(struct.pack("32B", *item_type_params_deprecated))
        # padding: [u32; 11],
        padding = self.padding or ([0] * 11)
        if len(padding) != 11:
            raise T2NErrorMisuse("padding must be array of 11 int")
        b_arr.extend(struct.pack("11I", *padding))
        binheader = bytes(b_arr)
        assert len(binheader) == 128, len(binheader)
        return binheader

    @property
    def torch_dtype_or_custom(self) -> T.Union[torch.dtype, "TractCustomTypes"]:
        """Return the torch dtype or custom tract type based on the header.

        This property interprets the ``item_type_vendor`` and ``item_type``
        fields to provide a convenient ``torch.dtype`` object
        for standard types or a ``TractCustomTypes`` enum member
        for tract-specific quantised formats.
        """
        if self.item_type_vendor == self.TRACT_ITEM_TYPE_VENDOR:
            return self.TractCustomTypes(self.item_type)
        return NUMPY_TO_TORCH_DTYPE[
            _numpy_dtype_make(int(self.item_type, 16), self.bits_per_item)
        ]

    @classmethod
    def from_dat_file(cls, file) -> "DatBinHeader":
        if isinstance(file, str):
            raise T2NErrorMisuse(
                "file parameter must be a file object not a file name"
            )

        [magic1, magic2, major, minor] = _fromfile(
            file, dtype=np.uint8, count=4
        )
        if magic1 != 0x4E or magic2 != 0xEF:
            raise T2NErrorMisuse("not a valid NNEF file")

        [data_length, rank] = _fromfile(file, dtype=np.uint32, count=2)

        if file.seekable():
            header_size = 128
            file_size = os.fstat(file.fileno()).st_size
            if file_size != header_size + data_length:
                raise T2NErrorMisuse(
                    "invalid tensor file; size does not match header info"
                )

        if rank > cls.MAX_TENSOR_RANK:
            raise T2NErrorMisuse(
                "tensor rank exceeds maximum possible "
                f"value of {cls.MAX_TENSOR_RANK}"
            )

        shape = _fromfile(file, dtype=np.uint32, count=cls.MAX_TENSOR_RANK)
        shape = shape[:rank]

        [bits, full_item_type] = _fromfile(file, dtype=np.uint32, count=2)
        # item_type_params_deprecated: [u8; 32],
        reserved = _fromfile(file, dtype=np.uint32, count=19).tobytes()
        b_item_type_params_deprecated = reserved[:32]
        b_padding = reserved[32:]
        b_full_item_type = full_item_type.tobytes()
        b_vendor_code = b_full_item_type[2:]
        item_type = b_full_item_type[:2]
        return DatBinHeader(
            bits_per_item=bits.tolist(),
            data_size_bytes=data_length.tolist(),
            dims=shape.tolist(),
            item_type=item_type.hex(),
            item_type_vendor=b_vendor_code,
            rank=rank.tolist(),
            version_maj=major.tolist(),
            version_min=minor.tolist(),
            item_type_params_deprecated=list(
                struct.unpack("32B", b_item_type_params_deprecated)
            ),
            padding=list(struct.unpack("11I", b_padding)),
            magic=(magic1.tobytes() + magic2.tobytes()).hex(),
        )

    @classmethod
    def from_dat(cls, filepath: T.Union[Path, str]) -> "DatBinHeader":
        if isinstance(filepath, str):
            filepath = Path(filepath)
        with filepath.open("rb") as fh:
            return cls.from_dat_file(fh)
