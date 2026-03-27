"""Configuration dataclasses for NeMo→NNEF export.

Defines lightweight, structured containers for both the CLI
(``NemoTractConfig`` and sub-configs) and the programmatic API
(``NemoExportConfig``). Kept separate from runtime logic to minimise
dependencies and improve clarity.

Lightweight enums that other modules need at import time (e.g.
``InspectFormat``) live here to avoid circular imports.
"""

from __future__ import annotations

import typing as T
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from torch_to_nnef.compress import DEFAULT_COMPRESSION_REGISTRY
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme


class InspectFormat(Enum):
    """Output formats for inspection results."""

    HUMAN = "human"
    JSON = "json"
    HUMAN_RICH = "human-rich"


@dataclass
class ModelSelectionConfig:
    """Model selection arguments."""

    model_slug: str = "*"
    model_path: T.Optional[str] = None


@dataclass
class OutputConfig:
    """Output destination arguments."""

    export_dir: Path = Path("")


@dataclass
class SubnetSelectionConfig:
    """Subnet filtering and composition."""

    skip_preprocessor: bool = False
    split_joint_decoder: bool = False
    only_subnets: T.Optional[T.List[str]] = None


@dataclass
class SdpaConfig:
    """SDPA-related export toggles."""

    force_sdpa_pytorch: bool = False
    tract_reify_sdpa: bool = False


@dataclass
class TractBinaryConfig:
    """Tract binary selection and IO-check policy."""

    tract_specific_version: T.Optional[str] = None
    tract_specific_path: T.Optional[str] = None
    tract_check_io_tolerance: T.Union[str, TractCheckTolerance] = (
        TractCheckTolerance.APPROXIMATE.value
    )


@dataclass
class NamingPrecisionConfig:
    """Naming scheme and precision."""

    naming_scheme: VariableNamingScheme = (
        VariableNamingScheme.NATURAL_VERBOSE_CAMEL
    )
    data_type: str = "float32"


@dataclass
class CompressionConfig:
    """Compression options for output artifacts."""

    compress_registry: str = DEFAULT_COMPRESSION_REGISTRY
    compress_method: T.Optional[str] = None
    dump_checked_io: bool = False


@dataclass
class InspectionConfig:
    """Inspection and dry-run options."""

    inspect_signatures: bool = False
    inspect_stages: T.Optional[T.List[str]] = None
    inspect_format: InspectFormat = InspectFormat.HUMAN_RICH
    inspect_output: T.Optional[Path] = None
    inspect_diff: bool = False
    shape_config: T.Optional[Path] = None
    dump_shape_config: T.Optional[Path] = None
    dry_run: bool = False


@dataclass
class LogConfig:
    """Logging verbosity."""

    verbose: bool = False


@dataclass
class NemoTractConfig:
    """Container for CLI arguments for NeMo→NNEF export.

    Attributes:
        model: Model selection arguments.
        output: Export destination arguments.
        subnet: Subnet filtering arguments.
        sdpa: SDPA-related export toggles.
        tract: Tract binary path/version and IO-check.
        naming: Naming scheme and precision.
        compression: Compression options.
        inspect: Inspection and dry-run options.
        log: Logging verbosity.
    """

    model: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    subnet: SubnetSelectionConfig = field(default_factory=SubnetSelectionConfig)
    sdpa: SdpaConfig = field(default_factory=SdpaConfig)
    tract: TractBinaryConfig = field(default_factory=TractBinaryConfig)
    naming: NamingPrecisionConfig = field(default_factory=NamingPrecisionConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    inspect: InspectionConfig = field(default_factory=InspectionConfig)
    log: LogConfig = field(default_factory=LogConfig)


@dataclass
class NemoExportConfig:
    """Configuration for the export_nemo_from_model programmatic API.

    All fields have sensible defaults so callers only specify what they
    actually need to override.

    Attributes:
        subnet: Subnet filtering and composition.
        compression: Compression options for NNEF artifacts.
        naming_scheme: NNEF variable naming scheme.
        data_type: Export dtype policy: float32 | float16 | mixed.
        pretrained_name: Human-readable label embedded in export metadata.
    """

    subnet: SubnetSelectionConfig = field(default_factory=SubnetSelectionConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    naming_scheme: VariableNamingScheme = (
        VariableNamingScheme.NATURAL_VERBOSE_CAMEL
    )
    data_type: str = "float32"
    pretrained_name: str = ""
