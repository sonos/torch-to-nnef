import tempfile
from pathlib import Path

import pytest
import torch

from torch_to_nnef.utils import SemanticVersion

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    cond_tract_gt_0_22_0,
)

try:
    import nemo
    import nemo.collections.asr as nemo_asr  # noqa: F401

    from torch_to_nnef.nemo_tract.axis_registry import (
        AxisSymbolRegistry,
        load_axis_symbol_registry,
    )
    from torch_to_nnef.nemo_tract.config import (
        CompressionConfig,
        NemoExportConfig,
        SubnetSelectionConfig,
    )
    from torch_to_nnef.nemo_tract.export import export_nemo_from_model
    from torch_to_nnef.nemo_tract.model_loader import (
        FAST_CONFORMER_TDT_LARGE,
        MARBLENET_VAD,
        NEMOTRON_0_6B,
        PARAKEET_V3_SLUG,
        QUARTZNET,
    )
    from torch_to_nnef.nemo_tract.provider import NemoProvider
    from torch_to_nnef.nemo_tract.constants import (
        LENGTH_INPUT_NAMES,
        LENGTH_OUTPUT_NAMES,
    )
    from torch_to_nnef.nemo_tract.registry_utils import (
        dump_registry_from_signatures,
        tie_batch_symbols_in_registry,
        validate_registry_against_signatures,
    )
    from torch_to_nnef.remodeler import Stage, save_config
    from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme
except ImportError as exp:
    print("disable test_nemo because:", exp)
    pytest.skip(
        reason="disabled since import of nemo_asr failed in some way",
        allow_module_level=True,
    )


ASSETS_DIR = Path(__file__).parent / "assets"


def _skip_unless_nemo_tract(inference_target):
    if not (
        cond_tract_gt_0_22_0(inference_target)
        and SemanticVersion.from_str(nemo.__version__) > "2.1.0"
    ):
        pytest.skip(
            "skip test for tract>0.22.0 && nemo>2.1 "
            "since tract needs fix & features"
        )


def _build_axis_registry(
    asr_model, inference_target, cfg: NemoExportConfig, shape_config=None
):
    """Build axis registry from model discovery or shape config file.

    Mirrors the logic in cli._build_axis_registry for test use.
    """
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=cfg.subnet.skip_preprocessor,
        split_joint_decoder=cfg.subnet.split_joint_decoder,
        float_dtype=(
            torch.float16 if cfg.data_type == "float16" else torch.float32
        ),
        only_subnets=cfg.subnet.only_subnets,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    if shape_config is None:
        default_reg = dump_registry_from_signatures(raw_sigs)
        return tie_batch_symbols_in_registry(default_reg)
    axis_reg = load_axis_symbol_registry(shape_config)
    validate_registry_against_signatures(raw_sigs, axis_reg)
    return axis_reg


def check_export_asr_model(
    model_slug,
    cfg: NemoExportConfig = None,
    shape_config: Path = None,
):
    """Export a NeMo ASR model using the public programmatic API.

    Uses NemoExportConfig + export_nemo_from_model to validate the full
    export pipeline including axis registry building and tract IO check.
    """
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    if cfg is None:
        cfg = NemoExportConfig()

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_slug, map_location="cpu"
    )
    asr_model.eval()

    axis_reg = _build_axis_registry(
        asr_model, inference_target, cfg, shape_config=shape_config
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=Path(tmpdir),
            axis_reg=axis_reg,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Existing: default export for each model
# ---------------------------------------------------------------------------
@pytest.mark.ci_skip
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(PARAKEET_V3_SLUG, id=PARAKEET_V3_SLUG),
        pytest.param(NEMOTRON_0_6B, id=NEMOTRON_0_6B),
        pytest.param(QUARTZNET, id=QUARTZNET),
        pytest.param(MARBLENET_VAD, id=MARBLENET_VAD),
        pytest.param(FAST_CONFORMER_TDT_LARGE, id=FAST_CONFORMER_TDT_LARGE),
    ],
)
def test_nemo_model_export(model):
    check_export_asr_model(model)


# ---------------------------------------------------------------------------
# Config variant tests — VAD-heavy for fast iteration
# ---------------------------------------------------------------------------
@pytest.mark.ci_skip
@pytest.mark.parametrize(
    "model_slug, cfg",
    [
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                subnet=SubnetSelectionConfig(skip_preprocessor=True),
            ),
            id="vad-skip-preprocessor",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(data_type="float16"),
            id="vad-float16",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                subnet=SubnetSelectionConfig(only_subnets=["preprocessor"]),
            ),
            id="vad-only-preprocessor",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                subnet=SubnetSelectionConfig(only_subnets=["encoder"]),
            ),
            id="vad-only-encoder",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                compression=CompressionConfig(compress_method="min_max_q4_0"),
            ),
            id="vad-quant-q4_0",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                compression=CompressionConfig(
                    compress_method="min_max_q4_0_all",
                ),
            ),
            id="vad-quant-q4_0-all",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(naming_scheme=VariableNamingScheme.RAW),
            id="vad-naming-raw",
        ),
        pytest.param(
            MARBLENET_VAD,
            NemoExportConfig(
                compression=CompressionConfig(dump_checked_io=True),
            ),
            id="vad-dump-checked-io",
        ),
        pytest.param(
            FAST_CONFORMER_TDT_LARGE,
            NemoExportConfig(
                subnet=SubnetSelectionConfig(split_joint_decoder=True),
            ),
            id="fast-conformer-split-decoder",
        ),
    ],
)
def test_nemo_export_config_variants(model_slug, cfg):
    check_export_asr_model(model_slug, cfg=cfg)


# ---------------------------------------------------------------------------
# Shape config tests — axis registry loaded from YAML
# ---------------------------------------------------------------------------
@pytest.mark.ci_skip
@pytest.mark.parametrize(
    "model_slug, shape_config, extra_cfg",
    [
        pytest.param(
            PARAKEET_V3_SLUG,
            ASSETS_DIR / "shapes.parakeet.yaml",
            NemoExportConfig(
                subnet=SubnetSelectionConfig(split_joint_decoder=True),
            ),
            id="parakeet-shapes-full",
        ),
        pytest.param(
            MARBLENET_VAD,
            ASSETS_DIR / "shapes.marblenet.collapsed.yaml",
            NemoExportConfig(),
            id="vad-collapsed",
        ),
    ],
)
def test_nemo_export_shape_config(model_slug, shape_config, extra_cfg):
    check_export_asr_model(model_slug, cfg=extra_cfg, shape_config=shape_config)


@pytest.mark.ci_skip
def test_nemo_export_vad_batch_collapsed():
    """Export VAD with batch-collapsed dims, bound scalar lengths, stripped outputs.

    Programmatically builds a collapsed registry from the discovered default,
    exercises the full BoundaryAdapter pipeline (collapse + bind + output filter).
    """
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    cfg = NemoExportConfig()
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=MARBLENET_VAD, map_location="cpu"
    )
    asr_model.eval()

    # Discover default registry
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=False,
        split_joint_decoder=False,
        float_dtype=torch.float32,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    default_reg = dump_registry_from_signatures(raw_sigs)
    default_reg = tie_batch_symbols_in_registry(default_reg)

    # Build collapsed registry: collapse all BATCH dims
    collapse_dims = {}
    for qname, axes in default_reg.symbols_per_input.items():
        batch_syms = [
            str(s) for s in axes.values() if str(s).upper().endswith("__BATCH")
        ]
        if batch_syms:
            collapse_dims[qname] = batch_syms

    # Bind scalar: for each length input, bind to the time dim of the
    # corresponding signal input in the same subnet.  After batch collapse
    # the length becomes a scalar derived from the signal's time extent.
    bind_to_dim = {}
    for qname, axes in default_reg.symbols_per_input.items():
        subnet, _, inp_name = qname.rpartition(".")
        if inp_name not in LENGTH_INPUT_NAMES:
            continue
        # Find a sibling input in the same subnet that has a TIME symbol
        for sibling_q, sibling_axes in default_reg.symbols_per_input.items():
            if not sibling_q.startswith(f"{subnet}.") or sibling_q == qname:
                continue
            time_syms = [
                (str(s), sibling_q.split(".", 1)[1])
                for s in sibling_axes.values()
                if str(s).upper().endswith("__TIME")
            ]
            if time_syms:
                sym, sibling_name = time_syms[0]
                bind_to_dim[qname] = f"{sibling_name}.{sym}"
                break

    # Verify every length input got a bind
    length_inputs = [
        q for q in default_reg.symbols_per_input
        if q.rpartition(".")[2] in LENGTH_INPUT_NAMES
    ]
    assert set(length_inputs) <= set(bind_to_dim), (
        f"some length inputs have no bind: "
        f"{set(length_inputs) - set(bind_to_dim)}"
    )

    # Strip length outputs: keep only non-length outputs per subnet
    outputs_keep = {}
    for subnet, keep_list in default_reg.outputs_keep_per_subnet.items():
        filtered = [o for o in keep_list if o not in LENGTH_OUTPUT_NAMES]
        if filtered and len(filtered) < len(keep_list):
            outputs_keep[subnet] = filtered
        else:
            outputs_keep[subnet] = keep_list

    collapsed_reg = AxisSymbolRegistry(
        symbols_per_input=default_reg.symbols_per_input,
        rank_per_input=default_reg.rank_per_input,
        bind_to_dim=bind_to_dim,
        input_collapse_dims=collapse_dims,
        renamed_symbols_per_subnet=default_reg.renamed_symbols_per_subnet,
        outputs_keep_per_subnet=outputs_keep,
        original_shape_per_input=default_reg.original_shape_per_input,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=Path(tmpdir),
            axis_reg=collapsed_reg,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Dry-run: dump shape config template and verify round-trip
# ---------------------------------------------------------------------------
@pytest.mark.ci_skip
def test_nemo_dump_shape_config_dry_run():
    """Dump a shape config YAML for VAD via dry-run and verify it round-trips.

    1. Discover signatures for VAD model
    2. Build default registry
    3. Serialize to YAML via save_config
    4. Re-load with load_axis_symbol_registry
    5. Validate against the same signatures
    """
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=MARBLENET_VAD, map_location="cpu"
    )
    asr_model.eval()

    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=False,
        split_joint_decoder=False,
        float_dtype=torch.float32,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    default_reg = dump_registry_from_signatures(raw_sigs)
    default_reg = tie_batch_symbols_in_registry(default_reg)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "shapes.vad.yaml"
        save_config(config_path, default_reg)

        # Verify the YAML is parseable and round-trips
        reloaded_reg = load_axis_symbol_registry(config_path)
        validate_registry_against_signatures(raw_sigs, reloaded_reg)

        # Verify the reloaded registry can be used for export
        cfg = NemoExportConfig()
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=Path(tmpdir) / "export",
            axis_reg=reloaded_reg,
            cfg=cfg,
        )
