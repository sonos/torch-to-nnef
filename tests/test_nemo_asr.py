import tempfile
from pathlib import Path

import pytest
import torch

from torch_to_nnef.utils import SemanticVersion

from .utils import (
    TRACT_INFERENCES_TO_TESTS_APPROX,
    check_model_io_test,
    cond_tract_gt_0_22_0,
)

try:
    import nemo
    import nemo.collections.asr as nemo_asr  # noqa: F401

    from torch_to_nnef.inference_target.tract import TractCheckTolerance
    from torch_to_nnef.nemo_tract import (
        PARAKEET_V3_SLUG,
        iter_export_params_for_generic_nemo_asr_model,
    )
    from torch_to_nnef.nemo_tract.axis_registry import (
        AxisSymbolRegistry,
        load_axis_symbol_registry,
    )
    from torch_to_nnef.nemo_tract.config import (
        CompressionConfig,
        NemoExportConfig,
        SubnetSelectionConfig,
    )
    from torch_to_nnef.nemo_tract.constants import (
        LENGTH_INPUT_NAMES,
        LENGTH_OUTPUT_NAMES,
    )
    from torch_to_nnef.nemo_tract.export import export_nemo_from_model
    from torch_to_nnef.nemo_tract.model_loader import (
        FAST_CONFORMER_TDT_LARGE,
        MARBLENET_VAD,
        NEMOTRON_0_6B,
        QUARTZNET,
    )
    from torch_to_nnef.nemo_tract.provider import NemoProvider
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


def _load_asr_model(model_slug):
    """Load a NeMo ASR model, trying ASRModel first then classification."""
    try:
        return nemo_asr.models.ASRModel.from_pretrained(
            model_name=model_slug, map_location=torch.device("cpu")
        )
    except FileNotFoundError:
        pass
    return nemo_asr.models.EncDecClassificationModel.from_pretrained(
        model_name=model_slug, map_location=torch.device("cpu")
    )


def _build_axis_registry(
    asr_model, inference_target, cfg: NemoExportConfig, shape_config=None
):
    """Build axis registry from model discovery or shape config file."""
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


# ---------------------------------------------------------------------------
# Legacy helper: per-subnet check via iter_export_params (no axis registry)
# ---------------------------------------------------------------------------
def check_export_asr_model_legacy(
    model_slug,
    skip_preprocessor=False,
    check_io_tolerance=None,
):
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)
    if check_io_tolerance is not None:
        inference_target = inference_target.with_check_io_tolerance(
            check_io_tolerance
        )
    asr_model = _load_asr_model(model_slug)

    # Build default axis registry so batch symbols are unified
    # (e.g. AUDIO_SIGNAL__BATCH and LENGTH__BATCH both become BATCH).
    # Without this, tract can't prove reshape symbol equalities.
    provider = NemoProvider(
        inference_target=inference_target,
        skip_preprocessor=skip_preprocessor,
        split_joint_decoder=False,
        float_dtype=torch.float32,
    )
    raw_sigs = provider.discover_signatures(asr_model, Stage.RAW)
    default_reg = dump_registry_from_signatures(raw_sigs)
    axis_reg = tie_batch_symbols_in_registry(default_reg)

    for export_params in iter_export_params_for_generic_nemo_asr_model(
        asr_model,
        inference_target,
        skip_preprocessor=skip_preprocessor,
        axis_registry=axis_reg,
    ):
        check_model_io_test(
            model=export_params.model,
            test_input=export_params.test_input,
            inference_target=export_params.inference_target,
            input_names=export_params.input_names,
            output_names=export_params.output_names,
            custom_extensions=export_params.custom_extensions,
            check_io_names_qte_match=False,
        )


# ---------------------------------------------------------------------------
# Public API helper: export_nemo_from_model + NemoExportConfig
# ---------------------------------------------------------------------------
def check_export_asr_model(
    model_slug,
    cfg: NemoExportConfig = None,
    shape_config: Path = None,
):
    """Export a NeMo ASR model using the public programmatic API."""
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    if cfg is None:
        cfg = NemoExportConfig()

    asr_model = _load_asr_model(model_slug)
    asr_model.eval()

    axis_reg = _build_axis_registry(
        asr_model, inference_target, cfg, shape_config=shape_config
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = Path(tmpdir) / "export"
        export_dir.mkdir()
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=export_dir,
            axis_reg=axis_reg,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Default export for each model (legacy per-subnet path)
# ---------------------------------------------------------------------------
def test_nemo_asr_parakeet_v3():
    check_export_asr_model_legacy(PARAKEET_V3_SLUG)


@pytest.mark.ci_skip
@pytest.mark.parametrize(
    "model_slug, check_io_tolerance",
    [
        pytest.param(
            NEMOTRON_0_6B,
            TractCheckTolerance.APPROXIMATE,
            id=NEMOTRON_0_6B,
        ),
        pytest.param(
            QUARTZNET,
            TractCheckTolerance.VERY,
            id=QUARTZNET,
        ),
        pytest.param(
            MARBLENET_VAD,
            TractCheckTolerance.APPROXIMATE,
            id=MARBLENET_VAD,
        ),
        pytest.param(
            FAST_CONFORMER_TDT_LARGE,
            TractCheckTolerance.APPROXIMATE,
            id=FAST_CONFORMER_TDT_LARGE,
        ),
    ],
)
def test_nemo_model_export(model_slug, check_io_tolerance):
    check_export_asr_model_legacy(
        model_slug, check_io_tolerance=check_io_tolerance
    )


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
            marks=pytest.mark.skip(reason="float16 export under investigation"),
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
            marks=pytest.mark.xfail(
                reason="quantized weights cause tract IO mismatch"
            ),
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
    """Export VAD with batch-collapsed dims, bound lengths.

    Builds a collapsed registry from the discovered default,
    exercises BoundaryAdapter (collapse + bind + output filter).
    """
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    cfg = NemoExportConfig()
    asr_model = _load_asr_model(MARBLENET_VAD)
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
    # corresponding signal input in the same subnet.
    bind_to_dim = {}
    for qname, _axes in default_reg.symbols_per_input.items():
        subnet, _, inp_name = qname.rpartition(".")
        if inp_name not in LENGTH_INPUT_NAMES:
            continue
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
        q
        for q in default_reg.symbols_per_input
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
        export_dir = Path(tmpdir) / "export"
        export_dir.mkdir()
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=export_dir,
            axis_reg=collapsed_reg,
            cfg=cfg,
        )


# ---------------------------------------------------------------------------
# Dry-run: dump shape config template and verify round-trip
# ---------------------------------------------------------------------------
@pytest.mark.ci_skip
def test_nemo_dump_shape_config_dry_run():
    """Dump shape config YAML for VAD and verify round-trip."""
    inference_target = TRACT_INFERENCES_TO_TESTS_APPROX[0]
    _skip_unless_nemo_tract(inference_target)

    asr_model = _load_asr_model(MARBLENET_VAD)
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
        export_dir = Path(tmpdir) / "export"
        export_dir.mkdir()
        cfg = NemoExportConfig()
        export_nemo_from_model(
            model=asr_model,
            target=inference_target,
            export_dir=export_dir,
            axis_reg=reloaded_reg,
            cfg=cfg,
        )
