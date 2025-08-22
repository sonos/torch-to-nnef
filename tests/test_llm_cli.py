import tempfile
from pathlib import Path
import itertools

import numpy as np
import pytest
from torch_to_nnef.compress import DEFAULT_COMPRESSION
from torch_to_nnef.inference_target.tract import TractCheckTolerance
from torch_to_nnef.llm_tract.config import ExportDirStruct, SmolSlugs
from torch_to_nnef.llm_tract.exporter import dump_llm
from torch_to_nnef.torch_graph.ir_naming import VariableNamingScheme

from .utils import IS_DEBUG, TRACT_INFERENCES_TO_TESTS_APPROX

try:
    from torch_to_nnef.llm_tract.config import LlamaSlugs
    from torch_to_nnef.llm_tract.exporter import LLMExporter

    LLMExporter.load(LlamaSlugs.DUMMY.value)
except ImportError as exp:
    print("disable test_llm_cli because:", exp)
    pytest.skip(
        reason="disabled since import of transformers failed in some way",
        allow_module_level=True,
    )

# test all tract supported version
SUPPORT_LLM_CLI_OPTS = [
    {"tract_specific_version": _.version}
    for _ in TRACT_INFERENCES_TO_TESTS_APPROX
    if _.version > "0.21.5"
]

# test all compression version
SUPPORT_LLM_CLI_OPTS += [{"compression_method": _} for _ in DEFAULT_COMPRESSION]

# test upcasting
SUPPORT_LLM_CLI_OPTS += [
    {
        "force_module_dtype": "f16",
        "force_f32_attention": True,
        "tract_check_io_tolerance": TractCheckTolerance.SUPER,
    },
    {
        "force_module_dtype": "f16",
        "force_f32_linear_accumulator": True,
        "tract_check_io_tolerance": TractCheckTolerance.SUPER,
    },
    {
        "force_module_dtype": "f16",
        "force_f32_normalization": True,
        "tract_check_io_tolerance": TractCheckTolerance.SUPER,
    },
]

# test device-map
SUPPORT_LLM_CLI_OPTS += [
    # {
    #     "device_map": "auto",
    # },
    # {
    #     "device_map": "t2n_auto",
    # },
    {
        "device_map": "t2n_offload_disk",
    },
]

# ensure dump with tokenizer works
SUPPORT_LLM_CLI_OPTS += [{"dump_with_tokenizer_and_conf": True}]
SUPPORT_LLM_CLI_OPTS += [{"export_dir_struct": ExportDirStruct.FLAT}]
SUPPORT_LLM_CLI_OPTS += [{"sample_generation_total_size": 8}]
SUPPORT_LLM_CLI_OPTS += [{"naming_scheme": VariableNamingScheme.NUMERIC}]


BASE_LLM_SLUGS_TO_TEST = [
    LlamaSlugs.DUMMY.value,
]


TEST_SPECS = list(
    itertools.product(BASE_LLM_SLUGS_TO_TEST, SUPPORT_LLM_CLI_OPTS)
)


@pytest.mark.parametrize(
    "model_slug,cli_kwargs",
    TEST_SPECS,
    ids=[
        ts[0] + "," + ",".join(f"{k}={v}" for k, v in ts[1].items())
        for ts in TEST_SPECS
    ],
)
def test_llama_export_from_llmexporter(model_slug, cli_kwargs):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        export_dirpath = td / "dump_here"
        dbg_path = (
            Path.cwd()
            / "failed_tests"
            / "test_llama_export_io_npz_from_LLMExporter"
        )
        # Fixed behavior change in: huggingface/transformers 4.53.0: Jun 26
        dump_llm(
            model_slug,
            export_dirpath=export_dirpath,
            debug_bundle_path=(dbg_path if IS_DEBUG else None),
            **cli_kwargs,
        )


def test_llama_export_io_npz():
    llm_exporter = LLMExporter.load(LlamaSlugs.DUMMY.value)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        total_tokens = 6
        p_npz, pp_npz, tg_npz = llm_exporter.dump_all_io_npz_kind(
            td, size=total_tokens
        )
        pdic = dict(**np.load(p_npz))
        ppdic = dict(**np.load(pp_npz))
        tgdic = dict(**np.load(tg_npz))
        token_gens_and_dic_list = [
            (total_tokens, pdic),
            (total_tokens // 2, ppdic),
            (1, tgdic),
        ]
        for n_tokens, dic in token_gens_and_dic_list:
            assert dic["input_ids"].shape == (1, n_tokens)
            assert dic["outputs"].shape == (1, 1, 32000)
            assert dic["out_cache_key_0"].shape == (1, 2, total_tokens, 4)
            assert dic["out_cache_value_0"].shape == (1, 2, total_tokens, 4)
