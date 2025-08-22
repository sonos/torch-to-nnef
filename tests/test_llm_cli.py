import tempfile
from pathlib import Path
import itertools

import numpy as np
import pytest

from .utils import IS_DEBUG, TRACT_INFERENCES_TO_TESTS_APPROX

try:
    from torch_to_nnef.llm_tract.config import LlamaSLugs
    from torch_to_nnef.llm_tract.exporter import LLMExporter

    LLMExporter.load(LlamaSLugs.DUMMY.value)
except ImportError as exp:
    print("disable test_llm_cli because:", exp)
    pytest.skip(
        reason="disabled since import of transformers failed in some way",
        allow_module_level=True,
    )

SUPPORT_LLM_INFERENCE_TARGETS = [
    _ for _ in TRACT_INFERENCES_TO_TESTS_APPROX if _.version > "0.21.5"
]

LLM_SLUGS_TO_TEST = [LlamaSLugs.DUMMY.value]


@pytest.mark.parametrize(
    "model_slug,inference_target",
    list(itertools.product(LLM_SLUGS_TO_TEST, SUPPORT_LLM_INFERENCE_TARGETS)),
    ids=LLM_SLUGS_TO_TEST,
)
def test_llama_export_from_llmexporter(model_slug, inference_target):
    llm_exporter = LLMExporter.load(model_slug)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        export_dirpath = td / "dump_here"
        dbg_path = (
            Path.cwd()
            / "failed_tests"
            / "test_llama_export_io_npz_from_LLMExporter"
        )
        # Regression: starting with huggingface/transformers 4.53.0: Jun 26
        #
        # Add a CI on this testing over each 10 minor versions except transformers
        llm_exporter.dump(
            export_dirpath=export_dirpath,
            debug_bundle_path=(dbg_path if IS_DEBUG else None),
        )


def test_llama_export_io_npz():
    llm_exporter = LLMExporter.load(LlamaSLugs.DUMMY.value)
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
