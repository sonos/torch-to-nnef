import tempfile
from pathlib import Path

import numpy as np

from torch_to_nnef.llm_tract.config import LlamaSLugs
from torch_to_nnef.llm_tract.exporter import LLMExporter

from .utils import TRACT_INFERENCES_TO_TESTS_APPROX

inference_targets = [
    (str(_), _)
    for _ in TRACT_INFERENCES_TO_TESTS_APPROX
    if _.version > "0.21.5"
]


def test_llama_export_io_npz_from_LLMExporter():
    llm_exporter = LLMExporter.load(LlamaSLugs.DUMMY.value)
    new_llm_exporter = LLMExporter(
        llm_exporter.hf_model_causal,
        llm_exporter.tokenizer,
        as_float16=False,
    )
    with tempfile.TemporaryDirectory() as td:
        export_dirpath = Path(td) / "dump_here"
        new_llm_exporter.dump(export_dirpath=export_dirpath)


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
            assert dic["outputs"].shape == (1, n_tokens, 32000)
            assert dic["out_cache_key_0"].shape == (1, 2, total_tokens, 4)
            assert dic["out_cache_value_0"].shape == (1, 2, total_tokens, 4)
