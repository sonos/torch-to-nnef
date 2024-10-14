import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from torch_to_nnef.llm_tract.config import LlamaSLugs
from torch_to_nnef.peft.cli import (
    NAME_PLACEHOLDER,
    PEFT_MAPPING_FILENAME,
    export_peft,
)
from torch_to_nnef.utils import cd

DEFAULT_MODEL_SLUG = os.environ.get("LLAMA_SLUG", LlamaSLugs.DUMMY.value)


def test_export_LoRA():
    causal_llama = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_SLUG)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    peft_model = get_peft_model(causal_llama, peft_config)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        statedict_filepath = td / "base_sd.pt"
        output_filepath = td / "output.peft.nnef.tgz"
        torch.save(peft_model.state_dict(), statedict_filepath)
        export_peft(statedict_filepath, output_filepath)
        with cd(td):
            subprocess.check_call(
                ["tar", "-xzf", str(output_filepath.absolute())]
            )
        lora_vars = {
            _.with_suffix("").name
            for _ in td.iterdir()
            if _.name.endswith(".dat")
        }
        assert lora_vars == {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        }
        with (td / PEFT_MAPPING_FILENAME).open("r", encoding="utf8") as fh:
            mapping = json.load(fh)
        assert mapping["method"] == "LoRA"
        assert mapping["mapping_table"][
            "base_model.model.model.layers.0.self_attn.v_proj.weight"
        ] == [
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
        ]


def test_export_DoRA():
    causal_llama = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL_SLUG)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_dora=True,
    )
    peft_model = get_peft_model(causal_llama, peft_config)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        statedict_filepath = td / "base_sd.pt"
        output_filepath = td / "output.peft.nnef.tgz"
        torch.save(peft_model.state_dict(), statedict_filepath)
        export_peft(
            statedict_filepath,
            output_filepath,
            patterns=[
                f"{NAME_PLACEHOLDER}.lora_A.default.weight$",
                f"{NAME_PLACEHOLDER}.lora_B.default.weight$",
                f"{NAME_PLACEHOLDER}.lora_magnitude_vector.default.weight$",
            ],
            method_name="DoRA",
        )
        with cd(td):
            subprocess.check_call(
                ["tar", "-xzf", str(output_filepath.absolute())]
            )
        lora_vars = {
            _.with_suffix("").name
            for _ in td.iterdir()
            if _.name.endswith(".dat")
        }
        assert lora_vars == {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_magnitude_vector.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default.weight",
        }
        with (td / PEFT_MAPPING_FILENAME).open("r", encoding="utf8") as fh:
            mapping = json.load(fh)
        assert mapping["method"] == "DoRA"
        assert mapping["mapping_table"][
            "base_model.model.model.layers.0.self_attn.v_proj.weight"
        ] == [
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight",
            "base_model.model.model.layers.0.self_attn.v_proj.lora_magnitude_vector.default.weight",
        ]
