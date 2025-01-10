import os
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers.generation.utils import DynamicCache
from transformers.models.phi import configuration_phi, modeling_phi

from tests.utils import TRACT_INFERENCES_TO_TESTS
from torch_to_nnef.export import export_model_to_nnef


class DummyModel(nn.Module):
    LAYER_IDX = 0

    def __init__(self) -> None:
        super().__init__()
        config = configuration_phi.PhiConfig()
        self.spda_att = modeling_phi.PhiSdpaAttention(
            config, layer_idx=self.LAYER_IDX
        )

    def forward(self, x, key_states, value_states):
        sh = x.shape
        old_states_sh = key_states.shape
        position_offset = old_states_sh[2]

        dyn_cache = DynamicCache()
        dyn_cache.update(key_states, value_states, layer_idx=self.LAYER_IDX)
        attention_mask = torch.full(
            (1, sh[0], sh[1], position_offset + sh[1]),
            fill_value=torch.finfo(torch.float32).max,
        )
        attention_mask = torch.tril(attention_mask)
        # part of interest {
        (attn_output, _, past_key_value) = self.spda_att(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=torch.arange(
                position_offset, position_offset + sh[1]
            ).unsqueeze(0),
            past_key_value=dyn_cache,
            use_cache=True,
        )
        # }
        return (
            attn_output,
            past_key_value.key_cache[self.LAYER_IDX],
            past_key_value.value_cache[self.LAYER_IDX],
        )


if hasattr(
    modeling_phi, "PhiSdpaAttention"
):  # new version of huggingface doesn't have it

    @pytest.mark.parametrize(
        "inference_target",
        [_ for _ in TRACT_INFERENCES_TO_TESTS if _.version >= "0.21.4"],
    )
    def test_phi_spda_attn(inference_target):
        mod = DummyModel()
        S = 6
        P = 10
        with torch.no_grad():
            test_input = (
                torch.rand(1, S, 2048),
                torch.rand(1, 32, P, 64),
                torch.rand(1, 32, P, 64),
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                export_path = Path(tmpdir) / "model.nnef"

                model = mod.eval()

                input_names = ["hidden_states", "past_keys", "past_values"]
                output_names = [
                    "final_attn_output",
                    "new_past_keys",
                    "new_past_values",
                ]
                dbg_name = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
                dbg_name = f"{dbg_name}_phi_spda"
                inference_target = deepcopy(inference_target)
                inference_target.dynamic_axes = {
                    "hidden_states": {1: "S"},
                    "past_keys": {2: "P"},
                    "past_values": {2: "P"},
                }
                export_model_to_nnef(
                    model=model,
                    args=test_input,
                    file_path_export=export_path,
                    input_names=input_names,
                    output_names=output_names,
                    # log_level=log.INFO,
                    inference_target=inference_target,
                    debug_bundle_path=(Path.cwd() / "failed_tests" / dbg_name)
                    if os.environ.get("DEBUG", False)
                    else None,
                )
