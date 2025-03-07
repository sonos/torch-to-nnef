from functools import partial
import typing as T

import logging
import torch

import inspect

try:
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
except ImportError as exp:
    raise ValueError(
        "Should be used with 'torch_to_nnef[llm_tract]' enabled"
    ) from exp

LOGGER = logging.getLogger(__name__)


def build_past_kv_list(
    args: T.Iterable[torch.Tensor],
) -> T.List[T.Tuple[torch.Tensor, torch.Tensor]]:
    past_key_values = []
    tup: T.List[torch.Tensor] = []
    for idx, k_or_v in enumerate(args):
        assert isinstance(k_or_v, torch.Tensor), k_or_v
        if idx % 2 == 0 and len(tup):
            assert len(tup) == 2
            past_key_values.append(tuple(tup))
            tup = []
        tup.append(k_or_v)
    assert len(tup) == 2
    past_key_values.append(tuple(tup))
    return past_key_values  # type: ignore


def build_past_kv_dyn_cache(args: T.Iterable[torch.Tensor]) -> DynamicCache:
    return DynamicCache.from_legacy_cache(tuple(build_past_kv_list(args)))


class TorchToNNEFWrappedLLM(torch.nn.Module):
    """Base module class for all LLM wrapping

    These wrapper are needed to ensure deterministic inputs/outputs
    graph signature and allow some modeling optimization of few architecture.

    """


class BaseCausalWithDynCacheAndTriu(TorchToNNEFWrappedLLM):
    """Assume common AutoModelForCausalLM arch.

    with :
    - .model
    - .lm_head

    """

    with_dyn_cache: bool = True

    def __init__(
        self, model: AutoModelForCausalLM, num_logits_to_keep: int = 1
    ):
        super().__init__()
        self.model = model
        self.num_logits_to_keep = num_logits_to_keep

    def forward(self, input_ids: torch.Tensor, *args):
        """same as calling without any smart caching mechanism self.model.model+lm_head and softmax.

        This export module is extremly ineficient because no caching can be provided ...

        """
        _, seq_length = input_ids.shape[:2]

        # BUILD cache {
        cache = build_past_kv_dyn_cache(args)
        # }
        past_key_values_length = cache.get_seq_length()

        # get pos ids {
        cache_position = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = cache_position.unsqueeze(0)
        inputs_embeds = self.model.model.embed_tokens(input_ids)

        attention_mask = (
            torch.triu(
                torch.full(
                    [seq_length, seq_length],
                    torch.finfo(inputs_embeds.dtype).min,
                ),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(inputs_embeds.dtype)
        # }

        hidden_states = inputs_embeds

        outputs = self.model.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            output_attentions=False,
            use_cache=True,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.model.lm_head(
            hidden_states[:, -self.num_logits_to_keep :, :]
        )

        # Extract cache {
        kv_cache_flat_list = [t for kv in cache.to_legacy_cache() for t in kv]
        # }
        return [logits] + kv_cache_flat_list


def _slice_hidden_state_to_lasts(
    mod, inputs, outputs, num_logits_to_keep: int = 1
):
    # pylint: disable-next=import-outside-toplevel
    from transformers.modeling_outputs import BaseModelOutputWithPast

    sliced_out = outputs[0][:, -num_logits_to_keep:]

    if hasattr(outputs, "hidden_states"):
        outputs.hidden_states = sliced_out
        return BaseModelOutputWithPast(
            last_hidden_state=sliced_out,
            past_key_values=outputs.past_key_values,
            hidden_states=sliced_out,
        )

    return (sliced_out, *outputs[1:])


class BaseCausal(TorchToNNEFWrappedLLM):
    def __init__(
        self, model, with_dyn_cache: bool = True, num_logits_to_keep: int = 1
    ):
        super().__init__()
        self.model = model
        self.with_dyn_cache = with_dyn_cache
        self.num_logits_to_keep = num_logits_to_keep
        sign = inspect.signature(model.forward)
        fkwargs = {}
        if "logits_to_keep" in sign.parameters:
            fkwargs["logits_to_keep"] = self.num_logits_to_keep
        elif "num_logits_to_keep" in sign.parameters:
            fkwargs["num_logits_to_keep"] = self.num_logits_to_keep
        else:
            if self.model.config.model_type == "openelm":
                self.model.transformer.register_forward_hook(
                    partial(
                        _slice_hidden_state_to_lasts,
                        num_logits_to_keep=num_logits_to_keep,
                    )
                )
            else:
                LOGGER.warning(
                    f"model of class: {model.__class__}.forward as no 'num_logits_to_keep'"
                    "so we inference exported may be suboptimal "
                )

        self.forward_kwargs = fkwargs

    def forward(self, input_ids: torch.Tensor, *args):
        # input_ids: [1, S] with torch.int64
        # past_key_values
        # past_key_values: Optional[List[torch.FloatTensor]] = None # type annotation in code WRONG
        if self.with_dyn_cache:
            past_key_values = build_past_kv_dyn_cache(args)
        else:
            past_key_values = build_past_kv_list(args)

        out_dic = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            **self.forward_kwargs,
        )

        if self.with_dyn_cache:
            kvs = [t for kv in past_key_values.to_legacy_cache() for t in kv]
        else:
            kvs = [k_or_v for kv in out_dic["past_key_values"] for k_or_v in kv]

        assert len(args) == len(kvs), f"{len(args) * 2} == {len(kvs)}"
        # key values, (32 tensors) of shape (1, 3, S, 64)
        return [out_dic["logits"]] + kvs
