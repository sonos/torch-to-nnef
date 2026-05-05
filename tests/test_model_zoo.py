"""Tests canonical models."""

import contextlib
import math
import os
import platform
import warnings
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from torch.jit._trace import TracerWarning
from torchaudio import models as audio_mdl

from torch_to_nnef.inference_target.tract import TractCheckTolerance

with contextlib.suppress(ImportError):
    from tests.shifted_window_attention_patch import (
        ExportableShiftedWindowAttention,
        ExportableSwinTransformerBlock,
    )

from torch_to_nnef.inference_target import TractNNEF

from .utils import (  # noqa: E402
    TRACT_INFERENCES_TO_TESTS_APPROX,
    TestSuiteInferenceExactnessBuilder,
    check_model_io_test,
    remove_weight_norm,
    set_seed,
)

set_seed(int(os.environ.get("SEED", 25)))

test_suite = TestSuiteInferenceExactnessBuilder(
    TRACT_INFERENCES_TO_TESTS_APPROX
)


def get_vision_model(vision_model_name, pretrained=True, progress=False):
    """Handle torchvison API evolution."""
    class_name = vision_model_name.lower()
    if not hasattr(torchvision.models, class_name):
        return
    mdl_cls = getattr(torchvision.models, class_name)
    if torchvision.__version__ < "0.13.0":
        model = mdl_cls(pretrained=pretrained, progress=progress)
    else:
        weights_enum = getattr(
            torchvision.models,
            f"{vision_model_name}_Weights",
        )
        model = mdl_cls(
            weights=weights_enum.DEFAULT if pretrained else None,
            progress=progress,
        )
    return model


def add_vision_model_test_suite(input_size, name):
    inps = torch.rand(1, 3, input_size, input_size)
    mdl = get_vision_model(name)
    base_name = name.lower()
    if mdl is None:
        print(
            f"missing '{base_name}' in vision "
            f"package: {torchvision.__version__}"
        )
        return
    test_suite.add(
        (inps,),
        mdl,
        test_name=base_name,
    )


add_vision_model_test_suite(224, "AlexNet")
add_vision_model_test_suite(256, "ResNet50")
add_vision_model_test_suite(256, "MNASNet1_0")
add_vision_model_test_suite(256, "EfficientNet_B0")
add_vision_model_test_suite(224, "ViT_B_16")
# swin_transformer {
# need slice with stride
if hasattr(torchvision.models, "swin_transformer"):
    torchvision.models.swin_transformer.ShiftedWindowAttention = (
        ExportableShiftedWindowAttention
    )
    torchvision.models.swin_transformer.SwinTransformerBlock = (
        ExportableSwinTransformerBlock
    )
    data = torch.rand(1, 3, 224, 224)
    mdl = torchvision.models.swin_t()  # pretrained=False
    mdl.eval()
    mdl(data)  # precompute attn mask and few shapes
    test_suite.add(
        data,
        mdl,
        test_name="swin_transformer",
        inference_conditions=lambda i: (
            isinstance(i, TractNNEF) and i.version > "0.19.0"
        ),
    )

# }


test_suite.add(
    torch.rand(1, 1, 100, 64),
    audio_mdl.DeepSpeech(64, n_hidden=256),
    test_name="deepspeech",
)

if hasattr(audio_mdl, "Conformer"):

    class ConformerWrapper(torch.nn.Module):
        """Wrap Conformer for export.

        Avoid returning length that is not edited
        torch_to_nnef forbid to return same tensor as inputed
        by the model as this means this output is not needed
        and may introduce silent variable name alterations.
        """

        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, x, length):
            out, _ = self.model(x, length)
            return out

    test_suite.add(
        (torch.rand(1, 100, 64), torch.tensor([100])),
        ConformerWrapper(
            audio_mdl.Conformer(
                64,
                num_heads=2,
                num_layers=2,
                ffn_dim=128,
                depthwise_conv_kernel_size=31,
            )
        ),
        test_name="conformer",
        inference_conditions=lambda i: (
            isinstance(i, TractNNEF) and i.version >= "0.21.2"
        ),
    )

if hasattr(audio_mdl, "ConvTasNet"):
    test_suite.add(
        (torch.rand(1, 1, 1024),),
        audio_mdl.ConvTasNet(
            num_sources=2,
            # encoder/decoder parameters
            enc_kernel_size=16,
            enc_num_feats=512,
            # mask generator parameters
            msk_kernel_size=3,
            msk_num_feats=128,
            msk_num_hidden_feats=512,
            msk_num_layers=2,
            msk_num_stacks=3,
            msk_activate="sigmoid",
        ),
        test_name="convtasnet",
    )


wav2vec2_model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
remove_weight_norm(wav2vec2_model)
wav2vec2_model.eval()


test_suite.add(
    (torch.rand(1, 1, 512),),
    wav2vec2_model.encoder,
    test_name="wav2vec2_encoder",
)
# test_suite.add(
#     (torch.rand(1, 16000),),
#     wav2vec2_model,
#     test_name="wav2vec2",
# ) # 1: eval() called on a Dummy op. This is a bug.

# export pretrained work but multi_head might give different values


# albert {


def inference_modifier_tract_tol_arm(inference_target):
    inference_target = deepcopy(inference_target)
    if (
        isinstance(inference_target, TractNNEF)
        and "arm" in platform.uname().machine.lower()
    ):
        inference_target.check_io_tolerance = TractCheckTolerance.SUPER
    return inference_target


def tract_upper_than_21_7_or_not_arm(inference_target):
    return (
        isinstance(inference_target, TractNNEF)
        and inference_target.version >= "0.21.7"
    ) or "arm" not in platform.uname().machine.lower()


try:
    from transformers import AlbertModel, AlbertTokenizer

    class ALBERTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AlbertModel.from_pretrained("albert-base-v2")

        def forward(self, *args):
            outputs = self.model(*args)
            last_hidden_states = outputs.last_hidden_state
            return last_hidden_states

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    inputs = tokenizer("Hello, I am happy", return_tensors="pt")

    test_suite.add(
        tuple(inputs.values()),
        ALBERTModel(),
        test_name="albert",
        inference_conditions=tract_upper_than_21_7_or_not_arm,
        inference_modifier=inference_modifier_tract_tol_arm,
    )
except ImportError:
    print("missing deps to test on albert model")

# }


# --- Mini image-generation architectures ------------------------------------
# Reproduce the operator mix of Stable Diffusion components (GroupNorm +
# self-attention with 1/sqrt(d) scaling + F.interpolate upsample + conv, plus
# sinusoidal timestep embedding + cross-attention for the UNet) on tiny
# models. Guards the work unblocked by tract's ``resize`` op and t2n's
# ``mul`` constant-fold dtype handling, without pulling diffusers or
# downloading the full SD weights.


class _MiniVAEMidBlock(torch.nn.Module):
    """GroupNorm + self-attention with 1/sqrt(d) scaling (AttnBlock-like)."""

    def __init__(self, channels: int = 8):
        super().__init__()
        self.channels = channels
        self.norm = torch.nn.GroupNorm(4, channels)
        self.q = torch.nn.Linear(channels, channels)
        self.k = torch.nn.Linear(channels, channels)
        self.v = torch.nn.Linear(channels, channels)
        self.proj_out = torch.nn.Linear(channels, channels)

    def forward(self, x):
        c = self.channels
        h_ = self.norm(x)
        b, _, h, w = x.shape
        h_ = h_.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        scale = c**-0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        out = self.proj_out(attn @ v).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out


class MiniVAEDecoderLike(torch.nn.Module):
    """Mid-block attention then two F.interpolate upsample + conv steps."""

    def __init__(self):
        super().__init__()
        self.mid = _MiniVAEMidBlock(channels=8)
        self.upconv1 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.upconv2 = torch.nn.Conv2d(8, 4, 3, padding=1)

    def forward(self, x):
        x = self.mid(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.upconv1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.upconv2(x)


test_suite.add(
    (torch.randn(1, 8, 4, 4),),
    MiniVAEDecoderLike(),
    test_name="mini_vae_decoder_like",
)


def _sin_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (same recipe as diffusers Timesteps)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _MiniResBlock(torch.nn.Module):
    """Conv + GroupNorm + SiLU + inject time emb + Conv + skip."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(4, in_ch)
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = torch.nn.Linear(time_dim, out_ch)
        self.norm2 = torch.nn.GroupNorm(4, out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            torch.nn.Identity()
            if in_ch == out_ch
            else torch.nn.Conv2d(in_ch, out_ch, 1)
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _MiniCrossAttn(torch.nn.Module):
    """Cross-attention of a 2D feature map against an encoder hidden state."""

    def __init__(self, channels: int, ctx_dim: int):
        super().__init__()
        self.channels = channels
        self.norm = torch.nn.GroupNorm(4, channels)
        self.q = torch.nn.Linear(channels, channels)
        self.k = torch.nn.Linear(ctx_dim, channels)
        self.v = torch.nn.Linear(ctx_dim, channels)
        self.proj_out = torch.nn.Linear(channels, channels)

    def forward(self, x, ctx):
        c = self.channels
        b, _, h, w = x.shape
        h_ = self.norm(x).permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = self.q(h_)
        k = self.k(ctx)
        v = self.v(ctx)
        attn = torch.softmax(q @ k.transpose(-1, -2) * (c**-0.5), dim=-1)
        out = self.proj_out(attn @ v).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out


class MiniUNetLike(torch.nn.Module):
    """UNet with timestep conditioning, one down + mid + one up block.

    Not shape-invariant to SD 1.5 but exercises the same operator mix: time
    embedding, cross-attn, resnet blocks, strided downsample, upsample.
    """

    def __init__(self, ch: int = 8, ctx_dim: int = 16):
        super().__init__()
        time_dim = 4 * ch
        self.ch = ch
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(ch, time_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_dim, time_dim),
        )
        self.down_res = _MiniResBlock(4, ch, time_dim)
        self.down_attn = _MiniCrossAttn(ch, ctx_dim)
        self.downsample = torch.nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.mid_res = _MiniResBlock(ch, ch, time_dim)
        self.mid_attn = _MiniCrossAttn(ch, ctx_dim)
        self.up_res = _MiniResBlock(ch * 2, ch, time_dim)
        self.up_attn = _MiniCrossAttn(ch, ctx_dim)
        self.out_conv = torch.nn.Conv2d(ch, 4, 3, padding=1)

    def forward(self, sample, timestep, encoder_hidden_states):
        t_emb = _sin_timestep_embedding(timestep, self.ch)
        if t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0)
        t_emb = self.time_mlp(t_emb)
        # Down
        h1 = self.down_res(sample, t_emb)
        h1 = self.down_attn(h1, encoder_hidden_states)
        h2 = self.downsample(h1)
        # Mid
        h2 = self.mid_res(h2, t_emb)
        h2 = self.mid_attn(h2, encoder_hidden_states)
        # Up
        h2 = F.interpolate(h2, scale_factor=2, mode="nearest")
        h = torch.cat([h2, h1], dim=1)
        h = self.up_res(h, t_emb)
        h = self.up_attn(h, encoder_hidden_states)
        return self.out_conv(h)


test_suite.add(
    (
        torch.randn(1, 4, 8, 8),
        torch.tensor([10.0]),
        torch.randn(1, 4, 16),
    ),
    MiniUNetLike(),
    test_name="mini_unet_like",
)


# Mini Flux-Schnell MM-DiT: same architecture as Flux-Schnell / SD3 (double-
# stream + single-stream transformer blocks, RoPE positions, fused qkv split)
# with a tiny config so it runs in the zoo env. Gated on diffusers import.
try:
    from diffusers import FluxTransformer2DModel

    class MiniFluxTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t = (
                FluxTransformer2DModel(
                    patch_size=1,
                    in_channels=64,
                    num_layers=2,
                    num_single_layers=2,
                    attention_head_dim=16,
                    num_attention_heads=4,
                    joint_attention_dim=64,
                    pooled_projection_dim=32,
                    guidance_embeds=False,
                    axes_dims_rope=(8, 4, 4),
                )
                .to(torch.float32)
                .eval()
            )

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
        ):
            return self.t(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

    test_suite.add(
        (
            torch.randn(1, 8, 64),
            torch.randn(1, 8, 64),
            torch.randn(1, 32),
            torch.tensor([10.0]),
            torch.zeros(8, 3),
            torch.zeros(8, 3),
        ),
        MiniFluxTransformer(),
        test_name="mini_flux_mm_dit",
    )
except ImportError:
    print("missing diffusers to test Flux mini transformer")


@pytest.mark.parametrize(
    "id,test_input,model,inference_target",
    test_suite.test_samples,
    ids=test_suite.ids,
)
def test_model_export(id, test_input, model, inference_target):
    """Test simple models."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=TracerWarning)
        check_model_io_test(
            model=model,
            test_input=test_input,
            inference_target=inference_target,
        )
