<!-- markdownlint-disable-file MD013 MD024 -->
# DeepFilterNet 3 export

Export DeepFilterNet 3's per-frame streaming model to NNEF and round-trip
it through tract end-to-end (`check_io=True` passes). The exported
artifact has the same shape as grazder's reference ONNX export
(`torch_streaming_model`): one audio frame in + 12 state tensors in, one
enhanced frame out + 12 updated state tensors out. The caller (tract's
pulse mode, a Python loop, or a Rust runtime) drives the per-frame loop
and threads the state.

## Why per-frame, not whole-buffer

The official DeepFilterNet 3 PyTorch model class (`df.deepfilternet3.DfNet`)
is **frequency-domain only**: STFT, ERB feature extraction, complex
deep-filter coefficients, and iSTFT all live in `libDF` (Rust) outside
the model. [@grazder's pure-torch reimplementation](https://github.com/grazder/DeepFilterNet)
provides:

- `TorchDFMinimalPipeline` -- the *outer* whole-waveform wrapper that
  internally chunks audio and loops over `torch_streaming_model`.
- `ExportableStreamingMinimalTorchDF` (alias `torch_streaming_model`) --
  the *inner* per-frame model with 12 explicit state tensors.

We export the **inner** model. The outer wrapper's loop unrolls into 52+
graph copies under tracing (3MB → 19MB+), and tract's `PushSliceUp`
declutter pass rejects the rolling-buffer slice fan-outs (no `begin=0`
sibling). Exporting per-frame matches grazder's ONNX deploy shape and
sidesteps both issues -- the rolling buffers become external state, not
in-graph slices.

## Variants

DeepFilterNet's synthesis stage maps a complex frame spectrum back to the
time domain. grazder's `frame_synthesis` (in `torch_streaming_model`)
implements this as a matrix multiply with a precomputed
`irfft_matrix = torch.linalg.pinv(rfft_matrix)`. That choice is
deliberate -- it round-trips through ONNX cleanly, where a real
`torch.fft.irfft` would not. The matrix is also a baked-in model
parameter (so its weights ship inside the artifact).

t2n natively supports `torch.fft.irfft` via the `tract_core_fft` op and
the view-tagged complex IR layout, so variant B replaces the matmul
with the real thing and ships a smaller, slightly faster artifact.

| Variant | iFFT path on synthesis | Status |
| ------- | ---------------------- | ------ |
| **A** (default) | matmul-iFFT with `pinv(rfft_matrix)`, verbatim from grazder | ✅ exports + tract round-trip passes |
| **B** | replace matmul-iFFT with `torch.fft.irfft` | ✅ exports + tract round-trip passes |

`tract >= 0.22.1` is required.

## Bench (single-frame inference, M4 Pro CPU, tract 0.22.1)

`tract dump --profile --json` aggregates per-op latency for a single
per-frame call. Real-time budget for streaming at 48 kHz is **10 ms** per
frame (480 samples).

### Deployment shape

The headline difference between the two paths is *what ships*:

| Path | Artifacts | DSP runtime |
| ---- | --------- | ----------- |
| NNEF (t2n) | 1 file: `deepfilternet3.nnef.tgz` | none -- STFT / iSTFT / ERB filterbank are *in* the graph |
| Official ONNX (`DfTract`) | 3 files: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx` | external `libDF` (Rust) for STFT / iSTFT / ERB |

For an edge / embedded / WebAssembly deploy this is the load-bearing
difference: NNEF runs the whole pipeline on a single inference runtime
(tract); the official ONNX deploy needs libDF compiled & glued to the
tract ONNX call site to handle the frequency-domain pre / post
processing.

### NN-only timings (apples-to-apples)

`bench.py` partitions tract's per-op profile by node-name prefix
(`enc__`, `erb_dec__`, `df_dec__` = NN; everything else = DSP) and
reports the NN-only sum alongside the full pipeline. The NN-only
column matches what the official ONNX bundle measures end-to-end
(since its ONNX graphs *only* contain the NN -- libDF handles the
DSP):

| Variant | NN-only | NN ops | RTFx (NN-only) |
| ------- | ------- | ------ | -------------- |
| NNEF matmul-iFFT (A) | 0.641 ms | 187 | **15.60×** |
| NNEF torch.fft.irfft (B) | **0.621 ms** | 187 | **16.10×** |
| ONNX official (3 components) | 0.617 ms | 245 | 16.20× |

NN-only: **NNEF-B is within 1 % of the ONNX deploy** (4 µs on 0.62 ms,
inside the trial-to-trial spread of ~5-10 µs). NNEF-A is 4 % slower,
attributable to its `(481, 2, 960) ≈ 3.7 MB` `irfft_matrix` matmul on
synthesis (replaced in B by tract's native `Fft`). At this level the
two runtimes are running essentially the same NN work; the deployment
shape is what differs.

The remaining gap is small enough to be inside noise, but a few
candidates if we want to push it further:

- tract's ONNX importer initializer-bakes weights into the model;
  the NNEF importer reads them as `variable` nodes. Closing this
  would mostly affect load time, but a couple of µs may be on the
  table per call.
- Our `ConvTranspose` emit handles asymmetric `output_padding` via
  asymmetric padding (added in this branch); ONNX maps it through
  a single op. Worth profiling whether tract's NNEF declutter
  rewrites it to the same einsum form.
- Op count: NNEF has 187 NN ops (3.4 µs/op avg), ONNX has 245
  (2.5 µs/op avg). NNEF has fewer, *heavier* ops -- some declutter
  passes that tract applies on ONNX may not run on NNEF.

### Full-pipeline timings (NNEF includes DSP in-graph; ONNX doesn't)

| Variant | Median | Min | p90 | Artifact size | Total ops | RTFx |
| ------- | ------ | --- | --- | ------------- | --------- | ---- |
| NNEF matmul-iFFT (A) | 0.763 ms | 0.754 ms | 0.770 ms | 13 MB | 275 | **13.10×** |
| NNEF torch.fft.irfft (B) | 0.717 ms | 0.713 ms | 0.739 ms | 8.9 MB | 290 | **13.95×** |
| ONNX official 3-comp (NN-only, libDF handles DSP elsewhere) | 0.617 ms | 0.613 ms | 0.621 ms | 11 MB | 245 | **16.20×** |

The two NNEF artifacts include the STFT analysis + iFFT synthesis +
ERB feature extract in-graph (per-frame). The official ONNX timings
*exclude* that DSP -- to get a like-for-like end-to-end you'd add
libDF's per-frame DSP cost on top of the ONNX number. We don't have
that number here; what we do have is that the full NNEF pipeline runs
in **0.72 ms / 14× real-time** without any DSP companion.

Variant B is ~3 % faster than A and **30 % smaller** (8.9 MB vs
13 MB), trading the baked `irfft_matrix` parameter for tract's
native `Fft` op.

## Run

```bash
# from the repo root
cd examples/speech_enhancement/deepfilternet

# 1. install pip deps (in-repo torch-to-nnef + upstream deepfilternet for
#    the pretrained DFN3 checkpoint and the `init_df()` loader)
pip install -r requirements.txt

# 2. clone grazder's fork so we can pick up torch_df_streaming_minimal.py
./bootstrap.sh

# variant A export (matmul-iFFT, verbatim from grazder, per-frame)
python export.py --out deepfilternet3.nnef.tgz

# variant B export (torch.fft.irfft, native t2n FFT path, per-frame)
python export_stft_variant.py --out deepfilternet3_stft.nnef.tgz

# (optional) per-frame ONNX export via grazder's symbolic-op hooks
# Note: this artifact does not load on tract -- the inline DFT shape
# inference disagrees with what tract's ONNX importer expects. Use
# the official 3-component bundle for tract-based comparisons.
# python export_onnx_baseline.py --out deepfilternet3.onnx

# (optional) for the 3-way bench, fetch the official ONNX components
# (NN-only, no FFT/ERB in graph) -- libDF handles DSP outside
curl -L -o DeepFilterNet3_onnx.tar.gz \
    https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3_onnx.tar.gz
mkdir -p _official_onnx && tar -xzf DeepFilterNet3_onnx.tar.gz -C _official_onnx

# 3-way per-frame speed comparison on tract
python bench.py \
    --nnef-a deepfilternet3.nnef.tgz \
    --nnef-b deepfilternet3_stft.nnef.tgz \
    --onnx-official-dir _official_onnx/tmp/export
```

The export uses `check_io=True` and a real tract round-trip; the script
fails loudly if NNEF execution on tract diverges from the PyTorch
reference per frame.

## Pretrained weights

`TorchDFMinimalPipeline()` calls `init_df()` from the upstream
`deepfilternet` package, which downloads and caches DFN3's weights on
first run.

## Known limits

- **Whole-waveform export** (the outer `TorchDFMinimalPipeline`)
  produces a NNEF that hits tract's `PushSliceUp` declutter assertion
  (`boundaries[0] == 0`) on rolling-buffer slice fan-outs. Fixing that
  is upstream (tract). The per-frame export avoids the issue entirely
  by making rolling buffers external state -- this is also the deploy
  shape grazder's reference ONNX uses.
- **Per-frame ONNX from torch.onnx** (`export_onnx_baseline.py`): the
  script produces a valid ONNX (~12 MB), but tract refuses its `DFT`
  op at the analyse stage (`outputs[0].shape[0] == Val(2)` unification
  failure). Use the official 3-component bundle for the
  tract-vs-tract speed comparison; that bundle works because all the
  FFT / ERB DSP lives in libDF (Rust), so the ONNX graphs themselves
  carry no `DFT` op.
- **Official ONNX symbolic dims**: tract's CLI can't always resolve
  the symbolic dimensions baked into the upstream `enc.onnx` /
  `erb_dec.onnx` / `df_dec.onnx` (`Relue0_dim_0`, etc.). `bench.py`
  pre-runs `onnxsim.simplify(..., overwrite_input_shapes={...})` to
  bake the per-frame concrete shapes before profiling.
