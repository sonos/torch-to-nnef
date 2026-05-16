<!-- markdownlint-disable-file MD013 MD024 -->
# DPDFNet 2 (16 kHz) export + Rust WAV cleaner

Export [CEVA's DPDFNet](https://github.com/ceva-ip/DPDFNet) (a modern
DeepFilterNet 2 successor with dual-path RNN blocks for stronger
cross-band modelling) to a **single** NNEF artifact and run it on
tract through a minimal Rust wrapper that takes a WAV in and writes a
cleaned WAV out.

The end-to-end deploy is:

```
            +-------------------------------------------+
noisy.wav --|  wav-cleaner  (1 binary, 1 NNEF artifact) |--> clean.wav
            +-------------------------------------------+
                            ^
                            |
                +-----------+-----------+
                | dpdfnet2.nnef.tgz     |  STFT + DPDFNet + iFFT in-graph
                +-----------------------+
```

No DSP companion library, no Python in the hot path. The NNEF artifact
bundles rolling-STFT analysis, the DPDFNet NN, iFFT synthesis, and
overlap-add, all in one tract-executable graph.

## Why DPDFNet over DFN3

DPDFNet2 (paper 2024, repo 2025) extends DeepFilterNet 2 with **dual-path
RNN blocks** between the ERB encoder and ERB/DF decoders. CEVA reports
quality wins across PESQ / STOI / SI-SNR on standard SE benchmarks. The
inner per-frame model exports cleanly to t2n NNEF; the outer streaming
pipeline (STFT / iFFT / overlap-add) is wrapped by us in `export.py`
the same way as DFN3 variant B.

## Inputs / outputs (per frame)

```text
in :  audio_frame[160]            float32  (16 kHz, 10 ms hop)
   +  stft_buf[320]               float32  (rolling input window)
   +  nn_state[45424]             float32  (flat DPDFNet state)
   +  ola_buf[320]                float32  (overlap-add buffer)

out:  enhanced_frame[160]         float32
   +  stft_buf'[320]
   +  nn_state'[45424]
   +  ola_buf'[320]
```

Initial state is zeros for all three state tensors (the upstream DPDFNet
initialises its norm-state internally, no metadata-baking needed for
this 16 kHz checkpoint).

## Bench (M4 Pro CPU, tract 0.22.1)

Per-frame budget at 16 kHz with hop=160 is **10 ms**.

| Path | Median | NN ops | DSP ops | RTFx |
| ---- | ------ | ------ | ------- | ---- |
| `dpdfnet2.nnef.tgz` (NN + STFT + iFFT in-graph) | **2.128 ms** | 666 (2.035 ms) | 129 (0.076 ms) | **4.70x** |
| ` ` -> NN-only portion | 2.035 ms | 666 | -- | 4.91x |
| ` ` -> DSP-only portion | 0.076 ms | -- | 129 | -- |

Hottest NN ops are the **DPRNN** bidirectional GRUs in the encoder
(four `OptMatMul` calls at ~42 us each, plus a `Scan` at 35 us). The
in-graph DSP costs 76 us / 3.6 % of the total -- well within the
"single artifact, no DSP companion" budget.

End-to-end on the Rust wrapper (3 seconds of audio): **4.44x
real-time**, **2.24 ms per frame**. The small overhead vs tract's
internal profiler (2.13 ms) is wrapper-side tensor construction and
WAV I/O.

## t2n bugs surfaced

DPDFNet's `DPRNNBlock` uses `einops.rearrange("b c t f -> (b f t) c")`
to collapse three axes for the inter-chunk RNN. einops builds the
target shape via a chain of in-place `aten::mul_` calls on a running
scalar tensor. t2n's IR `call_op` shape-inference replay was running
the in-place op directly, so every node in the chain aliased the same
storage and read back the final mutated value. Fixed in the same
branch by rerouting `aten::mul_` / `aten::div_` to their out-of-place
equivalents (`fix(torch_graph): route in-place mul_/div_ through
out-of-place ops`), with a regression test in
`tests/test_inplace_shape_arith.py`.

## Run

```bash
# from the repo root
cd examples/speech_enhancement/dpdfnet

# 1. install pip deps (in-repo torch-to-nnef + einops for the model + soundfile)
pip install -r requirements.txt

# 2. clone CEVA's repo + fetch the dpdfnet2 checkpoint from HuggingFace
./bootstrap.sh dpdfnet2

# 3. export the streaming NNEF artifact (passes through check_io=True against tract)
python export.py --out dpdfnet2.nnef.tgz

# 4. profile on tract (per-op latencies, NN vs DSP split)
python bench.py

# 5. build the Rust wav-cleaner and run it
cd wav-cleaner-rs
cargo build --release
./target/release/wav-cleaner --model ../dpdfnet2.nnef.tgz --in noisy.wav --out clean.wav
```

The Rust wrapper expects 16 kHz mono WAV input (PCM 16-bit or float32);
it writes 16-bit PCM. State is threaded automatically; an extra 2 frames
of silence are appended at the tail so the overlap-add buffer flushes
the last samples.

## Known limits

- **16 kHz only** in this example. The CEVA repo also ships
  `dpdfnet2_48khz_hr` / `dpdfnet8_48khz_hr` (960 hop, 320 window
  variant). The same `export.py` shape would work; just swap the
  `_build_dpdfnet2` factory for `dpdfnet_48khz_hr.py`'s and adjust
  `HOP_SIZE` / `N_FFT`. Not done here to keep the example focused.
- **Static batch size = 1**. DPDFNet is a per-frame streaming model;
  batched inference is unusual. The export uses static axes.
- **No tract pulse-mode**. The artifact is per-frame with explicit
  state I/O; the Rust wrapper threads state in a loop. Tract's pulse
  declutter could in principle compile the per-frame graph into a
  `model.run(audio_buffer) -> clean_buffer` call, which would
  eliminate the per-frame call overhead. Out of scope here; would
  benefit from tract upstream work.

## Files

```
examples/speech_enhancement/dpdfnet/
  bootstrap.sh         clone DPDFNet repo + download HF checkpoint
  export.py            wrap DPDFNet + STFT + iFFT, export to NNEF
  bench.py             tract per-op profile + NN/DSP split
  requirements.txt     pip deps (einops, soundfile, in-repo t2n)
  README.md            this file
  wav-cleaner-rs/      minimal tract-nnef Rust binary, WAV in / WAV out
    Cargo.toml
    src/main.rs
```
