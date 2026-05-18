<!-- markdownlint-disable-file MD013 MD024 -->
# DPDFNet export + Rust WAV cleaner

Export any [CEVA DPDFNet](https://github.com/ceva-ip/DPDFNet) variant
(a modern DeepFilterNet 2 successor with dual-path RNN blocks for
stronger cross-band modelling) to a **single** NNEF artifact, and run
it on tract through a minimal Rust wrapper that takes a WAV in and
writes a cleaned WAV out.

End-to-end deploy:

```
            +-------------------------------------------+
noisy.wav --|  wav-cleaner  (1 binary, 1 NNEF artifact) |--> clean.wav
            +-------------------------------------------+
                            ^
                            |
              +-------------+--------------+
              | <variant>.nnef.tgz         |  STFT + DPDFNet + iFFT in-graph
              | <variant>.json (manifest)  |
              +----------------------------+
```

No DSP companion library, no Python in the hot path. The NNEF artifact
bundles rolling-STFT analysis, the DPDFNet NN, iFFT synthesis, and
overlap-add, all in one tract-executable graph. A sidecar JSON manifest
next to the artifact carries the audio params (`sample_rate`, `n_fft`,
`hop_size`, `state_size`) so the bench script and the Rust wrapper
adapt to any variant without hard-coding shapes.

## Why DPDFNet over DFN3

DPDFNet (paper 2024, repo 2025) extends DeepFilterNet 2 with **dual-path
RNN blocks** between the ERB encoder and ERB/DF decoders. CEVA reports
quality wins across PESQ / STOI / SI-SNR on standard SE benchmarks. The
inner per-frame model exports cleanly to t2n NNEF; the outer streaming
pipeline (STFT / iFFT / overlap-add) is wrapped by us in `export.py`
the same way as DFN3 variant B.

## Supported variants

All six checkpoints CEVA ships on HuggingFace are exportable from the
same `export.py` via `--variant`:

| Variant              | Sample rate | hop | n_fft | DPRNN blocks |
| -------------------- | ----------- | --- | ----- | ------------ |
| `baseline`           | 16 kHz      | 160 | 320   | 0            |
| `dpdfnet2`           | 16 kHz      | 160 | 320   | 2            |
| `dpdfnet4`           | 16 kHz      | 160 | 320   | 4            |
| `dpdfnet8`           | 16 kHz      | 160 | 320   | 8            |
| `dpdfnet2_48khz_hr`  | 48 kHz      | 480 | 960   | 2            |
| `dpdfnet8_48khz_hr`  | 48 kHz      | 480 | 960   | 8            |

The 48 kHz HR variants need an extra fixup at load time: the upstream
`MagNorm48` / `SpecNorm48` register their `var0` / `s0` buffers via
`torch.full(shape, int)` which infers `int64`, and the forward then
divides a float32 tensor by the buffer (tract rejects "no super type
for F32 and I64"). `export.py` recasts those buffers to float32 before
export.

## Inputs / outputs (per frame)

```text
in :  audio_frame[hop_size]       float32
   +  stft_buf[n_fft]             float32  (rolling input window)
   +  nn_state[state_size]        float32  (flat DPDFNet state)
   +  ola_buf[n_fft]              float32  (overlap-add buffer)

out:  enhanced_frame[hop_size]    float32
   +  stft_buf'[n_fft]
   +  nn_state'[state_size]
   +  ola_buf'[n_fft]
```

Initial state is zeros for all three state tensors; the upstream
DPDFNet initialises its norm-state internally.

## Bench (M4 Pro CPU, tract 0.22.1)

Per-frame budget at the model's sample rate is `hop_size /
sample_rate` (10 ms for every variant CEVA ships).

| Variant              | Median   | NN ops          | DSP ops         | RTFx     |
| -------------------- | -------- | --------------- | --------------- | -------- |
| `dpdfnet2`           | 2.13 ms  | 666 (2.04 ms)   | 129 (0.08 ms)   | **4.70x** |
| `dpdfnet2_48khz_hr`  | 3.40 ms  | 691 (3.25 ms)   | 132 (0.12 ms)   | **2.94x** |

Hottest NN ops are the **DPRNN** bidirectional GRUs in the encoder; the
48 kHz HR additionally hits a heavier `ConvTranspose` in `erb_dec` and
its rfft / irfft cost is 2-3x the 16 kHz one (960-point vs 320-point).
In-graph DSP costs <4 % of the total in both cases, well within the
"single artifact, no DSP companion" budget.

End-to-end on the Rust wrapper, 3 seconds of synthetic noisy audio:

| Variant              | Per-frame | RTFx     |
| -------------------- | --------- | -------- |
| `dpdfnet2`           | 2.18 ms   | **4.55x** |
| `dpdfnet2_48khz_hr`  | 3.38 ms   | **2.94x** |

(Wrapper-side overhead vs tract's internal profiler is tensor
construction + WAV I/O.)

## Run

```bash
# from the repo root
cd examples/speech_enhancement/dpdfnet

# 1. install pip deps (in-repo torch-to-nnef + einops + soundfile)
pip install -r requirements.txt

# 2. clone CEVA's repo + fetch a checkpoint from HuggingFace
#    Pass the variant name; defaults to dpdfnet2 (16 kHz).
./bootstrap.sh dpdfnet2
./bootstrap.sh dpdfnet2_48khz_hr   # or any other variant

# 3. export the streaming NNEF artifact (check_io=True against tract).
#    Default --variant is dpdfnet2; output goes to <variant>.nnef.tgz
#    next to a <variant>.json manifest with the audio params.
python export.py --variant dpdfnet2
python export.py --variant dpdfnet2_48khz_hr

# 4. profile on tract (reads the sidecar manifest)
python bench.py --nnef dpdfnet2.nnef.tgz

# 5. build the Rust wav-cleaner once, then point it at any variant
cd wav-cleaner-rs
cargo build --release
./target/release/wav-cleaner \
    --model ../dpdfnet2.nnef.tgz \
    --in noisy_16k.wav \
    --out clean_16k.wav
./target/release/wav-cleaner \
    --model ../dpdfnet2_48khz_hr.nnef.tgz \
    --in noisy_48k.wav \
    --out clean_48k.wav
```

The Rust wrapper reads the variant manifest to pick up the sample rate,
hop, n_fft, and state size, so the same binary handles every variant.
Input WAV must match the variant's sample rate, mono int16 or float32;
output is 16-bit PCM. State is threaded automatically; an extra two
frames of silence are appended at the tail so the overlap-add buffer
flushes the last samples.

## Known limits

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
  export.py            wrap any DPDFNet variant + STFT + iFFT, export to NNEF
  bench.py             tract per-op profile + NN/DSP split (manifest-aware)
  requirements.txt     pip deps (einops, soundfile, in-repo t2n)
  README.md            this file
  wav-cleaner-rs/      minimal tract-nnef Rust binary, WAV in / WAV out;
                       reads the sidecar manifest so it handles every variant
    Cargo.toml
    src/main.rs
```
