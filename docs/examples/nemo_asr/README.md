# 10. Nemo ASR support

!!! abstract "Goals"

    By the end of this guide, you will know how to:

    1. :material-toolbox: Export a NeMo ASR model to NNEF using `t2n_export_nemo`
    2. :fontawesome-brands-rust: Run WAV inference from a minimal Rust binary
    3. :fontawesome-brands-python: Run inference from Python using `tract`
    4. :material-toolbox: Evaluate the exported model using Word Error Rate (WER)

!!! example "Prerequisite"

    - [ ] Basic Python knowledge
    - [ ] Basic Rust knowledge
    - [ ] Approximately 10 minutes to read this page

## Overview

This page documents the end-to-end workflow for exporting an [NVIDIA NeMo Automatic Speech Recognition](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) (ASR) model to **NNEF** using `torch-to-nnef`, running inference with [**tract**](https://github.com/sonos/tract), and evaluating the exported model against standard ASR benchmarks.



## Export a NeMo ASR model

The `t2n_export_nemo` command loads a pre-trained ASR model from the NeMo toolkit and exports it to the NNEF format.

If not already installed, install `torch_to_nnef` with the `nemo-tract` extra. This enables the NeMo-specific export command:

```bash
t2n_export_nemo \
    -e ./dump_parakeet_v3_06B \ # export directory
    --tract-specific-path $HOME/SONOS/src/tract/target/release/tract \ # optional path to tract binary
    -tt very # numerical tolerance for NeMo vs tract checks

# -s nvidia/parakeet-tdt-0.6b-v3 \ # optional explicit model slug
# -p ~/user/finetuned-parakeet.nemo \ # optional explicit path to .nemo file
# --compress-method min_max_q4_0_all # optional model compression
```

Since in this example no `-s` argument is provided, the command defaults to listing the known 'nemo' compatible models on HuggingFace Hub and Nemo registeries (we mostly tested parakeet and nemotron).

After the command completes, the export directory (e.g. ./dump_parakeet_v3_06B) will contain:

- The exported NNEF model files
- A model_config.json file describing the exported pipeline
- A export_config.json with all export options used
- A .log file with export details

Additional export options are available via:

```bash
t2n_export_nemo --help
```


Some NeMo preprocessing components are not yet fully supported by tract. In such cases, options such as `--skip-preprocessor` can be used to exclude those stages from the export.

### CLI flags quick reference

- `-e, --export-dir`: Output directory (must not pre-exist).
- `-s, --model-slug`: Explicit NeMo model slug; omit to choose interactively.
- `-p, --model-path`: Explicit local path to .nemo file.
- `--tract-specific-version` / `--tract-specific-path`: Select Tract version or binary.
- `--tract-reify-sdpa`: Enable SDPA reification where supported by selected Tract.
- `-tt, --tract-check-io-tolerance`: IO check strictness (`exact`, `approximate`, `loose`, or `skip`).
- `--skip-preprocessor`: Export only encoder/decoder/joint parts.
- `--split-joint-decoder`: Split `decoder` and `joint` into separate subnets.
- `--compress-registry` / `--compress-method`: Apply weight compression during export.

Run `t2n_export_nemo --help` for the full list of options.


## Shape configuration (boundary remodeler)

In many cases you will want to control the symbolic shapes and boundary transforms used during export (e.g., set a stable `BATCH` symbol, collapse size-1 dims, bind a scalar to a dynamic size, or keep only a subset of outputs). You can manage this via a YAML shape config file passed to the CLI.

Generate a starting template aligned to your model with:

```bash
t2n_export_nemo \
  --inspect-signatures \
  --dump-shape-config ./shapes.yaml \
  # ... your usual flags (model slug/path, etc.)
```

The generated `shapes.yaml` uses a nested layout per subnet:

- `inputs`: mapping of input-name → settings
- `renamed_symbols` (optional): `{ TARGET: [SOURCES...] }` aliasing of dynamic symbols
- `outputs_keep` (always present in the template): ordered list of output names to keep (default if omitted: keep all)

Per-input settings under `inputs`:

- `original_shape`: list of dims (ints or strings)
- `collapse_dims` (optional): list of symbols to collapse at the boundary
- `bind_scalar_to_dim_size` (optional): dynamic source as `subnet.input.SYMBOL`

Example (abbreviated):

```yaml
encoder:
  inputs:
    audio_signal:
      original_shape: [AUDIO_SIGNAL__BATCH, 128, AUDIO_SIGNAL__TIME]
      collapse_dims: [AUDIO_SIGNAL__BATCH]
    length:
      original_shape: [LENGTH__BATCH]
      collapse_dims: [LENGTH__BATCH]
      bind_scalar_to_dim_size: encoder.audio_signal.AUDIO_SIGNAL__TIME

decoder_joint:
  inputs:
    encoder_outputs:
      original_shape: [ENCODER_OUTPUTS__BATCH, 1024, ENCODER_OUTPUTS__TIME]
      collapse_dims: [ENCODER_OUTPUTS__BATCH, ENCODER_OUTPUTS__TIME]

decoder:
  renamed_symbols: { BATCH: [TARGETS__BATCH, STATES_0__BATCH, STATES_1__BATCH] }
  # Typical RNNT decoder outputs include: outputs, prednet_lengths, states_out
  # Keep only the ones you need (e.g., drop prednet_lengths)
  outputs_keep: [outputs, states_out]
  inputs:
    targets:
      original_shape: [TARGETS__BATCH, TARGETS__TIME]
      collapse_dims: [BATCH]
    states_0:
      original_shape: [2, STATES_0__BATCH, 640]
      collapse_dims: [BATCH]
    states_1:
      original_shape: [2, STATES_1__BATCH, 640]
      collapse_dims: [BATCH]
```

!!! note "Decoder: dropping prednet_lengths while keeping IO aligned"

    When you exclude `prednet_lengths` from decoder outputs via `outputs_keep`,
    also bind the `target_length` input to the TIME dimension of `targets` so it becomes
    an internal scalar (and is no longer exposed as an external input):

    ```yaml
    decoder:
      outputs_keep: [outputs, states_out]
      inputs:
        targets:
          original_shape: [TARGETS__BATCH, TARGETS__TIME]
          collapse_dims: []
        target_length:
          original_shape: [TARGET_LENGTH__BATCH]
          collapse_dims: []
          bind_scalar_to_dim_size: decoder.targets.TARGETS__TIME
        states_0:
          original_shape: [2, STATES_0__BATCH, 640]
          collapse_dims: [BATCH]
        states_1:
          original_shape: [2, STATES_1__BATCH, 640]
          collapse_dims: [BATCH]
    ```

    This keeps the external input/output quantities consistent and makes the
    boundary contract explicit: `target_length = size(targets, TIME)`.

Notes:

- Symbols are normalized to uppercase; `b`/`batch` become `BATCH`.
- `renamed_symbols` targets cannot include themselves in sources.
- `collapse_dims` requires the symbol to be dynamic on that input at the selected stage.
- `bind_scalar_to_dim_size` binds a dynamic size as an `int64` scalar.
- `outputs_keep` filters exported outputs; order follows the subnet’s original `output_names`. The template always includes it so you can easily trim.

---


## Audio preprocessing requirements

All supported NeMo ASR models expect audio input with the following characteristics:

- 16 kHz sample rate
- Mono channel
- WAV format

Ensure that all input audio conforms to these requirements before running inference.

---

!!! warning "next sections are limited to RNNT and TDT models."

    Due to limited time and resources, the following sections focus on RNNT and TDT models.
    Others are not guaranteed to work as is, but contributions are welcome!

---

## Example: Running a NeMo ASR model with tract


[in this example directory](https://github.com/sonos/torch-to-nnef/tree/main/docs/examples/nemo_asr)
The example uses a pre-trained ASR model from NVIDIA NeMo and shows how to perform inference using the exported NNEF artifacts.



## Run the exported model in Rust

To run the exported NeMo ASR model from Rust, add the tract-nemo crate to your Cargo.toml:

```toml
[dependencies]
tract-nemo = {
  git = "https://github.com/sonos/torch-to-nnef.git",
  branch = "main",
  subdir = "docs/examples/nemo_asr/"
}
```

Rust inference example
```rust
use tract_nemo::nemo_asr::NemoAsrModel;

fn main() -> tract_nemo::TractResult<()> {
    // Load the exported NeMo ASR model
    let model_path = "./dump_parakeet_v3_06B";
    let mut asr_model = NemoAsrModel::load(model_path)?;

    let input_wavs = vec![
        // paths to input WAV files
    ];

    // Run inference
    let transcripts = asr_model.infer_from_wav_paths(&input_wavs)?;

    // Display results
    for (i, t) in transcripts.iter().enumerate() {
        println!("Transcription[{}]: '{}'", i, t.text);

        // Each transcript also contains detailed items:
        // - token
        // - logit
        // - emitted_at_encoder_timestep
        // - emitted_at_encoder_timestep_iteration
    }

    Ok(())
}
```


## Run the exported model in Python

The exported NeMo ASR model can also be executed from Python using the tract-nemo Python bindings.

First, install the Python package:

```bash
pip install "git+https://github.com/sonos/torch-to-nnef.git@main#egg=nemo-asr-tract&subdirectory=docs/examples/nemo_asr/src/nemo_asr_py"
```

Python inference example
```python
import nemo_asr_tract

def main():
    # Load the exported NeMo ASR model
    model_path = "./dump_parakeet_v3_06B"
    asr_model = nemo_asr_tract.nemo_asr.NemoAsrModel.load(model_path)

    input_wavs = [
        "path/to/your/input1.wav",
        "path/to/your/input2.wav",
    ]

    # Run inference
    transcripts = asr_model.infer_from_wav_paths(input_wavs)

    # Display results
    for i, t in enumerate(transcripts):
        print(f"Transcription[{i}]: '{t.text}'")
        print(f"Items[{i}]: {t.items}")

if __name__ == "__main__":
    main()
```

## Evaluation

If not already installed you need to setup the same python package, as the one for running tract model, **with the `eval` extra** for evaluation:

```bash
pip install "git+https://github.com/sonos/torch-to-nnef.git@main#egg=nemo-asr-tract[eval]&subdirectory=docs/examples/nemo_asr/src/nemo_asr_py"

```

The Python tooling also supports evaluation of the exported model using standard ASR benchmarks and WER metrics.

### Run an ASR Open Leaderboard evaluation

```bash
nemo_tract_eval \
    -e ./dump_parakeet_v3_06B \
    -r ~/SONOS/data/test_asr_export_parakeet \
    --device 0
```


This command runs an evaluation following the same protocol as the [Hugging Face ASR Open Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

It produces, for each dataset:

- `.jsonl` manifest files containing predictions and references
- Per-dataset WER scores
- Aggregated summary metrics

Use `--help` to inspect all available evaluation options.

### Display sample-level differences between runners

```bash
nemo_tract_eval_compare_manifest \
    --results-dir ./../my-results-dir/ \
    --max-items 5
```

This command displays side-by-side comparisons (by default, NeMo vs tract) for a subset of samples, sorted by absolute WER difference.

### Recompute scores and display a summary table

```bash
nemo_tract_eval_score_manifest ./../my-results-dir/
```

This recomputes WER scores from the generated manifest files and prints a summary table. This is useful when experimenting with alternative scoring logic.

### Custom runner support

For more advanced use cases, the evaluation framework supports custom runners and datasets.

To define a new runner or model, inherit from the base class and implement the required methods:

```python
from nemo_asr_tract.eval.runner import AsRRunner

class MyCustomRunner(AsRRunner):
    def __init__(self, model: str, device: int = 0):
        super().__init__(model, device)

    def name(self) -> str:
        my_super_model_and_runner_name = "dummy"
        return clean_name(my_super_model_and_runner_name)

    @classmethod
    def load_from_path(
        cls,
        *,
        cfg: EvalConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AsrRunner":
        """Load the ASR runner from a model directory."""
        return cls(model, batch_size=cfg.batch_size)

    def transcribe_from_wav_paths(self, wav_paths: List[str]):
        return []

```

The custom runner can then be selected via the `--model_runner_class` argument in the evaluation CLI.


### Tracking runner issues

In the past we have observed some issues with the exported models, such as mismatches between NeMo and tract runner outputs, or unexpected WER scores. To help track and debug these issues, we maintain a script where we log any runner-related discrepancy when running on specific batch, with specific hardware target (due to Kernel precisions differences).
Here is a sample usage (it needs extra eval to run properly).

```bash
nemo_tract_eval_batch_align_checker \
    --results-dir ./../my-results-dir/ \
    --output-file ./runner_issues_log.jsonl
    --model-dir ../../assets/model \
    --dataset librispeech \
    --split test.clean \
    --sample-idx 1000 \
    -o ~/SONOS/data/2026_02_05_debug_batched_metal \
    [--force-cpu]
```
### Shapes config (shapes.yaml)

You can generate and apply a per-subnet shapes configuration to annotate symbols, collapse boundary dims, bind scalars to dim sizes, and (optionally) rename symbols for the Tract-facing contract.

Workflow

1) Dump a template (nested by subnet):

```
t2n_export_nemo \
  --model-slug nvidia/parakeet-tdt-0.6b-v3 \
  --export-dir ./noop \
  --inspect-signatures \
  --dump-shape-config shapes.yaml \
  --dry-run \
  --split-joint-decoder
```

2) Edit `shapes.yaml` (structured example):

```
encoder:
  audio_signal:
    original_shape: [AUDIO_SIGNAL__BATCH, 128, AUDIO_SIGNAL__TIME]
    collapse_dims: [AUDIO_SIGNAL__BATCH]
  length:
    original_shape: [LENGTH__BATCH]
    collapse_dims: [LENGTH__BATCH]
    bind_scalar_to_dim_size: encoder.audio_signal.AUDIO_SIGNAL__TIME

decoder:
  # Unify batch symbols for Tract-facing dynamic axes
  renamed_symbols: { BATCH: [TARGETS__BATCH, STATES_0__BATCH, STATES_1__BATCH] }
  targets:
    original_shape: [TARGETS__BATCH, TARGETS__TIME]
    # Alias 'BATCH' is accepted when listed in renamed_symbols
    collapse_dims: [BATCH]
  states_0:
    original_shape: [2, STATES_0__BATCH, 640]
    collapse_dims: [BATCH]
  states_1:
    original_shape: [2, STATES_1__BATCH, 640]
    collapse_dims: [BATCH]
  # Binding can also use alias symbols:
  #   bind_scalar_to_dim_size: decoder.targets.BATCH

joint:
  encoder_outputs:
    original_shape: [ENCODER_OUTPUTS__BATCH, 1024, ENCODER_OUTPUTS__TIME]
    collapse_dims: [ENCODER_OUTPUTS__BATCH, ENCODER_OUTPUTS__TIME]
  decoder_outputs:
    original_shape: [DECODER_OUTPUTS__BATCH, 640, DECODER_OUTPUTS__TIME]
    collapse_dims: [DECODER_OUTPUTS__TIME]
```

3) Inspect with config applied (human-rich):

```
t2n_export_nemo \
  --model-slug nvidia/parakeet-tdt-0.6b-v3 \
  --export-dir ./noop \
  --inspect-signatures \
  --inspect-stage final \
  --inspect-format human-rich \
  --shape-config shapes.yaml \
  --dry-run \
  --split-joint-decoder
```

4) Export with config:

```
t2n_export_nemo \
  --model-slug nvidia/parakeet-tdt-0.6b-v3 \
  --export-dir ./export_with_shapes \
  --shape-config shapes.yaml \
  --split-joint-decoder
```

Notes

- Dynamic symbols are namespaced by input for clarity (e.g., `TARGETS__BATCH`).
- At the export boundary:
  - Tuple inputs are flattened (`states_0`, `states_1`).
  - `collapse_dims` removes listed dynamic axes externally and reinserts them internally.
  - `bind_scalar_to_dim_size` removes the bound input and injects `shape(source)[axis]` as an int64 tensor (dynamic).
  - `renamed_symbols` only affects the Tract-facing dynamic axes (e.g., unify batch under `BATCH` for decoder); inspector remains namespaced.
