# Example using tract to run a simple Nemo ASR model

This example demonstrates how to use the `tract` library to run a simple Nemo ASR (Automatic Speech Recognition) model.
The model used in this example is a pre-trained ASR model from NVIDIA's Nemo toolkit.

## Prerequisites
- Rust programming language installed (https://www.rust-lang.org/tools/install)
- Python installed (for evaluation and instrumentation)

# Export a Nemo model

To export a Nemo ASR model, you can use the following Python code snippet. This code loads a pre-trained ASR model from the Nemo toolkit and exports it to NNEF format.

Install torch_to_nnef if you haven't already with feature `nemo-tarct`,
this will enable the `t2n_export_nemo` command.:

```bash
t2n_export_nemo \
    -s nvidia/parakeet-tdt-0.6b-v3 \ # nemo pretrained model name
    -e ./dump_parakeet_v3_06B \ # export directory name
    --tract-specific-path $HOME/SONOS/src/tract/target/release/tract \ # path to tract binary (optional)
    -tt very # tolerance of check between nemo and tract for each sub-model

# --compress-method min_max_q4_0_all  # can be used to cmporess the model
```

After running the above command, you will find the exported NNEF model files in the specified export directory (`./dump_parakeet_v3_06B` in this case),
along side the `model_config.json` file.

# About audio preprocessing

Model expects 16kHz mono audio input.


# Run the exported model in Rust

Using the crate in this directory, you can run the exported Nemo ASR model in Rust as follows:
You need to reference the crate `tract-nemo` in your `Cargo.toml`:

```toml
[dependencies]
tract-nemo = {
  git = "https://github.com/sonos/torch-to-nnef.git",
  branch = "main",
  subdir = "docs/examples/nemo_asr/"
}

```

Then, you can use the following Rust code to load the exported model and run inference:

```rust
use tract_nemo::nemo_asr::NemoAsrModel;

fn main() -> tract_nemo::TractResult<()> {
    // Load the exported Nemo ASR model
    let model_path = "./dump_parakeet_v3_06B";
    let mut asr_model = NemoAsrModel::load(model_path)?;

    let input_wavs = vec![/* your input wav paths here */];

    // Run inference
    let transcriptions = asr_model.infer_from_wav_paths(&input_wavs)?;

    // Print the transcriptions result
    for (i, t) in transcripts.iter().enumerate() {
        println!(
            "Transcription[{}]: '{}'",
            i,
            &t.text
        );
        // transcripts also contain list of items with:
        // token
        // logit
        // emitted_at_encoder_timestep
        // emitted_at_encoder_timestep_iteration
    }
    Ok(())
}
```
This code snippet demonstrates how to load the exported Nemo ASR model and run inference on a list of input WAV file paths.


# Run the exported model in Python

You can also run the exported Nemo ASR model in Python using the `tract` library. Here's an example:
First ensure to install the `tract-nemo` package:
```bash
pip install -e ./src/nemo_asr_py/
```

Then you can use the following Python code to load the exported model and run inference:
```python
import nemo_asr_tract

def main():
    # Load the exported Nemo ASR model
    model_path = "./dump_parakeet_v3_06B"
    asr_model = nemo_asr_tract.nemo_asr.NemoAsrModel.load(model_path)

    input_wavs = [
        "path/to/your/input1.wav",
        "path/to/your/input2.wav",
    ]

    # Run inference
    transcriptions = asr_model.infer_from_wav_paths(input_wavs)
    # Print the transcriptions result
    for i, t in enumerate(transcriptions):
        print(f"Transcription[{i}]: '{t.text}'")
        print(f"Items[{i}]: {t.items}")

if __name__ == "__main__":
    main()
```

This code snippet demonstrates how to load the exported Nemo ASR model and run inference on a list of input WAV file paths in Python.


# Evaluation

You can evaluate the quality of the ASR model ran in tract with the python package as follows (once package installed):

```bash
nemo_tract_eval -e ./dump_parakeet_v3_06B -r ~/SONOS/data/test_asr_export_parakeet --device 0
```
This run an evaluation following the same protocol as the `ASR Open Leaderboard` evaluation from HuggingFace.
