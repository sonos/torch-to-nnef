VAD Demo Modules and Local Build

Overview

- `audio.js`: Audio buffer utils (windowing, conversion) used by the demo.
- `mic.js`: Microphone capture using the Web Audio API.
- `plot.js`: Lightweight rendering helpers for audio and VAD probability charts.
- `session.js`: Manages model session lifecycle (load, init, run, dispose).
- `ui.js`: Wires DOM events to session and mic; handles basic controls/indicators.
- `wasm.js`: Wasm glue to load the `wasm-bindgen` package built from Rust.
- `index.js`: Entry that composes mic, session, plotting and UI.
- `vad_worklet.js`: AudioWorklet processor for low-latency microphone capture.

Build (Wasm)

Prereqs

- Rust toolchain (stable) and `wasm-pack`
- Binaryen’s `wasm-opt` for size-optimized builds (recommended)
  - macOS (brew): `brew install binaryen`
  - Linux: use your package manager or https://github.com/WebAssembly/binaryen
  - Ensure `~/.cargo/bin` is on your `PATH` (for `wasm-pack`)

Commands

1) From the VAD example crate, build the wasm package (release):

   ```bash
   cd ../../../examples/vad
   wasm-pack build --release --target web
   ```

   This produces `pkg/` with the wasm JS shims and `.wasm` binary.

2) Copy or symlink the generated `pkg` folder next to this README if your hosting setup expects it. The demo pages import from `./pkg/` by default.

Serve Locally

Use any static file server from `docs/html`.

```bash
cd ../../html
python3 -m http.server 8080
# open http://localhost:8080/vad/index.html
```

Notes

- Release builds are configured for small size in `examples/vad/Cargo.toml` (`opt-level=z`, thin LTO, `wasm-opt=true`).
- If you change the Rust code, re-run the wasm-pack build step and refresh the page.
