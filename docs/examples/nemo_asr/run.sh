wget -nc
python3 -m venv .venv
source .venv/bin/activate
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
pip install -e ../../../[nemo-tract]
t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e model --tract-specific-path $HOME/SONOS/src/tract/target/release/tract
cargo test -- --nocapture
# wasm-pack build --target web --out-dir ../../html
# rm ../../html/.gitignore ../../html/*.ts
# find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
