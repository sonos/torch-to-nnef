wget -nc
python3 -m venv .venv
source .venv/bin/activate
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
wget https://raw.githubusercontent.com/mozilla/DeepSpeech/master/data/smoke_test/LDC93S1.wav
mv 2086-149220-0033.wav data_smoke_test_LDC93S1.wav assets/
pip install -e ../../../[nemo-tract]
t2n_export_nemo -s nvidia/parakeet-tdt-0.6b-v3 -e assets/model --tract-specific-path $HOME/SONOS/src/tract/target/release/tract
cd ./src/nemo_asr/ && cargo test --release -- --nocapture && cd ../../
