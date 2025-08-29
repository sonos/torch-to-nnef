wget -nc https://upload.wikimedia.org/wikipedia/commons/5/55/Grace_Hopper.jpg
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 ./export.py
wasm-pack build --target web --out-dir ../../html
rm ../../html/.gitignore ../../html/*.ts
find ../../html/*.json -maxdepth 1 -type f -name '*.json' ! -name '1kclass.json' -delete
