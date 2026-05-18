#!/usr/bin/env bash
# Clone CEVA's DPDFNet repo (we need their PyTorch model classes for
# t2n export; the pip package only ships ONNX inference). Pin to the
# release we tested against.

set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
clone_dir="${here}/_dpdfnet_clone"
checkpoint_dir="${here}/_checkpoints"
ckpt_name="${1:-dpdfnet2}"

if [ ! -d "${clone_dir}" ]; then
    echo "[bootstrap] Cloning DPDFNet repo ..."
    git clone --depth 1 https://github.com/ceva-ip/DPDFNet.git "${clone_dir}"
else
    echo "[bootstrap] ${clone_dir} already exists; skipping clone."
fi

mkdir -p "${checkpoint_dir}"
ckpt_file="${checkpoint_dir}/${ckpt_name}.pth"
if [ ! -f "${ckpt_file}" ]; then
    echo "[bootstrap] Downloading checkpoint ${ckpt_name} from HuggingFace ..."
    curl -L -o "${ckpt_file}" \
        "https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/checkpoints/${ckpt_name}.pth"
else
    echo "[bootstrap] ${ckpt_file} already present; skipping download."
fi

echo "[bootstrap] Done."
echo "  repo:       ${clone_dir}"
echo "  checkpoint: ${ckpt_file}"
