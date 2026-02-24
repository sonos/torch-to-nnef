#!/usr/bin/env bash

set -euo pipefail

# Default: dry run
DRY_RUN=true

if [[ "${1:-}" == "--force" ]]; then
    DRY_RUN=false
fi

echo "Starting cleanup in: $(pwd)"
echo "Dry run: $DRY_RUN"
echo

# Directories to remove
DIR_PATTERNS=(
    ".venv"
    "target"
    "targets"
    "images"
    "assets"
)

# Files to remove
FILE_PATTERNS=(
    "*.onnx"
    "*.pt"
    "*.wav"
    "*.nnef.tgz"
)

remove_path() {
    local path="$1"
    if $DRY_RUN; then
        echo "[DRY RUN] Would remove: $path"
    else
        echo "Removing: $path"
        rm -rf -- "$path"
    fi
}

echo "Searching for directories..."
for pattern in "${DIR_PATTERNS[@]}"; do
    while IFS= read -r -d '' dir; do
        remove_path "$dir"
    done < <(find . -type d -name "$pattern" -print0)
done

echo
echo "Searching for files..."
for pattern in "${FILE_PATTERNS[@]}"; do
    while IFS= read -r -d '' file; do
        remove_path "$file"
    done < <(find . -type f -name "$pattern" -print0)
done

echo
echo "Cleanup complete."
if $DRY_RUN; then
    echo "Re-run with --force to actually delete."
fi
