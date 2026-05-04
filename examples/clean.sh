#!/usr/bin/env bash

set -euo pipefail

# Default: dry run
DRY_RUN=true
# Extra confirmation once before deleting any '*.nnef' directories
SPECIAL_CONFIRM_NNEF_DIRS=false
# Extra confirmation once before deleting any '*.nnef.tar' archives
SPECIAL_CONFIRM_NNEF_TAR=false

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
    "*.nnef"
)

# Files to remove
FILE_PATTERNS=(
    "*.onnx"
    "*.pt"
    "*.wav"
    "*.nnef.tgz"
    "*.nnef.tar"
)

remove_path() {
    local path="$1"
    if $DRY_RUN; then
        echo "[DRY RUN] Would remove: $path"
    else
        # Ask once before deleting any .nnef directory (archives unpacked)
        if [ -d "$path" ] && [[ "$path" == *.nnef ]] && [ "$SPECIAL_CONFIRM_NNEF_DIRS" = false ]; then
            read -r -p "This will permanently remove all matched *.nnef directories. Type 'yes' to proceed: " ans
            if [[ "$ans" != "yes" ]]; then
                echo "Skipping .nnef directories."
                SPECIAL_CONFIRM_NNEF_DIRS=skipped
                return
            fi
            SPECIAL_CONFIRM_NNEF_DIRS=true
        fi
        # Ask once before deleting any .nnef.tar archive files
        if [ -f "$path" ] && [[ "$path" == *.nnef.tar ]] && [ "$SPECIAL_CONFIRM_NNEF_TAR" = false ]; then
            read -r -p "This will permanently remove all matched *.nnef.tar archives. Type 'yes' to proceed: " ans
            if [[ "$ans" != "yes" ]]; then
                echo "Skipping .nnef.tar archives."
                SPECIAL_CONFIRM_NNEF_TAR=skipped
                return
            fi
            SPECIAL_CONFIRM_NNEF_TAR=true
        fi
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
