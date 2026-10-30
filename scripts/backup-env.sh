#!/usr/bin/env bash
# Snapshot the package list and make a restorable archive of .venv.
set -euo pipefail
cd "$(dirname "$0")/.."

export HSA_ENABLE_DXG_DETECTION=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0

STAMP="$(date +%Y%m%d_%H%M%S)"
SNAP="env-snapshots/$STAMP"
BACKUPS="$HOME/venv-backups"
mkdir -p "$SNAP" "$BACKUPS"

uv pip freeze --python .venv/bin/python > "$SNAP/requirements.lock.txt"
.venv/bin/python --version > "$SNAP/python-version.txt"
readlink -f .venv/bin/python > "$SNAP/python-path.txt"
cp pyproject.toml "$SNAP/pyproject.toml"
cp playground/common/jax_compat.py "$SNAP/jax_compat.py"
.venv/bin/python -c "import jax; print(jax.devices())" > "$SNAP/jax-devices.txt" 2>&1 || true

ARCHIVE="$BACKUPS/venv_$STAMP.tar.zst"
tar -I 'zstd -T0 -3' -cf "$ARCHIVE" .venv
ls -1t "$BACKUPS"/venv_*.tar.zst | tail -n +4 | xargs -r rm --

echo "Snapshot: $SNAP"
echo "Archive:  $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
