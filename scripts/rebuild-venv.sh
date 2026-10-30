#!/usr/bin/env bash
# Usage: ./scripts/rebuild-venv.sh env-snapshots/<stamp>
set -euo pipefail
cd "$(dirname "$0")/.."

SNAP="${1:?usage: rebuild-venv.sh env-snapshots/<stamp>}"
LOCK="$SNAP/requirements.lock.txt"
AMD_INDEX="https://stable.repo.amd.com/rocm/whl-next/"

[ -d .venv ] && mv .venv ".venv.broken.$(date +%Y%m%d_%H%M%S)"
uv venv --python 3.12 .venv

grep -viE '^(rocm|jax[-_]rocm)' "$LOCK" > /tmp/pypi.txt || true
grep -iE  '^(rocm|jax[-_]rocm)' "$LOCK" > /tmp/amd.txt

uv pip install --python .venv/bin/python --no-deps -r /tmp/pypi.txt
uv pip install --python .venv/bin/python --no-deps --index-url "$AMD_INDEX" -r /tmp/amd.txt

export HSA_ENABLE_DXG_DETECTION=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
.venv/bin/python -c "import jax; print(jax.devices())"
