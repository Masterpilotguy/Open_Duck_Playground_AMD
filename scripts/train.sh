#!/usr/bin/env bash
# Run Open Duck Playground training with the existing .venv. Never calls uv sync.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
SCRIPT="${SCRIPT:-playground/my_duck_mini/runner.py}"
TASK="${TASK:-flat_terrain_backlash}"
STEPS="${STEPS:-1000000}"

export HSA_ENABLE_DXG_DETECTION=1
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export UV_NO_SYNC=1
export PYTHONUNBUFFERED=1

# Refuse to train on CPU by accident
"$PY" -c "import sys, jax; d = jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform != 'cpu' for x in d) else 'No GPU visible to JAX - aborting.')"

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "Running: $PY $SCRIPT --task $TASK --num_timesteps $STEPS $* (log: $LOG)"
"$PY" "$SCRIPT" --task "$TASK" --num_timesteps "$STEPS" "$@" 2>&1 | tee "$LOG"
