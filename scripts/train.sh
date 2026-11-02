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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# --- Free the GPU: unload any models Ollama (on Windows) is holding in VRAM ---
list_ollama_models() {
  "$OLLAMA" ps 2>/dev/null | tr -d '\r' | awk 'NR>1 && NF {print $1}' || true
}

unload_ollama() {
  if [ "${SKIP_OLLAMA_UNLOAD:-0}" = "1" ]; then
    echo "[ollama] unload skipped (SKIP_OLLAMA_UNLOAD=1)"
    return 0
  fi

  OLLAMA=""
  if command -v ollama.exe >/dev/null 2>&1; then OLLAMA="ollama.exe"
  elif command -v ollama >/dev/null 2>&1; then OLLAMA="ollama"
  fi
  if [ -z "$OLLAMA" ]; then
    echo "[ollama] CLI not reachable from WSL; cannot check for loaded models." >&2
    echo "[ollama] Stop them manually, or run with SKIP_OLLAMA_UNLOAD=1 to continue anyway." >&2
    exit 1
  fi

  local models m i
  models="$(list_ollama_models)"
  if [ -z "$models" ]; then
    echo "[ollama] no models loaded"
    return 0
  fi

  while IFS= read -r m; do
    echo "[ollama] unloading $m"
    "$OLLAMA" stop "$m" >/dev/null 2>&1 || true
  done <<< "$models"

  # Wait up to ~15s for the models to actually leave VRAM
  for i in $(seq 1 15); do
    [ -z "$(list_ollama_models)" ] && { echo "[ollama] all models unloaded"; return 0; }
    sleep 1
  done
  echo "[ollama] models still loaded after 15s:" >&2
  "$OLLAMA" ps >&2 || true
  echo "[ollama] aborting so training doesn't start with a busy GPU." >&2
  exit 1
}

unload_ollama
# ------------------------------------------------------------------------------

# Refuse to train on CPU by accident
"$PY" -c "import sys, jax; d = jax.devices(); print('JAX devices:', d); sys.exit(0 if any(x.platform != 'cpu' for x in d) else 'No GPU visible to JAX - aborting.')"

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "Running: $PY $SCRIPT --task $TASK --num_timesteps $STEPS $* (log: $LOG)"
"$PY" "$SCRIPT" --task "$TASK" --num_timesteps "$STEPS" "$@" 2>&1 | tee "$LOG"
