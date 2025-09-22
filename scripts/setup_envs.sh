#!/usr/bin/env bash
set -euo pipefail

# Simple bootstrap for three environments: Keras, PyTorch, JAX
# Defaults: Keras CPU, Torch CPU, JAX CUDA12 (PJRT). Override with flags.
#
# Usage examples:
#   bash scripts/setup_envs.sh                # Torch CPU, JAX GPU
#   bash scripts/setup_envs.sh --torch-gpu    # Torch GPU (cu121 wheels)
#   bash scripts/setup_envs.sh --cpu-only     # Force CPU for Torch and JAX
#   bash scripts/setup_envs.sh --jax-cpu      # JAX CPU only

TORCH_GPU=0
JAX_GPU=1

for arg in "$@"; do
  case "$arg" in
    --cpu-only)
      TORCH_GPU=0
      JAX_GPU=0
      ;;
    --torch-gpu)
      TORCH_GPU=1
      ;;
    --torch-cpu)
      TORCH_GPU=0
      ;;
    --jax-gpu)
      JAX_GPU=1
      ;;
    --jax-cpu)
      JAX_GPU=0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

# echo "[1/3] Keras: creating venv vkeras and installing deps..."
# uv venv vkeras >/dev/null
# source vkeras/bin/activate
# uv pip install tensorflow wandb >/dev/null
# deactivate

echo "[2/3] Torch: creating venv vtorch and installing deps..."
uv venv vtorch >/dev/null
source vtorch/bin/activate
if [[ "$TORCH_GPU" == "1" ]]; then
  uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision wandb >/dev/null
else
  uv pip install torch torchvision wandb >/dev/null
fi
deactivate

echo "[3/3] JAX: creating venv vjax and installing deps..."
uv venv vjax >/dev/null
source vjax/bin/activate
if [[ "$JAX_GPU" == "1" ]]; then
  uv pip install --upgrade "jax[cuda12]" >/dev/null
else
  uv pip install jax >/dev/null
fi
uv pip install flax optax wandb pillow >/dev/null

echo "Verifying JAX devices..."
python - <<'PY'
import jax
print('JAX devices:', jax.devices())
print('JAX backend:', jax.default_backend())
PY
deactivate

echo "Done. Activate envs with:"
echo "  source vkeras/bin/activate   # Keras"
echo "  source vtorch/bin/activate   # PyTorch"
echo "  source vjax/bin/activate     # JAX"

