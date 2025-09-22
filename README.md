smolVAE — didactic VAEs in Keras, PyTorch, and JAX

This repo contains three small, standalone scripts that train a Variational Autoencoder (VAE) on MNIST and log to Weights & Biases (wandb):

- `keras_vae.py` (Keras with TensorFlow backend)
- `torch_vae.py` (PyTorch)
- `jax_vae.py` (JAX + Flax + Optax)

Each framework has its own isolated environment managed with uv, and all dependencies are declared in a single `pyproject.toml` via extras.

Prerequisites

- Python 3.10+
- uv installed (see https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- A Weights & Biases account: https://wandb.ai/ (CLI: `wandb login` or set `WANDB_API_KEY`)

Install environments

Create one venv per framework with straight‑forward installs.

Quick start (one command):
- `bash scripts/setup_envs.sh`
  - Defaults: Keras CPU, Torch CPU, JAX GPU (CUDA 12 PJRT)
  - Options:
    - `--torch-gpu` to install Torch CUDA wheels (cu121)
    - `--jax-cpu` to install JAX CPU only
    - `--cpu-only` to force CPU for both Torch and JAX

- Keras (TensorFlow):
  - `uv venv vkeras`
  - `source vkeras/bin/activate`
  - `uv pip install tensorflow wandb`
  - `deactivate`

- PyTorch:
  - `uv venv vtorch`
  - `source vtorch/bin/activate`
  - CPU: `uv pip install torch torchvision wandb`
  - GPU (one‑liner alternative): `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision wandb`
  - `deactivate`

- JAX (GPU, PJRT CUDA 12):
  - `uv venv vjax`
  - `source vjax/bin/activate`
  - `uv pip install --upgrade "jax[cuda12]"`
  - `uv pip install flax optax wandb pillow`
  - Verify: `python -c "import jax; print(jax.devices())"` → expect `CudaDevice`

Notes

- The first `uv sync` creates a `uv.lock`. Subsequent syncs are fast and reproducible.
- If you use `uv run` inside an activated venv, pass `--active` to avoid creating a `.venv` project env: e.g., `uv run --active python jax_vae.py`.

Run training

- Keras:
  - `source .venv-keras/bin/activate`
  - `python keras_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --project smolVAE`

- PyTorch:
  - `source .venv-torch/bin/activate`
  - `python torch_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --lr 1e-3 --project smolVAE`

- JAX:
  - `source vjax/bin/activate`
  - `python jax_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --lr 1e-3 --project smolVAE`

All scripts accept `--entity` to log under a specific wandb entity.

Smoke tests

- Use `--smoke` to run a minimal training loop (small batch, ~10 steps) for quick validation and CI. Examples:
  - `python keras_vae.py --smoke`
  - `python torch_vae.py --smoke`
  - `python jax_vae.py --smoke`

GPU notes

- PyTorch: Installing from the CUDA index URL bundles CUDA libs; only an NVIDIA driver is required.
- JAX: `jax[cuda12]` installs PJRT CUDA and NVIDIA CUDA libs with `jaxlib`. Ensure a recent NVIDIA driver. Pillow is required for image logging.
- TensorFlow: The `tensorflow` wheel expects system CUDA/cuDNN. If `tf.config.list_physical_devices('GPU')` is empty, verify your CUDA/cuDNN install.

What’s inside each script

- Simple MLP encoder/decoder (256→128 layers), Gaussian latent with reparameterization.
- Bernoulli likelihood with binary cross-entropy reconstruction + KL divergence.
- MNIST normalized to [0,1].
- Periodic wandb logging of losses and example reconstructions.

FAQ

- Can I run without activating environments? Yes. You can also do `uv run --extra keras python keras_vae.py`, but using three named venvs keeps frameworks isolated.
- Apple Silicon / GPU builds: For JAX GPU or Apple Metal wheels, follow upstream install guides and adjust the `jax` dependency accordingly.
