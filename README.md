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

We create three separate virtual environments and sync only the extra for that framework into each. Activation is used so `uv sync` installs into the active venv.

- Keras (TensorFlow):
  - `uv venv .venv-keras`
  - `source .venv-keras/bin/activate`
  - `uv sync --no-default-groups --extra keras --active`
  - `deactivate`

- PyTorch:
  - `uv venv .venv-torch`
  - `source .venv-torch/bin/activate`
  - `uv sync --no-default-groups --extra torch --active`
  - `deactivate`

- JAX (Flax + Optax + TFDS):
  - `uv venv .venv-jax`
  - `source .venv-jax/bin/activate`
  - `uv sync --no-default-groups --extra jax --active`
  - `deactivate`

Notes

- The first `uv sync` creates a `uv.lock`. Subsequent syncs are fast and reproducible.
- CPU wheels are used by default for JAX via `jax[cpu]`. If you need GPU, install the appropriate `jaxlib` wheel per JAX docs.

Run training

- Keras:
  - `source .venv-keras/bin/activate`
  - `python keras_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --project smolVAE`

- PyTorch:
  - `source .venv-torch/bin/activate`
  - `python torch_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --lr 1e-3 --project smolVAE`

- JAX:
  - `source .venv-jax/bin/activate`
  - `python jax_vae.py --epochs 5 --batch-size 128 --latent-dim 16 --lr 1e-3 --project smolVAE`

All scripts accept `--entity` to log under a specific wandb entity.

Smoke tests

- Use `--smoke` to run a minimal training loop (small batch, ~10 steps) for quick validation and CI. Examples:
  - `python keras_vae.py --smoke`
  - `python torch_vae.py --smoke`
  - `python jax_vae.py --smoke`

GPU notes

- PyTorch GPU: After syncing the `torch` extra, install CUDA-enabled wheels (bundled CUDA) inside the torch venv:
  - `source .venv-torch/bin/activate && uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision && deactivate`
- JAX GPU: After syncing the `jax` extra, install a CUDA-enabled `jaxlib` wheel matching your CUDA/cuDNN setup (example for CUDA 12):
  - `source .venv-jax/bin/activate && uv pip install --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html "jaxlib==0.4.23+cuda12.cudnn89" && deactivate`
- TensorFlow GPU: The `tensorflow` wheel expects system CUDA/cuDNN libraries (e.g., CUDA 11.8 + cuDNN 8.6 on Linux). If `tf.config.list_physical_devices('GPU')` is empty, verify your local CUDA/cuDNN installation.

What’s inside each script

- Simple MLP encoder/decoder (256→128 layers), Gaussian latent with reparameterization.
- Bernoulli likelihood with binary cross-entropy reconstruction + KL divergence.
- MNIST normalized to [0,1].
- Periodic wandb logging of losses and example reconstructions.

FAQ

- Can I run without activating environments? Yes. You can also do `uv run --extra keras python keras_vae.py`, but using three named venvs keeps frameworks isolated.
- Apple Silicon / GPU builds: For JAX GPU or Apple Metal wheels, follow upstream install guides and adjust the `jax` dependency accordingly.
