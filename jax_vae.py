import argparse
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
import wandb


class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class Decoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(28 * 28)(x)
        return x  # logits


class VAE(nn.Module):
    latent_dim: int

    def setup(self):
        self.enc = Encoder(self.latent_dim)
        self.dec = Decoder(self.latent_dim)

    def __call__(self, x, rng):
        mu, logvar = self.enc(x)
        eps = random.normal(rng, mu.shape)
        z = mu + jnp.exp(0.5 * logvar) * eps
        logits = self.dec(z)
        return logits, mu, logvar


def bce_with_logits(logits, targets):
    # logits: (B, 784), targets: (B, 784)
    return jnp.sum(jnp.clip(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def kl_div(mu, logvar):
    return -0.5 * jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar))


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    latent_dim: int = 16
    lr: float = 1e-3
    project: str = "smolVAE"
    entity: str | None = None
    smoke: bool = False


def prepare_ds(split: str, batch_size: int, shuffle: bool):
    ds = tfds.load("mnist", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10_000)
    ds = ds.map(lambda x, y: (tfds.as_dataframe.tf.cast(x, jnp.float32) / 255.0, y))  # type: ignore
    ds = ds.map(lambda x, y: (jnp.expand_dims(x, -1), y))  # add channel
    ds = ds.batch(batch_size)
    ds = tfds.as_numpy(ds)
    return ds


def numpy_iterator(split: str, batch_size: int, shuffle: bool):
    import tensorflow as tf  # used only for dataset ops if available

    AUTOTUNE = tf.data.AUTOTUNE
    ds = tfds.load("mnist", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10_000)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return tfds.as_numpy(ds)


def create_state(rng, cfg: Config):
    model = VAE(cfg.latent_dim)
    dummy_x = jnp.zeros((1, 28, 28, 1), dtype=jnp.float32)
    params = model.init(rng, dummy_x, rng)["params"]
    tx = optax.adam(cfg.lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


@jax.jit
def train_step(state, x, rng):
    def loss_fn(params):
        logits, mu, logvar = state.apply_fn({"params": params}, x, rng)
        logits = logits.reshape((x.shape[0], -1))
        targets = x.reshape((x.shape[0], -1))
        recon = bce_with_logits(logits, targets)
        kl = kl_div(mu, logvar)
        loss = (recon + kl) / x.shape[0]
        return loss, (recon / x.shape[0], kl / x.shape[0])

    (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, recon, kl


@jax.jit
def eval_step(state, x, rng):
    logits, mu, logvar = state.apply_fn({"params": state.params}, x, rng)
    logits = logits.reshape((x.shape[0], -1))
    targets = x.reshape((x.shape[0], -1))
    recon = bce_with_logits(logits, targets)
    kl = kl_div(mu, logvar)
    loss = (recon + kl) / x.shape[0]
    return loss


def main():
    parser = argparse.ArgumentParser(description="Didactic JAX (Flax) VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--project", type=str, default="smolVAE")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke test (few steps, small batch)")
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        project=args.project,
        entity=args.entity,
        smoke=args.smoke,
    )

    wandb.init(project=cfg.project, entity=cfg.entity, config=cfg.__dict__)

    # GPU info
    platform = jax.default_backend()
    xla_backend = jax.lib.xla_bridge.get_backend().platform
    wandb.log({"jax_backend": platform, "xla_platform": xla_backend})

    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    state, model = create_state(init_rng, cfg)

    global_step = 0
    epochs = 1 if cfg.smoke else cfg.epochs
    max_train_steps = 10 if cfg.smoke else None
    max_eval_steps = 3 if cfg.smoke else None
    for epoch in range(epochs):
        # Fresh iterators each epoch
        train_ds = numpy_iterator("train", 32 if cfg.smoke else cfg.batch_size, shuffle=True)
        test_ds = numpy_iterator("test", 32 if cfg.smoke else cfg.batch_size, shuffle=False)

        # Train
        step = 0
        for batch in train_ds:
            x = batch[0].astype(np.float32)
            rng, step_rng = random.split(rng)
            state, loss, recon, kl = train_step(state, jnp.array(x), step_rng)
            if global_step % 100 == 0:
                wandb.log({"loss": float(loss), "recon": float(recon), "kl": float(kl), "epoch": epoch}, step=global_step)
            global_step += 1
            step += 1
            if max_train_steps is not None and step >= max_train_steps:
                break

        # Eval
        eval_loss = 0.0
        n = 0
        step = 0
        for batch in test_ds:
            x = batch[0].astype(np.float32)
            bs = x.shape[0]
            rng, step_rng = random.split(rng)
            l = eval_step(state, jnp.array(x), step_rng)
            eval_loss += float(l) * bs
            n += bs
            step += 1
            if max_eval_steps is not None and step >= max_eval_steps:
                break
        eval_loss /= max(1, n)
        wandb.log({"eval_loss": eval_loss, "epoch": epoch}, step=global_step)

        # Log a few reconstructions
        sample_iter = iter(numpy_iterator("test", 16, shuffle=False))
        sample = next(sample_iter)[0].astype(np.float32)
        rng, step_rng = random.split(rng)
        logits, _, _ = model.apply({"params": state.params}, jnp.array(sample), step_rng)
        recon = jax.nn.sigmoid(logits).reshape((-1, 28, 28, 1))
        images = []
        for i in range(min(16, recon.shape[0])):
            images.append(wandb.Image(sample[i], caption=f"orig_{i}"))
            images.append(wandb.Image(np.array(recon[i]), caption=f"recon_{i}"))
        wandb.log({"examples": images}, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main()
