import os
import argparse

import wandb

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers
import tensorflow as tf


def build_vae(latent_dim: int = 16):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)

    def sample(args):
        z_m, z_lv = args
        eps = tf.random.normal(shape=tf.shape(z_m))
        return z_m + tf.exp(0.5 * z_lv) * eps

    z = layers.Lambda(sample, name="z")([z_mean, z_logvar])

    # Decoder
    dec_in = keras.Input(shape=(latent_dim,))
    y = layers.Dense(128, activation="relu")(dec_in)
    y = layers.Dense(256, activation="relu")(y)
    y = layers.Dense(28 * 28, activation=None)(y)
    outputs = layers.Activation("sigmoid")(y)
    outputs = layers.Reshape((28, 28, 1))(outputs)
    decoder = keras.Model(dec_in, outputs, name="decoder")

    recon = decoder(z)
    vae = keras.Model(inputs, recon, name="vae")

    # Loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def vae_loss(x, x_hat):
        recon_loss = bce(tf.reshape(x, (-1, 28 * 28)), tf.reshape(x_hat, (-1, 28 * 28))) * 28 * 28
        kl = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)
        return tf.reduce_mean(recon_loss + kl)

    vae.add_loss(vae_loss(inputs, recon))
    vae.add_metric(tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=1)), name="kl", aggregation="mean")
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return vae


def load_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    return x_train, x_test


def main():
    parser = argparse.ArgumentParser(description="Didactic Keras VAE on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--project", type=str, default="smolVAE")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke test (few steps, small batch)")
    args = parser.parse_args()

    # GPU info/logging
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    wandb.init(project=args.project, entity=args.entity, config={
        "framework": "keras-tf",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "optimizer": "adam",
        "dataset": "mnist",
        "gpu_detected": len(gpus) > 0,
    })

    model = build_vae(args.latent_dim)
    x_train, x_test = load_data()
    # Build tf.data pipelines to support steps_per_epoch
    batch_size = 32 if args.smoke else args.batch_size
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq=100),
        wandb.keras.WandbModelCheckpoint(filepath="wandb-model")
    ]

    steps_per_epoch = 5 if args.smoke else None
    validation_steps = 2 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Log reconstructions
    x_hat = model.predict(x_test[:16], verbose=0)
    images = []
    for i in range(16):
        images.append(wandb.Image(x_test[i], caption=f"orig_{i}"))
        images.append(wandb.Image(x_hat[i], caption=f"recon_{i}"))
    wandb.log({"examples": images})

    wandb.finish()


if __name__ == "__main__":
    main()
