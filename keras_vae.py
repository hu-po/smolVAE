import os
import argparse

import wandb

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import layers
import tensorflow as tf


class VAE(keras.Model):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.flatten = layers.Flatten()
        self.enc_dense1 = layers.Dense(256, activation="relu")
        self.enc_dense2 = layers.Dense(128, activation="relu")
        self.z_mean = layers.Dense(latent_dim)
        self.z_logvar = layers.Dense(latent_dim)
        self.dec_dense1 = layers.Dense(128, activation="relu")
        self.dec_dense2 = layers.Dense(256, activation="relu")
        self.dec_out = layers.Dense(28 * 28, activation="sigmoid")

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="recon")
        self.kl_tracker = keras.metrics.Mean(name="kl")

    def encode(self, x):
        x = self.flatten(x)
        x = self.enc_dense1(x)
        x = self.enc_dense2(x)
        mu = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        y = self.dec_dense1(z)
        y = self.dec_dense2(y)
        y = self.dec_out(y)
        return tf.reshape(y, (-1, 28, 28, 1))

    def call(self, inputs, training=False):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def compute_losses(self, x, x_hat, mu, logvar):
        x = tf.reshape(x, (-1, 28 * 28))
        x_hat = tf.reshape(x_hat, (-1, 28 * 28))
        recon = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_hat))
        kl = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar))
        loss = (recon + kl) / tf.cast(tf.shape(x)[0], tf.float32)
        return loss, recon / tf.cast(tf.shape(x)[0], tf.float32), kl / tf.cast(tf.shape(x)[0], tf.float32)

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            x_hat, mu, logvar = self(x, training=True)
            loss, recon, kl = self.compute_losses(x, x_hat, mu, logvar)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {"loss": self.loss_tracker.result(), "recon": self.recon_tracker.result(), "kl": self.kl_tracker.result()}

    def test_step(self, data):
        x, _ = data
        x_hat, mu, logvar = self(x, training=False)
        loss, recon, kl = self.compute_losses(x, x_hat, mu, logvar)
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {"loss": self.loss_tracker.result(), "recon": self.recon_tracker.result(), "kl": self.kl_tracker.result()}


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

    model = VAE(args.latent_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    x_train, x_test = load_data()
    # Build tf.data pipelines to support steps_per_epoch
    batch_size = 32 if args.smoke else args.batch_size
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq=100),
        wandb.keras.WandbModelCheckpoint(filepath="wandb-model.keras")
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
    x_hat, _, _ = model.predict(x_test[:16], verbose=0)
    images = []
    for i in range(16):
        images.append(wandb.Image(x_test[i], caption=f"orig_{i}"))
        images.append(wandb.Image(x_hat[i], caption=f"recon_{i}"))
    wandb.log({"examples": images})

    wandb.finish()


if __name__ == "__main__":
    main()
