import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

import wandb


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        logits = self.dec(z)
        return logits.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


@dataclass
class Config:
    batch_size: int = 128
    epochs: int = 5
    latent_dim: int = 16
    lr: float = 1e-3
    project: str = "smolVAE"
    entity: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    smoke: bool = False


def loss_fn(x, logits, mu, logvar):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon + kl) / x.size(0), recon / x.size(0), kl / x.size(0)


def get_dataloaders(batch_size: int):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


def train(cfg: Config):
    wandb.init(project=cfg.project, entity=cfg.entity, config=cfg.__dict__)
    device = torch.device(cfg.device)
    model = VAE(cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader, test_loader = get_dataloaders(32 if cfg.smoke else cfg.batch_size)

    if torch.cuda.is_available():
        wandb.log({"cuda": True, "cuda_name": torch.cuda.get_device_name(0)})
    else:
        wandb.log({"cuda": False})

    global_step = 0
    epochs = 1 if cfg.smoke else cfg.epochs
    max_steps = 10 if cfg.smoke else None
    for epoch in range(epochs):
        model.train()
        step = 0
        for x, _ in train_loader:
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, recon, kl = loss_fn(x, logits, mu, logvar)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if global_step % 100 == 0:
                wandb.log({"loss": loss.item(), "recon": recon.item(), "kl": kl.item(), "epoch": epoch}, step=global_step)
            global_step += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break

        # Eval
        model.eval()
        eval_loss = 0.0
        eval_steps = 3 if cfg.smoke else None
        step = 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits, mu, logvar = model(x)
                loss, recon, kl = loss_fn(x, logits, mu, logvar)
                eval_loss += loss.item() * x.size(0)
                step += 1
                if eval_steps is not None and step >= eval_steps:
                    break
        eval_loss /= len(test_loader.dataset)
        wandb.log({"eval_loss": eval_loss, "epoch": epoch}, step=global_step)

        # Log a small grid of reconstructions
        x, _ = next(iter(test_loader))
        x = x.to(device)[:16]
        logits, _, _ = model(x)
        x_hat = torch.sigmoid(logits)
        grid = vutils.make_grid(torch.cat([x, x_hat], dim=0), nrow=16)
        wandb.log({"recons": [wandb.Image(grid, caption=f"epoch_{epoch}")]}, step=global_step)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Didactic PyTorch VAE on MNIST")
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
    train(cfg)


if __name__ == "__main__":
    main()
