#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LATENT_DIM = 32

# ==========================================================
# NATVAEv6 – Decoder + AR Prior uyumlu sürüm
# ==========================================================

class DecoderSimple(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, z):
        return self.dec(z)


class ARPrior(nn.Module):
    """ Autoregressive Prior – generate_with_natrvae_v6 ile birebir aynı """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.shared = nn.ModuleList([
            nn.Sequential(
                nn.Linear(i if i > 0 else 1, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
            )
            for i in range(latent_dim)
        ])
        self.out = nn.ModuleList([
            nn.Linear(64 if i > 0 else 1, 2)
            for i in range(latent_dim)
        ])

    def forward(self, z):
        batch = z.shape[0]
        mus = []
        lvs = []
        for i in range(self.latent_dim):
            if i == 0:
                inp = torch.zeros(batch, 1)
            else:
                inp = z[:, :i]

            # reduce dimensionality
            if inp.shape[1] > 1:
                inp = inp.mean(dim=1, keepdim=True)

            h = self.shared[i](inp)
            out = self.out[i](h)
            mus.append(out[:, 0:1])
            lvs.append(out[:, 1:2])

        mus = torch.cat(mus, dim=1)
        lvs = torch.cat(lvs, dim=1)
        return mus, lvs


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.lv = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.enc(x)
        return self.mu(h), self.lv(h)


# ==========================================================
# Training
# ==========================================================

DATASETS = ["CICIDS2017", "UNSW-NB15"]


def train(ds):
    print(f"\n--- NATVAEv6 Training: {ds} ---")

    path = f"data/{ds}_processed.npy"
    if not os.path.exists(path):
        print("❌ Veri yok:", path)
        return

    data = np.load(path).astype(np.float32)
    input_dim = data.shape[1]

    enc = Encoder(input_dim)
    dec = DecoderSimple(input_dim)
    prior = ARPrior(LATENT_DIM)

    params = list(enc.parameters()) + list(dec.parameters())
    optim = torch.optim.Adam(params, lr=1e-3)

    loader = DataLoader(
        TensorDataset(torch.tensor(data)),
        batch_size=128,
        shuffle=True
    )

    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        total = 0
        for batch, in loader:
            mu, lv = enc(batch)
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            z = mu + eps * std

            recon = dec(z)
            recon_loss = ((recon - batch) ** 2).mean()

            kl = 0.5 * torch.mean(torch.exp(lv) + mu**2 - 1 - lv)

            loss = recon_loss + 0.01 * kl
            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} | Loss={total:.4f}")

    os.makedirs("models", exist_ok=True)
    out = f"models/{ds}_natrvae_v6.pth"
    torch.save({
        "enc": enc.state_dict(),
        "dec": dec.state_dict(),
        "prior": prior.state_dict(),
    }, out)

    print("✓ Saved:", out)


if __name__ == "__main__":
    for ds in DATASETS:
        train(ds)
