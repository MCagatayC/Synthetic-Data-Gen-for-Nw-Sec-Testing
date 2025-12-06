#!/usr/bin/env python3
import os
import numpy as np
import torch
import pandas as pd
import pickle

LATENT_DIM = 32
DATASETS = ["CICIDS2017", "UNSW-NB15"]
MODEL_DIR = "models"

class DecoderSimple(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(LATENT_DIM, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, out_dim)
        )
    def forward(self, z):
        return self.dec(z)


class ARPrior(torch.nn.Module):
    """
    ARPrior which matches the checkpoint:
    - shared[i] first Linear expects in_features = i (if i>0) else 1, out=128,
      then Linear(128,64).
    - out[0] is Linear(1,2) (uses raw inp), out[i>0] is Linear(64,2) (uses shared output).
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # build shared modules: shared[i] expects input dim = i (or 1 for i==0)
        shared = []
        for i in range(latent_dim):
            in_feat = i if i > 0 else 1
            seq = torch.nn.Sequential(
                torch.nn.Linear(in_feat, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU()
            )
            shared.append(seq)
        self.shared = torch.nn.ModuleList(shared)

        # build out modules:
        # out[0]: takes raw inp (1) => Linear(1,2)
        # out[i>0]: takes shared output (64) => Linear(64,2)
        outs = []
        for i in range(latent_dim):
            if i == 0:
                outs.append(torch.nn.Linear(1, 2))
            else:
                outs.append(torch.nn.Linear(64, 2))
        self.out = torch.nn.ModuleList(outs)

    def forward(self, z):
        # z: (batch, latent_dim) - partially filled during sampling
        batch = z.shape[0]
        mus = []
        lvs = []
        for i in range(self.latent_dim):
            if i == 0:
                inp = torch.zeros(batch, 1, device=z.device, dtype=z.dtype)
                # for i==0 the checkpoint expects out[0] to consume the raw inp (1d)
                out = self.out[0](inp)
            else:
                inp = z[:, :i]               # shape (batch, i)
                # feed inp directly to shared[i] which expects i features
                h = self.shared[i](inp)      # shape (batch, 64)
                out = self.out[i](h)         # out layer expects 64 -> 2
            mus.append(out[:, 0:1])   # (batch,1)
            lvs.append(out[:, 1:2])   # (batch,1)
        mus = torch.cat(mus, dim=1)   # (batch, latent_dim)
        lvs = torch.cat(lvs, dim=1)
        return mus, lvs


def generate(ds_name, n_samples=5000):
    proc = f"data/{ds_name}_processed.npy"
    scaler_file = f"models/{ds_name}_scaler.pkl"
    model_file = f"models/{ds_name}_natrvae_v6.pth"

    if not os.path.exists(proc) or not os.path.exists(model_file) or not os.path.exists(scaler_file):
        print("Eksik dosya, atlanıyor:", ds_name)
        return

    data = np.load(proc).astype(np.float32)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    # load checkpoint (expects dict with keys "dec" and "prior")
    ck = torch.load(model_file, map_location="cpu")

    dec = DecoderSimple(data.shape[1])
    prior = ARPrior()

    # load weights (these keys must exist in your checkpoint)
    dec.load_state_dict(ck["dec"])
    prior.load_state_dict(ck["prior"])
    dec.eval()
    prior.eval()

    zs = []
    batch = 512

    with torch.no_grad():
        for bstart in range(0, n_samples, batch):
            b = min(batch, n_samples - bstart)
            # initialize zero z; will fill coordinates autoregressively
            z = torch.zeros(b, LATENT_DIM)

            for i in range(LATENT_DIM):
                mus, lvs = prior(z)     # prior uses current partial z
                mu_i = mus[:, i]
                lv_i = lvs[:, i]
                std = torch.exp(0.5 * lv_i)
                z[:, i] = mu_i + torch.randn_like(std) * std

            zs.append(z)

        zs = torch.cat(zs, dim=0).numpy()
        syn = dec(torch.tensor(zs)).numpy()

    # inverse transform if scaler available
    try:
        inv = scaler.inverse_transform(syn)
    except Exception:
        inv = syn

    os.makedirs("synthetic_data", exist_ok=True)
    try:
        cols = pd.read_csv(f"models/{ds_name}_columns.csv", header=None).iloc[:, 0]
        df = pd.DataFrame(inv, columns=cols)
    except Exception:
        df = pd.DataFrame(inv)
    out = f"synthetic_data/{ds_name}_synthetic_natrvae_v6.csv"
    df.to_csv(out, index=False)
    print("✓ NATVAEv6 sentetik kaydedildi:", out)


if __name__ == "__main__":
    for ds in DATASETS:
        generate(ds, n_samples=5000)
