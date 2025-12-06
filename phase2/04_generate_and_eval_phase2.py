#!/usr/bin/env python3
import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from pathlib import Path

# ==========================
# CONFIG
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS = ["CICIDS2017", "UNSW-NB15"]

MODEL_DIR = Path("models")
NATVAE_DIR = Path("natvae_models")
SCALER_DIR = Path("scalers")
LATENT_DIR = Path("latents_phase2")
COLUMNS_DIR = Path("columns")
RESULT_DIR = Path("results_eval")

RESULT_DIR.mkdir(exist_ok=True)

LATENT_DIM = 32  # phase2 VAE NATVAE ortak latent boyutu


# ==========================
# VAE MODELİ
# ==========================
class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(128, LATENT_DIM)
        self.logvar = nn.Linear(128, LATENT_DIM)

        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )

    def decode(self, z):
        return self.dec(z)


# ==========================
# NATVAE MODELİ
# (senin eğittiğin NATVAEv6 formatına uygun)
# ==========================
class NATVAEv6(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.mu = nn.Linear(256, LATENT_DIM)
        self.logvar = nn.Linear(256, LATENT_DIM)

        self.prior = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM)
        )

        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def decode(self, z):
        return self.dec(z)


# ==========================
# STATE DICT SAFE LOAD
# ==========================
def load_state_safe(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    try:
        model.load_state_dict(ckpt, strict=False)
        return True, ""
    except Exception as e:
        return False, str(e)


# ==========================
# EVALUATE FUNCTION
# ==========================
def evaluate(name: str):
    print("\n==============================")
    print(f">>> Reconstruction: {name}")
    print("==============================")

    t0 = time.time()

    # ------------------------
    # 1) Columns
    # ------------------------
    col_file = COLUMNS_DIR / f"{name}_columns.csv"
    if not col_file.exists():
        print(f"[ERROR] Kolon dosyası bulunamadı: {col_file}")
        return

    columns = pd.read_csv(col_file, header=None)[0].tolist()

    # ------------------------
    # 2) Scaler
    # ------------------------
    scaler_path = SCALER_DIR / f"{name}_scaler.pkl"
    if not scaler_path.exists():
        print(f"[ERROR] Scaler bulunamadı: {scaler_path}")
        return

    scaler = joblib.load(scaler_path)
    input_dim = scaler.scale_.shape[0]  # MinMaxScaler için doğru yöntem
    print(f"[OK] Kolon sayısı = {input_dim}")

    # ------------------------
    # 3) Latent
    # ------------------------
    latent_path = LATENT_DIR / f"{name}_latent.npy"
    if not latent_path.exists():
        print(f"[ERROR] Latent bulunamadı: {latent_path}")
        return

    latent = np.load(latent_path)
    print(f"[OK] Latent yüklendi: {latent.shape}")

    # ------------------------
    # 4) VAE Decode
    # ------------------------
    print("\n--- VAE Reconstruction Başlıyor ---")
    vae_model_path = MODEL_DIR / f"{name}_vae.pth"

    if not vae_model_path.exists():
        print(f"[WARN] VAE modeli yok, atlanıyor: {vae_model_path}")
    else:
        vae = VAE(input_dim).to(DEVICE)
        ok, err = load_state_safe(vae, vae_model_path)

        if not ok:
            print(f"[ERROR] VAE yüklenemedi: {err}")
        else:
            with torch.no_grad():
                z = torch.tensor(latent, dtype=torch.float32, device=DEVICE)
                out = vae.decode(z).cpu().numpy()

            reconstructed = scaler.inverse_transform(out)
            out_path = RESULT_DIR / f"{name}_vae_reconstructed.csv"
            pd.DataFrame(reconstructed, columns=columns).to_csv(out_path, index=False)

            print(f"[OK] VAE reconstruction kaydedildi: {out_path}")

    # ------------------------
    # 5) NATVAEv6 Decode
    # ------------------------
    print("\n--- NATVAEv6 Reconstruction Başlıyor ---")
    natvae_path = NATVAE_DIR / f"{name}_natvae_v6.pth"

    if not natvae_path.exists():
        print(f"[WARN] NATVAE modeli yok: {natvae_path}")
    else:
        natvae = NATVAEv6(input_dim).to(DEVICE)
        ok, err = load_state_safe(natvae, natvae_path)

        if not ok:
            print(f"[ERROR] NATVAE yüklenemedi: {err}")
        else:
            with torch.no_grad():
                z = torch.tensor(latent, dtype=torch.float32, device=DEVICE)
                out = natvae.decode(z).cpu().numpy()

            reconstructed = scaler.inverse_transform(out)
            out_path = RESULT_DIR / f"{name}_natvae_reconstructed.csv"
            pd.DataFrame(reconstructed, columns=columns).to_csv(out_path, index=False)

            print(f"[OK] NATVAE reconstruction kaydedildi: {out_path}")

    dt = time.time() - t0
    print(f"\n[DONE] {name} tamamlandı ({dt:.1f} s)")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    for ds in DATASETS:
        evaluate(ds)

    print("\nTAMAMLANDI ✔")
