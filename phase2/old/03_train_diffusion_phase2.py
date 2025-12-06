import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------
# Basit bir Diffusion Model (DDPM) — Phase2 için minimal eğitim modeli
# -------------------------------------------------------------------

class SimpleDiffusionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------
# Phase2: Preprocess edilmiş verileri ve latent vektörleri yükle
# -------------------------------------------------------------------
DATASETS = ["CICIDS2017", "UNSW-NB15"]

PROCESSED_DIR = "data"
LATENT_DIR = "latent_dataset"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_phase2_data(ds):
    processed_path = os.path.join(PROCESSED_DIR, f"{ds}_processed.npy")
    latent_path = os.path.join(LATENT_DIR, f"{ds}_latent.npy")

    if not os.path.exists(processed_path):
        print(f"❌ Preprocess dosyası bulunamadı: {processed_path}")
        return None, None

    if not os.path.exists(latent_path):
        print(f"❌ Latent dosyası bulunamadı: {latent_path}")
        return None, None

    print(f"✓ {ds} yükleniyor...")
    data = np.load(processed_path).astype(np.float32)
    latent = np.load(latent_path).astype(np.float32)

    return data, latent


# -------------------------------------------------------------------
# Diffusion Eğitimi
# -------------------------------------------------------------------
def train_diffusion(ds_name, data):
    input_dim = data.shape[1]
    print(f"\n--- {ds_name} Diffusion Eğitimi Başlıyor ---")
    print(f"Veri boyutu: {data.shape}")

    model = SimpleDiffusionModel(input_dim)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    EPOCHS = 10

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for batch, in loader:
            noise = torch.randn_like(batch)
            optim.zero_grad()

            pred = model(batch + noise)
            loss = ((pred - batch) ** 2).mean()

            loss.backward()
            optim.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} → Loss: {total_loss:.4f}")

    # Model kaydet
    save_path = os.path.join(MODEL_DIR, f"{ds_name}_diffusion.pth")
    torch.save(model.state_dict(), save_path)

    print(f"✓ Diffusion modeli kaydedildi → {save_path}\n")


# -------------------------------------------------------------------
# ANA ÇALIŞMA
# -------------------------------------------------------------------
if __name__ == "__main__":
    for ds in DATASETS:
        data, latent = load_phase2_data(ds)
        if data is None:
            continue

        # Diffusion model preprocess edilmiş veriye göre eğitiliyor
        train_diffusion(ds, data)

    print("\n--- ✔ Diffusion Phase2 Tamamlandı ---")
