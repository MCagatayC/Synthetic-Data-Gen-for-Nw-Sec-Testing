import torch
import numpy as np
from torch import nn
import os

# --- KONFİGÜRASYON ---
DATASETS = ["CICIDS2017", "UNSW-NB15"]
LATENT_DIR = "latent_dataset"
os.makedirs(LATENT_DIR, exist_ok=True)

# --- VAE Sınıfı (Yükleme için minimal tanım) ---
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

def process(ds_name):
    data_path = f"data/{ds_name}_processed.npy"
    model_path = f"models/{ds_name}_vae.pth"

    if not os.path.exists(data_path):
        print(f"⚠ UYARI: Preprocess edilmiş veri bulunamadı: {data_path}")
        return

    if not os.path.exists(model_path):
        print(f"⚠ UYARI: VAE modeli bulunamadı: {model_path}")
        return

    # Veri yükle
    data = np.load(data_path)
    vae = VAE(data.shape[1])
    vae.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    vae.eval()

    # Latent oluştur
    with torch.no_grad():
        mu, _ = vae.encode(torch.tensor(data, dtype=torch.float32))

    latent_file = os.path.join(LATENT_DIR, f"{ds_name}_latent.npy")
    np.save(latent_file, mu.numpy())
    print(f"{ds_name} Latent oluşturuldu: {mu.shape} → {latent_file}")

if __name__ == "__main__":
    for ds in DATASETS:
        process(ds)
