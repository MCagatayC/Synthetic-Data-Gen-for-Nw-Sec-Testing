#!/usr/bin/env python3
import os, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATASETS = ["CICIDS2017", "UNSW-NB15"]
LATENT_DIR = "latent_dataset"; MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)

class Diffusion(nn.Module):
    def __init__(self, latent_dim=32, hidden=128, timesteps=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, x, t):
        t_emb = t.unsqueeze(1).float() / float(self.timesteps)
        inp = torch.cat([x, t_emb], dim=1)
        return self.net(inp)

def load_latent(ds):
    path = os.path.join(LATENT_DIR, f"{ds}_latent.npy")
    if not os.path.exists(path):
        print(f"⚠ Latent bulunamadı: {path}"); return None
    arr = np.load(path).astype(np.float32)
    return arr

def train(ds_name):
    latent = load_latent(ds_name)
    if latent is None: return
    print(f"\n--- {ds_name} Diffusion Eğitimi --- | Latent shape: {latent.shape}")
    device = torch.device("cpu")
    latent_tensor = torch.tensor(latent)
    dataset = TensorDataset(latent_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    model = Diffusion(latent_dim=latent.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = model.timesteps
    EPOCHS = 20
    for ep in range(1, EPOCHS+1):
        total_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            t = torch.randint(0, T, (batch.shape[0],))
            noise = torch.randn_like(batch)
            alpha = 1 - (t.float()/T * 0.02)
            alpha = alpha.unsqueeze(1)
            x_noised = batch * torch.sqrt(alpha) + noise * torch.sqrt(1 - alpha)
            pred_noise = model(x_noised, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep}/{EPOCHS} - Loss: {total_loss/len(loader):.6f}")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{ds_name}_diffusion.pth"))
    print(f"✓ Kaydedildi: models/{ds_name}_diffusion.pth")

if __name__ == "__main__":
    for ds in DATASETS: train(ds)
