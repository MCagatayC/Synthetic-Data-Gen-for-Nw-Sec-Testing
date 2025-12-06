import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time

# --- KONFİGÜRASYON ---
DATASETS = ["CICIDS2017", "UNSW-NB15"]
LATENT_DIM = 32
NAT_NOISE = 0.005
BETA = 0.1
EPOCHS = 50
BATCH_SIZE = 128

class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(128, LATENT_DIM)
        self.logvar = nn.Linear(128, LATENT_DIM)
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std
    def decode(self, z): return self.dec(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

def train(ds_name):
    path = f"data/{ds_name}_processed.npy"
    if not os.path.exists(path):
        print(f"⚠ UYARI: Preprocess edilmiş veri bulunamadı: {path}. {ds_name} atlanıyor.")
        return
    
    data = np.load(path)
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = VAE(data.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\n--- {ds_name} VAE Eğitimi Başlıyor ---")
    start = time.time()
    
    for ep in range(EPOCHS):
        total_loss = 0
        for x, in loader:
            x_noisy = x + torch.randn_like(x) * NAT_NOISE
            recon, mu, logvar = model(x_noisy)
            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + BETA * kld
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep+1)%10==0: 
            print(f"Epoch {ep+1}: Loss {total_loss/len(dataset):.4f}")
        
    duration = (time.time()-start)/3600
    model_file = f"models/{ds_name}_vae.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_file)
    print(f"Tamamlandı ({duration:.4f} saat) → {model_file}")
    
    os.makedirs("results", exist_ok=True)
    with open(f"results/{ds_name}_metrics.txt", "w") as f:
        f.write(f"VAE Time: {duration:.4f} h\n")

if __name__ == "__main__":
    for ds in DATASETS:
        train(ds)
