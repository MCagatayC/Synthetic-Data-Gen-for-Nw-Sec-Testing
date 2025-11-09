import time
import os
from prepare_data import load_and_scale_data, torch, np, nn

# --- VAE Mimarisi (v3 - 01... ile aynı) ---
class VAE_v3(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )
    def encode(self, x):
        h = self.encoder(x); return self.mu(h), self.logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); return self.decode(z), mu, logvar
# --- Bitiş VAE v3 Mimarisi ---

def main():
    print("--- Adım 2 (v3 Mimarisi): Gizli Alan Veri Seti Oluşturuluyor ---")
    
    # YENİ: 4 değer döndürüyor
    df_orig, data_scaled, _, _ = load_and_scale_data() 
    if df_orig is None: return

    input_dim = data_scaled.shape[1] # 82 olmalı
    latent_dim = 32 # YENİ: 32 ile tutarlı hale getirildi

    vae = VAE_v3(input_dim, latent_dim) # YENİ: VAE_v3
    vae.load_state_dict(torch.load('github_repo/models/vae.pth'))
    vae.eval()
    print(f"Eğitilmiş VAE (v3) modeli yüklendi. Input: {input_dim} -> Latent: {latent_dim}")

    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    with torch.no_grad():
        mu, _ = vae.encode(data_tensor)
    
    latent_data = mu.numpy()

    os.makedirs('github_repo/latent_dataset', exist_ok=True)
    np.save('github_repo/latent_dataset/latent_data.npy', latent_data)
    print(f"Gizli alan veri seti 'github_repo/latent_dataset/latent_data.npy' olarak kaydedildi. Shape: {latent_data.shape}")

if __name__ == "__main__":
    main()