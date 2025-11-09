import time
import os
import pandas as pd
from prepare_data import load_and_scale_data, save_scaler, torch, nn, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# --- VAE Mimarisi (v3 - Aynı) ---
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

# YENİ: Beta-VAE Kayıp Fonksiyonu
def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Beta çarpanı ile KLD'nin etkisini azalt
    return recon_loss + beta * kld

def main():
    print("--- Adım 1 (v4 Mimarisi / Beta-VAE): VAE Eğitimi Başlatılıyor ---")
    
    df_orig, data_scaled, scaler, column_names = load_and_scale_data()
    if df_orig is None: return
    
    input_dim = data_scaled.shape[1]
    latent_dim = 32
    epochs = 150 # YENİ: Epoch artırıldı
    batch_size = 128
    beta_value = 0.1 # YENİ: Beta değeri (KLD'nin etkisini %10'a düşür)

    print(f"Yeni Input Dim: {input_dim}, Latent Dim: {latent_dim}, Beta: {beta_value}")

    tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=True)

    vae = VAE_v3(input_dim, latent_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    start_time = time.time()
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, in loader:
            recon_batch, mu, logvar = vae(batch)
            
            # YENİ: Beta değeri kayıp fonksiyonuna iletildi
            loss = vae_loss(recon_batch, batch, mu, logvar, beta=beta_value)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}") 

    end_time = time.time()
    training_time = (end_time - start_time) / 3600.0
    print(f"VAE (v4 - Beta) eğitimi tamamlandı. Toplam süre: {training_time:.4f} saat.")
    
    os.makedirs('github_repo/models', exist_ok=True)
    torch.save(vae.state_dict(), 'github_repo/models/vae.pth')
    save_scaler(scaler, 'github_repo/models/scaler.pkl')
    pd.Series(column_names).to_csv('github_repo/models/column_names.csv', index=False, header=False)
    
    print("VAE (v4) modeli, scaler ve sütun isimleri 'github_repo/models/' klasörüne kaydedildi.")
    
    os.makedirs('github_repo/benchmarks', exist_ok=True)
    with open('github_repo/benchmarks/training_times.txt', 'w') as f:
        f.write(f"VAE_v4_Training_Time_Hours: {training_time}\n")
        f.write(f"VAE_v4_Input_Dim: {input_dim}\n")
        f.write(f"VAE_v4_Beta: {beta_value}\n")

if __name__ == "__main__":
    main()