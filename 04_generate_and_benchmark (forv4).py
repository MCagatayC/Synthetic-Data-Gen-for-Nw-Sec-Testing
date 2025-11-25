import time
import os
from prepare_data import (
    load_and_scale_data, load_scaler, 
    torch, nn, np, pd, plt, sns
)

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

# --- Transformer Denoiser Mimarisi (Aynı) ---
class TransformerDenoiser(nn.Module):
    def __init__(self, data_dim=32, timesteps=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.timesteps = timesteps
        self.time_emb = nn.Embedding(timesteps, d_model)
        self.data_emb = nn.Linear(data_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, data_dim)

    def forward(self, x_data, t):
        x_emb = self.data_emb(x_data.unsqueeze(1)); t_embedding = self.time_emb(t).unsqueeze(1)
        x_in = x_emb + t_embedding; transformer_out = self.transformer(x_in)
        output = self.output(transformer_out.squeeze(1)); return output
# --- Bitiş Transformer Mimarisi ---

def main():
    print("--- Adım 4 (v3 Mimarisi): VAE-LDM ile Veri Üretme ---")
    
    # YENİ: 4 değer döndürüyor
    df_real, _, scaler, column_names = load_and_scale_data() 
    if df_real is None: return

    input_dim = column_names.size # 82 olmalı
    latent_dim = 32 # YENİ: 32
    timesteps = 100
    num_samples = 5000 
    
    latent_model = TransformerDenoiser(data_dim=latent_dim, timesteps=timesteps)
    latent_model.load_state_dict(torch.load('github_repo/models/latent_diffusion.pth'))
    latent_model.eval()
    print("Latent *Transformer* Difüzyon modeli yüklendi.")

    vae = VAE_v3(input_dim, latent_dim) # YENİ: VAE_v3
    vae.load_state_dict(torch.load('github_repo/models/vae.pth'))
    vae.eval()
    print("VAE (v3) modeli yüklendi.")
    
    print(f"{num_samples} adet gizli alan vektörü üretiliyor (Sampling)...")
    beta = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1. - beta
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    with torch.no_grad():
        x = torch.randn(num_samples, latent_dim) 
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long)
            z = torch.randn_like(x) if t > 0 else 0
            alpha = alphas[t]; alpha_hat = alphas_cumprod[t]; beta_t = beta[t]
            pred_noise = latent_model(x, t_tensor)
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred_noise) + torch.sqrt(beta_t) * z
    
    latent_samples = x 
    print("Gizli alan vektörleri üretildi.")
    
    with torch.no_grad():
        synthetic_scaled_data = vae.decode(latent_samples).numpy()

    # --- YENİ: Ters Ölçekleme ve One-Hot Geri Dönüşümü ---
    # Scaler sadece sayısal sütunlara uygulanmıştı
    # Sentetik veriyi DataFrame'e koyup, scaler'ı sadece o sütunlara uygulamalıyız
    
    # 1. Sütun isimlerini yükle (01_train_vae.py'de kaydettik)
    # column_names_loaded = pd.read_csv('github_repo/models/column_names.csv', header=None).squeeze().tolist()
    # (Yukarıdaki satır, column_names'i zaten aldığımız için gereksiz, ama sağlaması)
    
    df_synthetic_scaled = pd.DataFrame(synthetic_scaled_data, columns=column_names)
    
    # 2. Orijinal sayısal sütunları bul (df_real'dan)
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    
    # 3. Scaler'ı SADECE sayısal sütunlara uygula
    df_synthetic_final = df_synthetic_scaled.copy()
    df_synthetic_final[numeric_cols] = scaler.inverse_transform(df_synthetic_scaled[numeric_cols])
    
    # 4. Kategorik sütunları (One-Hot) en yakın kategoriye yuvarla (Opsiyonel ama iyi pratik)
    categorical_cols_encoded = df_synthetic_final.columns.drop(numeric_cols)
    df_synthetic_final[categorical_cols_encoded] = df_synthetic_final[categorical_cols_encoded].round()
    
    # --- Bitiş Ters Ölçekleme ---

    csv_path = 'github_repo/synthetic_data/synthetic_traffic_VAE-LDM.csv'
    df_synthetic_final.to_csv(csv_path, index=False)
    print(f"Üretilen sentetik veri (v3) şuraya kaydedildi: {csv_path}")

    plot_path = 'github_repo/benchmarks/VAE-LDM_vs_Real.png'
    plt.figure()
    
    # Grafik için ORİJİNAL sayısal sütunlardan birini kullanalım
    feature_to_plot = numeric_cols[0] # 'Time' veya 'Length' olmalı
    
    sns.kdeplot(df_real[feature_to_plot], label='Gerçek Veri', fill=True)
    sns.kdeplot(df_synthetic_final[feature_to_plot], label='VAE-LDM (v3 Mimarisi)', fill=True)
    plt.title(f"Hibrit VAE-LDM (v3) vs Gerçek Veri ({feature_to_plot} Dağılımı)")
    plt.legend()
    plt.savefig(plot_path)
    print(f"Karşılaştırma grafiği şuraya kaydedildi: {plot_path}")
    plt.close()
    
if __name__ == "__main__":
    main()