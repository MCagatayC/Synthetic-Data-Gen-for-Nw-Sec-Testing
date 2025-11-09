from prepare_libs import load_and_scale_data, torch, nn, DataLoader, TensorDataset, plt, sns, np, pd

# Veriyi hazırla
df, data_scaled, scaler = load_and_scale_data()
input_dim = data_scaled.shape[1]
latent_dim = 32  # artırıldı

# PyTorch dataset
tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_data), batch_size=64, shuffle=True)

# VAE bileşenleri
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
            # Sigmoid kaldırıldı
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Kayıp fonksiyonu (MSE olarak güncellendi)
vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')  # güncellendi
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

# Eğitim döngüsü
vae.train()
for epoch in range(50):
    total_loss = 0
    for batch, in loader:
        recon_batch, mu, logvar = vae(batch)
        loss = vae_loss(recon_batch, batch, mu, logvar)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.2f}")

# Sentetik veri üretimi ve görselleştirme
vae.eval()
with torch.no_grad():
    z = torch.randn(5000, latent_dim)  # artırıldı
    samples = vae.decode(z).numpy()
    samples = scaler.inverse_transform(samples)

plt.figure()
sns.kdeplot(df.iloc[:, 0], label='Gerçek', fill=True)
sns.kdeplot(samples[:, 0], label='VAE Üretilmiş', fill=True)
plt.title("VAE ile Üretilmiş vs Gerçek Veri")
plt.legend()
plt.savefig("vae_output-2.png")
print("Grafik vae_output-2.png dosyasına kaydedildi.")
plt.close()
