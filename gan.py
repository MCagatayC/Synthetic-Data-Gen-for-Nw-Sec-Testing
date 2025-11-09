from prepare_libs import load_and_scale_data, torch, nn, DataLoader, TensorDataset, plt, sns, np, pd

# Veriyi hazırla
df, data_scaled, scaler = load_and_scale_data()
input_dim = data_scaled.shape[1]
latent_dim = 32

# PyTorch dataset
tensor_data = torch.tensor(data_scaled, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_data), batch_size=64, shuffle=True)

# GAN bileşenleri/home/dolly/Desktop/
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Modelleri oluştur ve optimizasyon ayarlarını yap
G = Generator()
D = Discriminator()
g_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# Eğitim döngüsü
for epoch in range(50):
    for real_batch, in loader:
        real_labels = torch.ones(len(real_batch), 1)
        fake_labels = torch.zeros(len(real_batch), 1)

        # Discriminator eğitimi
        z = torch.randn(len(real_batch), latent_dim)
        fake_data = G(z)
        d_real = D(real_batch)
        d_fake = D(fake_data.detach())
        d_loss = loss_fn(d_real, real_labels) + loss_fn(d_fake, fake_labels)
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # Generator eğitimi
        z = torch.randn(len(real_batch), latent_dim)
        fake_data = G(z)
        d_fake = D(fake_data)
        g_loss = loss_fn(d_fake, real_labels)
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

# Sentetik veri üretimi ve görselleştirme
with torch.no_grad():
    gen_data = G(torch.randn(1000, latent_dim)).numpy()
    gen_data = scaler.inverse_transform(gen_data)

plt.figure()
sns.kdeplot(df.iloc[:, 0], label='Gerçek', fill=True)
sns.kdeplot(gen_data[:, 0], label='GAN Üretilmiş', fill=True)
plt.title("GAN ile Üretilmiş vs Gerçek Veri")
plt.legend()
#plt.show()
plt.savefig("gan_output.png")
print("Grafik gan_output.png dosyasına kaydedildi.")
plt.close()
