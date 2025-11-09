from prepare_libs import load_and_scale_data, torch, nn, DataLoader, TensorDataset, plt, sns, np, pd

# Veriyi hazırla
df, data_scaled, scaler = load_and_scale_data()
input_dim = data_scaled.shape[1]

# Basitleştirilmiş Denoising Diffusion Probabilistic Model (DDPM)
timesteps = 100
beta = torch.linspace(1e-4, 0.02, timesteps)
alphas = 1. - beta
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Basit MLP noise tahmin modeli
class MLPDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        t_emb = t.unsqueeze(1).float() / timesteps
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

model = MLPDenoiser()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
data_loader = DataLoader(data_tensor, batch_size=64, shuffle=True)

# Eğitim döngüsü
for epoch in range(10):
    for x0 in data_loader:
        t = torch.randint(0, timesteps, (x0.shape[0],))
        noise = torch.randn_like(x0)
        alpha_t = alphas_cumprod[t].unsqueeze(1)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

        pred_noise = model(xt, t)
        loss = nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Sampling (gürültüden veriye dönüş)
model.eval()
with torch.no_grad():
    x = torch.randn(1000, input_dim)
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((x.shape[0],), t, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else 0
        alpha = alphas[t]
        alpha_hat = alphas_cumprod[t]
        beta_t = beta[t]
        pred_noise = model(x, t_tensor)
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred_noise) + torch.sqrt(beta_t) * z
    x_gen = scaler.inverse_transform(x.numpy())

plt.figure()
sns.kdeplot(df.iloc[:, 0], label='Gerçek', fill=True)
sns.kdeplot(x_gen[:, 0], label='Diffusion Üretilmiş', fill=True)
plt.title("Diffusion ile Üretilmiş vs Gerçek Veri")
plt.legend()
#plt.show()
plt.savefig("diffusion_output.png")
print("Grafik diffusion_output.png dosyasına kaydedildi.")
plt.close()
