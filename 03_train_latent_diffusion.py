import time
from prepare_data import torch, nn, np, DataLoader, TensorDataset

# --- Transformer Denoiser Mimarisi (Aynı) ---
class TransformerDenoiser(nn.Module):
    def __init__(self, data_dim=32, timesteps=100, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.timesteps = timesteps
        self.time_emb = nn.Embedding(timesteps, d_model)
        self.data_emb = nn.Linear(data_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, data_dim)

    def forward(self, x_data, t):
        x_emb = self.data_emb(x_data.unsqueeze(1))
        t_embedding = self.time_emb(t).unsqueeze(1)
        x_in = x_emb + t_embedding
        transformer_out = self.transformer(x_in)
        output = self.output(transformer_out.squeeze(1))
        return output
# --- Bitiş Transformer Mimarisi ---

def main():
    print("--- Adım 3 (v3 Verisi): Latent Transformer Difüzyon Eğitimi ---")
    
    latent_data = np.load('github_repo/latent_dataset/latent_data.npy')
    data_tensor = torch.tensor(latent_data, dtype=torch.float32)
    
    latent_dim = latent_data.shape[1] # Bu 32 olmalı
    timesteps = 100
    epochs = 50
    batch_size = 128 # Batch size'ı artıralım

    print(f"Latent Dim (data_dim) için eğitim: {latent_dim}")
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

    model = TransformerDenoiser(data_dim=latent_dim, timesteps=timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    beta = torch.linspace(1e-4, 0.02, timesteps)
    alphas_cumprod = torch.cumprod(1. - beta, dim=0)

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x0 in data_loader:
            t = torch.randint(0, timesteps, (x0.shape[0],))
            noise = torch.randn_like(x0)
            alpha_t = alphas_cumprod[t].unsqueeze(1)
            xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
            pred_noise = model(xt, t)
            loss = nn.functional.mse_loss(pred_noise, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss / len(data_loader):.6f}")

    end_time = time.time()
    training_time = (end_time - start_time) / 3600.0
    print(f"Latent Transformer Difüzyon (v3) eğitimi tamamlandı. Toplam süre: {training_time:.4f} saat.")
    
    torch.save(model.state_dict(), 'github_repo/models/latent_diffusion.pth')
    print("Yeni Latent Difüzyon modeli 'github_repo/models/latent_diffusion.pth' olarak kaydedildi.")

    # Benchmark dosyasını güncelle
    try:
        with open('github_repo/benchmarks/training_times.txt', 'r') as f:
            lines = f.readlines()
        # Sadece VAE süresini ve input dim'i tut
        filtered_lines = [l for l in lines if 'VAE_v3' in l]
        with open('github_repo/benchmarks/training_times.txt', 'w') as f:
            f.writelines(filtered_lines)
            f.write(f"Latent_Transformer_Diffusion_v3_Time_Hours: {training_time}\n")
    except FileNotFoundError:
         with open('github_repo/benchmarks/training_times.txt', 'a') as f:
            f.write(f"Latent_Transformer_Diffusion_v3_Time_Hours: {training_time}\n")

if __name__ == "__main__":
    main()