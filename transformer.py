from prepare_libs import load_and_scale_data, torch, nn, DataLoader, TensorDataset, plt, sns, np, pd

# Veriyi hazırla
df, data_scaled, scaler = load_and_scale_data()
input_dim = data_scaled.shape[1]

# Sequence oluşturmak için pencereli veri (Transformer sıralı veri bekler)
def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        sequences.append(seq)
    return np.stack(sequences)

# Sequence'leri oluştur
sequences = create_sequences(data_scaled)
tensor_data = torch.tensor(sequences, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)

# Transformer tabanlı sequence-to-sequence Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output = nn.Linear(64, input_dim)

    def forward(self, x):
        x_emb = self.embedding(x)  # [batch, seq_len, 64]
        x_emb = x_emb.permute(1, 0, 2)  # [seq_len, batch, 64]
        memory = self.encoder(x_emb)
        out = self.decoder(x_emb, memory)
        out = out.permute(1, 0, 2)
        return self.output(out)

# Model, optimizer ve kayıp fonksiyonu
model = TransformerAutoencoder(input_dim=input_dim, seq_len=10)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

# Eğitim döngüsü
for epoch in range(100):
    total_loss = 0
    for batch, in loader:
        recon = model(batch)
        loss = loss_fn(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Sentetik veri üretimi
model.eval()
with torch.no_grad():
    idx = np.random.choice(len(data_scaled) - 10, 10)
    seeds = np.stack([data_scaled[i:i+10] for i in idx])
    seeds_tensor = torch.tensor(seeds, dtype=torch.float32)

    # İsteğe bağlı: Hafif gürültü ekleyerek çeşitlilik sağlanabilir
    # seeds_tensor += torch.randn_like(seeds_tensor) * 0.01

    gen_seq = model(seeds_tensor).numpy()
    gen_flat = gen_seq.reshape(-1, input_dim)
    x_gen = scaler.inverse_transform(gen_flat)

# Karşılaştırmalı görselleştirme
plt.figure(figsize=(10, 5))
sns.kdeplot(df.iloc[:, 0], label='Gerçek', fill=True)
sns.kdeplot(x_gen[:, 0], label='Transformer Üretilmiş', fill=True)
plt.title("Transformer ile Üretilmiş vs Gerçek Veri")
plt.xlabel("Time")
plt.ylabel("Density")
plt.legend()
plt.savefig("transformer_output.png")
print("Grafik transformer_output.png dosyasına kaydedildi.")
plt.close()
