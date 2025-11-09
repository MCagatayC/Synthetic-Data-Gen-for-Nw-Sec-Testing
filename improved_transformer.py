from prepare_libs import load_and_scale_data, torch, nn, DataLoader, TensorDataset, plt, sns, np, pd

# Veriyi hazırla
df, data_scaled, scaler = load_and_scale_data()
input_dim = data_scaled.shape[1]

# Sequence oluşturmak için pencereli veri
def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        sequences.append(seq)
    return np.stack(sequences)

sequences = create_sequences(data_scaled)
tensor_data = torch.tensor(sequences, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor_data), batch_size=32, shuffle=True)

# İYİLEŞTİRİLMİŞ Transformer Autoencoder
class ImprovedTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Geliştirilmiş embedding katmanı
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # Daha etkili encoder-decoder yapısı
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Çıktı katmanı iyileştirmeleri
        self.output = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x_emb = self.embedding(x)
        encoded = self.encoder(x_emb)
        decoded = self.decoder(x_emb, encoded)
        return self.output(decoded)

# Model, optimizer ve kayıp fonksiyonu
model = ImprovedTransformerAutoencoder(input_dim=input_dim, seq_len=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
loss_fn = nn.MSELoss()

# Erken durdurma mekanizması
best_loss = float('inf')
patience = 10
patience_counter = 0

# Eğitim döngüsü
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch, in loader:
        recon = model(batch)
        loss = loss_fn(recon, batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    scheduler.step(avg_loss)
    
    # Validation loss hesaplama
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch, in loader:
            recon = model(batch)
            val_loss += loss_fn(recon, batch).item()
        val_loss /= len(loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Erken durdurma kontrolü
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# En iyi modeli yükle
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Gelişmiş sentetik veri üretimi
with torch.no_grad():
    idx = np.random.choice(len(data_scaled) - 10, 50, replace=False)
    seeds = np.stack([data_scaled[i:i+10] for i in idx])
    seeds_tensor = torch.tensor(seeds, dtype=torch.float32)
    
    noise_level = 0.02 * seeds_tensor.std()
    seeds_tensor += torch.randn_like(seeds_tensor) * noise_level
    
    gen_seq = model(seeds_tensor).numpy()
    gen_flat = gen_seq.reshape(-1, input_dim)
    x_gen = scaler.inverse_transform(gen_flat)

# Gelişmiş görselleştirme
plt.figure(figsize=(15, 6))
for i in range(min(3, input_dim)):
    plt.subplot(1, 3, i+1)
    sns.kdeplot(df.iloc[:, i], label='Real', fill=True)
    sns.kdeplot(x_gen[:, i], label='Generated', fill=True)
    plt.title(f"Distribution of Feature {i+1} Dağılımı")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
plt.tight_layout()
plt.savefig("improved_transformer_output.png")
print("Grafik improved_transformer_output.png dosyasına kaydedildi.")
plt.close()