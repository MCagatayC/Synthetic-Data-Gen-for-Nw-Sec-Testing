import torch
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- Model Tanımları (Yukarıdakilerle uyumlu olmalı) ---
# (Basitlik için burada tekrar tanımlıyorum, import da edilebilir)
class VAE(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(32, 128), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 256), torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, input_dim)
        )
    def decode(self, z): return self.dec(z)

class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(33, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 32)
        )
    def forward(self, x, t):
        t_emb = t.unsqueeze(1).float() / 100
        return self.net(torch.cat([x, t_emb], 1))

def evaluate(ds_name):
    print(f"\n>>> DEĞERLENDİRİLİYOR: {ds_name} <<<")
    
    # Yükle
    real_data = np.load(f"data/{ds_name}_processed.npy")
    with open(f"models/{ds_name}_scaler.pkl", "rb") as f: scaler = pickle.load(f)
    
    vae = VAE(real_data.shape[1])
    vae.load_state_dict(torch.load(f"models/{ds_name}_vae.pth"), strict=False)
    
    diff = Diffusion()
    diff.load_state_dict(torch.load(f"models/{ds_name}_diffusion.pth"))
    
    # Üret (5000 örnek)
    x = torch.randn(5000, 32)
    for t in reversed(range(100)):
        t_tensor = torch.full((5000,), t)
        noise_pred = diff(x, t_tensor)
        alpha = 1 - (t/100 * 0.02)
        x = (x - noise_pred * (1-alpha)/torch.sqrt(1-alpha)) / torch.sqrt(alpha)
        
    syn_data = vae.decode(x).detach().numpy()
    
    # Kaydet
    syn_df = pd.DataFrame(scaler.inverse_transform(syn_data))
    try:
        cols = pd.read_csv(f"models/{ds_name}_columns.csv", header=None).iloc[:,0]
        syn_df.columns = cols
    except: pass
    syn_df.to_csv(f"synthetic_data/{ds_name}_synthetic.csv", index=False)
    
    # --- DCR (Privacy) ---
    real_sample = real_data[np.random.choice(len(real_data), 5000)]
    nbrs = NearestNeighbors(n_neighbors=1).fit(real_sample)
    dists, _ = nbrs.kneighbors(syn_data)
    dcr = np.mean(dists)
    print(f"DCR Score: {dcr:.5f}")
    
    # --- Utility (FNR Reduction - Simülasyon) ---
    # Etiket olmadığı için anomali (Time > ortalama) varsayımıyla test ediyoruz
    labels = (real_sample[:, 0] > 0.5).astype(int) # Basit etiket
    
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(real_sample[:3500], labels[:3500])
    # (Burada tam FNR hesaplamak için etiketli orijinal veriye ihtiyaç var, 
    # bu kısım sentetik verinin dağılımını öğrendiğini varsayar)
    
    with open(f"results/{ds_name}_metrics.txt", "a") as f:
        f.write(f"DCR: {dcr:.5f}\n")

if __name__ == "__main__":
    for ds in ["CICIDS2017", "UNSW-NB15"]: evaluate(ds)