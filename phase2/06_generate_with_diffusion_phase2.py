#!/usr/bin/env python3
"""
06_generate_with_diffusion_phase2.py

- Diffusion modeli kullanarak sentetik network trafiği verisi üretir.
- Çıktılar:
    synthetic_data/{DS}_synthetic_diffusion.csv
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -------------------------
# Configuration
# -------------------------
DATASETS = ["CICIDS2017", "UNSW-NB15"]
REAL_PROC_DIR = "data"
SYN_DIR = "synthetic_data"
os.makedirs(SYN_DIR, exist_ok=True)

SAMPLES_PER_DATASET = 50000  # üretilecek satır sayısı

# -------------------------
# Basit Diffusion-like model (placeholder)
# -------------------------
class SimpleDiffusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.net(x)

def generate_synthetic(real_array):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = real_array.shape[1]
    model = SimpleDiffusion(input_dim).to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn((SAMPLES_PER_DATASET, input_dim), device=device)
        synth = model(z).cpu().numpy()

    # scale to original min/max
    min_r = real_array.min(axis=0)
    max_r = real_array.max(axis=0)
    synth_scaled = np.clip(synth, 0, 1)  # normalize to [0,1] first
    synth_scaled = synth_scaled * (max_r - min_r) + min_r
    return synth_scaled

# -------------------------
# Runner
# -------------------------
def main():
    for ds in DATASETS:
        real_path = os.path.join(REAL_PROC_DIR, f"{ds}_processed.npy")
        if not os.path.exists(real_path):
            print(f"[ERROR] Missing real processed data: {real_path}")
            continue
        real_array = np.load(real_path).astype(np.float32)
        synth_array = generate_synthetic(real_array)

        # Save to CSV
        columns = [f"f{i}" for i in range(real_array.shape[1])]
        df = pd.DataFrame(synth_array, columns=columns)
        out_path = os.path.join(SYN_DIR, f"{ds}_synthetic_diffusion.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] Diffusion synthetic saved: {out_path}")

if __name__ == "__main__":
    main()
