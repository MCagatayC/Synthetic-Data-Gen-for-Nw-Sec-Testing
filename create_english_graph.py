import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# --- 1. Dosya Yolları ---
REAL_DATA_CSV = 'data/Midterm_53_group.csv'
SYNTHETIC_DATA_CSV = 'github_repo/synthetic_data/synthetic_traffic_VAE-LDM.csv'
OUTPUT_IMAGE_NAME = "VAE-LDM_vs_Real_v4_EN.png" # Yeni dosya adı

# --- 2. Veri Yükleme ---
try:
    df_real = pd.read_csv(REAL_DATA_CSV, usecols=['Time'])
    df_synthetic = pd.read_csv(SYNTHETIC_DATA_CSV, usecols=['Time'])
    print(f"Gerçek veri yüklendi: {REAL_DATA_CSV}")
    print(f"Sentetik veri yüklendi: {SYNTHETIC_DATA_CSV}")
except FileNotFoundError as e:
    print(f"HATA: Gerekli CSV dosyası bulunamadı. {e}")
    exit()
except Exception as e:
    print(f"Veri yüklenirken bir hata oluştu: {e}")
    exit()

# --- 3. Stil Ayarları ---
sns.set_style("whitegrid")
palette = sns.color_palette("deep", 2)
real_color = palette[0]
vae_color = palette[1]
plt.figure(figsize=(10, 6))
sns.set_context("paper", font_scale=1.2)

# --- 4. Grafiği Çizme ---

# 'Gerçek Veri' (Mavi)
sns.kdeplot(
    df_real['Time'], 
    label="Real Data",              # İngilizce Etiket
    color=real_color,
    fill=True, 
    alpha=0.3,
    linewidth=1.75
)

# 'VAE-LDM' (Turuncu)
sns.kdeplot(
    df_synthetic['Time'], 
    # --- DÜZELTME BURADA ---
    label="VAE-LDM (v4 Architecture)", # v3 -> v4 olarak güncellendi
    # -------------------------
    color=vae_color,
    fill=True, 
    alpha=0.3,
    linewidth=1.75
)

# --- 5. Başlıkları ve Eksenleri Ayarlama ---
plt.title("Hybrid VAE-LDM (v4) vs Real Data (Time Distribution)", fontsize=16) # Başlık da v4 oldu
plt.xlabel("Time", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 1600)
plt.ylim(0, 0.003) 
plt.legend(fontsize=12, loc='upper right')

# --- 6. Kaydetme ---
plt.savefig(OUTPUT_IMAGE_NAME, dpi=300, bbox_inches='tight')

print(f"\nGrafik başarıyla (v4 etiketli) '{OUTPUT_IMAGE_NAME}' olarak kaydedildi.")
