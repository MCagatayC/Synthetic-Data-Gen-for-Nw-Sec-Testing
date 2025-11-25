import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

DATA_DIR = '/media/dolly/1TB/tez-projesi/data'
CSV_NAME = os.path.join(DATA_DIR, 'Midterm_53_group.csv') 

def load_and_scale_data(): # Artık 'numeric_only' parametresi yok
    if not os.path.exists(CSV_NAME):
        print(f"HATA: {CSV_NAME} dosyası bulunamadı. Lütfen dataset'i '/media/dolly/1TB/tez-projesi/data/' klasörüne kopyalayın.")
        return None, None, None, None
        
    df = pd.read_csv(CSV_NAME)
    df_clean = df.dropna().reset_index(drop=True)
    
    # Sayısal ve Kategorik sütunları ayır
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    print(f"Sayısal sütunlar: {list(numeric_cols)}")
    print(f"Kategorik sütunlar: {list(categorical_cols)}")
    
    # --- One-Hot Encoding ---
    # Boyutun patlamaması için, her kategorik sütunda sadece en yaygın 20 değeri al
    # Geri kalanları 'Other' olarak birleştir
    df_encoded = pd.DataFrame(index=df_clean.index)
    
    for col in categorical_cols:
        top_categories = df_clean[col].value_counts().nlargest(20).index
        # En yaygın 20 dışındakileri 'Other' yap
        df_clean[col] = df_clean[col].apply(lambda x: x if x in top_categories else 'Other')
        
    # One-hot encoding uygula
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, dummy_na=False)
    
    # Orijinal sayısal sütunları (zaten df_encoded içindeler) ölçekle
    scaler = MinMaxScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    print(f"Veri işlendi. Yeni Shape (One-Hot sonrası): {df_encoded.shape}")
    
    # Orijinal df'i (grafik çizimi için) ve ölçeklenmiş veriyi döndür
    # YENİ: .astype(np.float32) ekleyerek object tipinden kurtul
    return df_clean, df_encoded.values.astype(np.float32), scaler, df_encoded.columns

# Scaler'ı kaydet/yükle
def save_scaler(scaler, path):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler şuraya kaydedildi: {path}")

def load_scaler(path):
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler şuradan yüklendi: {path}")
    return scaler