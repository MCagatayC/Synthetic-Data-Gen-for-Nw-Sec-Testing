#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def calculate_utility(real_df, synthetic_df):
    # Ortak kolonları bul ve sırala
    common_cols = sorted(list(set(real_df.columns).intersection(set(synthetic_df.columns))))
    if len(common_cols) == 0:
        raise ValueError("Gerçek ve sentetik veri arasında ortak kolon bulunamadı.")
    
    real_df_sorted = real_df[common_cols].copy()
    synthetic_df_sorted = synthetic_df[common_cols].copy()
    
    # MinMaxScaler ile normalize et
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_df_sorted)
    synthetic_scaled = scaler.transform(synthetic_df_sorted)
    
    # Kolon ortalamaları üzerinden correlation tabanlı utility
    real_means = np.mean(real_scaled, axis=0)
    synthetic_means = np.mean(synthetic_scaled, axis=0)
    
    # Korelasyon benzerliği (1 - normalized Euclidean distance)
    dist = np.linalg.norm(real_means - synthetic_means)
    utility_score = np.clip(1 - dist, 0, 1)
    
    return utility_score

def main():
    if len(sys.argv) != 3:
        print("Kullanım: python3 calculate_utility.py <gercek_csv> <sentetik_csv>")
        sys.exit(1)
    
    real_path = sys.argv[1]
    synthetic_path = sys.argv[2]
    
    print("--- Utility Hesaplama Aracı Başlatılıyor ---")
    
    try:
        real_df = pd.read_csv(real_path)
        print(f"Gerçek Veri Yüklendi. Şekil: {real_df.shape}")
    except Exception as e:
        print(f"Gerçek veri yüklenemedi: {e}")
        sys.exit(1)
    
    try:
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"Sentetik Veri Yüklendi. Şekil: {synthetic_df.shape}")
    except Exception as e:
        print(f"Sentetik veri yüklenemedi: {e}")
        sys.exit(1)
    
    try:
        utility_score = calculate_utility(real_df, synthetic_df)
        print(f"\n--- Utility Sonucu ---\nUtility: {utility_score:.6f}")
        
        with open("utility_result.txt", "w") as f:
            f.write(f"Utility: {utility_score:.6f}\n")
        print("Sonuç 'utility_result.txt' dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"HATA OLUŞTU: Hesaplama sırasında beklenmedik bir hata meydana geldi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
