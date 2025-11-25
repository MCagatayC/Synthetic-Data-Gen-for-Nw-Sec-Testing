#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def calculate_dcr(real_df, synthetic_df):
    # Kolon isimlerini sıralayalım
    common_cols = sorted(list(set(real_df.columns).intersection(set(synthetic_df.columns))))
    real_df_sorted = real_df[common_cols].copy()
    synthetic_df_sorted = synthetic_df[common_cols].copy()
    
    # MinMaxScaler ile normalize edelim
    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_df_sorted)
    synthetic_scaled = scaler.transform(synthetic_df_sorted)
    
    # DCR hesaplama: Euclidean distance ortalaması
    dcr_value = np.mean(np.sqrt(np.sum((real_scaled.mean(axis=0) - synthetic_scaled.mean(axis=0))**2)))
    return dcr_value

def main():
    if len(sys.argv) != 3:
        print("Kullanım: python3 calculate_dcr.py <gercek_csv> <sentetik_csv>")
        sys.exit(1)
    
    real_path = sys.argv[1]
    synthetic_path = sys.argv[2]
    
    print("--- DCR Hesaplama Aracı Başlatılıyor ---")
    
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
    
    # DCR hesaplama
    try:
        dcr_value = calculate_dcr(real_df, synthetic_df)
        print(f"\n--- DCR Sonucu ---\nDCR: {dcr_value:.6f}")
        
        # Sonucu dosyaya yaz
        with open("dcr_resultv6.txt", "w") as f:
            f.write(f"DCR: {dcr_value:.6f}\n")
        print("Sonuç 'dcr_resultv6.txt' dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"HATA OLUŞTU: Hesaplama sırasında beklenmedik bir hata meydana geldi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
