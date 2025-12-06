#!/usr/bin/env python3
"""
01_prepare_phase2.py
- Veri setlerini yükler (CICIDS2017, UNSW-NB15),
- Temizler, kategorikleri one-hot yapar,
- MinMax ile ölçekler,
- Aşağıdaki dosyaları üretir:
    data/{DS}_processed.npy
    models/{DS}_scaler.pkl
    models/{DS}_columns.csv
- Büyük dosyalar için MAX_SAMPLES ile örnekleme yapar.
"""
import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

MAX_SAMPLES = 200_000
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

def load_cicids2017():
    base = "data/CICIDS2017"
    if not os.path.exists(base):
        print("❌ CICIDS2017 klasörü bulunamadı:", base); return None
    files = glob.glob(os.path.join(base, "*.csv"))
    if not files:
        print("❌ CICIDS2017 CSV bulunamadı."); return None

    # Eğer tek dosya varsa tamam, değilse parçalayarak oku
    if len(files) == 1:
        print("✓ CICIDS2017 tek dosya bulundu:", os.path.basename(files[0]))
        df = pd.read_csv(files[0], nrows=MAX_SAMPLES)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    print(f"✓ CICIDS2017 çoklu dosya ({len(files)}) — parça parça okunuyor.")
    per_file = max(1, int(MAX_SAMPLES / len(files)))
    chunks = []
    for f in files:
        try:
            c = pd.read_csv(f, nrows=per_file)
            c.columns = c.columns.str.strip()
            chunks.append(c)
        except Exception as e:
            print("Dosya okunamadı:", f, e)
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def load_unsw_nb15():
    base = "data/UNSW-NB15"
    if not os.path.exists(base):
        print("❌ UNSW-NB15 klasörü bulunamadı:", base); return None
    files = glob.glob(os.path.join(base, "*.csv"))
    if not files:
        print("❌ UNSW-NB15 CSV bulunamadı."); return None
    target = next((f for f in files if "training" in f.lower()), files[0])
    print("✓ UNSW-NB15 dosyası seçildi:", os.path.basename(target))
    df = pd.read_csv(target, nrows=MAX_SAMPLES)
    if "id" in df.columns: df.drop(columns=["id"], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

def process_dataset(name, df):
    print(f"\n🚀 İşleniyor: {name} | Ham shape: {df.shape}")

    # Sadece sayısal ve kategorik ayrımı
    numerics = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricals = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Kategorik kolonlar için top-k koruma (top10)
    for col in categoricals:
        topk = df[col].value_counts().nlargest(10).index
        df[col] = df[col].apply(lambda x: x if x in topk else "Other")

    # One-hot
    df_encoded = pd.get_dummies(df, columns=categoricals, dummy_na=False)

    # Scale
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df_encoded.values.astype(np.float32))

    # Save
    np.save(f"data/{name}_processed.npy", arr)
    with open(f"models/{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    pd.Series(df_encoded.columns).to_csv(f"models/{name}_columns.csv", index=False, header=False)

    print(f"✔ Tamamlandı: data/{name}_processed.npy | Shape: {arr.shape}")

if __name__ == "__main__":
    cic = load_cicids2017()
    if cic is not None:
        process_dataset("CICIDS2017", cic)
    unsw = load_unsw_nb15()
    if unsw is not None:
        process_dataset("UNSW-NB15", unsw)

