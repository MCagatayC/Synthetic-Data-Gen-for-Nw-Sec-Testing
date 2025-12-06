import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import glob

MAX_SAMPLES = 200000  # RAM kısıtı için

def load_cicids2017():
    """
    Yeni Kaggle dataset'i (ericanacletoribeiro/cicids2017-cleaned-and-preprocessed)
    ile uyumlu hale getirildi.
    """
    base = "data/CICIDS2017"

    # Önce klasör var mı kontrol et
    if not os.path.exists(base):
        print("❌ CICIDS2017 klasörü bulunamadı:", base)
        return None

    # Tüm CSV dosyalarını bul
    files = glob.glob(os.path.join(base, "*.csv"))
    if not files:
        print("❌ CICIDS2017 CSV dosyası bulunamadı.")
        return None

    # Eğer sadece 1 dosya varsa (Yeni Kaggle dataset'i)
    if len(files) == 1:
        print(f"✓ CICIDS2017 tek dosya bulundu: {os.path.basename(files[0])}")
        df = pd.read_csv(files[0], nrows=MAX_SAMPLES)
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

    # Birden çok dosya varsa (eski CICIDS2017 formatı)
    df_list = []
    print(f"✓ {len(files)} CICIDS2017 dosyası bulundu. Bölerek yükleniyor...")
    per_file = int(MAX_SAMPLES / len(files)) + 1

    for f in files:
        try:
            chunk = pd.read_csv(f, nrows=per_file)
            chunk.columns = chunk.columns.str.strip()
            df_list.append(chunk)
        except Exception as e:
            print("Dosya okunamadı:", f, "| Hata:", e)

    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def load_unsw_nb15():
    base = "data/UNSW-NB15"
    files = glob.glob(os.path.join(base, "*.csv"))

    if not files:
        print("❌ UNSW-NB15 CSV bulunamadı.")
        return None

    # training-set varsa onu al
    target = next((f for f in files if "training" in f.lower()), files[0])

    print(f"✓ UNSW-NB15 dosyası: {os.path.basename(target)}")

    df = pd.read_csv(target, nrows=MAX_SAMPLES)
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    return df


def process_dataset(name, df):
    print(f"\n🚀 İşleniyor: {name} | Ham veri: {df.shape}")

    numerics = df.select_dtypes(include=[np.number]).columns
    categoricals = df.select_dtypes(exclude=[np.number]).columns

    # Kategorik kolon temizleme (top 10 sınıf bırak)
    for col in categoricals:
        top10 = df[col].value_counts().nlargest(10).index
        df[col] = df[col].apply(lambda x: x if x in top10 else "Other")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categoricals)

    # Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_encoded.values.astype(np.float32))

    # Save output
    np.save(f"data/{name}_processed.npy", data_scaled)
    print(f"📁 Kaydedildi: data/{name}_processed.npy")

    with open(f"models/{name}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    pd.Series(df_encoded.columns).to_csv(f"models/{name}_columns.csv", index=False, header=False)

    print(f"✔ Tamamlandı → {name} | Shape: {data_scaled.shape}")


if __name__ == "__main__":
    df1 = load_cicids2017()
    if df1 is not None:
        process_dataset("CICIDS2017", df1)

    df2 = load_unsw_nb15()
    if df2 is not None:
        process_dataset("UNSW-NB15", df2)
