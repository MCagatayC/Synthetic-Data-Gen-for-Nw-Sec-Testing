import pandas as pd
import os

# Dataset'in indirildiği dizin
dataset_path = '/home/dolly/.cache/kagglehub/datasets/ravikumargattu/network-traffic-dataset/versions/2'

dataset_file = os.path.join(dataset_path, 'Midterm_53_group.csv')  # Dosya adını kontrol et, gerekirse değiştir

# Veriyi yükle
try:
    df = pd.read_csv(dataset_file)  # Eğer dosya .csv ise
    print("Dataset başarıyla yüklendi!")
except Exception as e:
    print(f"Hata: {e}")

# Eğer veri yüklendiyse, ilk 5 satırı yazdır
if 'df' in locals():
    print("\nİlk 5 Satır:")
    print(df.head())  # İlk 5 satır

    # Genel veri kontrolü
    print("\nVeri Çerçevesi Özeti:")
    print(df.info())  # Veri tipi ve boş değerler hakkında bilgi verir
clear

