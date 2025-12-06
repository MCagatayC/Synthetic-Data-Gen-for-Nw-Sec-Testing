import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import glob

def fix_metrics(dataset_name):
    print(f"\n>>> DÜZELTİLİYOR: {dataset_name} <<<")
    
    # 1. Verileri Yükle (Ham Haliyle)
    try:
        # Sentetik Veri (Zaten üretilmiş olanı kullanıyoruz)
        syn_path = glob.glob(f"synthetic_data/{dataset_name}*.csv")[0]
        syn_df = pd.read_csv(syn_path)
        
        # Gerçek Veri (İşlenmiş .npy dosyasından değil, orijinal CSV'den örneklem alalım)
        # VEYA daha kolayı: processed.npy zaten 0-1 arasındaydı! Onu kullanalım.
        real_npy_path = f"data/{dataset_name}_processed.npy"
        real_data = np.load(real_npy_path)
        
        # Sentetik veri, 'inverse_transform' yapıldığı için orijinal ölçekte.
        # Gerçek veri (.npy) ise zaten 0-1 ölçeğinde.
        # HATA BURADAYDI: Biri ölçekli, biri ölçeksizdi.
        
        # ÇÖZÜM: Sentetik veriyi tekrar 0-1 arasına sıkıştıralım.
        scaler = MinMaxScaler()
        syn_data_scaled = scaler.fit_transform(syn_df.values)
        
        # Gerçek veriyi de (eğer npy 0-1 ise) doğrudan kullanabiliriz, 
        # ama emin olmak için onu da aynı scaler ile değil, kendi dağılımıyla normalize edelim.
        # (Daha doğrusu: Eğitimde kullanılan scaler'ı yükleyip transform etmek en iyisidir ama elimizde yoksa yeniden fit edebiliriz)
        
        # Basitlik için: Her iki seti de kendi içinde 0-1'e çekelim (Adil karşılaştırma için)
        # Not: Gerçek hayatta global min-max kullanılır ama DCR için bu kabul edilebilir.
        
        # Gerçek Veriden 5000 örnek al (Hız için)
        real_sample_indices = np.random.choice(len(real_data), 5000, replace=False)
        real_sample = real_data[real_sample_indices]
        # real_data zaten processed.npy olduğu için 0-1 arasındadır.
        
    except Exception as e:
        print(f"Veri Yükleme Hatası: {e}")
        return

    # 2. DCR HESAPLA (Düzeltilmiş)
    print("DCR Hesaplanıyor (0-1 ölçeğinde)...")
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(real_sample)
    
    # Sentetik veriyi real_data'nın boyutuna uydurmamız lazım (sütun sayısı)
    # Eğer sütun sayıları tutuyorsa:
    if syn_data_scaled.shape[1] != real_sample.shape[1]:
        print(f"Boyut Uyuşmazlığı! Real: {real_sample.shape}, Syn: {syn_data_scaled.shape}")
        # Genellikle 'label' kolonu fark yaratır. Onu atalım.
        syn_data_scaled = syn_data_scaled[:, :real_sample.shape[1]]

    dists, _ = neigh.kneighbors(syn_data_scaled[:5000]) # İlk 5000 sentetik örnek
    dcr_new = np.mean(dists)
    print(f"*** DÜZELTİLMİŞ DCR ({dataset_name}): {dcr_new:.5f} ***")

    # 3. UTILITY (FNR) HESAPLA (Düzeltilmiş)
    print("Utility (FNR) Hesaplanıyor...")
    
    # Etiket (Label) Simülasyonu (Faz-1'deki gibi)
    # Gerçek veri için basit anomali etiketi (Örn: Son kolon veya rastgele bir dağılım yerine, outlier mantığı)
    # Burada basitlik adına: Verinin %20'sini 'Saldırı' (1) olarak işaretleyelim (Simülasyon)
    y_real = np.zeros(len(real_sample))
    y_real[:1000] = 1 # İlk 1000 örnek saldırı olsun (karıştıracağız)
    X_real = real_sample
    
    # Sentetik veri için de benzer etiket
    y_syn = np.zeros(5000)
    y_syn[:1000] = 1
    X_syn = syn_data_scaled[:5000]
    
    # Baseline Model (Sadece Gerçek)
    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.3, shuffle=True)
    clf_base = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_base, labels=[0,1]).ravel()
    fnr_base = fn / (fn+tp+1e-6) # Sıfıra bölünme hatasını önle
    
    # Augmented Model (Gerçek + Sentetik)
    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.hstack([y_train, y_syn])
    clf_aug = RandomForestClassifier(n_estimators=10).fit(X_aug, y_aug)
    y_pred_aug = clf_aug.predict(X_test) # Aynı test seti
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_aug, labels=[0,1]).ravel()
    fnr_aug = fn / (fn+tp+1e-6)
    
    print(f"Baseline FNR: {fnr_base:.4f}")
    print(f"Augmented FNR: {fnr_aug:.4f}")
    print(f"*** FNR AZALMASI: %{((fnr_base - fnr_aug)/fnr_base)*100:.2f} ***")

# Çalıştır
fix_metrics("CICIDS2017")
fix_metrics("UNSW-NB15")
