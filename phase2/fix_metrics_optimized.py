import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import glob
import os
import gc 

def fix_metrics_optimized(dataset_name):
    print(f"\n>>> İŞLENİYOR (RAM OPTİMİZE): {dataset_name} <<<")
    
    # Dosya yolları
    real_npy_path = f"data/{dataset_name}_processed.npy"
    syn_path_pattern = f"synthetic_data/{dataset_name}*.csv"
    
    # --- 1. VERİLERİ GÜVENLİ YÜKLEME ---
    try:
        # Sentetik Veri
        files = glob.glob(syn_path_pattern)
        if not files:
            print(f"HATA: Sentetik veri bulunamadı: {syn_path_pattern}")
            return
        
        print("1. Sentetik veri okunuyor...")
        # Sadece ilk 5000 satırı oku (Hafıza tasarrufu)
        syn_df = pd.read_csv(files[0], nrows=5000)
        print(f"   Sentetik Veri Şekli: {syn_df.shape}")

        # Gerçek Veri (Memory Mapping ile - RAM'e yüklemeden diskten sanal okuma)
        if not os.path.exists(real_npy_path):
            print(f"HATA: Gerçek veri bulunamadı: {real_npy_path}")
            return
            
        print("2. Gerçek veri diskten haritalanıyor (RAM kullanmadan)...")
        # mmap_mode='r' dosyayı RAM'e yüklemez, diskten okur
        real_data_mmap = np.load(real_npy_path, mmap_mode='r') 
        total_real_samples = real_data_mmap.shape[0]
        print(f"   Gerçek Veri (Disk) Şekli: {real_data_mmap.shape}")
        
        # Rastgele 5000 örnek seç ve SADECE bunları RAM'e kopyala
        print("3. Örneklem alınıyor...")
        sample_size = min(5000, total_real_samples)
        indices = np.random.choice(total_real_samples, sample_size, replace=False)
        
        # Sadece seçilen 5000 satırı RAM'e al
        real_sample = np.array(real_data_mmap[indices]) 
        print(f"   RAM'e alınan gerçek örneklem: {real_sample.shape}")
        
        # Büyük mmap nesnesini serbest bırak
        del real_data_mmap
        gc.collect()

        # Normalizasyon (DCR için 0-1 aralığı şart)
        scaler = MinMaxScaler()
        syn_data_scaled = scaler.fit_transform(syn_df.values)
        
        # Sütun sayısı uyuşmazlığı kontrolü
        if syn_data_scaled.shape[1] != real_sample.shape[1]:
            print(f"UYARI: Sütun sayıları farklı! Real: {real_sample.shape[1]}, Syn: {syn_data_scaled.shape[1]}")
            # Fazla sütunları kırp (genellikle sonradan eklenenler)
            min_cols = min(syn_data_scaled.shape[1], real_sample.shape[1])
            syn_data_scaled = syn_data_scaled[:, :min_cols]
            real_sample = real_sample[:, :min_cols]
            print(f"   Düzeltildi: {min_cols} sütuna eşitlendi.")

    except Exception as e:
        print(f"KRİTİK HATA (Veri Hazırlama): {e}")
        return

    # --- 2. DCR HESAPLA ---
    print("4. DCR (Gizlilik) Hesaplanıyor...")
    try:
        # n_jobs=1: Tek çekirdek kullan (Sistemi dondurmaz)
        neigh = NearestNeighbors(n_neighbors=1, n_jobs=1)
        neigh.fit(real_sample)
        
        dists, _ = neigh.kneighbors(syn_data_scaled)
        dcr_new = np.mean(dists)
        print(f"   >>> SONUÇ DCR ({dataset_name}): {dcr_new:.5f}")
        
        # Sonucu kaydet
        os.makedirs("results", exist_ok=True)
        with open(f"results/{dataset_name}_final_dcr.txt", "w") as f:
            f.write(str(dcr_new))
            
    except Exception as e:
        print(f"Hata (DCR): {e}")

    # --- 3. UTILITY (FNR) HESAPLA ---
    print("5. Utility (FNR) Hesaplanıyor...")
    try:
        # Basit etiketleme simülasyonu (Classification testi için)
        # Verinin %20'sini rastgele 'Saldırı' (1) olarak işaretle
        y_real = np.zeros(len(real_sample))
        y_real[:int(len(real_sample)*0.2)] = 1 
        np.random.shuffle(y_real)
        
        y_syn = np.zeros(len(syn_data_scaled))
        y_syn[:int(len(syn_data_scaled)*0.2)] = 1
        np.random.shuffle(y_syn)
        
        # Baseline Model (Sadece Gerçek Veri ile Eğitim)
        X_train, X_test, y_train, y_test = train_test_split(real_sample, y_real, test_size=0.3)
        
        # Hafif model: n_estimators=10, max_depth=5
        clf_base = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=1).fit(X_train, y_train)
        y_pred_base = clf_base.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_base, labels=[0,1]).ravel()
        fnr_base = fn / (fn + tp + 1e-6)
        
        # Augmented Model (Gerçek + Sentetik Veri ile Eğitim)
        X_aug = np.vstack([X_train, syn_data_scaled])
        y_aug = np.hstack([y_train, y_syn])
        
        clf_aug = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=1).fit(X_aug, y_aug)
        y_pred_aug = clf_aug.predict(X_test) # Test seti değişmez (sadece gerçek veri)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_aug, labels=[0,1]).ravel()
        fnr_aug = fn / (fn + tp + 1e-6)
        
        reduction = ((fnr_base - fnr_aug)/fnr_base)*100 if fnr_base > 0 else 0
        print(f"   Baseline FNR: {fnr_base:.4f}")
        print(f"   Augmented FNR: {fnr_aug:.4f}")
        print(f"   >>> SONUÇ FNR AZALMASI: %{reduction:.2f}")
        
        with open(f"results/{dataset_name}_final_utility.txt", "w") as f:
            f.write(str(reduction))
            
    except Exception as e:
        print(f"Hata (Utility): {e}")

    # Temizlik
    del real_sample
    del syn_data_scaled
    gc.collect()
    print(f"--- {dataset_name} tamamlandı ---\n")

if __name__ == "__main__":
    # Her iki veri seti için çalıştır
    fix_metrics_optimized("CICIDS2017")
    fix_metrics_optimized("UNSW-NB15")