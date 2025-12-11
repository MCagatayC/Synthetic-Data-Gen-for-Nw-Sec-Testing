import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, entropy
from sklearn.preprocessing import MinMaxScaler

# --- AYARLAR ---
DATASETS = ["CICIDS2017", "UNSW-NB15"]
RESULTS_DIR = "results_comprehensive"
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_kl_divergence(p, q):
    """İki dağılım arasındaki KL Divergence skorunu hesaplar."""
    # Olasılık dağılımına çevir (Histogram)
    p_hist, bin_edges = np.histogram(p, bins=50, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    
    # Sıfıra bölünme hatasını önlemek için epsilon ekle
    p_hist = np.where(p_hist == 0, 1e-10, p_hist)
    q_hist = np.where(q_hist == 0, 1e-10, q_hist)
    
    return entropy(p_hist, q_hist)

def train_and_evaluate_tstr(real_data, syn_data, target_col=None):
    """
    TSTR (Train on Synthetic, Test on Real) Analizi.
    Eğer target_col yoksa (unsupervised), son sütunu etiket varsayalım veya anomali tespiti yapalım.
    Bu kodda son sütunu etiket varsayıyoruz.
    """
    # Veriyi X ve y olarak ayır
    X_real = real_data.iloc[:, :-1].values
    y_real = real_data.iloc[:, -1].values
    
    X_syn = syn_data.iloc[:, :-1].values
    y_syn = syn_data.iloc[:, -1].values
    
    # Gerçek veriyi Train/Test olarak böl (Test seti sabit kalacak)
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )
    
    # 1. TRTR (Train Real, Test Real) - Baseline
    rf_real = RandomForestClassifier(n_estimators=50, n_jobs=1, max_depth=10)
    rf_real.fit(X_train_real, y_train_real)
    y_pred_real = rf_real.predict(X_test_real)
    
    scores_real = {
        "f1": f1_score(y_test_real, y_pred_real, average='macro'),
        "precision": precision_score(y_test_real, y_pred_real, average='macro'),
        "recall": recall_score(y_test_real, y_pred_real, average='macro')
    }
    
    # 2. TSTR (Train Synthetic, Test Real) - Bizim Model
    rf_syn = RandomForestClassifier(n_estimators=50, n_jobs=1, max_depth=10)
    # Sentetik veride etiketlerin float olmasını engelle (classification için int lazım)
    y_syn = y_syn.astype(int)
    rf_syn.fit(X_syn, y_syn)
    y_pred_syn = rf_syn.predict(X_test_real) # TEST GERÇEK VERİDE YAPILIYOR
    
    scores_syn = {
        "f1": f1_score(y_test_real, y_pred_syn, average='macro'),
        "precision": precision_score(y_test_real, y_pred_syn, average='macro'),
        "recall": recall_score(y_test_real, y_pred_syn, average='macro')
    }
    
    return scores_real, scores_syn

def evaluate_dataset(dataset_name):
    print(f"\n--- Analiz Ediliyor: {dataset_name} ---")
    
    # Dosya Yolları (Önceki betiklerin çıktılarına göre)
    # Not: Sentetik veri CSV olarak kaydedilmişti. Gerçek veri npy idi.
    # Karşılaştırma için gerçek veriyi de CSV'ye veya DataFrame'e çevirmemiz lazım.
    
    try:
        # Sentetik Veriyi Oku (İlk 10.000 satır yeterli analiz için)
        syn_path = f"synthetic_data/{dataset_name}_synthetic_natrvae_v6.csv"
        # Eğer bu dosya yoksa, en son ürettiğimiz sentetik veriyi bulun
        import glob
        files = glob.glob(f"synthetic_data/{dataset_name}*.csv")
        if not files:
            print(f"Hata: Sentetik veri bulunamadı ({dataset_name})")
            return
        syn_df = pd.read_csv(files[0], nrows=10000)
        
        # Gerçek Veriyi Oku (processed.npy üzerinden)
        real_npy_path = f"data/{dataset_name}_processed.npy"
        real_data_arr = np.load(real_npy_path, mmap_mode='r')
        
        # Rastgele 10.000 gerçek örnek seç
        idx = np.random.choice(real_data_arr.shape[0], 10000, replace=False)
        real_df = pd.DataFrame(real_data_arr[idx])
        # Sütun isimlerini eşle (Eğer mümkünse)
        real_df.columns = syn_df.columns
        
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return

    results = {}

    # --- 1. İSTATİSTİKSEL TESTLER (KS ve KL) ---
    print("1. İstatistiksel Testler (KS & KL) yapılıyor...")
    ks_scores = []
    kl_scores = []
    
    # Her sütun için tek tek hesapla
    for col in real_df.columns:
        r_col = real_df[col].values
        s_col = syn_df[col].values
        
        # KS Test
        ks_stat, _ = ks_2samp(r_col, s_col)
        ks_scores.append(ks_stat)
        
        # KL Divergence
        kl_stat = calculate_kl_divergence(r_col, s_col)
        kl_scores.append(kl_stat)
        
    results['avg_ks_score'] = np.mean(ks_scores)
    results['avg_kl_divergence'] = np.mean(kl_scores)
    print(f"   Avg KS Score: {results['avg_ks_score']:.4f} (Hedef < 0.1)")
    print(f"   Avg KL Div: {results['avg_kl_divergence']:.4f} (Hedef < 0.1)")

    # --- 2. TSTR ANALİZİ (Machine Learning Utility) ---
    print("2. TSTR Analizi (ML Utility) yapılıyor...")
    # Etiket sütunu oluşturmamız lazım. 
    # CICIDS ve UNSW processed verisinde son sütun genellikle Label'dır (veya one-hot encoded label).
    # Basitlik için son sütunu hedef alalım. Eğer son sütun sürekli (continuous) ise
    # median'dan bölerek binary classification yapalım.
    
    # Etiketleri binary yap (Classification testi için)
    threshold_r = real_df.iloc[:, -1].median()
    real_df.iloc[:, -1] = (real_df.iloc[:, -1] > threshold_r).astype(int)
    
    threshold_s = syn_df.iloc[:, -1].median()
    syn_df.iloc[:, -1] = (syn_df.iloc[:, -1] > threshold_s).astype(int)
    
    scores_real, scores_syn = train_and_evaluate_tstr(real_df, syn_df)
    
    results['tstr_real'] = scores_real
    results['tstr_syn'] = scores_syn
    
    print("   Real Data F1:", scores_real['f1'])
    print("   Syn Data F1: ", scores_syn['f1'])
    print(f"   Utility Loss: {scores_real['f1'] - scores_syn['f1']:.4f}")

    # Sonuçları Kaydet
    with open(f"{RESULTS_DIR}/{dataset_name}_comprehensive_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    for ds in DATASETS:
        evaluate_dataset(ds)