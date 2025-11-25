import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import os

# --- 1. Proje Ayarları (Bizim Projemize Göre Düzeltildi) ---
REAL_DATA_PATH = 'data/Midterm_53_group.csv' 
SYNTHETIC_DATA_PATH = 'github_repo/synthetic_data/synthetic_traffic_VAE-LDM.csv'
SAMPLE_SIZE = 5000 
POSSIBLE_TIME_COLUMNS = ['Time', 'time', 'Timestamp']

def find_feature_to_plot(df_columns):
    """'Time'  özelliğini çeşitli olası adlarla arar."""
    for col_name in POSSIBLE_TIME_COLUMNS:
        if col_name in df_columns:
            print(f"Grafik için '{col_name}' özelliği bulundu.")
            return col_name
    
    print(f"UYARI: {POSSIBLE_TIME_COLUMNS} özellikleri bulunamadı.")
    return None

def load_and_preprocess(real_path, synthetic_path, sample_size):
    """Verileri yükler, Gerçek veriyi One-Hot yöntemiyle işler ve örnekler."""
    
    # --- Gerçek Veriyi Yükle ve İşle ---
    if not os.path.exists(real_path):
        print(f"HATA: '{real_path}' bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return None, None, None, None, 0
    print(f"'{real_path}' yükleniyor ve işleniyor (One-Hot Encoding)...")
    real_df = pd.read_csv(real_path)
    real_df_clean = real_df.dropna().reset_index(drop=True)

    # One-Hot Encoding mantığını (prepare_data.py'den) yeniden uygula
    numeric_cols = real_df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = real_df_clean.select_dtypes(exclude=[np.number]).columns
    
    real_processed_df = real_df_clean.copy()
    for col in categorical_cols:
        top_categories = real_df_clean[col].value_counts().nlargest(20).index
        real_processed_df[col] = real_processed_df[col].apply(lambda x: x if x in top_categories else 'Other')
    
    real_processed_df = pd.get_dummies(real_processed_df, columns=categorical_cols, dummy_na=False)
    
    # --- Sentetik Veriyi Yükle ---
    if not os.path.exists(synthetic_path):
        print(f"HATA: '{synthetic_path}' bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return None, None, None, None, 0
    print(f"'{synthetic_path}' yükleniyor...")
    synthetic_df = pd.read_csv(synthetic_path)

    # --- Verileri Hizala ve Örnekle ---
    common_cols = list(set(real_processed_df.columns) & set(synthetic_df.columns))
    
    if not common_cols:
        print("HATA: Gerçek ve sentetik veriler arasında ortak sütun bulunamadı.")
        return None, None, None, None, 0
        
    real_data_aligned = real_processed_df[common_cols].astype(np.float32)
    synthetic_data_aligned = synthetic_df[common_cols].astype(np.float32)
    
    n_features = len(common_cols)
    print(f"Karşılaştırma için {n_features} ortak özellik bulundu.")

    if n_features < 2:
        print(f"HATA: PCA/t-SNE  için en az 2 ortak özelliğe ihtiyaç var.")
        return None, None, None, None, 0

    feature_to_plot = find_feature_to_plot(common_cols)
    if feature_to_plot is None:
        feature_to_plot = common_cols[0] 
        print(f"KDE grafiği  için ilk sütun olan '{feature_to_plot}' kullanılacak.")

    real_sample = real_data_aligned.sample(n=min(sample_size, len(real_data_aligned)), random_state=42)
    synthetic_sample = synthetic_data_aligned.sample(n=min(sample_size, len(synthetic_data_aligned)), random_state=42)

    real_sample['label'] = 'Real Data'
    synthetic_sample['label'] = 'VAE-LDM (v4)'
    
    unscaled_data_with_labels = pd.concat([real_sample, synthetic_sample], ignore_index=True)
    
    labels = unscaled_data_with_labels['label']
    unscaled_data_for_scaling = unscaled_data_with_labels.drop(columns=['label'])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(unscaled_data_for_scaling)
    
    print("Veri yüklendi ve ölçeklendi.")
    return scaled_data, labels, unscaled_data_with_labels, feature_to_plot, n_features

def plot_univariate_distributions(unscaled_data_with_labels, feature, filename='univariate_comparison.png'):
    """Tek değişkenli dağılımı (KDE) çizer ve kaydeder."""
    print(f"'{feature}' özelliği için 1D KDE grafiği  oluşturuluyor... (univariate_comparison.png)")
    
    sns.set_style("whitegrid")
    palette = {"Real Data": "#3470a3", "VAE-LDM (v4)": "#e07a3e"}
    plt.figure(figsize=(10, 6))
    sns.set_context("paper", font_scale=1.2)
    
    sns.kdeplot(
        data=unscaled_data_with_labels,
        x=feature,
        hue='label', 
        palette=palette,
        fill=True, 
        alpha=0.3,
        linewidth=1.75
    )
    
    plt.title(f"Distribution Comparison for '{feature}' Feature (VAE-LDM v4)", fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1600)
    plt.ylim(0, 0.003) 
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grafik kaydedildi: {filename}")

# --- GÜNCELLENMİŞ FONKSİYON ---
def plot_pca_and_tsne(data, labels, n_features, filename='pca_tsne_comparison.png'):
    """PCA  ve t-SNE  grafiklerini ALT ALTA (dikey) çizer."""
    
    # --- DEĞİŞİKLİK BURADA (1): 2 Satır, 1 Sütun ---
    # Figür boyutu (genişlik, yükseklik) dikey olarak ayarlandı.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
    # -----------------------------------------------
    
    sns.set_style("whitegrid") 
    
    real_indices = (labels == 'Real Data')
    synthetic_indices = (labels == 'VAE-LDM (v4)')
    colors = {'Real Data': '#3470a3', 'VAE-LDM (v4)': '#e07a3e'}

    # --- PCA PLOT (ax1 - Üstteki Grafik) ---
    print(f"PCA  analizi ({n_features} özellik) başlıyor (Hızlı)...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)
    
    ax1.scatter(pca_results[real_indices, 0], pca_results[real_indices, 1], 
                label='Real Data', alpha=0.5, c=colors['Real Data'], s=20)
    ax1.scatter(pca_results[synthetic_indices, 0], pca_results[synthetic_indices, 1], 
                label='VAE-LDM (v4)', alpha=0.5, c=colors['VAE-LDM (v4)'], s=20)
    ax1.set_title(f'PCA of Real vs. Synthetic Data ({n_features} Features)', fontsize=18)
    ax1.set_xlabel('Principal Component 1', fontsize=14)
    ax1.set_ylabel('Principal Component 2', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    print("PCA tamamlandı.")

    # --- t-SNE PLOT (ax2 - Alttaki Grafik) ---
    print(f"t-SNE  analizi ({n_features} özellik) başlıyor (Bu işlem yavaş olabilir)...")
    perplexity_val = min(40, len(data) - 2) 
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, n_iter=300, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(data)
    
    ax2.scatter(tsne_results[real_indices, 0], tsne_results[real_indices, 1], 
                label='Real Data', alpha=0.5, c=colors['Real Data'], s=20)
    ax2.scatter(tsne_results[synthetic_indices, 0], tsne_results[synthetic_indices, 1], 
                label='VAE-LDM (v4)', alpha=0.5, c=colors['VAE-LDM (v4)'], s=20)
    ax2.set_title(f't-SNE of Real vs. Synthetic Data ({n_features} Features)', fontsize=18)
    ax2.set_xlabel('t-SNE Component 1', fontsize=14)
    ax2.set_ylabel('t-SNE Component 2', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    print("t-SNE tamamlandı.")

    plt.suptitle('High-Dimensional Fidelity Comparison (VAE-LDM v4)', fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA/t-SNE grafiği kaydedildi: {filename}")
# --- GÜNCELLENMİŞ FONKSİYON BİTİŞİ ---

def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        
        scaled_data, labels, unscaled_data_with_labels, feature_to_plot, n_features = load_and_preprocess(
            REAL_DATA_PATH, SYNTHETIC_DATA_PATH, SAMPLE_SIZE
        )
        
        if scaled_data is None:
            print("İşlem durduruldu. Lütfen dosya yollarını ve içeriğini kontrol edin.")
            return

        plot_univariate_distributions(unscaled_data_with_labels, feature_to_plot, filename='univariate_comparison.png')
        plot_pca_and_tsne(scaled_data, labels, n_features, filename='pca_tsne_comparison.png')

    print("\n--- Grafik Oluşturma Tamamlandı ---")
    print("Tüm grafikler başarıyla oluşturuldu ve kaydedildi.")
    print("Lütfen 'univariate_comparison.png' ve 'pca_tsne_comparison.png' dosyalarını Overleaf projenize yükleyin.")

if __name__ == '__main__':
    main()
