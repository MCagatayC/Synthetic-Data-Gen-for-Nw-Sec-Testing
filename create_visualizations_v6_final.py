#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import os

# -------------------------------
# 1. Proje Ayarları
# -------------------------------
REAL_DATA_PATH = '/media/dolly/1TB/tez-projesi/data/Midterm_53_group.csv'
SYNTHETIC_DATA_PATH = '/media/dolly/1TB/tez-projesi/github_repo/synthetic_data/synthetic_traffic_VAE-LDM.csv'
SAMPLE_SIZE = 5000
POSSIBLE_TIME_COLUMNS = ['Time', 'time', 'Timestamp']


# -------------------------------
# Yardımcı Fonksiyonlar
# -------------------------------
def find_feature_to_plot(df_columns):
    for col_name in POSSIBLE_TIME_COLUMNS:
        if col_name in df_columns:
            return col_name
    return None


def load_and_preprocess(real_path, synthetic_path, sample_size):
    """Verileri yükler, One-Hot encoding yapar ve ortak kolonlara göre hizalar."""

    if not all(os.path.exists(p) for p in [real_path, synthetic_path]):
        print("HATA: Dosyalardan biri bulunamadı.")
        return None, None, None, None, 0

    real_df = pd.read_csv(real_path).dropna().reset_index(drop=True)
    synthetic_df = pd.read_csv(synthetic_path)

    # Kolon türleri
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    categorical_cols = real_df.select_dtypes(exclude=[np.number]).columns

    real_processed_df = real_df.copy()

    # Kategorik kolonları 20 sınıfa düşür ve temizle
    for col in categorical_cols:
        top_categories = real_processed_df[col].value_counts().nlargest(20).index
        real_processed_df[col] = real_processed_df[col].apply(
            lambda x: x if x in top_categories else 'Other'
        )

    # One-hot encoding
    real_processed_df = pd.get_dummies(real_processed_df, columns=categorical_cols, dummy_na=False)

    # Ortak kolonlar
    common_cols = list(set(real_processed_df.columns) & set(synthetic_df.columns))

    if len(common_cols) < 2:
        print("UYARI: Ortak kolon bulunamadı.")
        return None, None, None, None, 0

    real_aligned = real_processed_df[common_cols].astype(np.float32)
    synthetic_aligned = synthetic_df[common_cols].astype(np.float32)

    n_features = len(common_cols)

    # Sampling
    real_sample = real_aligned.sample(n=min(sample_size, len(real_aligned)), random_state=42)
    synthetic_sample = synthetic_aligned.sample(n=min(sample_size, len(synthetic_aligned)), random_state=42)

    real_sample['label'] = 'Real Data'
    synthetic_sample['label'] = 'VAE-LDM (v6)'

    full_df = pd.concat([real_sample, synthetic_sample], ignore_index=True)

    labels = full_df['label']
    df_for_scaling = full_df.drop(columns=['label'])

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_for_scaling)

    feature_to_plot = find_feature_to_plot(common_cols)

    return scaled_data, labels, full_df, feature_to_plot, n_features


# -------------------------------
# Tek Değişkenli Dağılım (KDE)
# -------------------------------
def plot_univariate_distributions(unscaled_df, feature, filename):
    sns.set_style("whitegrid")
    palette = {"Real Data": "#3ea3e0", "VAE-LDM (v6)": "#FD972B"}

    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=unscaled_df,
        x=feature,
        hue='label',
        palette=palette,
        fill=True,
        alpha=0.3,
        linewidth=1.75
    )

    plt.title(f"Distribution Comparison for '{feature}' Feature (VAE-LDM v6)", fontsize=16)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Grafik kaydedildi: {filename}")


# -------------------------------
# PCA + t-SNE
# -------------------------------
def plot_pca_and_tsne(data, labels, n_features, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
    sns.set_style("whitegrid")

    real_idx = labels == 'Real Data'
    syn_idx = labels == 'VAE-LDM (v6)'
    colors = {'Real Data': '#3470a3', 'VAE-LDM (v6)': '#e07a3e'}

    # --- PCA ---
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)

    ax1.scatter(pca_results[real_idx, 0], pca_results[real_idx, 1], c=colors['Real Data'], s=20, alpha=0.5, label='Real Data')
    ax1.scatter(pca_results[syn_idx, 0], pca_results[syn_idx, 1], c=colors['VAE-LDM (v6)'], s=20, alpha=0.5, label='VAE-LDM (v6)')
    ax1.set_title("PCA (Real vs. Synthetic)")
    ax1.legend()

    # --- t-SNE (Düzeltilmiş parametreler ile) ---
    perplexity_val = min(30, len(data) - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity_val,
        random_state=42,
        init='pca',
        learning_rate=200.0
    )

    tsne_results = tsne.fit_transform(data)

    ax2.scatter(tsne_results[real_idx, 0], tsne_results[real_idx, 1], c=colors['Real Data'], s=20, alpha=0.5, label='Real Data')
    ax2.scatter(tsne_results[syn_idx, 0], tsne_results[syn_idx, 1], c=colors['VAE-LDM (v6)'], s=20, alpha=0.5, label='VAE-LDM (v6)')
    ax2.set_title("t-SNE (Real vs. Synthetic)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"PCA/t-SNE grafiği kaydedildi: {filename}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scaled_data, labels, unscaled_df, feature_to_plot, n_features = load_and_preprocess(
            REAL_DATA_PATH, SYNTHETIC_DATA_PATH, SAMPLE_SIZE
        )

        if scaled_data is None:
            print("İşlem durduruldu.")
            return

        if feature_to_plot:
            plot_univariate_distributions(unscaled_df, feature_to_plot, 'univariate_comparison(NAT-VAEv6).png')

        plot_pca_and_tsne(scaled_data, labels, n_features, 'pca_tsne_comparison(NAT-VAEv6).png')

    print("\n--- Grafik Oluşturma Tamamlandı ---")


if __name__ == "__main__":
    main()
