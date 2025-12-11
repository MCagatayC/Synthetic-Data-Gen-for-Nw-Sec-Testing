import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# --- AYARLAR ---
# ============================
DATASETS = ["CICIDS2017", "UNSW-NB15"]

RESULT_DIR = "results_comprehensive_v2"
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================
# --- KL Divergence ---
# ============================
def calculate_kl_divergence(p, q):
    try:
        p_hist, bins = np.histogram(p, bins=40, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        p_hist = np.where(p_hist == 0, 1e-9, p_hist)
        q_hist = np.where(q_hist == 0, 1e-9, q_hist)

        return entropy(p_hist, q_hist)
    except:
        return np.nan


# ============================
# --- TSTR ---
# ============================
def run_tstr(real_df, syn_df):
    # Son sütun target
    X_real = real_df.iloc[:, :-1].values
    y_real = real_df.iloc[:, -1].astype(int).values

    X_syn = syn_df.iloc[:, :-1].values
    y_syn = syn_df.iloc[:, -1].astype(int).values

    # Böl
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42
    )

    # --- TRTR ---
    clf_real = RandomForestClassifier(n_estimators=40, n_jobs=1, max_depth=8)
    clf_real.fit(X_train_real, y_train_real)
    f1_r = f1_score(y_test_real, clf_real.predict(X_test_real), average="macro")

    # --- TSTR ---
    clf_syn = RandomForestClassifier(n_estimators=40, n_jobs=1, max_depth=8)
    clf_syn.fit(X_syn, y_syn)
    f1_s = f1_score(y_test_real, clf_syn.predict(X_test_real), average="macro")

    return f1_r, f1_s


# ============================
# --- Dataset Değerlendirme ---
# ============================
def evaluate_dataset(name):
    print(f"\n--- Analiz: {name} ---")

    # SENTETİK YÜKLEME
    import glob
    syn_files = glob.glob(f"synthetic_data/{name}*.csv")

    if not syn_files:
        print(f"HATA: Sentetik veri yok → {name}")
        return None

    syn_df = pd.read_csv(syn_files[0], nrows=8000)

    # GERÇEK VERİ YÜKLEME
    real_path = f"data/{name}_processed.npy"
    if not os.path.exists(real_path):
        print(f"HATA: Gerçek veri yok → {real_path}")
        return None

    real_np = np.load(real_path, mmap_mode="r")
    idx = np.random.choice(real_np.shape[0], 8000, replace=False)
    real_df = pd.DataFrame(real_np[idx], columns=syn_df.columns)

    # Target binary yapılır
    real_df.iloc[:, -1] = (real_df.iloc[:, -1] > real_df.iloc[:, -1].median()).astype(int)
    syn_df.iloc[:, -1] = (syn_df.iloc[:, -1] > syn_df.iloc[:, -1].median()).astype(int)

    # ============================
    # --- KS & KL ---
    # ============================
    ks_vals = []
    kl_vals = []

    for col in real_df.columns[:-1]:  # label hariç
        r = real_df[col].values
        s = syn_df[col].values

        ks_stat, _ = ks_2samp(r, s)
        ks_vals.append(ks_stat)

        kl_vals.append(calculate_kl_divergence(r, s))

    avg_ks = float(np.mean(ks_vals))
    avg_kl = float(np.nanmean(kl_vals))

    # ============================
    # --- TSTR ---
    # ============================
    f1_real, f1_syn = run_tstr(real_df, syn_df)
    util_loss = f1_real - f1_syn

    # SONUÇ OBJESİ
    return {
        "dataset": name,
        "avg_ks": avg_ks,
        "avg_kl": avg_kl,
        "real_f1": float(f1_real),
        "syn_f1": float(f1_syn),
        "utility_loss": float(util_loss),
        "ks_list": ks_vals,
        "kl_list": kl_vals
    }


# ============================
# --- HEATMAP ÇİZ ---
# ============================
def save_heatmap(results):
    ks_data = {
        r["dataset"]: r["ks_list"] for r in results
    }

    df_heat = pd.DataFrame.from_dict(ks_data, orient="index")

    plt.figure(figsize=(10, 4))
    sns.heatmap(df_heat, cmap="viridis", cbar=True)
    plt.title("KS Statistic Heatmap Across Features")
    plt.xlabel("Feature Index")
    plt.ylabel("Dataset")
    plt.tight_layout()

    heat_path = f"{RESULT_DIR}/ks_kl_heatmap.png"
    plt.savefig(heat_path, dpi=300)
    plt.close()


# ============================
# --- LaTeX Tablosu Yaz ---
# ============================
def save_latex_table(results):
    df = pd.DataFrame(results)[["dataset", "avg_ks", "avg_kl", "real_f1", "syn_f1", "utility_loss"]]

    table_path = f"{RESULT_DIR}/metrics_table.tex"

    with open(table_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f",
                            caption="Comprehensive Evaluation Metrics",
                            label="tab:comprehensive_eval"))

    print(f"\nLaTeX tablo oluşturuldu → {table_path}")


# ============================
# --- MAIN ---
# ============================
if __name__ == "__main__":
    all_results = []

    for ds in DATASETS:
        res = evaluate_dataset(ds)
        if res is not None:
            all_results.append(res)

    if all_results:
        save_heatmap(all_results)
        save_latex_table(all_results)

        print("\nTamamlandı ✔")
    else:
        print("\nHiçbir dataset işlenemedi!")
