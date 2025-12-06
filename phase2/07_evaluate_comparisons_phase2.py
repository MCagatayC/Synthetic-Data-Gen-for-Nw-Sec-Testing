#!/usr/bin/env python3
"""
07_evaluate_comparisons_phase2.py (npy uyumlu)

- Input expectations:
    * Real processed data: data/{DS}_processed.npy  (NumPy array, scaled)
    * Diffusion synthetic CSV: synthetic_data/{DS}_synthetic_diffusion.csv
    * NATVAEv6 synthetic CSV: synthetic_data/{DS}_synthetic_natrvae_v6.csv
    * Column names: models/{DS}_columns.csv
    * Scaler (optional): models/{DS}_scaler.pkl

- Outputs (all to results/):
    * {ds}_{method}_pca_tsne.png
    * {ds}_{method}_univariate_ks.csv
    * {ds}_{method}_hist_columns.png (sample of columns)
    * {ds}_{method}_dcr.txt
    * {ds}_{method}_utility.csv
    * {ds}_summary_row appended to results/summary_evaluation_table.csv
"""

import os
import numpy as np
import pandas as pd
import pickle
import math
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------
# Configuration
# -------------------------
DATASETS = ["CICIDS2017", "UNSW-NB15"]
SYN_FILES = {
    "diffusion": "synthetic_data/{ds}_synthetic_diffusion.csv",
    "natrvae":  "synthetic_data/{ds}_synthetic_natrvae_v6.csv"
}
REAL_PROC = "data/{ds}_processed.npy"
COLS_FILE = "models/{ds}_columns.csv"
SCALER_FILE = "models/{ds}_scaler.pkl"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sampling and plotting settings
MAX_POINTS_PLOT = 2000
TSNE_PERPLEXITY = 30
TSNE_ITER = 500

# Utility classifier settings
CLASSIFIER = RandomForestClassifier
RF_PARAMS = {"n_estimators": 50, "random_state": 42}

# -------------------------
# Helpers
# -------------------------
def load_real(ds):
    path = REAL_PROC.format(ds=ds)
    if not os.path.exists(path):
        print(f"[ERROR] Missing real: {path}")
        return None
    try:
        arr = np.load(path)
        arr = arr.astype(np.float32)
        return arr
    except Exception as e:
        print(f"[ERROR] Failed loading real {path}: {e}")
        return None

def load_synth(ds, tag):
    path = SYN_FILES[tag].format(ds=ds)
    if not os.path.exists(path):
        print(f"[WARN] Missing synthetic: {path}")
        return None
    try:
        df = pd.read_csv(path)
        return df.values.astype(np.float32), df.columns.tolist()
    except Exception as e:
        print(f"[ERROR] Failed to load synth {path}: {e}")
        return None

def load_cols(ds):
    path = COLS_FILE.format(ds=ds)
    if os.path.exists(path):
        try:
            cols = pd.read_csv(path, header=None).iloc[:,0].tolist()
            return cols
        except Exception:
            return None
    return None

def maybe_inverse_scale(scaler_path, data):
    if not os.path.exists(scaler_path):
        return data
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler.inverse_transform(data)
    except Exception:
        return data

def safe_savefig(fig, outpath):
    try:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"[WARN] Failed to save figure {outpath}: {e}")

# -------------------------
# Core metrics / plots
# -------------------------
def pca_tsne_plot(real, synth, ds, tag):
    try:
        n = min(MAX_POINTS_PLOT, real.shape[0], synth.shape[0])
        X = np.vstack([real[:n], synth[:n]])
        labels = np.array([0]*n + [1]*n)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=min(TSNE_PERPLEXITY, max(5, n//3)), max_iter=TSNE_ITER)
        X_tsne = tsne.fit_transform(X)

        fig, axs = plt.subplots(1,2, figsize=(12,5))
        axs[0].scatter(X_pca[:n,0], X_pca[:n,1], s=4, label='real')
        axs[0].scatter(X_pca[n:,0], X_pca[n:,1], s=4, label='synth')
        axs[0].set_title(f"PCA - {ds} - {tag} (n={n})")
        axs[0].legend(loc='best', fontsize='small')

        axs[1].scatter(X_tsne[:n,0], X_tsne[:n,1], s=4, label='real')
        axs[1].scatter(X_tsne[n:,0], X_tsne[n:,1], s=4, label='synth')
        axs[1].set_title(f"t-SNE - {ds} - {tag}")
        axs[1].legend(loc='best', fontsize='small')

        out = os.path.join(RESULTS_DIR, f"{ds}_{tag}_pca_tsne.png")
        safe_savefig(fig, out)
        plt.close(fig)
        print(f"[OK] PCA/t-SNE plot saved: {out}")
        return out
    except Exception as e:
        print(f"[ERROR] PCA/t-SNE failed for {ds} {tag}: {e}")
        return None

def univariate_ks_tests(real, synth, ds, tag, col_names=None):
    try:
        cols = real.shape[1]
        rows = []
        for i in range(cols):
            try:
                r = real[:, i]
                s = synth[:, i]
                r = r[np.isfinite(r)]
                s = s[np.isfinite(s)]
                if len(r) < 10 or len(s) < 10:
                    rows.append((i, None, None))
                    continue
                stat, p = ks_2samp(r, s)
                rows.append((i, float(stat), float(p)))
            except Exception:
                rows.append((i, None, None))
        df = pd.DataFrame(rows, columns=["col_idx", "ks_stat", "pvalue"])
        if col_names and len(col_names) >= df.shape[0]:
            df["col_name"] = col_names[:df.shape[0]]
        out = os.path.join(RESULTS_DIR, f"{ds}_{tag}_univariate_ks.csv")
        df.to_csv(out, index=False)
        print(f"[OK] KS tests saved: {out}")
        return out
    except Exception as e:
        print(f"[ERROR] KS tests failed for {ds} {tag}: {e}")
        return None

def dcr(real, synth, ds, tag, k=1, samples=5000):
    try:
        n = min(samples, real.shape[0], synth.shape[0])
        r_idx = np.random.choice(real.shape[0], n, replace=(n>real.shape[0]))
        s_idx = np.random.choice(synth.shape[0], n, replace=(n>synth.shape[0]))
        real_sub = real[r_idx]
        synth_sub = synth[s_idx]
        nbrs = NearestNeighbors(n_neighbors=k).fit(real_sub)
        dists, _ = nbrs.kneighbors(synth_sub)
        mean_d = float(np.mean(dists))
        out = os.path.join(RESULTS_DIR, f"{ds}_{tag}_dcr.txt")
        with open(out, "w") as f:
            f.write(f"DCR_mean_{k}nn: {mean_d}\n")
        print(f"[OK] DCR saved: {out} (mean {mean_d:.6f})")
        return mean_d, out
    except Exception as e:
        print(f"[ERROR] DCR failed for {ds} {tag}: {e}")
        return None, None

def utility_classification(real, synth, ds, tag, test_size=0.4, random_state=42):
    try:
        n = min(50000, real.shape[0])
        X_train, X_test = train_test_split(real[:n], test_size=test_size, random_state=random_state)
        thresh = np.median(X_train[:,0])
        y_train = (X_train[:,0] > thresh).astype(int)
        y_test = (X_test[:,0] > thresh).astype(int)

        clf = CLASSIFIER(**RF_PARAMS)
        clf.fit(X_train, y_train)

        # Real test
        probs_real = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X_test)
        preds_real = (probs_real >= 0.5).astype(int) if probs_real.ndim>1 or probs_real.dtype!='int' else probs_real
        acc_real = accuracy_score(y_test, preds_real)
        prec_real = precision_score(y_test, preds_real, zero_division=0)
        rec_real = recall_score(y_test, preds_real, zero_division=0)
        f1_real = f1_score(y_test, preds_real, zero_division=0)
        try:
            auc_real = roc_auc_score(y_test, probs_real)
        except Exception:
            auc_real = None

        # Synthetic test
        n_s = min(synth.shape[0], len(X_test))
        synth_test = synth[:n_s]
        if synth_test.shape[1] != real.shape[1]:
            c = min(synth_test.shape[1], real.shape[1])
            synth_test = synth_test[:, :c]
            X_test_sub = X_test[:, :c]
        else:
            X_test_sub = X_test[:n_s]

        y_synth = (synth_test[:,0] > thresh).astype(int)
        try:
            probs_synth = clf.predict_proba(synth_test)[:,1] if hasattr(clf, "predict_proba") else clf.predict(synth_test)
            preds_synth = (probs_synth >= 0.5).astype(int) if probs_synth.ndim>1 or probs_synth.dtype!='int' else probs_synth
        except Exception:
            preds_synth = clf.predict(synth_test)

        acc_synth = accuracy_score(y_synth, preds_synth)
        prec_synth = precision_score(y_synth, preds_synth, zero_division=0)
        rec_synth = recall_score(y_synth, preds_synth, zero_division=0)
        f1_synth = f1_score(y_synth, preds_synth, zero_division=0)
        try:
            auc_synth = roc_auc_score(y_synth, probs_synth)
        except Exception:
            auc_synth = None

        out = os.path.join(RESULTS_DIR, f"{ds}_{tag}_utility.csv")
        df = pd.DataFrame([
            ("accuracy", acc_real, acc_synth),
            ("precision", prec_real, prec_synth),
            ("recall", rec_real, rec_synth),
            ("f1", f1_real, f1_synth),
            ("auc", auc_real, auc_synth),
        ], columns=["metric", "real_test", "synth_test"])
        df.to_csv(out, index=False)
        print(f"[OK] Utility saved: {out}")

        return {
            "acc_real": acc_real, "prec_real": prec_real, "rec_real": rec_real, "f1_real": f1_real, "auc_real": auc_real,
            "acc_synth": acc_synth, "prec_synth": prec_synth, "rec_synth": rec_synth, "f1_synth": f1_synth, "auc_synth": auc_synth,
            "out_csv": out
        }
    except Exception as e:
        print(f"[ERROR] Utility eval failed for {ds} {tag}: {e}")
        return None

def hist_compare(real, synth, ds, tag, max_cols=6):
    try:
        vars_real = np.nanvar(real, axis=0)
        idx = np.argsort(vars_real)[::-1][:max_cols]
        fig, axs = plt.subplots(math.ceil(len(idx)/2), 2, figsize=(10, 4*math.ceil(len(idx)/2)))
        axs = axs.flatten()
        for i, col_idx in enumerate(idx):
            try:
                axs[i].hist(real[:,col_idx], bins=80, alpha=0.6, label='real', density=True)
                axs[i].hist(synth[:,col_idx], bins=80, alpha=0.6, label='synth', density=True)
                axs[i].set_title(f"col {col_idx}")
                axs[i].legend()
            except Exception:
                axs[i].text(0.1, 0.5, "plot error")
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
        out = os.path.join(RESULTS_DIR, f"{ds}_{tag}_hist_cols.png")
        safe_savefig(fig, out)
        plt.close(fig)
        print(f"[OK] Hist plot saved: {out}")
        return out
    except Exception as e:
        print(f"[ERROR] Hist compare failed for {ds} {tag}: {e}")
        return None

# -------------------------
# Runner
# -------------------------
def run_dataset(ds):
    real = load_real(ds)
    if real is None:
        print(f"[SKIP] No real for {ds}")
        return None

    col_names = load_cols(ds)
    synths = {}
    for tag in SYN_FILES.keys():
        loaded = load_synth(ds, tag)
        if loaded is not None:
            synths[tag] = loaded
        else:
            print(f"[WARN] No synth for {ds} method {tag}")

    summary_rows = []
    for tag, val in synths.items():
        synth_arr, synth_cols = val
        min_cols = min(real.shape[1], synth_arr.shape[1])
        if synth_arr.shape[1] != real.shape[1]:
            print(f"[INFO] Aligning dims for {ds} {tag}: real {real.shape[1]} vs synth {synth_arr.shape[1]} -> using first {min_cols} columns")
            real_a = real[:, :min_cols]
            synth_a = synth_arr[:, :min_cols]
        else:
            real_a, synth_a = real, synth_arr

        synth_inv = maybe_inverse_scale(SCALER_FILE.format(ds=ds), synth_a)
        real_inv = maybe_inverse_scale(SCALER_FILE.format(ds=ds), real_a)

        pca_png = pca_tsne_plot(real_a, synth_a, ds, tag)
        ks_csv = univariate_ks_tests(real_a, synth_a, ds, tag, col_names)
        dcr_mean, dcr_txt = dcr(real_a, synth_a, ds, tag)
        util = utility_classification(real_a, synth_a, ds, tag)
        hist_png = hist_compare(real_a, synth_a, ds, tag)

        summary_rows.append({
            "dataset": ds,
            "method": tag,
            "pca_tsne_png": pca_png,
            "ks_csv": ks_csv,
            "dcr_mean": dcr_mean,
            "dcr_txt": dcr_txt,
            "utility_csv": util["out_csv"] if util else None,
            "acc_real": util["acc_real"] if util else None,
            "acc_synth": util["acc_synth"] if util else None,
            "hist_png": hist_png
        })

    return summary_rows

def run_all():
    all_rows = []
    for ds in DATASETS:
        print(f"\n=== Evaluating dataset: {ds} ===")
        rows = run_dataset(ds)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        print("[INFO] No evaluations completed.")
        return

    df = pd.DataFrame(all_rows)
    summary_out = os.path.join(RESULTS_DIR, "summary_evaluation_table.csv")
    df.to_csv(summary_out, index=False)
    print(f"[OK] Summary saved → {summary_out}")

if __name__ == "__main__":
    run_all()
