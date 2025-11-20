# Towards Delivering High-Quality Realism at High Speed: Synthetic Traffic for Modern Security Testing

**Authors:** Mahmut Çağatay Çoban, Enis Karaarslan  
**Institution:** Mugla Sitki Kocman University (MSKU)

This repository contains the official implementation, source code, and supplementary materials for the research paper: **"Towards Delivering High-Quality Realism at High Speed: Synthetic Traffic for Modern Security Testing"**.

## 🚀 Project Overview

The escalating sophistication of cyber threats necessitates robust security testing, yet this process is constrained by the scarcity of realistic, privacy-compliant data. While **Diffusion Models** offer superior fidelity (FID=5.2), their training time (22.7 hours) is impractical for rapid deployment.

This project introduces a novel **Hybrid VAE-Latent Diffusion (NAT-VAEv6)** model that solves this "quality-speed" dilemma.

### Key Achievements
* **⚡ 93.3% Speedup:** Reduced total training time from **22.7 hours** (Standard Diffusion) to **~1.51 hours** (NAT-VAEv6).
* **📈 High Fidelity:** Successfully replicated the complex, multi-modal distribution of real network traffic without "Mode Collapse".
* **🛡️ Security Utility:** Achieved a **23% reduction** in False Negative Rates (FNR) and a **35% increase** in Zero-Day attack detection when used to augment IDS training.
* **🔧 Technical Innovations:**
    * **Feature Enrichment:** Expanded feature space from 3 to **82 dimensions** via One-Hot Encoding.
    * **Stabilization:** Implemented **Beta-VAE ($\beta=0.1$)** loss to balance reconstruction and regularization.

---

## 📂 Repository Structure

**Note:** This repository uses **Git Large File Storage (LFS)** to manage datasets and model checkpoints.

```text
.
├── 01_train_vae.py                  # Step 1: Train the stabilized Beta-VAE (NAT-VAEv6)
├── 02_create_latent_dataset.py      # Step 2: Compress raw data into 32-dim latent space
├── 03_train_latent_diffusion.py     # Step 3: Train Transformer-based Diffusion on latent space
├── 04_generate_and_benchmark.py     # Step 4: Generate synthetic data and plot distributions
├── calculate_dcr.py                 # Utility: Calculate Distance to Closest Record (Privacy Risk)
├── calculate_utility.py             # Utility: Calculate IDS Performance (FNR Reduction)
├── create_visualizations_v6_final.py# Utility: Generate PCA/t-SNE and KDE plots
├── prepare_data.py                  # Helper: Data loading, One-Hot Encoding, Scaling
├── requirements.txt                 # Python dependencies
├── data/
│   └── Midterm_53_group.csv         # Original Dataset (Requires LFS)
└── github_repo/
    ├── benchmarks/                  # Comparative plots (PCA, t-SNE, KDE)
    ├── models/                      # Trained .pth models and scalers (Requires LFS)
    └── synthetic_data/              # Generated Synthetic Traffic .csv (Requires LFS)
