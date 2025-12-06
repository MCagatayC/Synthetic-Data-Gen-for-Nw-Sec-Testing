import os
import shutil

PHASE2_DIRS = [
    "data",
    "models",
    "latent_dataset",
    "synthetic_data",
    "results"
]

def ensure_dirs():
    for d in PHASE2_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Klasör hazır: {d}")

def clean_temp():
    temp_dirs = ["__pycache__", ".ipynb_checkpoints"]
    for t in temp_dirs:
        if os.path.exists(t):
            shutil.rmtree(t)
            print(f"✓ Temizlendi: {t}")

if __name__ == "__main__":
    print("\n=== Phase2 Kurulum Başlıyor ===\n")
    ensure_dirs()
    clean_temp()
    print("\n=== ✔ Phase2 Setup Tamamlandı ===")
