import os
import shutil
import zipfile
import subprocess
from kaggle import KaggleApi

# ============================================
# 1 — KLASÖRLERİ OLUŞTUR
# ============================================
REQUIRED_DIRS = ["data", "models", "latent_dataset", "synthetic_data", "results"]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"Klasör kontrol edildi/oluşturuldu: {d}")

print("\n✔ Klasörler hazır.\n")


# ============================================
# 2 — KAGGLE API DOĞRULAMA
# ============================================
print("Kaggle API doğrulanıyor...")

try:
    api = KaggleApi()
    api.authenticate()
    print("✔ Kaggle API doğrulandı.\n")
except Exception as e:
    print("❌ Kaggle API doğrulanamadı!", e)
    exit()


# ============================================
# 3 — DATASET TANIMLARI (Phase2 güncellenmiş)
# ============================================
DATASETS = {
    "CICIDS2017": {
        "ref": "ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
    },
    "UNSW-NB15": {
        "ref": "mrwellsdavid/unsw-nb15"
    }
}


# ============================================
# 4 — DATASET İNDİRME & EXTRACT FONKSİYONU
# ============================================
def download_and_extract(dataset_name, ref):
    print(f"\n>>> {dataset_name} indiriliyor...")
    print(f"Dataset URL: https://www.kaggle.com/datasets/{ref}")

    temp_dir = f"_temp_{dataset_name}"
    extract_dir = f"_extracted_{dataset_name}"

    # Eski dosyaları temizle
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    os.makedirs(temp_dir, exist_ok=True)

    # Kaggle CLI ile ZIP indir
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", ref, "-p", temp_dir],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ Kaggle indirme hatası:", result.stderr)
        return None

    # ZIP dosyasını bul
    zips = [f for f in os.listdir(temp_dir) if f.endswith(".zip")]
    if not zips:
        print("❌ ZIP bulunamadı!")
        return None

    zip_path = os.path.join(temp_dir, zips[0])
    print(f"ZIP bulundu: {zips[0]}")

    # Extract et
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    print("✓ Çıkarıldı:", extract_dir)
    return extract_dir


# ============================================
# 5 — CSV DOSYALARINI DATA/ ALTINA TAŞI
# ============================================
for name, info in DATASETS.items():
    extracted = download_and_extract(name, info["ref"])
    if not extracted:
        continue

    # Tüm CSV dosyalarını bul
    csv_files = []
    for root, _, files in os.walk(extracted):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        print(f"❌ {name} için CSV bulunamadı!")
        continue

    target_dir = os.path.join("data", name)
    os.makedirs(target_dir, exist_ok=True)

    for src in csv_files:
        dst = os.path.join(target_dir, os.path.basename(src))
        shutil.move(src, dst)
        print(f"→ Taşındı: {os.path.basename(src)} → {target_dir}")

print("\n--- ✔ Kurulum Tamamlandı ✔ ---")
