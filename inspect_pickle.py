#!/usr/bin/env python3
# inspect_pickle.py
# Kullanım: python3 inspect_pickle.py github_repo/models/scaler.pkl

import sys
import pickletools
import re
from pathlib import Path

def disassemble_pickle(path):
    print("\n=== PICKLE DISASSEMBLY (pickletools.dis) ===\n")
    try:
        with open(path, "rb") as f:
            data = f.read()
        # pickletools.dis yazdırır — çok uzun olabilir ama faydalıdır
        pickletools.dis(data)
    except Exception as e:
        print(f"[!] pickletools.dis hata verdi: {e}")

def extract_unicode_tokens(path, max_lines=200):
    print("\n=== BULUNAN UNICODE TOKENS (sınırlı çıktı) ===\n")
    try:
        with open(path, "rb") as f:
            data = f.read()
        found = []
        count = 0
        for opcode, arg, pos in pickletools.genops(data):
            # arg genelde string ise burada gelir
            name = getattr(opcode, "name", str(opcode))
            if name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE"):
                s = arg
                # bazı arg'lar bytes olabilir; str'e çevir
                if isinstance(s, bytes):
                    try:
                        s = s.decode("utf-8", errors="replace")
                    except:
                        s = str(s)
                # Basit filtre: çok uzun olmayan, okunaklı stringleri al
                if 0 < len(s) <= 300:
                    found.append((pos, name, s))
                    count += 1
                    if count >= max_lines:
                        break
        if not found:
            print("[i] Unicode token bulunamadı veya çok uzun token'lar var.")
            return []
        for pos, opname, s in found:
            print(f"{pos:06d}  {opname:20s}  {s}")
        return found
    except Exception as e:
        print(f"[!] token çıkarma sırasında hata: {e}")
        return []

def heuristic_versions(found_tokens):
    print("\n=== HEURISTIK: Olası sürüm / modül tespitleri ===\n")
    version_regex = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
    suspects = []
    for pos, opname, s in found_tokens:
        low = s.lower()
        if "sklearn" in low or "scikit" in low or "numpy" in low or "scipy" in low:
            suspects.append((pos, s))
            continue
        # versiyon benzeri stringler
        m = version_regex.search(s)
        if m:
            suspects.append((pos, s))
    if not suspects:
        print("[i] Heuristik olarak bir şey bulunamadı.")
    else:
        for pos, s in suspects:
            print(f"{pos:06d}   {s}")
    return suspects

def main():
    if len(sys.argv) != 2:
        print("Kullanım: python3 inspect_pickle.py <scaler.pkl yolu>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[!] Dosya bulunamadı: {path}")
        sys.exit(1)

    # 1) Disassemble (uzun olabilir)
    disassemble_pickle(path)

    # 2) Unicode token çıkar
    tokens = extract_unicode_tokens(path, max_lines=400)

    # 3) Heuristik sürüm/modül tespiti
    heuristic_versions(tokens)

    print("\n=== BİTTİ ===\n")
    print("Not: Bu script pkl'i deserialize etmez. Çıktıyı buraya kopyalarsan, ben hangi numpy/scikit-learn sürümlerinin gerektiğini tespit etmene yardımcı olurum.")

if __name__ == "__main__":
    main()
