import pandas as pd
from pathlib import Path

data_folder = Path("data")

files = sorted(data_folder.glob("*.parquet"))

all_data = []

print("=" * 70)
print("VERİ SETİ HAZIRLANIYOR")
print("=" * 70)

for file in files:
    print(f"\nOkunuyor: {file.name}")

    df = pd.read_parquet(file)

    df["Target"] = df["Label"].apply(
        lambda x: "BENIGN" if x == "Benign" else "ATTACK"
    )

    all_data.append(df)

    print(f"Satır sayısı: {len(df):,}")

print("\nVeriler birleştiriliyor...")

data = pd.concat(all_data, ignore_index=True)

print("\n" + "=" * 70)
print("BİRLEŞTİRİLMİŞ VERİ SETİ")
print("=" * 70)

print(f"Satır sayısı: {len(data):,}")
print(f"Sütun sayısı: {len(data.columns)}")

print("\nTarget dağılımı:")
print(data["Target"].value_counts())

print("\nTarget yüzdeleri:")
print(data["Target"].value_counts(normalize=True) * 100)

print("\nOrijinal Label dağılımı:")
print(data["Label"].value_counts())

print("\nEksik değer toplamı:")
print(data.isnull().sum().sum())

print("\nHazırlama işlemi tamamlandı.")