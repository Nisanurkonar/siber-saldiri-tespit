import pandas as pd
from pathlib import Path

data_folder = Path("data")

files = sorted(data_folder.glob("*.parquet"))

print("=" * 70)
print("CICIDS2017 VERİ SETİ KONTROLÜ")
print("=" * 70)

total_rows = 0

for file in files:
    print(f"\nDosya: {file.name}")

    df = pd.read_parquet(file)

    print(f"Satır sayısı: {len(df):,}")
    print(f"Sütun sayısı: {len(df.columns)}")

    print("Label dağılımı:")
    print(df["Label"].value_counts())

    total_rows += len(df)

print("\n" + "=" * 70)
print(f"TOPLAM SATIR: {total_rows:,}")
print(f"TOPLAM DOSYA: {len(files)}")
print("=" * 70)