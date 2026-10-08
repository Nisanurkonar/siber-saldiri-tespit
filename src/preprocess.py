import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("data")

files = sorted(data_folder.glob("*.parquet"))

all_data = []

print("=" * 70)
print("VERİ ÖN İŞLEME")
print("=" * 70)

for file in files:
    print(f"Okunuyor: {file.name}")

    df = pd.read_parquet(file)

    df["Target"] = np.where(
        df["Label"] == "Benign",
        0,
        1
    )

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("\nBirleştirilmiş veri boyutu:")
print(data.shape)

constant_columns = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

data = data.drop(columns=constant_columns)

data = data.drop(columns=["Label"])

print("\nSabit sütunlar çıkarıldı.")
print("Label sütunu çıkarıldı.")

print("\nYeni veri boyutu:")
print(data.shape)

print("\nTarget dağılımı:")
print(data["Target"].value_counts())

print("\nTarget yüzdeleri:")
print(data["Target"].value_counts(normalize=True) * 100)

X = data.drop(columns=["Target"])
y = data["Target"]

print("\nÖzellik matrisi X:")
print(X.shape)

print("\nHedef değişken y:")
print(y.shape)

print("\nEksik değer:")
print(X.isnull().sum().sum())

print("\nSonsuz değer:")
print(np.isinf(X).sum().sum())

print("\n" + "=" * 70)
print("ÖN İŞLEME KONTROLÜ TAMAMLANDI")
print("=" * 70)