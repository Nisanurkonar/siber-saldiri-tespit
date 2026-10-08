import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("data")

files = sorted(data_folder.glob("*.parquet"))

print("=" * 70)
print("ÖZELLİKLERİN KONTROLÜ")
print("=" * 70)

all_data = []

for file in files:
    print(f"Okunuyor: {file.name}")
    df = pd.read_parquet(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("\nVeri boyutu:")
print(data.shape)

print("\nEksik değer toplamı:")
print(data.isnull().sum().sum())

numeric_data = data.select_dtypes(include=[np.number])

print("\nSayısal özellik sayısı:")
print(len(numeric_data.columns))

print("\nSonsuz değer kontrolü:")

inf_counts = np.isinf(numeric_data).sum()

inf_counts = inf_counts[inf_counts > 0]

if len(inf_counts) == 0:
    print("Sonsuz değer bulunamadı.")
else:
    print(inf_counts)

print("\nSabit değerli sütunlar:")

constant_columns = []

for column in numeric_data.columns:
    if numeric_data[column].nunique() <= 1:
        constant_columns.append(column)

if len(constant_columns) == 0:
    print("Sabit değerli sütun bulunamadı.")
else:
    for column in constant_columns:
        print(column)

print("\nLabel ve Target sütunları:")
print(data["Label"].value_counts())

print("\nTarget dağılımı:")

data["Target"] = np.where(
    data["Label"] == "Benign",
    "BENIGN",
    "ATTACK"
)

print(data["Target"].value_counts())

print("\n" + "=" * 70)
print("ÖZELLİK KONTROLÜ TAMAMLANDI")
print("=" * 70)