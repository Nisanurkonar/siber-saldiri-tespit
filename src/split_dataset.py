import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

data_folder = Path("data")

files = sorted(data_folder.glob("*.parquet"))

all_data = []

print("=" * 70)
print("EĞİTİM / TEST VERİSİ AYIRMA")
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

X = data.drop(columns=["Target"])
y = data["Target"]

print("\nToplam veri:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEğitim verisi:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nTest verisi:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

print("\nEğitim Target dağılımı:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\nTest Target dağılımı:")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)

print("\n" + "=" * 70)
print("EĞİTİM / TEST AYIRMA TAMAMLANDI")
print("=" * 70)