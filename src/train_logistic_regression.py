import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


data_folder = Path("data")
results_folder = Path("results")

results_folder.mkdir(exist_ok=True)

files = sorted(data_folder.glob("*.parquet"))

all_data = []

print("=" * 70)
print("LOGISTIC REGRESSION MODELİ")
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
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEğitim verisi:")
print(X_train.shape)

print("\nTest verisi:")
print(X_test.shape)

print("\nEğitim için dengeli örnekleme yapılıyor...")

train_data = X_train.copy()
train_data["Target"] = y_train.values

benign = train_data[train_data["Target"] == 0]
attack = train_data[train_data["Target"] == 1]

sample_size = min(len(benign), len(attack), 300000)

benign_sample = benign.sample(
    n=sample_size,
    random_state=42
)

attack_sample = attack.sample(
    n=sample_size,
    random_state=42
)

train_sample = pd.concat(
    [benign_sample, attack_sample],
    ignore_index=True
)

train_sample = train_sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_train_sample = train_sample.drop(columns=["Target"])
y_train_sample = train_sample["Target"]

print("\nModel eğitim verisi:")
print(X_train_sample.shape)

print("\nSınıf dağılımı:")
print(y_train_sample.value_counts())

print("\nStandardScaler uygulanıyor...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_sample)
X_test_scaled = scaler.transform(X_test)

print("Ölçeklendirme tamamlandı.")

print("\nLogistic Regression eğitiliyor...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train_sample)

print("Model eğitimi tamamlandı.")

print("\nTest verisi üzerinde tahmin yapılıyor...")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 70)
print("MODEL SONUÇLARI")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

report = classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "ATTACK"]
)

print("\nClassification Report:")
print(report)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

results_file = results_folder / "logistic_regression_results.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("LOGISTIC REGRESSION MODEL SONUÇLARI\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(report)

    f.write("\nConfusion Matrix\n")
    f.write("=" * 60 + "\n")
    f.write(str(cm))

print(f"\nSonuçlar kaydedildi: {results_file}")

print("\nConfusion Matrix grafiği oluşturuluyor...")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "ATTACK"]
)

disp.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

confusion_matrix_file = (
    results_folder / "logistic_regression_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion Matrix kaydedildi: "
    f"{confusion_matrix_file}"
)

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION ANALİZİ TAMAMLANDI")
print("=" * 70)