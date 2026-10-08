import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
print("RANDOM FOREST MODELİ")
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

print("\nOrijinal eğitim verisi:")
print(X_train.shape)

print("\nÖrnekleme yapılıyor...")

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

print("\nEğitim sınıf dağılımı:")
print(y_train_sample.value_counts())

print("\nRandom Forest eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train_sample, y_train_sample)

print("Model eğitimi tamamlandı.")

print("\nTest verisi üzerinde tahmin yapılıyor...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

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

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "ATTACK"]
)

print(report)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

results_file = results_folder / "random_forest_results.txt"

with open(results_file, "w", encoding="utf-8") as f:
    f.write("RANDOM FOREST MODEL SONUÇLARI\n")
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

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

confusion_matrix_file = results_folder / "random_forest_confusion_matrix.png"

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Confusion Matrix kaydedildi: {confusion_matrix_file}")

print("\nFeature Importance hesaplanıyor...")

feature_importance = pd.DataFrame({
    "Feature": X_train_sample.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance_file = results_folder / "random_forest_feature_importance.csv"

feature_importance.to_csv(
    feature_importance_file,
    index=False
)

top_features = feature_importance.head(20)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest - Top 20 Feature Importance")

plt.tight_layout()

feature_importance_image = results_folder / "random_forest_feature_importance.png"

plt.savefig(
    feature_importance_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Feature Importance tablosu kaydedildi: {feature_importance_file}")
print(f"Feature Importance grafiği kaydedildi: {feature_importance_image}")

print("\n" + "=" * 70)
print("RANDOM FOREST ANALİZİ TAMAMLANDI")
print("=" * 70)