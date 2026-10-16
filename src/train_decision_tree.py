
import gc
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

data_folder = Path("data")
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

data_files = sorted(data_folder.glob("*.parquet"))

if not data_files:
    raise FileNotFoundError("data klasöründe Parquet dosyası bulunamadı.")

dataframes = []

for file in data_files:
    print(f"Veri yükleniyor: {file.name}")
    df = pd.read_parquet(file)

    if "Label" not in df.columns:
        raise ValueError(f"{file.name} dosyasında Label sütunu bulunamadı.")

    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)
del dataframes
gc.collect()

data["Label"] = data["Label"].astype(str).str.strip()
data["Target"] = (
    data["Label"].str.upper() != "BENIGN"
).astype("int8")

print("\nEtiket dağılımı:")
print(data["Target"].value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

if data["Target"].nunique() != 2:
    raise ValueError("Veri setinde iki sınıf bulunamadı.")

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

X = data.drop(
    columns=["Label", "Target"] + constant_columns,
    errors="ignore"
)

y = data["Target"].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(0)

del data
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

del X, y
gc.collect()

print("\nEğitim ve test verileri hazırlandı.")
print(f"Eğitim verisi: {len(X_train):,}")
print(f"Test verisi: {len(X_test):,}")
print("\nEğitim etiket dağılımı:")
print(y_train.value_counts())

train_data = X_train.copy()
train_data["Target"] = y_train.to_numpy()

benign = train_data[train_data["Target"] == 0]
attack = train_data[train_data["Target"] == 1]

sample_size = min(len(benign), len(attack))

if sample_size == 0:
    raise ValueError(
        "Dengeli eğitim verisi oluşturulamadı. "
        "Yukarıdaki etiket dağılımını kontrol edin."
    )

benign_sample = benign.sample(n=sample_size, random_state=42)
attack_sample = attack.sample(n=sample_size, random_state=42)

balanced_train = pd.concat(
    [benign_sample, attack_sample],
    ignore_index=True
).sample(frac=1, random_state=42)

X_train_balanced = balanced_train.drop(columns=["Target"])
y_train_balanced = balanced_train["Target"]

print(f"\nDengelenmiş eğitim verisi: {len(X_train_balanced):,}")
print("Dengeli sınıf dağılımı:")
print(y_train_balanced.value_counts())

del train_data, benign, attack
del benign_sample, attack_sample, balanced_train
del X_train, y_train
gc.collect()

print("\nDecision Tree modeli eğitiliyor...")

model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_balanced, y_train_balanced)

print("Model eğitimi tamamlandı. Tahmin yapılıyor...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Score": [accuracy, precision, recall, f1, roc_auc]
})

results_text = f"""
DECISION TREE MODEL SONUCLARI

Egitim verisi: {len(X_train_balanced)}
Test verisi: {len(X_test)}

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Karmasiklik Matrisi:
{matrix}

Siniflandirma Raporu:
{classification_report(y_test, y_pred, labels=[0, 1], target_names=["BENIGN", "ATTACK"], zero_division=0)}
"""

(results_folder / "decision_tree_results.txt").write_text(
    results_text,
    encoding="utf-8"
)

metrics.to_csv(
    results_folder / "decision_tree_metrics.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["BENIGN", "ATTACK"]
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title("Decision Tree - Karmasiklik Matrisi")
fig.tight_layout()
fig.savefig(
    results_folder / "decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metrics["Metric"], metrics["Score"])

ax.set_ylim(0, 1.08)
ax.set_ylabel("Skor")
ax.set_title("Decision Tree - Model Metrikleri")

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015,
        f"{height:.4f}",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(
    results_folder / "decision_tree_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

importance = pd.DataFrame({
    "Feature": X_train_balanced.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    results_folder / "decision_tree_feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 7))
top_features = importance.head(15).sort_values("Importance")
ax.barh(top_features["Feature"], top_features["Importance"])
ax.set_xlabel("Özellik Önemi")
ax.set_title("Decision Tree - En Önemli 15 Özellik")
fig.tight_layout()
fig.savefig(
    results_folder / "decision_tree_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print(results_text)
print("\nOluşturulan dosyalar:")
print("- results/decision_tree_results.txt")
print("- results/decision_tree_metrics.csv")
print("- results/decision_tree_confusion_matrix.png")
print("- results/decision_tree_metrics.png")
print("- results/decision_tree_feature_importance.csv")
print("- results/decision_tree_feature_importance.png")
