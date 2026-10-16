
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_folder = Path("results")

results_file = results_folder / "logistic_regression_results.txt"

if not results_file.exists():
    raise FileNotFoundError(
        f"Sonuç dosyası bulunamadı: {results_file}"
    )

metrics = {}

with open(results_file, "r", encoding="utf-8") as file:
    for line in file:
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC"
            ]:
                metrics[key] = float(value)

expected_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

missing_metrics = [
    metric for metric in expected_metrics
    if metric not in metrics
]

if missing_metrics:
    raise ValueError(
        f"Sonuç dosyasında eksik metrikler var: {missing_metrics}"
    )

metrics_df = pd.DataFrame({
    "Metric": expected_metrics,
    "Score": [metrics[metric] for metric in expected_metrics]
})

csv_file = results_folder / "logistic_regression_metrics.csv"
metrics_df.to_csv(csv_file, index=False)

plt.figure(figsize=(9, 5))

bars = plt.bar(
    metrics_df["Metric"],
    metrics_df["Score"]
)

plt.ylim(0, 1.05)
plt.ylabel("Skor")
plt.title("Logistic Regression - Performans Metrikleri")
plt.xticks(rotation=20)

for bar, score in zip(bars, metrics_df["Score"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.4f}",
        ha="center"
    )

plt.tight_layout()

image_file = results_folder / "logistic_regression_metrics.png"

plt.savefig(
    image_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("LOGISTIC REGRESSION SONUÇ ANALİZİ")
print("=" * 60)

print(metrics_df.to_string(index=False))

print(f"\nMetrik tablosu kaydedildi: {csv_file}")
print(f"Metrik grafiği kaydedildi: {image_file}")

print("\nAnaliz tamamlandı.")
