
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_folder = Path("results")
results_folder.mkdir(exist_ok=True)

results_file = results_folder / "decision_tree_results.txt"

if not results_file.exists():
    raise FileNotFoundError(
        "decision_tree_results.txt bulunamadı. Önce train_decision_tree.py dosyasını çalıştır."
    )

results_text = results_file.read_text(encoding="utf-8")

metrics = {}

for line in results_text.splitlines():
    if ":" not in line:
        continue

    name, value = line.split(":", 1)
    name = name.strip()
    value = value.strip()

    if name in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        metrics[name] = float(value)

metric_order = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

missing_metrics = [metric for metric in metric_order if metric not in metrics]

if missing_metrics:
    raise ValueError(
        f"Sonuç dosyasında eksik metrikler var: {missing_metrics}"
    )

metrics_df = pd.DataFrame({
    "Metric": metric_order,
    "Score": [metrics[metric] for metric in metric_order]
})

csv_file = results_folder / "decision_tree_metrics.csv"
metrics_df.to_csv(csv_file, index=False)

plt.figure(figsize=(9, 5))

bars = plt.bar(
    metrics_df["Metric"],
    metrics_df["Score"]
)

plt.ylim(0, 1.08)
plt.ylabel("Skor")
plt.title("Decision Tree - Model Metrikleri")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.015,
        f"{height:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

metrics_plot = results_folder / "decision_tree_metrics.png"

plt.savefig(
    metrics_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("DECISION TREE METRIKLERI")
print(metrics_df.to_string(index=False))
print(f"\nMetrik tablosu kaydedildi: {csv_file}")
print(f"Metrik grafiği kaydedildi: {metrics_plot}")
