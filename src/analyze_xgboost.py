
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

results_folder = Path("results")
results_file = results_folder / "xgboost_results.txt"

if not results_file.exists():
    raise FileNotFoundError(
        "XGBoost sonuç dosyası bulunamadı: results/xgboost_results.txt"
    )

results_text = results_file.read_text(encoding="utf-8")

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

metrics = []

for metric_name in metric_names:
    prefix = f"{metric_name}:"

    for line in results_text.splitlines():
        if line.strip().startswith(prefix):
            score = float(line.split(":", 1)[1].strip())
            metrics.append({
                "Metric": metric_name,
                "Score": score
            })
            break
    else:
        raise ValueError(f"{metric_name} metriği sonuç dosyasında bulunamadı.")

metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    results_folder / "xgboost_metrics.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metrics_df["Metric"], metrics_df["Score"])

ax.set_ylim(0, 1.08)
ax.set_ylabel("Skor")
ax.set_title("XGBoost - Model Metrikleri")

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
    results_folder / "xgboost_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("XGBoost sonuç analizi tamamlandı.")
print("\nModel metrikleri:")
print(metrics_df.to_string(index=False))
print("\nOluşturulan dosyalar:")
print("- results/xgboost_metrics.csv")
print("- results/xgboost_metrics.png")
