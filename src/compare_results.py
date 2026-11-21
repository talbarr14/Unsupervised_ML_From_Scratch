import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent
results_folder = project_folder / "results"

datasets = ["Dataset 1", "Dataset 2", "Dataset 3"]

algorithms = [
    "K-means",
    "Single Linkage",
    "Complete Linkage",
    "Average Linkage",
    "Centroid Linkage"
]

silhouette_scores = np.array([
    [0.8198, 0.3741, 0.6903],  # K-means
    [0.8198, -0.0650, 0.6903], # Single
    [0.8198, 0.2639, 0.6903],  # Complete
    [0.8198, 0.3480, 0.6903],  # Average
    [0.8198, 0.3525, 0.6903]   # Centroid
])

original_points = [500, 500, 500]
outliers_removed = [14, 19, 15]
cleaned_points = [486, 481, 485]

kmeans_iterations = [2, 6, 3]
kmeans_wcss = [1373.4413, 3262.7565, 5064.9797]


fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(algorithms))
width = 0.25

for i in range(len(datasets)):
    bars = ax.bar(
        x + (i - 1) * width,
        silhouette_scores[:, i],
        width,
        label=datasets[i]
    )

    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

ax.axhline(0, color="black", linewidth=0.8)

ax.set_xlabel("Clustering Algorithm")
ax.set_ylabel("Silhouette Coefficient")
ax.set_title("Silhouette Coefficient Comparison Across 3 Datasets")

ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=15, ha="right")

ax.set_ylim(-0.15, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    results_folder / "silhouette_comparison.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(datasets, outliers_removed)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.3,
        str(int(height)),
        ha="center"
    )

ax.set_xlabel("Dataset")
ax.set_ylabel("Number of Outliers Detected")
ax.set_title("Outlier Detection Across 3 Datasets")
ax.set_ylim(0, 25)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    results_folder / "outlier_comparison.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(datasets, kmeans_wcss)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 75,
        f"{height:.2f}",
        ha="center"
    )

ax.set_xlabel("Dataset")
ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)")
ax.set_title("K-means WCSS Across 3 Datasets")
ax.set_ylim(0, 6000)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    results_folder / "kmeans_wcss_comparison.png",
    dpi=300
)
plt.close()


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(datasets, kmeans_iterations)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.1,
        str(int(height)),
        ha="center"
    )

ax.set_xlabel("Dataset")
ax.set_ylabel("Number of Iterations")
ax.set_title("K-means Convergence Across 3 Datasets")
ax.set_ylim(0, 8)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.savefig(
    results_folder / "kmeans_iterations_comparison.png",
    dpi=300
)
plt.close()


summary = pd.DataFrame({
    "Dataset": datasets,
    "Original Points": original_points,
    "Outliers Removed": outliers_removed,
    "Cleaned Points": cleaned_points,
    "K-means Iterations": kmeans_iterations,
    "K-means WCSS": kmeans_wcss
})

summary.to_csv(
    results_folder / "experiment_summary.csv",
    index=False
)

print("All comparison charts generated successfully!")
print("\nExperiment Summary:")
print(summary.to_string(index=False))
print("\nFiles saved to:", results_folder)
