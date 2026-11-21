import numpy as np
import pandas as pd
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

dataset_number = int(input("Enter dataset number (1, 2, or 3): "))

if dataset_number not in [1, 2, 3]:
    raise ValueError("Dataset number must be 1, 2, or 3.")

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def silhouette_score(points, labels):
    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        raise ValueError("Silhouette requires at least 2 clusters.")

    scores = []

    for i in range(len(points)):
        current_cluster = labels[i]

        same_cluster_indices = np.where(
            labels == current_cluster
        )[0]

        if len(same_cluster_indices) == 1:
            scores.append(0.0)
            continue

        within_distances = []

        for j in same_cluster_indices:
            if i != j:
                within_distances.append(
                    euclidean_distance(points[i], points[j])
                )

        a = np.mean(within_distances)

        other_cluster_distances = []

        for cluster in unique_clusters:
            if cluster == current_cluster:
                continue

            other_indices = np.where(labels == cluster)[0]

            distances = [
                euclidean_distance(points[i], points[j])
                for j in other_indices
            ]

            other_cluster_distances.append(np.mean(distances))

        b = min(other_cluster_distances)

        if max(a, b) == 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)

        scores.append(s)

    return np.mean(scores)


result_files = {
    "K-means": f"dataset{dataset_number}_kmeans.csv",
    "Single Linkage": f"dataset{dataset_number}_hierarchical_single.csv",
    "Complete Linkage": f"dataset{dataset_number}_hierarchical_complete.csv",
    "Average Linkage": f"dataset{dataset_number}_hierarchical_average.csv",
    "Centroid Linkage": f"dataset{dataset_number}_hierarchical_centroid.csv"
}

print("\nSilhouette Coefficient Comparison")
print("---------------------------------")

results = {}

for algorithm, filename in result_files.items():
    file_path = project_folder / "results" / filename

    df = pd.read_csv(file_path)
    points = df[["X", "Y", "Z"]].to_numpy()
    labels = df["Cluster"].to_numpy()

    score = silhouette_score(points, labels)
    results[algorithm] = score

    print(f"{algorithm}: {score:.4f}")

highest_score = max(results.values())

best_algorithms = [
    algorithm
    for algorithm, score in results.items()
    if abs(score - highest_score) < 0.0001
]

print("\nHighest Silhouette Score:", round(highest_score, 4))
print("Best performing algorithm(s):")

for algorithm in best_algorithms:
    print("-", algorithm)