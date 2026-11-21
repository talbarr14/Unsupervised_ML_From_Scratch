import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

data_file = project_folder / "data" / "dataset1_cleaned.csv"
df = pd.read_csv(data_file)
points = df[["X", "Y", "Z"]].to_numpy()


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def hierarchical_clustering(points, k, linkage="single"):
    n = len(points)

    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n.")

    valid_methods = ["single", "complete", "average", "centroid"]

    if linkage not in valid_methods:
        raise ValueError("Invalid linkage method.")

    clusters = [[i] for i in range(n)]
    sizes = np.ones(n, dtype=int)
    centroids = points.copy()
    active = np.ones(n, dtype=bool)

    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(points[i], points[j])
            distances[i, j] = d
            distances[j, i] = d

    np.fill_diagonal(distances, np.inf)

    number_of_clusters = n

    while number_of_clusters > k:

        flat_index = np.argmin(distances)
        i, j = np.unravel_index(flat_index, distances.shape)

        size_i = sizes[i]
        size_j = sizes[j]

        new_centroid = (
            size_i * centroids[i] +
            size_j * centroids[j]
        ) / (size_i + size_j)

        for other in range(n):
            if not active[other] or other == i or other == j:
                continue

            if linkage == "single":
                new_distance = min(
                    distances[i, other],
                    distances[j, other]
                )

            elif linkage == "complete":
                new_distance = max(
                    distances[i, other],
                    distances[j, other]
                )

            elif linkage == "average":
                new_distance = (
                    size_i * distances[i, other] +
                    size_j * distances[j, other]
                ) / (size_i + size_j)

            elif linkage == "centroid":
                new_distance = euclidean_distance(
                    new_centroid, centroids[other]
                )

            distances[i, other] = new_distance
            distances[other, i] = new_distance

        clusters[i].extend(clusters[j])
        clusters[j] = []

        sizes[i] = size_i + size_j
        sizes[j] = 0
        centroids[i] = new_centroid

        active[j] = False
        distances[j, :] = np.inf
        distances[:, j] = np.inf
        distances[i, i] = np.inf

        number_of_clusters -= 1

    labels = np.zeros(n, dtype=int)
    final_clusters = [c for c in clusters if len(c) > 0]

    for cluster_index, members in enumerate(final_clusters):
        for point_index in members:
            labels[point_index] = cluster_index

    return labels


k = int(input("Enter the number of clusters (k): "))

methods = ["single", "complete", "average", "centroid"]

for method in methods:
    print(f"\nRunning {method} linkage...")

    labels = hierarchical_clustering(
        points, k, linkage=method
    )

    print(f"Results for {method} linkage:")

    for i in range(k):
        cluster_points = points[labels == i]
        print(f"Cluster {i + 1}: {len(cluster_points)} points")

    results_df = df.copy()
    results_df["Cluster"] = labels + 1

    results_df.to_csv(
        project_folder / "results" /
        f"dataset1_hierarchical_{method}.csv",
        index=False
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.get_cmap("tab20", k)

    for i in range(k):
        cluster_points = points[labels == i]

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color=colors(i),
            s=15,
            alpha=0.65,
            label=f"Cluster {i + 1}"
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Hierarchical Clustering - {method.title()} "
        f"Linkage (k={k})"
    )
    ax.legend()

    plt.savefig(
        project_folder / "results" /
        f"dataset1_hierarchical_{method}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

print("\nAll four linkage methods completed.")
print("Plots and results saved in the results folder.")