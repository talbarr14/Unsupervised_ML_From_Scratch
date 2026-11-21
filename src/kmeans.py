import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

dataset_number = int(input("Enter dataset number (1, 2, or 3): "))

if dataset_number not in [1, 2, 3]:
    raise ValueError("Dataset number must be 1, 2, or 3.")

data_file = (
    project_folder / "data" /
    f"dataset{dataset_number}_cleaned.csv"
)

df = pd.read_csv(data_file)
points = df[["X", "Y", "Z"]].to_numpy()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def kmeans(points, k, max_iterations=100, tolerance=0.0001):
    n = len(points)

    if k < 1 or k > n:
        raise ValueError("k must be between 1 and the number of points.")

    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=k, replace=False)
    centroids = points[indices].copy()

    for iteration in range(max_iterations):

        labels = []

        for point in points:
            distances = []

            for centroid in centroids:
                distance = euclidean_distance(point, centroid)
                distances.append(distance)

            closest_cluster = np.argmin(distances)
            labels.append(closest_cluster)

        labels = np.array(labels)

        new_centroids = centroids.copy()

        for cluster_index in range(k):
            cluster_points = points[labels == cluster_index]

            if len(cluster_points) > 0:
                new_centroids[cluster_index] = np.mean(
                    cluster_points, axis=0
                )

        movements = []

        for i in range(k):
            movement = euclidean_distance(
                centroids[i], new_centroids[i]
            )
            movements.append(movement)

        centroids = new_centroids

        if max(movements) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
    else:
        print(f"Reached maximum of {max_iterations} iterations.")

    final_labels = []

    for point in points:
        distances = [
            euclidean_distance(point, centroid)
            for centroid in centroids
        ]
        final_labels.append(np.argmin(distances))

    return np.array(final_labels), centroids

def calculate_wcss(points, labels, centroids):
    wcss = 0.0

    for i in range(len(points)):
        centroid = centroids[labels[i]]
        distance = euclidean_distance(points[i], centroid)
        wcss += distance ** 2

    return wcss

k = int(input("Enter the number of clusters (k): "))

labels, centroids = kmeans(points, k)

wcss = calculate_wcss(points, labels, centroids)
print(f"Within-Cluster Sum of Squares (WCSS): {wcss:.4f}")

print("\nK-means Clustering Results")
print("--------------------------")
print("Total points:", len(points))
print("Number of clusters:", k)

for i in range(k):
    cluster_size = np.sum(labels == i)
    print(f"\nCluster {i + 1}:")
    print("Number of points:", cluster_size)
    print("Centroid:", centroids[i])

results_df = df.copy()
results_df["Cluster"] = labels + 1

results_df.to_csv(
    project_folder / "results" /
    f"dataset{dataset_number}_kmeans.csv",
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

ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    c="black",
    marker="X",
    s=180,
    label="Centroids"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Dataset {dataset_number}: K-means Clustering (k={k})")
ax.legend()

plt.savefig(
    project_folder / "results" /
    f"dataset{dataset_number}_kmeans.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
