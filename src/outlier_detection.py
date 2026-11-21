import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent.parent

data_file = project_folder / "data" / "dataset1.csv"
df = pd.read_csv(data_file)
points = df[["X", "Y", "Z"]].to_numpy()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_knn_distances(points, k=5):
    scores = []

    for i in range(len(points)):
        distances = []

        for j in range(len(points)):
            if i != j:
                distance = euclidean_distance(points[i], points[j])
                distances.append(distance)

        distances.sort()
        average_distance = np.mean(distances[:k])
        scores.append(average_distance)

    return np.array(scores)

scores = calculate_knn_distances(points, k=5)

q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)
iqr = q3 - q1

threshold = q3 + 1.5 * iqr

outlier_mask = scores > threshold
detected_outliers = points[outlier_mask]
cleaned_points = points[~outlier_mask]

# Print results
print("Original dataset size:", len(points))
print("Outlier threshold:", round(threshold, 4))
print("Number of detected outliers:", len(detected_outliers))
print("Cleaned dataset size:", len(cleaned_points))

if len(detected_outliers) == 0:
    print("No outliers detected.")
else:
    print("\nDetected outliers:")
    for point in detected_outliers:
        print(point)

cleaned_df = pd.DataFrame(
    cleaned_points, columns=["X", "Y", "Z"]
)
cleaned_df.to_csv(
    project_folder / "data" / "dataset1_cleaned.csv",
    index=False
)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

normal = points[~outlier_mask]

ax.scatter(
    normal[:, 0], normal[:, 1], normal[:, 2],
    c="blue", s=15, alpha=0.6, label="Normal Points"
)

if len(detected_outliers) > 0:
    ax.scatter(
        detected_outliers[:, 0],
        detected_outliers[:, 1],
        detected_outliers[:, 2],
        c="red", s=50, label="Detected Outliers"
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dataset 1: Outlier Detection")
ax.legend()

plt.savefig(
    project_folder / "results" / "dataset1_outliers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()