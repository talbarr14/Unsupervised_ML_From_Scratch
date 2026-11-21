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
    f"dataset{dataset_number}.csv"
)

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

threshold = q3 + 3.0 * iqr

outlier_mask = scores > threshold
detected_outliers = points[outlier_mask]
cleaned_points = points[~outlier_mask]

injected_mask = np.zeros(len(points), dtype=bool)
injected_mask[-10:] = True

true_positives = np.sum(outlier_mask & injected_mask)

false_positives = np.sum(outlier_mask & ~injected_mask)

false_negatives = np.sum(~outlier_mask & injected_mask)

precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0
)

recall = true_positives / (
    true_positives + false_negatives
)

print("\nOutlier Detection Validation")
print("----------------------------")
print("Injected candidate outliers:", 10)
print("Injected points detected:", true_positives)
print("Regular points flagged:", false_positives)
print("Injected points missed:", false_negatives)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

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
    project_folder / "data" /
    f"dataset{dataset_number}_cleaned.csv",
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
ax.set_title(f"Dataset {dataset_number}: Outlier Detection")
ax.legend()

plt.savefig(
    project_folder / "results" /
    f"dataset{dataset_number}_outliers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()