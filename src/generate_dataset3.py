import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(456)

# Create three elongated clusters in different directions

# Cluster 1: Elongated along the X-axis
cluster1 = np.random.normal(
    loc=[2, 2, 2],
    scale=[3.0, 0.7, 0.7],
    size=(165, 3)
)

# Cluster 2: Elongated along the Y-axis
cluster2 = np.random.normal(
    loc=[10, 10, 10],
    scale=[0.7, 3.0, 0.7],
    size=(165, 3)
)

# Cluster 3: Elongated along the Z-axis
cluster3 = np.random.normal(
    loc=[18, 2, 10],
    scale=[0.7, 0.7, 3.0],
    size=(160, 3)
)

# Generate 10 potential outliers
outliers = np.random.uniform(
    low=[-10, -10, -10],
    high=[30, 20, 25],
    size=(10, 3)
)

# Combine all 500 points
S = np.vstack((cluster1, cluster2, cluster3, outliers))

# Locate project directory
project_folder = Path(__file__).resolve().parent.parent

# Create DataFrame
df = pd.DataFrame(S, columns=["X", "Y", "Z"])

# Save dataset
output_file = project_folder / "data" / "dataset3.csv"
df.to_csv(output_file, index=False)

# Print information
print("Dataset 3 generated successfully!")
print("Total points:", len(S))
print("Dataset shape:", S.shape)
print("\nFirst 5 points:")
print(df.head())

# Visualize original dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["X"],
    df["Y"],
    df["Z"],
    color="blue",
    s=15,
    alpha=0.65
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dataset 3: 500 Elongated 3D Points")

plt.savefig(
    project_folder / "results" / "dataset3_original.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
