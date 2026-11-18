import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

# Generate 3 groups of points in 3D space
cluster1 = np.random.normal(
    loc=[2, 2, 2], scale=1.0, size=(165, 3)
)

cluster2 = np.random.normal(
    loc=[10, 10, 10], scale=1.0, size=(165, 3)
)

cluster3 = np.random.normal(
    loc=[18, 2, 10], scale=1.0, size=(160, 3)
)

# Generate 10 potential outliers
outliers = np.random.uniform(
    low=[-10, -10, -10],
    high=[30, 20, 25],
    size=(10, 3)
)

S = np.vstack((cluster1, cluster2, cluster3, outliers))

project_folder = Path(__file__).resolve().parent.parent
output_file = project_folder / "data" / "dataset1.csv"

df = pd.DataFrame(S, columns=["X", "Y", "Z"])
df.to_csv(output_file, index=False)

print("Total points:", len(S))
print("Dataset shape:", S.shape)
print(df.head())

# Visualize the dataset
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["X"], df["Y"], df["Z"], s=15, alpha=0.7)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dataset 1: 500 Points in 3D Space")

plt.show()