import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(123)

cluster1 = np.random.normal(
    loc=[2, 2, 2],
    scale=1.7,
    size=(165, 3)
)

cluster2 = np.random.normal(
    loc=[5, 5, 5],
    scale=1.8,
    size=(165, 3)
)

cluster3 = np.random.normal(
    loc=[8, 2, 5],
    scale=1.5,
    size=(160, 3)
)

outliers = np.random.uniform(
    low=[-10, -10, -10],
    high=[20, 20, 20],
    size=(10, 3)
)

S = np.vstack((cluster1, cluster2, cluster3, outliers))

project_folder = Path(__file__).resolve().parent.parent

df = pd.DataFrame(S, columns=["X", "Y", "Z"])

output_file = project_folder / "data" / "dataset2.csv"
df.to_csv(output_file, index=False)

print("Dataset 2 generated successfully!")
print("Total points:", len(S))
print("Dataset shape:", S.shape)
print("\nFirst 5 points:")
print(df.head())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["X"],
    df["Y"],
    df["Z"],
    s=15,
    alpha=0.65,
    color="blue"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Dataset 2: 500 Overlapping 3D Points")

plt.savefig(
    project_folder / "results" / "dataset2_original.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()