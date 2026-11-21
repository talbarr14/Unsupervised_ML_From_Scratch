Unsupervised Machine Learning From Scratch

Implementation and Evaluation of K-means and Hierarchical Agglomerative Clustering Algorithms

A Python-based machine learning project developed for CS 634: Data Mining at the New Jersey Institute of Technology (NJIT).

This project explores how different unsupervised learning algorithms perform on datasets with varying cluster structures. All clustering algorithms, outlier detection methods, and evaluation metrics were implemented from scratch using Python and NumPy, without relying on prebuilt machine learning libraries such as Scikit-learn.

Project Overview

The objective of this project is to implement, evaluate, and compare unsupervised machine learning algorithms on 3 synthetic datasets containing 500 points each in 3-dimensional Euclidean space.

The project investigates how differences in cluster separation, overlap, and shape affect clustering performance.

Algorithms implemented:

K-means clustering

Hierarchical agglomerative clustering with 4 linkage methods

Distance-based outlier detection

Silhouette coefficient evaluation

Technologies Used

Python — Core algorithm implementation

NumPy — Numerical calculations and array operations

Pandas — Dataset handling and CSV operations

Matplotlib — Data visualization and comparison charts

Git/GitHub — Version control and project documentation

Datasets

Three synthetic datasets were generated to test clustering algorithms under different conditions.

Dataset

Description

Original Points

Outliers Detected

Cleaned Points

Dataset 1

Well-separated Gaussian clusters

500

14

486

Dataset 2

Partially overlapping Gaussian clusters

500

19

481

Dataset 3

Elongated, non-spherical clusters

500

15

485

All datasets were generated using fixed random seeds to ensure reproducibility.

Dataset 1: Well-Separated Clusters

Three Gaussian clusters were generated with centers far apart from one another. This dataset serves as a baseline for evaluating how well clustering algorithms identify clearly defined groups.

Dataset 2: Overlapping Clusters

Three Gaussian clusters were generated with closer centers and larger standard deviations, introducing overlap between groups.

This dataset evaluates the ability of clustering algorithms to distinguish groups when cluster boundaries are less clearly defined.

Dataset 3: Elongated Clusters

Three clusters were generated with different standard deviations along the X, Y, and Z axes, producing elongated, non-spherical distributions.

This dataset evaluates how clustering algorithms handle variations in cluster shape.

Methodology

1. Outlier Detection

A distance-based outlier detection algorithm was implemented using the average Euclidean distance to each point's 5 nearest neighbors.

The algorithm:

Calculates distances between points.

Identifies the 5 nearest neighbors of each point.

Computes each point's average neighbor distance.

Determines an outlier threshold using the interquartile range (IQR).

Removes detected outliers before clustering.

The threshold is defined as:

Threshold = Q3 + 3.0 × IQR

This conservative threshold was applied consistently across all 3 datasets.

2. K-means Clustering

K-means was implemented from scratch using iterative centroid updates.

The algorithm:

Randomly selects k initial centroids from the dataset.

Assigns each point to its nearest centroid using Euclidean distance.

Recalculates centroids using the mean of assigned points.

Repeats until convergence or the maximum iteration count is reached.

K-means was evaluated using the Silhouette coefficient and Within-Cluster Sum of Squares (WCSS).

3. Hierarchical Agglomerative Clustering

Hierarchical agglomerative clustering was implemented from scratch.

Initially, every point represents an individual cluster. The algorithm repeatedly merges the closest pair of clusters until the user-specified number of clusters remains.

Four linkage methods were implemented:

Single linkage: Minimum distance between points in different clusters

Complete linkage: Maximum distance between points in different clusters

Average linkage: Average distance between points in different clusters

Centroid linkage: Euclidean distance between cluster centroids

4. Silhouette Coefficient

The Silhouette coefficient was implemented from scratch to evaluate cluster cohesion and separation.

For each point:

s(i) = (b(i) - a(i)) / max(a(i), b(i))

Where:

a(i) is the average distance to other points in the same cluster.

b(i) is the minimum average distance to points in another cluster.

The average Silhouette coefficient is used to compare clustering quality. Higher values generally indicate better-separated and more cohesive clusters.

Experimental Results

Silhouette Coefficient Comparison

Algorithm

Dataset 1

Dataset 2

Dataset 3

K-means

0.8198

0.3741

0.6903

Single Linkage

0.8198

-0.0650

0.6903

Complete Linkage

0.8198

0.2639

0.6903

Average Linkage

0.8198

0.3480

0.6903

Centroid Linkage

0.8198

0.3525

0.6903

Scores are rounded to 4 decimal places.



K-means Performance

Metric

Dataset 1

Dataset 2

Dataset 3

Points Clustered

486

481

485

Iterations to Convergence

2

6

3

WCSS

1373.4413

3262.7565

5064.9797



Key Findings

1. Well-separated clusters produce strong clustering results.

On Dataset 1, all 5 methods achieved approximately the same Silhouette coefficient of 0.8198, demonstrating that clearly separated groups are relatively straightforward to identify.

2. Overlapping clusters reveal differences between algorithms.

On Dataset 2, K-means achieved the highest Silhouette coefficient of 0.3741.

Single linkage performed poorly, producing one cluster containing 479 points and 2 singleton clusters. This demonstrates the chaining effect, a known limitation of single linkage when groups are connected by nearby points.

3. Elongated clusters can still be identified successfully.

On Dataset 3, all methods achieved approximately the same Silhouette coefficient of 0.6903.

Despite the non-spherical cluster shapes, the groups remained sufficiently separated for all methods to produce strong results.

4. Algorithm performance depends on data structure.

No clustering method is universally best. Cluster separation, overlap, and shape influence the quality of the resulting partitions.

Project Structure

Unsupervised_ML_From_Scratch/
├── data/
│   ├── dataset1.csv
│   ├── dataset1_cleaned.csv
│   ├── dataset2.csv
│   ├── dataset2_cleaned.csv
│   ├── dataset3.csv
│   └── dataset3_cleaned.csv
├── results/
│   ├── dataset1_*.csv
│   ├── dataset1_*.png
│   ├── dataset2_*.csv
│   ├── dataset2_*.png
│   ├── dataset3_*.csv
│   ├── dataset3_*.png
│   ├── silhouette_comparison.png
│   ├── outlier_comparison.png
│   ├── kmeans_wcss_comparison.png
│   ├── kmeans_iterations_comparison.png
│   └── experiment_summary.csv
├── src/
│   ├── generate_data.py
│   ├── generate_dataset2.py
│   ├── generate_dataset3.py
│   ├── outlier_detection.py
│   ├── kmeans.py
│   ├── hierarchical.py
│   ├── silhouette.py
│   └── compare_results.py
└── README.md

Installation and Usage

1. Clone the repository

git clone https://github.com/YOUR-USERNAME/Unsupervised_ML_From_Scratch.git
cd Unsupervised_ML_From_Scratch

Replace YOUR-USERNAME with the appropriate GitHub username.

2. Install dependencies

python -m pip install numpy pandas matplotlib

3. Generate the datasets

python src/generate_data.py
python src/generate_dataset2.py
python src/generate_dataset3.py

4. Detect and remove outliers

python src/outlier_detection.py

Enter the dataset number (1, 2, or 3) when prompted. Repeat for each dataset.

5. Run K-means

python src/kmeans.py

Enter the dataset number and the desired number of clusters (k).

6. Run hierarchical clustering

python src/hierarchical.py

Enter the dataset number and k. The program executes all 4 hierarchical linkage methods.

7. Evaluate clustering quality

python src/silhouette.py

Enter the dataset number to calculate and compare Silhouette coefficients.

8. Generate comparison charts

python src/compare_results.py

This generates the final comparison charts and experiment summary using the recorded experiment results.

Conclusion

This project demonstrates the implementation and evaluation of unsupervised machine learning algorithms without relying on prebuilt machine learning libraries.

The experiments show that K-means and hierarchical clustering can produce similar results on well-separated datasets, while overlapping cluster structures can cause significant differences in performance.

The project also highlights the importance of outlier detection, appropriate linkage selection, and quantitative evaluation when analyzing unlabeled data.

Author

Developed by Tal Barr
New Jersey Institute of Technology (NJIT)
CS 634 — Data Mining
