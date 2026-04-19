---
tags:
  - evaluation
  - clustering
  - unsupervised
cssclasses:
  - term-table
---
Clustering metrics are quantitative measures used to evaluate the performance and quality of clustering algorithms.

## When to use which metric

| Metric | When to use |
|---|---|
| Silhouette | No ground truth; how well each point fits its cluster vs the next nearest. |
| Davies-Bouldin | No ground truth; lower = better separation. |
| Calinski-Harabasz | No ground truth; higher = better-defined clusters. |
| Dunn | No ground truth; tight clusters that are far apart. |
| Inertia | Compactness. Used in the elbow method for choosing $k$. |
| Adjusted Rand Index (ARI) | Ground truth available; similarity vs a random baseline. |
| Normalized Mutual Information (NMI) | Ground truth available; mutual information normalized by entropy. |
| Fowlkes-Mallows | Ground truth available; geometric mean of precision and recall. |
| Homogeneity / Completeness / V-measure | Ground truth available; class purity, cluster purity, and their harmonic mean. |

## Internal Metrics — Intrinsic (no ground truth)

These metrics evaluate clustering performance based only on the data and the clustering result, without requiring true labels.

### Silhouette Coefficient

Measures how similar an object is to its own cluster compared to other clusters. Values range from −1 to 1: +1 = good separation and cohesion, 0 = sample is close to the decision boundary between clusters, −1 = bad separation (too close to a neighboring cluster).

$$\text{Silhouette}(i) = \frac{b(i) - a(i)}{\max{a(i), b(i)}}$$

Where:
- $a(i)$ is the mean distance between point $i$ and all other points in the same cluster.
- $b(i)$ is the mean distance between point $i$ and all points in the nearest neighboring cluster.

The Silhouette score for a clustering is the mean Silhouette coefficient over all samples:

$$\text{Silhouette Score} = \frac{1}{n} \sum_{i=1}^{n} \text{Silhouette}(i)$$

### Davies-Bouldin Index

Measures the average similarity between clusters, where similarity is the ratio of within-cluster distances to between-cluster distances. Lower values indicate better clustering (0 is the minimum).

$$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Where:
- $k$ is the number of clusters.
- $\sigma_i$ is the average distance of points in cluster $i$ to its centroid.
- $d(c_i, c_j)$ is the distance between centroids of clusters $i$ and $j$.

### Calinski-Harabasz Index (Variance Ratio Criterion)

Ratio of between-cluster variance to within-cluster variance. Higher values indicate better-defined clusters.

$$\text{CH} = \frac{\text{Tr}(B_k) / (k-1)}{\text{Tr}(W_k) / (n-k)}$$

Where:
- $B_k$ is the between-cluster dispersion matrix.
- $W_k$ is the within-cluster dispersion matrix.
- $\text{Tr}$ is the trace of a matrix.
- $n$ is the total number of samples.
- $k$ is the number of clusters.

### Dunn Index

Ratio of the minimum inter-cluster distance to the maximum intra-cluster distance. Higher values indicate better clustering.

$$\text{Dunn} = \frac{\min_{i \neq j} d(c_i, c_j)}{\max_{1 \leq k \leq K} \text{diam}(c_k)}$$

Where:
- $d(c_i, c_j)$ is the distance between clusters $i$ and $j$.
- $\text{diam}(c_k)$ is the diameter of cluster $k$ (maximum distance between any two points in the cluster).

### Inertia (Within-cluster Sum of Squares)

Sum of squared distances of samples to their closest cluster center. Lower values indicate more compact clusters.

$$\text{Inertia} = \sum_{i=1}^{n} \min_{c_j \in C} ||x_i - c_j||^2$$

Where:
- $x_i$ is the data point.
- $c_j$ is the center of cluster $j$.
- $C$ is the set of all cluster centers.

## External Metrics — ground truth required

These metrics compare clustering results against known true labels.

### Adjusted Rand Index (ARI)

Measures the similarity between two clusterings, adjusted for chance. Values range from −1 to 1; 1 = perfect agreement, 0 = random assignment, negative = worse than random.

$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{\frac{1}{2} \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}$$

Where:
- $n_{ij}$ is the number of objects in both class $i$ and cluster $j$.
- $a_i$ is the number of objects in class $i$.
- $b_j$ is the number of objects in cluster $j$.
- $n$ is the total number of objects.

### Normalized Mutual Information (NMI)

Measures the mutual information between the clustering assignment and the ground truth, normalized by the average entropy of both. Values range from 0 to 1; 1 = perfect agreement.

$$\text{NMI}(U, V) = \frac{2 \times I(U, V)}{H(U) + H(V)}$$

Where:
- $I(U, V)$ is the mutual information between clusterings $U$ and $V$.
- $H(U)$ and $H(V)$ are the entropies of $U$ and $V$.

### Fowlkes-Mallows Score

Geometric mean of precision and recall. Values range from 0 to 1; 1 = perfect agreement.

$$\text{FMI} = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}$$

Where TP, FP, and FN are derived from counting pairs of points:
- **True Positive (TP)** — in the same cluster in both clusterings.
- **False Positive (FP)** — same cluster in the predicted clustering but not in the ground truth.
- **False Negative (FN)** — different clusters in the predicted clustering but in the same cluster in the ground truth.

### Homogeneity, Completeness, and V-measure

- **Homogeneity** — each cluster contains only members of a single class (values 0 to 1).
- **Completeness** — all members of a class are assigned to the same cluster (values 0 to 1).
- **V-measure** — harmonic mean of homogeneity and completeness (values 0 to 1).

$$\text{V-measure} = 2 \times \frac{\text{homogeneity} \times \text{completeness}}{\text{homogeneity} + \text{completeness}}$$

### Contingency Matrix

A table showing the distribution of data points across predicted clusters and true classes. Not a metric itself, but the foundation for many external metrics.

## Determining the Optimal Number of Clusters

- **Elbow Method** — plot inertia against the number of clusters and look for the "elbow" point.
- **Silhouette Method** — choose the number of clusters that maximizes the Silhouette score.
- **Gap Statistic** — compare within-cluster dispersion to that expected under a null reference distribution.

## Links

- [Scikit-learn documentation on clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [Wikipedia: Cluster Analysis](https://en.wikipedia.org/wiki/Cluster_analysis)
