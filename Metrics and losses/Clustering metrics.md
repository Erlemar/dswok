---
tags:
  - evaluation
  - metric
  - clustering
  - unsupervised
---
Clustering metrics are quantitative measures used to evaluate the performance and quality of clustering algorithms.

### Internal Metrics/Intrinsic (No Ground Truth Required)

These metrics evaluate clustering performance based only on the data and the clustering result, without requiring true labels:

1. **Silhouette Coefficient**: Measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1: +1 - good separation and cohesion, 0 - sample is close to the decision boundary between clusters, -1 - bad separation (too close to the points in a neighboring cluster).

$$\text{Silhouette}(i) = \frac{b(i) - a(i)}{\max{a(i), b(i)}}$$

Where:

- $a(i)$ is the mean distance between point $i$ and all other points in the same cluster
- $b(i)$ is the mean distance between point $i$ and all points in the nearest neighboring cluster

The Silhouette score for a clustering is the mean Silhouette coefficient over all samples:

$$\text{Silhouette Score} = \frac{1}{n} \sum_{i=1}^{n} \text{Silhouette}(i)$$

2. **Davies-Bouldin Index**: Measures the average similarity between clusters, where similarity is defined as the ratio between within-cluster distances and between-cluster distances. Lower values indicate better clustering (0 is minimum).

$$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

Where:

- $k$ is the number of clusters
- $\sigma_i$ is the average distance of all points in cluster $i$ to their cluster centroid
- $d(c_i, c_j)$ is the distance between centroids of clusters $i$ and $j$

3. **Calinski-Harabasz Index** (Variance Ratio Criterion): Ratio of between-cluster variance to within-cluster variance. Higher values indicate better-defined clusters.

$$\text{CH} = \frac{\text{Tr}(B_k) / (k-1)}{\text{Tr}(W_k) / (n-k)}$$

Where:

- $B_k$ is the between-cluster dispersion matrix
- $W_k$ is the within-cluster dispersion matrix
- $\text{Tr}$ is the trace of a matrix
- $n$ is the total number of samples
- $k$ is the number of clusters

4. **Dunn Index**: Ratio of the minimum inter-cluster distance to the maximum intra-cluster distance. Higher values indicate better clustering.

$$\text{Dunn} = \frac{\min_{i \neq j} d(c_i, c_j)}{\max_{1 \leq k \leq K} \text{diam}(c_k)}$$

Where:

- $d(c_i, c_j)$ is the distance between clusters $i$ and $j$
- $\text{diam}(c_k)$ is the diameter of cluster $k$ (maximum distance between any two points in the cluster)

5. **Inertia** (Within-cluster Sum of Squares): Sum of squared distances of samples to their closest cluster center. Lower values indicate more compact clusters.

$$\text{Inertia} = \sum_{i=1}^{n} \min_{c_j \in C} ||x_i - c_j||^2$$

Where:

- $x_i$ is the data point
- $c_j$ is the center of cluster $j$
- $C$ is the set of all cluster centers

### External Metrics (Ground Truth Required)

These metrics compare clustering results against known true labels:

1. **Adjusted Rand Index (ARI)**: Measures the similarity between two clusterings, adjusted for chance. Values range from -1 to 1, with 1 indicating perfect agreement, 0 indicating random assignment, and negative values indicating assignments worse than random.

$$\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}{\frac{1}{2} \left[ \sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2} \right] - \left[ \sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2} \right] / \binom{n}{2}}$$

Where:

- $n_{ij}$ is the number of objects that are in both class $i$ and cluster $j$
- $a_i$ is the number of objects in class $i$
- $b_j$ is the number of objects in cluster $j$
- $n$ is the total number of objects

2. **Normalized Mutual Information (NMI)**: Measures the mutual information between the clustering assignment and the ground truth, normalized by the average entropy of both. Values range from 0 to 1, with 1 indicating perfect agreement.

$$\text{NMI}(U, V) = \frac{2 \times I(U, V)}{H(U) + H(V)}$$

Where:

- $I(U, V)$ is the mutual information between clusterings $U$ and $V$
- $H(U)$ and $H(V)$ are the entropies of $U$ and $V$ respectively

3. **Fowlkes-Mallows Score**: Geometric mean of precision and recall. Values range from 0 to 1, with 1 indicating perfect agreement.

$$\text{FMI} = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}$$

Where TP, FP, and FN are derived from counting pairs of points that are:

- True Positive (TP): in the same cluster in both clusterings
- False Positive (FP): in the same cluster in the predicted clustering but not in the ground truth
- False Negative (FN): in different clusters in the predicted clustering but in the same cluster in the ground truth

4. **Homogeneity, Completeness, and V-measure**:
    - **Homogeneity**: Each cluster contains only members of a single class (values from 0 to 1)
    - **Completeness**: All members of a given class are assigned to the same cluster (values from 0 to 1)
    - **V-measure**: Harmonic mean of homogeneity and completeness (values from 0 to 1)

$$\text{V-measure} = 2 \times \frac{\text{homogeneity} \times \text{completeness}}{\text{homogeneity} + \text{completeness}}$$

5. **Contingency Matrix**: A table showing the distribution of data points across predicted clusters and true classes. Not a metric itself, but the foundation for many external metrics.

### Determining Optimal Number of Clusters

- **Elbow Method**: Plot inertia against number of clusters and look for the "elbow" point
- **Silhouette Method**: Choose the number of clusters that maximizes the Silhouette score
- **Gap Statistic**: Compare within-cluster dispersion to that expected under a null reference distribution

## Links

- [Scikit-learn documentation on clustering metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [Wikipedia: Cluster Analysis](https://en.wikipedia.org/wiki/Cluster_analysis)
