---
tags:
  - unsupervised
---
Dimensionality reduction is the process of reducing the number of features (dimensions) in a dataset while preserving as much relevant information as possible. It transforms high-dimensional data into a lower-dimensional representation that captures the essential structure and patterns in the original data.

### What is Dimensionality Reduction Used For?

* Curse of Dimensionality Mitigation
* Visualization of high-dimensional data in 2D or 3D space
* Noise Reduction
* Storage and Computational Efficiency
* Feature engineering

### Main Approaches to Dimensionality Reduction

#### Linear Methods

* Principal Component Analysis (PCA): finds the directions (principal components) along which the data varies the most. It projects the data onto these directions to create a lower-dimensional representation.
* Linear Discriminant Analysis (LDA): finds the linear combinations of features that best separate different classes.
* Independent Component Analysis (ICA): separates multivariate signals into additive, independent components. It assumes that the observed data is a linear mixture of independent source signals.
* Factor Analysis: models observed variables as linear combinations of unobserved latent factors plus noise. It's similar to PCA but includes a noise model.

#### Non-Linear Methods

* t-Distributed Stochastic Neighbor Embedding (t-SNE): preserves local neighborhood structure when mapping to lower dimensions. It's particularly effective for visualization but can be computationally expensive.
* Uniform Manifold Approximation and Projection (UMAP): preserves both local and global structure better than t-SNE and is generally faster. It's based on manifold learning and topological data analysis.
* Autoencoders: neural networks that learn to compress and reconstruct data. The compressed representation in the middle layer serves as the reduced-dimensional representation.
* Kernel PCA: extends PCA to non-linear dimensionality reduction using the kernel trick. It implicitly maps data to a higher-dimensional space where linear PCA is applied.

### Covariance

Covariance measures the joint variability of two random variables. It indicates the direction of the linear relationship between them.
For a dataset with features $X_1, X_2, ..., X_p$, the covariance between features $X_i$ and $X_j$ is:

$$\text{Cov}(X_i, X_j) = \frac{1}{n-1} \sum_{k=1}^{n} (x_{ki} - \bar{x}_i)(x_{kj} - \bar{x}_j)$$

Where:

- $n$ is the number of samples
- $x_{ki}$ is the k-th observation of feature $i$
- $\bar{x}_i$ is the mean of feature $i$

For a dataset $X$ with $n$ samples and $p$ features, the covariance matrix $C$ is a $p \times p$ symmetric matrix:

$$C_{ij} = \text{Cov}(X_i, X_j)$$

In matrix form, if $X$ is the centered data matrix (mean-subtracted): $$C = \frac{1}{n-1} X^T X$$
### PCA and Covariance

PCA is fundamentally linked to the covariance matrix of the data (assuming data is centered). The principal components are the eigenvectors of the covariance matrix, and the corresponding eigenvalues represent the variance captured by each principal component.

#### How PCA works

1. **Center the data**: Subtract the mean from each feature
2. **Compute covariance matrix**: $C = \frac{1}{n-1} X^T X$
3. **Find eigenvalues and eigenvectors**: $C \mathbf{v} = \lambda \mathbf{v}$
4. **Sort by eigenvalues**: Sort the eigenvectors in descending order based on their corresponding eigenvalues. The eigenvector with the largest eigenvalue is the first principal component. Basically, the first principal component finds the direction with the highest variation.
5. **Project data**: Transform data using selected eigenvectors

### Eigenvalues and Eigenvectors

Eigenvector is the direction that doesn't change when the linear transformation is applied. Eigenvalue is the scaling factor applied in the eigenvector direction

#### Singular Value Decomposition (SVD) and PCA
PCA can also be performed using SVD on the (centered) data matrix $X$ directly, without explicitly forming the covariance matrix. If $X = U S Vᵀ$ is the SVD of $X$:
*   The principal components are the columns of $V$.
*   The singular values in $S$ are related to the eigenvalues of the covariance matrix ($λ = s² / (n-1)`$).
Using SVD is often numerically more stable, especially for high-dimensional data.

### Evaluation Metrics

### 1. Explained Variance Ratio

Proportion of total variance explained by selected components: $$\text{EVR} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{p} \lambda_i}$$

### 2. Reconstruction Error

Mean squared error between original and reconstructed data: $$\text{MSE} = \frac{1}{n} ||\mathbf{X} - \mathbf{X}_{\text{reconstructed}}||_F^2$$
### Practical considerations
- Standardize features before applying PCA if they have different scales
- PCA works best when features are correlated
## Links

- [A Tutorial on Principal Component Analysis](https://arxiv.org/abs/1404.1100)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [The Elements of Statistical Learning - Chapter 14](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Pattern Recognition and Machine Learning - Chapter 12](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)