---
aliases:
  - knn
tags:
  - classification
  - regression
  - supervised
---
K-Nearest Neighbors (KNN) is a simple, non-parametric algorithm used for both classification and regression tasks. It makes predictions for a new data point based on the K closest data points (neighbors).
Not to be confused with [[K-means clustering]]. KNN is a supervised algorithm for classification/regression finding nearest neighbors, K-means is an unsupervised algorithm for clustering finding cluster centroids.

## How It Works

1. Choose the number K of neighbors
2. Calculate the distance between the query instance and all training samples
3. Sort the distances and determine the K nearest neighbors
4. For classification: 
   - Aggregate the class labels of K neighbors
   - Return the majority vote as the prediction $\hat{y} = \text{mode}(y_i), i \in K \text{ nearest neighbors}$
5. For regression:
   - Aggregate the values of K neighbors
   - Return the mean value as the prediction $\hat{y} = \frac{1}{K} \sum_{i \in K \text{ nearest neighbors}} y_i$

As KNN algorithm does not build a model during the training phase, it is called a lazy learner. The algorithm memories the entire training dataset and performs an action on the dataset at the time of prediction.
![[Pasted image 20240720174904.png]]
### Distance Metrics

1. Euclidean Distance:
   $d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$

2. Manhattan Distance:
   $d(x, y) = \sum_{i=1}^n |x_i - y_i|$

3. Minkowski Distance:
   $d(x, y) = (\sum_{i=1}^n |x_i - y_i|^p)^{\frac{1}{p}}$

### Computational Approaches
1. Brute Force: Calculates distances to all points. Simple but inefficient for large datasets.
2. [K-D Tree](https://scikit-learn.org/stable/modules/neighbors.html#k-d-tree): Space-partitioning data structure for organizing points in K-dimensional space: "The basic idea is that if point 𝐴 is very distant from point 𝐵, and point 𝐵 is very close to point 𝐶, then we know that points 𝐴 and 𝐶 are very distant, without having to explicitly calculate their distance". Efficient for low-dimensional data.
3. [Ball Tree](https://scikit-learn.org/stable/modules/neighbors.html#ball-tree): Hierarchical data structure using hyper-spheres. Efficient for high-dimensional data.

## Advantages

1. Simple implement
2. No assumptions about data distribution (non-parametric)
5. Can capture complex decision boundaries

## Disadvantages

1. Computationally expensive for large datasets
2. Sensitive to the scale of features
3. Curse of dimensionality: performance degrades with high-dimensional data

## Choosing K

1. Odd K for binary classification to avoid ties
2. Square root of N (where N is the number of samples) as a rule of thumb
3. Use cross-validation to find optimal K
4. Elbow Method: calculate error for different values of K, make a plot, look for an "elbow" on it - where the error begins to level off.
![[Pasted image 20240720172840.png]]

> [!example]- Code example
> ```python
> import numpy as np
> from collections import Counter
> 
> class KNN:
>     def __init__(self, k=3):
>         self.k = k
> 
>     def fit(self, X, y):
>         self.X_train = X
>         self.y_train = y
> 
>     def euclidean_distance(self, x1, x2):
>         return np.sqrt(np.sum((x1 - x2) ** 2))
> 
>     def predict(self, X):
>         predictions = [self._predict(x) for x in X]
>         return np.array(predictions)
> 
>     def _predict(self, x):
>         distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
>         k_indices = np.argsort(distances)[:self.k]
>         k_nearest_labels = [self.y_train[i] for i in k_indices]
>         most_common = Counter(k_nearest_labels).most_common(1)
>         return most_common[0][0]

## Links
* [Sklearn documentation](https://scikit-learn.org/stable/modules/neighbors.html)
