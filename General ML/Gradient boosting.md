---
tags:
  - algorithm
  - ensemble
  - model
  - supervised
aliases:
  - GBM
  - Gradient Boosting Machine
---
Gradient Boosting is an ensemble machine learning technique (boosting) that combines weak learners (typically shallow [[Decision Tree]]s) sequentially to create a strong predictive model. Each new tree is trained using gradient descent to correct errors made by previously trained trees.

## How It Works
1. Initialize the model with a constant value
2. For each iteration for a given loss function $L(y, F(x))$ and current model $F_m(x)$:
   - Calculate the negative gradient of the loss function   $g_i = -[∂L(y_i, F(x_i)) / ∂F(x_i)]$
   - Fit a regression tree $h_m(x)$ to $g_i$
   - Perform line search to find optimal step size $\large \rho_t = \underset{\rho}{\arg\min} \ \sum_{i = 1}^{n} L(y_i, \hat{f}(x_i) +  \rho \cdot h(x_i, \theta))$
   - Optionally, instead of the previous step, we could compute the optimal value for each leaf j = 1 to J: $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$ 
   - Update the model with learning rate η (0 < η ≤ 1) $F_m(x) = \hat{f}(x) + η \cdot ρ_m h_m(x)$
3. Repeat until convergence or maximum iterations are reached
4. The final model is the sum of all trees: $F_M(x) = F_0(x) + \sum_{m=1}^M η \cdot \rho_m h_m(x)$

## Advantages
1. Can optimize various loss functions (e.g., ranking, Poisson regression)
2. Handles mixed data types well

## Disadvantages
1. More prone to overfitting, especially on noisy data
2. Requires careful tuning of hyperparameters; training can take longer

### Loss Functions
- Regression: L2 (MSE), L1 (MAE), Huber loss
- Binary Classification: Log loss, Exponential loss
- Multiclass Classification: Multi-class log loss
- Ranking: LambdaMART, LambdaRank

### [[Regularization]]
1. Shrinkage (Learning Rate): Scales the contribution of each tree
2. Subsampling: Use only a fraction of the training data for each tree
3. Early Stopping: Stop training when validation error stops improving
4. L1/L2 [[Regularization]] on leaf weights

### Feature Importance
- Split-based Importance: Measures how often a feature is used to split the data across all trees; weighted by the improvement in the model as a result of each split.
- Gain-based Importance: Measures the average gain of splits that use the feature.

### Data processing
- Learns the best direction (left or right child) for missing values in each split
- Categorical features: one-hot encoding is inefficient; label encoding or target encoding is better. CatBoost has in-built specialized handling

## Classification vs Regression
- Regression: Typically uses a mean squared error or mean absolute error as a loss function
- Classification: Often uses log loss (binary) or multi-class log loss

## Popular Implementations

### XGBoost
- Sparse-aware split finding
- Weighted quantile sketch for approximate tree learning: doesn't try all possible splits, but bins feature values and finds the best split points among these bins
- Level-wise tree growth strategy(BFS): the tree is built one level at a time, across all leaves at the current depth before moving to the next level
- Later, leaf-wise tree grows was added (DFS)

### LightGBM
- Gradient-based One-Side Sampling (GOSS): keeps all instances with large gradients (which are more important for training) and randomly samples a subset of instances with small gradients
- Exclusive Feature Bundling (EFB): bundles mutually exclusive features (features that rarely take non-zero values simultaneously) into a single feature.
- Leaf-wise tree growth (DFS): the tree grows by choosing the leaf with the maximum delta loss to split.
- Histogram-based algorithm: Instead of using exact values for continuous features, buckets continuous feature values into discrete bins.

### CatBoost
- Symmetric trees: usually, trees are built sequentially, which can lead to a shift in predictions as the model evolves. Symmetric trees are built in a way that makes them immune to the order of data samples - each split is on the same attribute, with the same threshold
- Ordered boosting: Instead of using the current model to calculate gradients for all observations, it uses a separate model for each example - these models are trained on a subset of the data that comes before the current example in a random permutation.
- Native handling of categorical features - a combination of one-hot encoding and an advanced mean encoding technique

### Gradient Boosting vs. AdaBoost
- Gradient Boosting fits new models to the residuals of the previous models, AdaBoost adjusts the weights of misclassified instances.
- Gradient Boosting is more flexible in terms of loss function choice, AdaBoost was initially designed for binary classification but can be adapted for other tasks.
- AdaBoost uses an exponential loss function

### Other types of boosting
* Newton Boosting uses both first and second-order derivatives of the loss function
- Stochastic Gradient Boosting introduces randomness by using a random subset of the training data to build each tree
- Extreme Gradient Boosting (like XGBoost) uses both approaches, uses a more regularized model formalization to control overfitting

## DART (Dropouts meet Multiple Additive Regression Trees)
DART applies the concept of dropout, originally from neural networks, to gradient boosting decision trees.
During each iteration, DART randomly drops a fraction of the trees that have already been built, and the new tree is built to correct the errors of this "thinned" ensemble. For prediction, all trees are used. DART improves regularization and reduces overfitting.

## Links
- [Explained.ai: Gradient Boosting](https://explained.ai/gradient-boosting/)
- [MLcourse.ai lesson](https://mlcourse.ai/book/topic10/topic10_gradient_boosting.html)

