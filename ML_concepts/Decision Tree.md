---
tags:
  - algorithm
  - model
  - supervised
---
A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

## Structure 
* **Root Node**: The topmost node representing the entire dataset.
* **Internal Nodes**: Nodes that  represent the features of the dataset and are used to make decisions.
* **Branches**: Connections between nodes, representing decision rules.
* **Leaf Nodes**: Terminal nodes that provide the final output (class label (mode of the training instances) or value (mean of the training instances)).

## Training Process
1. Feature Selection: Choose the best attribute to split the data.
2. Split Point Decision: Determine the best split point for the chosen feature.
3. Splitting: Divide the dataset based on the chosen feature and split point: left child note contains data points where the feature value is less then the split point, right child node - with values more than the split point.
4. Recursion: Repeat steps 1-3 for each child node until stopping criteria are met.

### Splitting Criteria
* Classification Trees:
	* Gini Impurity: $G = 1 - \sum\limits_k (p_k)^2$
	* Entropy: $S = -\sum_{i=1}^{N}p_i \log_2{p_i}$
* Regression Trees:
	* Variance (equivalent to MSE): $D = \frac{1}{\ell} \sum\limits_{i =1}^{\ell} (y_i - \frac{1}{\ell} \sum\limits_{j=1}^{\ell} y_j)^2$

### Stopping Criteria
* Maximum depth reached
* Minimum number of samples in a node
* Minimum decrease in impurity
* All samples in a node belong to the same class

## Ensemble Methods Using Decision Trees
* [[Random Forest]]
* [[Gradient boosting]]

## Advantages
* Easy to understand and interpret
* Requires little data preprocessing
* Can handle both numerical and categorical data - Can be visualized easily ## Disadvantages - Can create overly complex trees that do not generalize well (overfitting) - Can be unstable; small variations in the data can result in a completely different tree - Biased towards features with more levels (in categorical variables) - Cannot predict beyond the range of the training data (for regression tasks)

**Disadvantages**
- Can create overly complex trees that do not generalize well (overfitting)
- Can be unstable; small variations in the data can result in a completely different tree
- Cannot predict beyond the range of the training data (for regression tasks)

## [[Regularization]]
* Pruning: Removing branches that provide little predictive power
* Setting minimum number of samples required at a leaf node
* Setting maximum depth of the tree
* Setting maximum number of features to consider for splitting

## Feature Importance
Decision Trees provide built-in feature importance: Importance is calculated based on how much each feature decreases the weighted impurity.

## Example of entropy calculation:
20 balls: 9 blue, 11 yellow.
$S_0 = -\frac{9}{20}\log_2{\frac{9}{20}}-\frac{11}{20}\log_2{\frac{11}{20}} \approx 1$
Now 13 balls: 8 blue and 5 yellow: $S_1 = -\frac{5}{13}\log_2{\frac{5}{13}}-\frac{8}{13}\log_2{\frac{8}{13}} \approx 0.96$

## Links
* [Explained.ai: Decision Tree](https://mlu-explain.github.io/decision-tree/)
