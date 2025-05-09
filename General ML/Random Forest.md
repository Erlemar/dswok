---
tags:
  - algorithm
  - ensemble
  - model
  - supervised
aliases:
  - RF
---
Random Forest is an ensemble learning method (bagging) that constructs a multitude of [[Decision Tree|decision trees]] at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
## How It Works
- Create multiple decision trees using bootstrap samples (with replacement) of the training data.
- For each split in each tree, consider only a random subset of features.
- For classification, use majority voting of trees; for regression, use the average prediction.
## Advantages
- Reduces overfitting compared to individual [[Decision Tree]]s.
- Handles high-dimensional data well due to the sampling
## Disadvantages
* Less interpretable than a single [[Decision Tree]].
* Computationally more intensive than a single [[Decision Tree]].
## Prerequisites for Good Performance
- Presence of actual signal in the features
- Low correlation between predictions (and errors) of individual trees. [[Decision Tree]] has high variance by definition, so using random sampling ensures low correlation between the individual trees.

### Links
* [Explained.ai: Random Forest](https://mlu-explain.github.io/random-forest/)
