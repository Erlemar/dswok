---
tags:
  - concept
  - evaluation
---
The bias-variance trade-off is a fundamental concept in machine learning that describes the balance between a model's ability to fit the training data (low bias) and its ability to generalize to new, unseen data (low variance).

### Bias
- Bias is the difference between the average prediction of the model and the correct value which we are trying to predict.
- A different explanation: imagine we train model on different datasets, bias measures how far off the predictions of these different models from the correct values.
- Model with a high bias is too simple and underfits the data, has strong model assumptions.
- Examples of high-bias models: [[Linear Regression]], [[Logistic regression]], linear discriminant analysis.

### Variance
- Variance is the variability of model prediction for a given data point or a value which tells us the spread of our data.
- A different explanation: imagine we train model on different datasets, variance measures how much the predictions for a certain sample differ between different models
- Model with a high variance is too complex and overfits the data, does not generalize well to unseen data.
- Examples of high-variance models: [[Decision tree]]s (when deep), [[K-Nearest Neighbors]] with small k, [[SVM|Support Vector Machine]]s.

### Trade-off
- As bias decreases, variance tends to increase, and vice versa. The goal is to find the "sweet spot" that minimizes both bias and variance.
- This trade-off in complexity is why there is a trade-off between bias and variance. An algorithm can't be more complex and less complex at the same time.
- Bagging ([[Random Forest]]) has higher bias and lower variance: averaging multiple [[Decision Tree]] aims to reduce variance at the cost of the small bias increase. Two independent trees will have the same bias (and their average will be the same), but models make different mistakes, so averaging them reduced the overall error - variance.
- Boosting ([[Gradient boosting]]) has lower bias and higher variance: models are complex, so they are prone to overfitting, but each model is optimized (starts from better points), so they are less likely to introduce bias

## Total Error Decomposition
$Total Error = Bias² + Variance + Irreducible Error$

Where:
- $Bias²$: How far off the model's predictions are from the correct values.
- $Variance$: How much the predictions for a given point vary between different realizations of the model.
- $Irreducible Error$: The noise in the true relationship that cannot be reduced by any model.

$Err(x) = E[(Y - \hat{f}(x))^2] = (E[\hat{f}(x)] - f(x))^2 + E[(\hat{f}(x) - E[\hat{f}(x)])^2] + \sigma^2_e$


https://mlu-explain.github.io/bias-variance/

![[Pasted image 20240707190658.png]]
## Underfitting vs. Overfitting
- Underfitting: Models with high bias and low variance. Unable to capture the underlying pattern of the data.
- Overfitting: Models with low bias and high variance. Captures the noise along with the underlying pattern in data.
![[Pasted image 20240707190740.png]]
![[Pasted image 20240707192228.png]]


## Strategies for Managing Bias-Variance Trade-off
1. Cross-Validation
2. [[Regularization]]
3. Ensemble methods:
   - Bagging: Reduces variance (e.g., [[Random Forest]])
   - Boosting: Reduces bias, may increase variance (e.g., [[Gradient boosting]])
4. Feature selection and engineering
5. Adjusting model complexity (e.g., tree depth, number of hidden layers)

## Links
- [Bias-Variance Tradeoff Explanation](https://scott.fortmann-roe.com/docs/BiasVariance.html)
- [MLU Explain: Bias-Variance](https://mlu-explain.github.io/bias-variance/)