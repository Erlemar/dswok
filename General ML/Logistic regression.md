---
tags:
  - algorithm
  - model
  - supervised
---
Logistic regression is a statistical method used for binary classification problems, modeling the probability of an instance belonging to a particular class. Common variations of Logistic regression: Binary (two possible outcomes), Multinomial (more than two unordered outcomes), Ordinal (more than two ordered outcomes).
Logistic regression is similar to [[Linear Regression]] but uses a logistic (sigmoid) function on the top of it.

$p(y=1|x) = \frac{1}{1 + e^{-z}}$
where $z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ$

Logit function: $log(\frac{p}{(1-p)}) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ$

The logit is the logarithm of the odds. Logits transform probabilities (from 0 to 1) to real numbers. The sigmoid function is the inverse of the logit function. It maps any real number to a probability between 0 and 1.

Odds represent the ratio of the probability of an event occurring to the probability of it not occurring.
If p = 0.75, odds = $\frac{0.75}{1-0.75}$ = 3 - This means the event is 3 times more likely to occur than not to occur.

## Assumptions
1. Independence of observations
2. Little or no multicollinearity among independent variables
3. Linearity in the logit for continuous variables

## Maximum Likelihood Estimation (MLE)
Logistic regression parameters are typically estimated using MLE.

### Likelihood Function
$L(β) = ∏ᵢ p(xᵢ)^yᵢ * (1-p(xᵢ))^{1-yᵢ}$

### Log-Likelihood
$ll(β) = Σᵢ [yᵢ log(p(xᵢ)) + (1-yᵢ) log(1-p(xᵢ))]$

## Gradient Descent for Logistic Regression

In practice, gradient descent is often used to find the optimal parameters.

Update rule: $β = β - α * ∇J(β)$
Where $α$ is the learning rate and $∇J(β)$ is the gradient of the cost function.

## Interpretation of Coefficients
- $βᵢ$: Log-odds increase for a one-unit increase in $xᵢ$, holding other variables constant
- $exp(βᵢ)$: Odds ratio for a one-unit increase in $xᵢ$. If $exp(β)$ = 1.2, a one-unit increase in $x$ multiplies the odds by 1.2 (20% increase).

## Advantages
- Simple and interpretable
- Provides probability outputs
- Less prone to overfitting compared to more complex models

## Limitations
- Assumes linearity in log-odds
- Cannot capture complex, non-linear relationships without feature engineering

> [!example]- Code example
> ```python
>import numpy as np
>
>class LogisticRegression:
>    def __init__(self, learning_rate: float=0.01, n_iterations: int=1000, threshold: int=0.5):
>        self.learning_rate = learning_rate
>        self.n_iterations = n_iterations
>        self.weights = None
>        self.bias = None
>        self.threshold = threshold
>
>    def sigmoid(self, z):
>        return 1 / (1 + np.exp(-z))
>
>    def fit(self, X, y):
>        n_samples, n_features = X.shape
>        self.weights = np.zeros(n_features)
>        self.bias = 0
>
>        for _ in range(self.n_iterations):
>            prediction = np.dot(X, self.weights) + self.bias
>            y_pred = self.sigmoid(prediction)
>
>            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
>            db = (1 / n_samples) * np.sum(y_pred - y)
>
>            self.weights -= self.learning_rate * dw
>            self.bias -= self.learning_rate * db
>
>    def predict(self, X):
>        prediction = np.dot(X, self.weights) + self.bias
>        y_predicted = self.sigmoid(prediction)
>        return [1 if i > self.threshold else 0 for i in y_predicted]
> ```

## Links
* [Explained.ai: Logistic Regression](https://mlu-explain.github.io/logistic-regression/)
