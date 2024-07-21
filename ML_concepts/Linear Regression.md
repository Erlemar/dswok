---
tags:
  - algorithm
  - model
  - supervised
---

Linear regression is a supervised algorithm or a statistical method that learns to model a dependent variable (target) as a function of some independent variables (features) by finding a line (or surface) that best "fits" the data. For linear regression, we assume the target to be continuous (a number), for [[Logistic regression]] we assume target to be discrete (has a finite number of classes).

### Simple Linear Regression
$y = β₀ + β₁x + ε$

### Multiple Linear Regression
$y = Xβ + ε$

Where:
- $y$ is the $n×1$ vector of dependent variable observations
- $X$ is the $n×(p+1)$ matrix of independent variables (including a column of 1s for the intercept)
- $β$ is the $(p+1)×1$ vector of coefficients
- $ε$ is the $n×1$ vector of error terms

## Learning the coefficients
### Ordinary Least Squares (OLS) / Analytics solution

OLS is the most common method for estimating the parameters of a linear regression model.
Objective: Minimize the sum of squared residuals: $min Σ(yᵢ - (β₀ + β₁x₁ᵢ + ... + βₚxₚᵢ))²$
$β̂ = (X'X)^{⁻¹}X'y$

Where:
- $β̂$ is the vector of estimated coefficients
- $X'$ is the transpose of $X$
- $(X'X)^{⁻¹}$ is the inverse of $X'X$

Properties of OLS estimators under the classical assumptions:
1. Best Linear Unbiased Estimator (BLUE)
2. Consistent
3. Asymptotically normal

### Estimation of σ²
$σ̂² = \frac{ESS}{n}$
**Coefficient validity**
In statistics, the coefficients are usually paired with their p-values. These p-values come from null hypothesis statistical tests: t-tests are used to measure whether a given coefficient is significantly different than zero (the null hypothesis that a particular coefficient $βi$​ equals zero), while F tests are used to measure whether _any_ of the terms in a regression model are significantly different from zero.

### Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is an alternative method to OLS for estimating the parameters of a linear regression model. It's particularly useful when dealing with non-normal error distributions. MLE finds the parameter values that maximize the likelihood of observing the given data.

**Likelihood Function for Linear Regression**
Assuming normally distributed errors:
$L(β, σ² | y, X) = ∏ᵢ (\frac{1}{√(2πσ²)}) * exp(\frac{-(yᵢ - xᵢ'β)²}{(2σ²)})$

Log-likelihood:
$ln L(β, σ² | y, X) = -n/2 * ln(2π) - n/2 * ln(σ²) - 1/(2σ²) * Σᵢ (yᵢ - xᵢ'β)²$

$ln L(β, σ² | y, X) = -\frac{n}{2} * ln(2π) - \frac{n}{2} * ln(σ²) - \frac{1}{(2σ²)} * Σᵢ (yᵢ - xᵢ'β)²$

**Comparison with OLS**
- Under normality assumption, MLE and OLS produce the same estimates for β
- MLE can be extended to non-normal error distributions
- MLE provides a framework for hypothesis testing and model selection (e.g., likelihood ratio tests)
### Gradient Descent for Linear Regression
When dealing with large datasets, the analytical solution may be computationally expensive. Gradient descent is an iterative optimization algorithm used to find the minimum of the cost function.

## Assumptions
- Linearity: The relationship between $X$ and $Y$ is linear.
- Homoscedasticity: Constant variance of residuals $(Var(ε|X) = σ²)$. This means that error distribution is consistent or all values of the features, there should be no discernible patterns.
- No Multicollinearity: Independent variables shouldn't be highly correlated with each other. This can be checked using correlation matrices or Variance Inflation Factor (VIF).
- Normality: Residuals are normally distributed $(ε ~ N(0, σ²))$. This can be checked Q-Q plots of the residuals or by histograms, or through statistical tests such as the Kolmogorov-Smirnov test. Relevant for MLE only
- Independence: Observations are independent of each other
- No Exogeneity: $E(ε|X) = 0$, meaning the errors are uncorrelated with the predictors


Consequences of violating assumptions:
- Violating linearity: Biased and inconsistent estimates
- Violating independence: Incorrect standard errors, inefficient estimates
- Violating homoscedasticity: Inefficient estimates, incorrect standard errors
- Violating normality: Hypothesis tests may be invalid for small samples
- Perfect multicollinearity: Unable to estimate unique coefficients

## Model Evaluation Metrics
1. R-squared (Coefficient of Determination): $R² = 1 - \frac{SSR}{SST}$ Where SSR is the sum of squared residuals and SST is the total sum of squares
2. Adjusted R-squared: $Adj R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]$
3. Mean Squared Error (MSE): $MSE = Σ\frac{(yᵢ - ŷᵢ)²}{n}$
4. Root Mean Squared Error (RMSE): $RMSE = √MSE$
5. Mean Absolute Error (MAE): $MAE = Σ\frac{|yᵢ - ŷᵢ|}{n}$

## Hypothesis Testing
1. t-test for individual coefficients:
   $H₀: βᵢ = 0$
   $t = \frac{β̂ᵢ}{SE(β̂ᵢ)}$
2. F-test for overall model significance:
   $F = (SSR / p) / (SSE / (n - p - 1))$
   $F = \frac{\frac{SSR}{p}}{\frac{SSE}{(n - p - 1)}}$

## Confidence Intervals
CI for $βᵢ$: $β̂ᵢ ± t(α/2, n-p-1) * SE(β̂ᵢ)$

## Extensions and [[Regularization]]
1. Ridge Regression (L2): $β̂ridge = argmin(||y - Xβ||² + λ||β||²)$
2. Lasso Regression (L1): $β̂lasso = argmin(||y - Xβ||² + λ||β||₁)$
3. Elastic Net: Combination of L1 and L2 penalties

## Modeling complex relationships
### Polynomial Regression
Extends linear regression to model non-linear relationships:
$y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε$

### Interaction Terms
Allows for modeling the combined effect of two or more variables:
$y = β₀ + β₁x₁ + β₂x₂ + β₃(x₁*x₂) + ε$

> [!example]- Code example
> ```python
> import numpy as np
> 
> class LinearRegression:
>     def __init__(self, learning_rate: float=0.01, n_iterations: int=1000):
>         self.learning_rate = learning_rate
>         self.n_iterations = n_iterations
>         self.weights = None
>         self.bias = None
> 
>     def fit(self, X, y):
>         n_samples, n_features = X.shape
>         self.weights = np.zeros((1, n_features))
>         self.bias = 0
> 
>         for _ in range(self.n_iterations):
>             prediction = np.dot(X, self.weights) + self.bias
> 
>             dw = (1 / n_samples) * np.dot(X.T, (prediction - y))
>             db = (1 / n_samples) * np.sum(prediction - y)
> 
>             self.weights -= self.learning_rate * dw
>             self.bias -= self.learning_rate * db
> 
>     def predict(self, X):
>         return np.dot(X, self.weights) + self.bias
> 
>     def mse(self, X, y):
>         y_predicted = self.predict(X)
>         return np.mean((y - y_predicted) ** 2)
> ```

## Links
* [Explained.ai: Linear Regression](https://mlu-explain.github.io/linear-regression/)
* [MLcourse.ai lesson](https://mlcourse.ai/book/topic04/topic04_intro.html)
