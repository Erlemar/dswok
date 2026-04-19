---
aliases: 
tags:
  - tabular-ml
  - evaluation
cssclasses:
  - term-table
---
Regression metrics are quantitative measures used to evaluate the performance of regression models, which predict continuous values rather than discrete classes.

## When to use which metric

| Metric | When to use |
|---|---|
| MAE | Interpretable error in target units; less sensitive to outliers. |
| MSE | Penalizes large errors heavily; differentiable (often used as loss). |
| RMSE | Same scale as MSE penalty but in target units; comparable to MAE. |
| MAPE | Percentage error. Positive targets, not close to zero. |
| R² | Proportion of variance explained (1 = perfect, 0 = predicting the mean, <0 = worse). |
| Adjusted R² | R² with a penalty for adding uninformative predictors. |
| Pearson r | Strength and direction of a linear relationship (-1 to +1). |

## Error-Based Metrics

### Mean Absolute Error (MAE)

Average of the absolute differences between predicted and actual values. Easy to understand — expressed in the same units as the target variable.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Mean Squared Error (MSE)

Average of the squared differences between predicted and actual values. More sensitive to outliers. Differentiable, hence often used as a [[General losses|loss function]].

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Root Mean Squared Error (RMSE)

Square root of MSE, returning the error metric to the original scale of the target variable. More sensitive to outliers than MAE, but is in the same unit as the target.

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Mean Absolute Percentage Error (MAPE)

Average of the absolute percentage errors. A MAPE of 10% means that, on average, the prediction is 10% off from the true value. Biased toward under-prediction. Better used for positive targets not close to zero.

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

## Goodness-of-Fit Metrics

### Coefficient of Determination (R²)

Proportion of variance in the dependent variable that is predictable from the independent variables. 1 means perfect prediction, 0 means the model's performance is equivalent to predicting the mean. Can be negative if the model is worse than a horizontal line. An R² of 0.75 means that 75% of the variance in the target variable can be explained by the model's features. R² tends to increase (or stay the same) as more features are added, even if those features aren't useful. A model with higher R² will have lower MSE, and vice versa.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{SSE}}{\text{SST}}$$

### Adjusted R²

Modifies R² to account for the number of predictors in the model. Penalizes the addition of predictors that don't improve the model. Adjusted R² is always less than or equal to R².

$$\text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

## Correlation, Covariance, and R²

**Covariance** measures how two variables change together (the joint variability of two variables).

$$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})$$

Covariance can range from −∞ to +∞:
- Positive values indicate the variables tend to move in the same direction.
- Negative values indicate they tend to move in opposite directions.
- Values near zero indicate little linear relationship.

The Pearson **correlation** coefficient standardizes the covariance by dividing it by the product of the standard deviations of both variables. It measures the strength and direction of the linear relationship between two continuous variables.

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2 \sum_{i=1}^{n} (Y_i - \bar{Y})^2}}$$

Where $\sigma_X$ and $\sigma_Y$ are the standard deviations of $X$ and $Y$. This standardization constrains the correlation coefficient to range from −1 to +1:

- $r = +1$ indicates a perfect positive linear relationship.
- $r = -1$ indicates a perfect negative linear relationship.
- $r = 0$ indicates no linear relationship.

In simple linear regression (one independent variable), R² equals the square of the correlation coefficient:

$$R^2 = r^2$$

In multiple regression, R² generalizes this concept:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}$$

Here R² equals the square of the multiple correlation coefficient — the correlation between observed and predicted values.

## Links
- [Scikit-learn Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Paper: R-squared is More Informative than SMAPE, MAE, MAPE, MSE and RMSE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8279135/)
- [The Little Book of ML Metrics](https://github.com/NannyML/The-Little-Book-of-ML-Metrics)
