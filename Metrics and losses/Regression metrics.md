---
aliases: 
tags:
  - evaluation
  - metric
  - regression
---
Regression metrics are quantitative measures used to evaluate the performance of regression models, which predict continuous values rather than discrete classes.

## Error-Based Metrics

1. **Mean Absolute Error (MAE)**: average of the absolute differences between predicted and actual values. It is easy to understand as it is expressed in the same units as the target variable.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
2. **Mean Squared Error (MSE)**: the average of the squared differences between predicted and actual values. More sensitive to outliers. Differentiable, hence often used as a [[Losses|loss function]].

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3. **Root Mean Squared Error (RMSE)**: the square root of MSE, returning the error metric to the original scale of the target variable. More sensitive to outliers than MAE, but is in the same unit as the target variable.

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

4. **Mean Absolute Percentage Error (MAPE)**: the average of the absolute percentage errors. A MAPE of 10% means that, on average, the prediction is 10% off from the true value. Biased towards under-prediction. Better used for positive targets and not close to zero.

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
## Goodness-of-Fit Metrics

5. **Coefficient of Determination (R²)**: represents the proportion of variance in the dependent variable that is predictable from the independent variables. 1 means perfect prediction, 0 means the model's performance is equivalent to predicting mean. Can be negative, if the model is worse than a horizontal line. An R² of 0.75 means that 75% of the variance in the target variable can be explained by the model's features. R² tends to increase (or stay the same) as more features are added to the model, even if those features are not actually useful. A model with a higher R² will have a lower MSE, and vice versa.
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{SSE}}{\text{SST}}$$

6. **Adjusted R²**: modifies the R² metric to account for the number of predictors in the model. Penalizes the addition of predictors that don't improve the model. Adjusted R² is always less than or equal to R².

$$\text{Adjusted } R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$
## Correlation, Covariance, and R²
Based on my research, I'll explain the relationship between correlation, covariance, and the coefficient of determination (R²).

# Relationship Between Correlation, Covariance, and R²

**Covariance** measures how two variables change together (the joint variability of two variables).
$$\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})$$

Covariance can range from -∞ to +∞, with:
- Positive values indicating the variables tend to move in the same direction
- Negative values indicating the variables tend to move in opposite directions
- Values near zero indicating little linear relationship

The Pearson **correlation** coefficient standardizes the covariance by dividing it by the product of the standard deviations of both variables. It measures the strength and direction of the linear relationship between two continuous variables.

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2 \sum_{i=1}^{n} (Y_i - \bar{Y})^2}}$$

Where:

- $\sigma_X$ and $\sigma_Y$ are the standard deviations of X and Y

This standardization constrains the correlation coefficient to range from -1 to +1:

- r = +1 indicates a perfect positive linear relationship
- r = -1 indicates a perfect negative linear relationship
- r = 0 indicates no linear relationship


In simple linear regression (with one independent variable), R² equals the square of the correlation coefficient:
$$R^2 = r^2$$

In multiple regression (with multiple independent variables), R² is a generalization of this concept:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}$$

In this case, R² equals the square of the multiple correlation coefficient, which is the correlation between the observed values and the predicted values.

## Links
- [Scikit-learn Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Paper: R-squared is More Informative than SMAPE, MAE, MAPE, MSE and RMSE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8279135/)
- [The Little Book of ML Metrics](https://github.com/NannyML/The-Little-Book-of-ML-Metrics)
