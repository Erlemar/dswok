---
tags:
  - loss
  - concept
aliases:
  - loss function
  - loss functions
---
Loss functions (also called objective functions or cost functions) are mathematical measures of the error between predicted and actual values. They quantify how well a model is performing and provide the optimization signal for training.

For domain-specific losses, see [[NLP losses]] and [[Computer vision losses]]. For evaluation metrics, see [[Metrics and losses]].

## Cross-Entropy Loss

Cross-entropy loss (or log loss) measures the performance of a classification model whose output is a probability value between 0 and 1.

**Binary Cross-Entropy**

$$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

**Categorical Cross-Entropy** is used for multi-class problems, where each sample belongs to a single class.

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Where:

- $y$ is the ground truth label
- $\hat{y}$ is the predicted probability
- $N$ is the number of samples
- $C$ is the number of classes

- **Label Smoothing Cross-Entropy**: Helps prevent overconfidence by replacing one-hot encoded ground truth with a mixture of the original labels and a uniform distribution.

$$\tilde{y}_{i,c} = (1 - \alpha) \cdot y_{i,c} + \alpha \cdot \frac{1}{C}$$

## Mean Squared Error (MSE)

MSE measures the average of the squares of the errors between predicted and actual values.

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Variants:**

- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a measure in the same units as the target variable.
- **Mean Absolute Error (MAE)**: Uses absolute differences instead of squared differences, making it less sensitive to outliers.

$$L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

## Kullback-Leibler Divergence (KL Divergence)

KL divergence measures how one probability distribution diverges from a second, expected probability distribution.

$$L_{\text{KL}} = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)$$

Where:

- $P$ is the true distribution
- $Q$ is the approximated distribution

**Applications:**

- Variational autoencoders (VAEs)
- Knowledge distillation
- Distribution matching

**Variants:**

- **Reverse KL Divergence**: Swaps the order of the distributions, yielding different behavior.
- **Jensen-Shannon Divergence**: A symmetrized and smoothed version of KL divergence.

## Regularization Losses

These are typically added to the main loss function to prevent overfitting and improve generalization.

### L1 Regularization (Lasso)

$$L_{\text{L1}} = \lambda \sum_{i=1}^{n} |w_i|$$

**Properties:**

- Encourages sparse weights (many weights become exactly zero)
- Less sensitive to outliers than L2
- Can be used for feature selection

### L2 Regularization (Ridge)

$$L_{\text{L2}} = \lambda \sum_{i=1}^{n} w_i^2$$

**Properties:**

- Penalizes large weights more heavily
- Rarely sets weights to exactly zero
- More stable solutions than L1 regularization

### Elastic Net

$$L_{\text{ElasticNet}} = \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$

**Properties:**

- Combines L1 and L2 regularization
- Can select groups of correlated features
- More robust than either L1 or L2 alone

## Links

- [PyTorch Loss Functions Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [TensorFlow Loss Functions Guide](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
- [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
