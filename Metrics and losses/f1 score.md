---
aliases:
  - f1
tags:
  - evaluation
  - metric
  - classification
---
The F1 score is an evaluation metric for classification models that combines precision and recall into a single value, providing a balanced assessment of model performance. It is particularly useful for imbalanced datasets where accuracy alone might be misleading.

The F1 score is calculated as the harmonic mean of precision and recall:

$$F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
F1 score varies from 0 to 1, where 1 represents perfect precision and recall. F1 uses the harmonic mean instead of arithmetic mean to punish extreme values. Precision 1 and recall 0 result in a mean of 0.5, but f1 of 0.

F1 is usually used for imbalanced datasets, when both precision and recall are important. But it ignores TN, so it can't be used when TN is important. And it is necessary to tune the classification threshold

## Variations

**F-beta Score** is a generalized version of the F1 score that introduces a parameter β to control the relative importance of precision and recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}$$
With β = 2, recall is weighted twice as much as precision.

In multi-class scenarios, **macro F1** averages the F1 scores calculated independently for each class:
$$\text{Macro F1} = \frac{1}{n} \sum_{i=1}^{n} F1_i$$
**Micro F1** aggregates the contributions of all classes to compute a single F1 score:
$$\text{Micro F1} = \frac{2 \cdot \sum_{i=1}^{n} \text{TP}_i}{2 \cdot \sum_{i=1}^{n} \text{TP}_i + \sum_{i=1}^{n} \text{FP}_i + \sum_{i=1}^{n} \text{FN}_i}$$
## Links
- [Scikit-learn documentation on F1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- [Wikipedia: F-score](https://en.wikipedia.org/wiki/F-score)
