---
aliases:
  - CM
tags:
  - evaluation
  - metric
  - classification
---
A Confusion Matrix is a table used to evaluate the performance of a classification model on a set of data for which the true values are known. It provides a detailed breakdown of how well the model predicts the classes.
## Basic Structure

For a binary classification problem, a confusion matrix is a 2×2 table:

|                    | Predicted Positive   | Predicted Negative   |
|--------------------|----------------------|----------------------|
| **Actual Positive**| True Positive (TP)   | False Negative (FN)  |
| **Actual Negative**| False Positive (FP)  | True Negative (TN)   |
![[Pasted image 20250508082351.png]]
Where:
- **True Positive (TP)**: Model correctly predicts the positive class
- **True Negative (TN)**: Model correctly predicts the negative class
- **False Positive (FP)**: Model incorrectly predicts the positive class (Type I error). Example: model predicts that a healthy patient is ill.
- **False Negative (FN)**: Model incorrectly predicts the negative class (Type II error). Example: model predicts that an ill patient is healthy.

For multi-class problems, the matrix expands to n×n dimensions, where n is the number of classes.

### Derived Metrics
#### Accuracy
Overall correctness of the model or the proportion of total predictions that were correct. Suitable when class distributions are balanced and the cost of misclassifying different classes is similar.
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Error Rate
Overall error of the model
   $$\text{Error Rate} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy}$$


#### Precision
Precision or Positive Predictive Value.
The proportion of actually positive instances among all instances predicted as positive (how many retrieved items are relevant). Suitable when the cost of a False Positive is high, for example spam detection.
   $$\text{Precision} = \frac{TP}{TP + FP}$$
#### Recall
Recall or Sensitivity, True Positive Rate
The proportion of correctly instances samples among all positive instances (how many relevant items were retrieved). Suitable when the cost of a False Negative is high, for example medical diagnosis or fraud detection.
   $$\text{Recall} = \frac{TP}{TP + FN}$$
#### Specificity
Specificity or True Negative Rate
The proportion of correctly instances samples among all negative instances. Suitable when correctly identifying negative instances is crucial, for example in medicine.
   $$\text{Specificity} = \frac{TN}{TN + FP}$$

#### False Positive Rate
The proportion of incorrectly instances samples among all negative instances. Suitable when "false alarms" are important.
   $$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

#### False Negative Rate
The proportion of incorrectly instances samples among all positive instances.
   $$\text{FNR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$$

#### F1 Score
Harmonic mean of precision and recall. Suitable when you need a balance between Precision and Recall, especially useful for imbalanced datasets where one might be high at the expense of the other.
   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

#### F-beta Score
A generalized version of the F1 score that introduces a parameter β to control the relative importance of precision and recall. With β = 2, recall is weighted twice as much as precision.

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}$$

#### Matthews Correlation Coefficient
A correlation coefficient between the observed and predicted binary classifications. A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and -1 indicates total disagreement between prediction and observation.
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

#### Cohen's Kappa
Measures agreement between predicted and actual classifications, adjusting for agreement by chance
    $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
    Where $p_o$ is the observed agreement and $p_e$ is the expected agreement by chance.

## Multi-class Metric Aggregation

For multi-class confusion matrices, metrics can be aggregated in several ways:

1. **Macro-averaging**: Calculate metrics for each class independently and take the unweighted mean
2. **Micro-averaging**: Aggregate the contributions of all classes to compute a single metric
3. **Weighted-averaging**: Calculate metrics for each class and take their average weighted by support (number of instances)
## ROC and AUC

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings. The Area Under the Curve (AUC) provides a single measure of classifier performance across all possible classification thresholds.

- **AUC = 0.5**: Performance is no better than random
- **AUC = 1.0**: Perfect classifier
- **AUC > 0.8**: Generally considered good performance

## Precision-Recall Curve

For imbalanced datasets, the Precision-Recall curve often provides more informative insights than the ROC curve. The area under this curve, known as Average Precision (AP), summarizes the precision-recall tradeoff.

## Links
- [Scikit-learn documentation on confusion matrices](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
- [Google ML Crash Course on Classification](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
- [Confusion matrix page on Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Precision & Recall at mlu-explain](https://mlu-explain.github.io/precision-recall/)
- 
