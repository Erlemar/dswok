---
aliases:
  - model validation
tags:
  - concept
  - evaluation
  - optimization
---
Model validation is the process of assessing how well a trained machine learning model performs on unseen data. It's crucial for evaluating the model's generalization ability and ensuring it's not overfitting or underfitting. It involves splitting the dataset into two or more parts, training the model on one of them and calculating metrics on all of them

## Purpose
- Estimate model performance on unseen data, ensure model reliability and robustness
- Detect overfitting and underfitting
- Compare different models or algorithmic approaches
- Tune hyperparameters

## Validation approaches
### Train-validation-test split
* The classical way of validation - splitting the dataset into three parts. Training set is used for training the model, validation set is used for optimizing model performance, test set is used to assess model's unbiased performance. We should check the performance on the test set only after we finish experiments on the validation set.
* Sometimes only two parts are used - train and validation/test
### Bootstrapping
* Data is samples with replacement to create multiple subsets of data
* Useful for estimating model variance and creating confidence intervals
### Cross-validation
* Repeatedly splitting data into training and validation sets randomly or based on certain principles.
* Usually training and validation metrics are calculated on each split, then mean value and standard deviation of metric are calculated
* Often a separate holdout/test set is used to measure the unbiased performance
* For relatively small datasets cross-validation is preferred to single train-validation-test split. However, when the data numbers millions of rows, it is okay to use a single split
### Adversarial validation
* This is a separate technique - it is used to check model generalization by creating a binary classification problem between the training and test sets.
* It works like this: combine training and test data with labels 0 and 1 respectively, train a binary classifier and measure AUC.
* High AUC indicated distribution shift between train and test data, low AUC mean the sets are similar.
* Features with high feature importance may have the highest distribution shift

## Ways of splitting the data
* Random spitting: split data randomly
* Stratified: split data keeping the percentage of each class in each subset similar
* Leave-One-Out: split data keeping only one sample in validation
* Group: split data keeping each group in one subset only. For example, can be used to ensure that the data belonging to each user is only in one subset to avoid leakage
* [[Time-series validation]]: Keeps the temporality of the data, useful when there are time dependencies

### Links
* [Explained.ai: The Importance of Data Splitting](https://mlu-explain.github.io/train-test-validation/)
* [Sklearn documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)
