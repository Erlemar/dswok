---
tags:
  - interview
---
Q: What is model cannibalization?
A: Model cannibalization occurs when a new/updated ML model negatively impacts the existing models within the same system. This happens when models have overlapping functionalities (for example, both models recommend something, and users prefer one of them). It can lead to reduced effectiveness of existing models, potential resource waste, and confusion in model selection.

Q: Is it possible to build a linear model to predict XOR operation?
A: No, it is not possible to build a linear model to accurately predict the XOR operation. This is because the XOR operation is not linearly separable. In a linear model, the decision boundary is a straight line (or a hyperplane in higher dimensions), but the XOR function requires a non-linear decision boundary. But adding polynomial features can help capture non-linear relationships.

Q: You are given two time series: BTC prices and news sentiments. How would you measure if there’s a price prediction signal in sentiments? How can you verify the causal relation?
A: EDA to observe patterns. Granger causality test. Control for confounders to check that both time-series don't depend on the same external variables. Check if news lag features can predict future price and not vise versa. Causal Machine Learning.