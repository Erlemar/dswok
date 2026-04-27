---
tags:
  - interview
  - interview-questions
---
## Model fundamentals

Q: Is it possible to build a linear model to predict the XOR operation?
A: No. XOR is not linearly separable: no hyperplane separates $(0,0), (1,1)$ from $(0,1), (1,0)$. Adding an interaction feature such as $x_1 \cdot x_2$, or any polynomial feature map, makes the problem linearly separable in the expanded space: essentially what a kernel [[SVM]] does via the kernel trick.

Q: Beyond the train/val gap, what else signals overfitting or underfitting?
A: Train and validation errors, both high and close together, suggest underfitting: adding complexity still helps both. A widening train–validation gap over epochs suggests overfitting. Other signals: unstable coefficients across CV folds, heavy reliance on a handful of noisy features, and a big performance gap between near-identical model complexities. More data and [[Regularization]] address variance; richer features or more complex models address bias.

Q: What is the bias (intercept) term in a linear model, and why is it needed?
A: The intercept lets the decision boundary shift away from the origin. Without it, the model is forced to predict zero when all inputs are zero.

Q: Why do ensembles usually beat single models?
A: Averaging decorrelated errors reduces variance (bagging, [[Random Forest]]), and sequentially fitting residuals reduces bias ([[Gradient boosting]]). The gain depends on how independent the base learners are: averaging different models works better than averaging similar models.

Q: When to use bagging vs boosting?
A: Bagging helps high-variance base learners (deep trees in [[Random Forest]], neural nets with different seeds) and handles noisy labels better. Boosting helps bias-limited problems where the base learner is weak (shallow trees); [[Gradient boosting]] is often the strongest classical baseline on structured tabular data. Boosting is more sensitive to label noise and needs explicit regularization via learning rate, tree depth, and early stopping.

Q: Why can a deeper decision tree overfit while a shallow one underfits?
A: A shallow [[Decision Tree]] partitions the feature space too coarsely to capture real structure (high bias). A very deep tree can create leaves with a single training sample and memorize noise (high variance). Beyond depth and pruning, single trees are unstable: small changes in the training set can change their structure significantly, which is why they are usually ensembled ([[Random Forest]], [[Gradient boosting]]).

Q: What is the difference between parametric and non-parametric models?
A: Parametric models have a fixed-capacity ([[Linear Regression]], [[Logistic regression]]): small parameter count, better sample efficiency, cheaper deployment, but limited flexibility. Non-parametric models ([[K-Nearest Neighbors]], [[Decision Tree]]) grow capacity with data: more flexible but memory-heavy at inference and weaker at extrapolating outside of the training range.

Q: How to choose between generative and discriminative models?
A: Discriminative models estimate $P(y \mid x)$ directly and tend to have higher metrics when there is a lot of data available (logistic regression, most neural nets). Generative models estimate $P(x, y)$ and help when data is scarce, features may be missing at inference, or synthetic samples are needed (Naive Bayes, LDA, GMM, VAE).

Q: What is the curse of dimensionality?
A: As dimensionality grows, data becomes sparse and distances between points concentrate (nearest and farthest neighbors start to look similar, see [[Distance calculation]]). Consequences: [[K-Nearest Neighbors]] degrades, density estimation needs exponentially more samples, and any model that relies on local geometry struggles. Mitigations: feature selection, [[Dimensionality Reduction]] (PCA, UMAP), or learning representations that embed the data in a lower-dimensional manifold.

## Data and features

Q: How do you handle missing values in a tabular model?
A: Match the strategy to the missingness mechanism. For MCAR (missing completely at random), row-drop or mean/median imputation is acceptable; for MAR (missing at random), model-based imputation (k-NN, iterative methods like MICE) or adding a missing-indicator feature works better; for MNAR (missing not at random), the missingness itself carries signal — encode it as a feature. XGBoost, LightGBM, and CatBoost handle missing values natively, which often beats manual imputation on tree models.

Q: How would you encode a high-cardinality categorical feature?
A: One-hot works up to a few dozen levels; beyond that, feature hashing gives bounded memory, frequency encoding is trivial and sometimes enough, target (mean) encoding captures predictive signal but leaks easily, and learned embeddings let deep models recover structure. Target encoding must be fit inside CV folds (or with out-of-fold averaging and smoothing), otherwise the encoded column leaks the label and inflates validation scores.

Q: When does feature scaling matter, and when does it not?
A: It matters for distance-based models ([[K-Nearest Neighbors]], [[K-means clustering]], [[SVM]] with an RBF kernel — see [[Distance calculation]]), for PCA, and for gradient-based optimization where unscaled features make the loss surface ill-conditioned. It does not matter for [[Decision Tree]], [[Random Forest]], or [[Gradient boosting]], which split on thresholds regardless of scale.

Q: How do outliers affect different model types?
A: Linear models with squared-error losses are sensitive because a few extreme points can dominate the objective. Distance-based methods ([[K-Nearest Neighbors]], [[K-means clustering]]) are also sensitive because outliers distort neighborhoods and centroids. Tree ensembles are more robust but not immune. Mitigations: robust losses (Huber, quantile) when outliers are real rare events, clipping or winsorizing when the tail is noise, transformation (for example, log), or removal if the outlier is a data-quality error.

Q: What is multicollinearity, and when should you care?
A: Two or more input features are strongly linearly related, which inflates the variance of [[Linear Regression]] coefficients and makes individual coefficients unstable and hard to interpret. Detect with a correlation matrix or variance inflation factor (VIF > 5–10 is a common flag). Fixes: drop one feature, combine them (PCA, sum), or apply L2 [[Regularization]]. Tree models are mostly unaffected on accuracy, but split gain gets spread across correlated features, so importance scores understate each one.

Q: What is the difference between feature importance and causal importance?
A: Feature importance (split gain, permutation importance, SHAP) reflects what the model uses for prediction, not what causes the outcome. Correlated features split gains, and a strong confounder can dominate importance while having no causal role. Causal estimates require an intervention, a randomized experiment, or an identification strategy (instrumental variables, difference-in-differences, uplift modeling).

Q: How do you handle cold start for new users or new items?
A: For users, fall back to content or context features (device, geography, referrer) and a popular-items baseline until enough interactions accumulate. For items, use content features (text, image, metadata embeddings) and nearest-item lookup against items with known engagement. [[Two-tower]] models can be trained to handle cold start explicitly by forcing content features alongside ID embeddings.

## Validation and metrics

Q: Which cross-validation variant for which problem?
A:
- K-fold: usual tabular data.
- Stratified K-fold: imbalanced classes.
- Group K-fold: multiple rows share a unit that must stay on one side of the split (user, patient, session).
- Time-series / walk-forward: temporal data, to preserve ordering. See [[Time-series validation]].
- Nested CV: hyperparameter tuning plus more realistic generalization estimate in the same run.

See [[Validation]] for the broader picture.

Q: How do you detect and prevent data leakage?
A: Check that every feature is computable at prediction time using only information available then; fit preprocessing (scalers, target encoders, imputers) inside CV folds rather than on the full dataset; verify no group (user, entity, session) appears in both train and test; and treat suspiciously high AUC as a leakage signal until disproved. See [[Validation]].

Q: How do you detect train/test distribution shift?
A: Adversarial validation: train a binary classifier to distinguish train from test; an AUC well above 0.5 indicates a shift, and the highest-importance features point to what changed. Complement with PSI or KL divergence on key features and feature-wise drift analysis.

Q: When to use calibration vs ranking?
A: ROC-AUC measures pairwise ranking (the probability that a random positive scores higher than a random negative) and is invariant to monotonic transforms of the score. PR-AUC summarizes precision-recall behavior across thresholds and emphasizes the positive class. Calibration is a separate concept: do predicted probabilities match observed frequencies? Calibration matters for thresholding, cost-sensitive decisions, and blending probabilities across models; ranking alone suffices for top-k selection. Isotonic regression or Platt scaling fix miscalibration without changing the ranking.

Q: What are the use cases of ROC-AUC vs PR-AUC?
A: ROC-AUC reports ranking across all thresholds and is stable when the positive class is well represented. PR-AUC focuses on the positive class and tracks model usefulness better when positives are rare, since it ignores the flood of true negatives that keeps ROC-AUC flattering. Under heavy imbalance, usually prefer PR-AUC.

Q: How do you diagnose miscalibration in a classifier?
A: Bin the predicted probabilities (equal-width or equal-frequency), plot a reliability diagram of observed positive rate vs mean predicted probability per bin, and summarize the deviation as Expected Calibration Error (ECE) or Brier score. Overconfidence shows up as a curve below the diagonal (predicted 0.8 but observed 0.6); underconfidence shows up above it. A model can rank perfectly and still be badly calibrated.

Q: How do you handle class imbalance?
A: Class-weighted loss, resampling (up, down, SMOTE), decision-threshold tuning, or switching to metrics that survive imbalance (PR-AUC over ROC-AUC, [[f1 score]], recall at a fixed precision). Threshold tuning and metric choice often matter more than resampling, and SMOTE on high-dimensional data can create implausible synthetic points.

Q: When does accuracy mislead, and what do you use instead?
A: Under heavy class imbalance (a 99% negative dataset scores 99% for free), when positive and negative errors carry different costs, and when the positive class is ill-defined. Use [[Confusion matrix]]-derived metrics — precision, recall, [[f1 score]], PR-AUC — or a business metric such as cost per false positive.

Q: Offline metrics improved, but the online A/B test was flat or negative — what happened?
A: The offline metric is a proxy that doesn't track the business metric closely enough. Training data came from a different policy than the one now being A/B tested, so offline evaluation was optimistic (policy mismatch). Train/serve feature skew means the model sees different inputs online. The A/B segment differs from the offline sample. Novelty effects inflate the initial lift and fade. Mitigations: counterfactual or off-policy evaluation, shadow deployment with real-time feature logging, and picking an offline proxy that has historically correlated with the target online metric on past launches. See [[AB Tests]] for the full online-experimentation reference.

## Model selection and tuning

Q: Which hyperparameter tuning approaches to use?
A: Grid search for small, low-dimensional spaces; random search when you have to tune many hyperparameters; Bayesian optimization (Optuna, scikit-optimize) for expensive objectives; Hyperband or BOHB when training cost varies widely, and bad configurations can be early-stopped. Bayesian methods assume a reasonably smooth objective; very noisy metrics can mislead them.

Q: How to choose L1 vs L2 vs ElasticNet vs early stopping?
A: L1 for sparsity and implicit feature selection; L2 for general shrinkage and stabilizing coefficients under multicollinearity; ElasticNet when both matter; early stopping as implicit regularization on gradient-trained models. See [[Regularization]] for more information.

Q: How do you compare two models fairly?
A: Use the same splits, preprocessing, and evaluation metric. Report variance across folds or seeds, not just the mean. For significance, a paired test on per-fold scores (or a bootstrap over the test set) gives a lightweight signal, with the usual caveats about CV-fold dependence. Compare compute cost and inference latency alongside quality.

## Unsupervised learning

Q: How do you evaluate a clustering model when you have no labels?
A: With no labels, internal metrics (silhouette score, Davies-Bouldin, Calinski-Harabasz) trade off compactness within clusters against separation between clusters. If a labeled subset exists, extrinsic metrics (adjusted Rand index, V-measure, cluster purity) are stronger. Beyond metrics: inspect cluster sizes, stability under subsampling, and whether the clusters map to human-meaningful segments. Silhouette favors convex, globular clusters, so it can pick $k=2$ on data whose real structure is five non-convex clusters. See [[K-means clustering]].

Q: When would you use dimensionality reduction, and which technique?
A: For compression and denoising before a downstream model (PCA when the structure is roughly linear), for visualization of high-dimensional data (t-SNE and UMAP, where UMAP often preserves more global structure than t-SNE), and to break the [[Distance calculation]] issues that hurt high-dimensional [[K-Nearest Neighbors]] or [[K-means clustering]]. Avoid using t-SNE distances as features; the algorithm optimizes for local neighborhoods, and the distances it produces aren't meaningful. See [[Dimensionality Reduction]].

## Production and deployment

Q: What is training-serving skew, and how do you catch it?
A: The offline pipeline and the online feature service compute features differently — different libraries, different missing-value handling, timezone mismatches, clipping or scaling applied in one but not the other. Catch it by logging served features and replaying them through the training pipeline, comparing distributions, and running a shadow model in production before launch.

Q: Covariate shift, label shift, and concept drift — what's the difference?
A:
- Covariate shift (data drift): $P(x)$ changes, but $P(y \mid x)$ still holds. Inputs look different, the mapping is intact. Sometimes harmless.
- Label shift (prior shift): $P(y)$ changes, but $P(x \mid y)$ stays — class prevalence moves (fraud spikes, churn seasonality). Fixable by reweighting.
- Concept drift: $P(y \mid x)$ itself changes — the same features now predict a different outcome (user preferences shift, fraud strategies mutate). This typically hurts and requires updating the model.

Detect concept drift via rolling performance on delayed labels and backtests of the current model on recent windows; detect distribution-only shifts via PSI, KL divergence, or feature-wise distribution checks.

Q: What are the causes of label lag and feedback loops?
A: Delayed ground truth means the most recent training labels are missing or noisy, and a model trained naively lags the true signal. Feedback loops appear when model decisions shape future data: a ranking model never exposes certain items, so they disappear from training; a fraud model blocks borderline transactions that would have labeled themselves. Mitigations include explore-exploit policies, counterfactual logging, and inverse-propensity reweighting.

Q: What is model cannibalization?
A: A new or updated model degrades the performance of other models in the same system by competing for the same surface: two recommenders fighting for the same feed slot, or a new ranker absorbing clicks that would otherwise have been credited to another model. The symptom is that the new model looks good in isolation, but the overall product metric is flat or down. Monitoring should include per-model attribution, not just the new model's metric.

## Applied reasoning

Q: You are given two time series: BTC prices and news sentiments. How would you measure if there's a price prediction signal in sentiments? How can you verify the causal relation?
A: Start with EDA for lead-lag patterns and cross-correlation at several lags. Test Granger causality from sentiment to price, and run the reverse direction as a sanity check — if price also Granger-causes sentiment, that could be feedback, shared drivers, or different lag structures, so don't take the forward test at face value. Control for obvious confounders (overall market moves, macro news) so that both series aren't simply reacting to a third variable. Then test whether lagged sentiment features predict future returns out-of-sample with [[Time-series validation]], never leaking future information. Granger causality is a statistical association under prediction, not true causation; for stronger claims, move to causal ML methods (double ML, synthetic controls) or a randomized intervention, which, for public markets, isn't feasible but for on-platform experiments is.
