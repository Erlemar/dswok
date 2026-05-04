---
aliases:
  - Probability calibration
  - Calibrated probabilities
  - Score calibration
tags:
  - evaluation
  - concept
---
A classifier is calibrated if its predicted probabilities match observed frequencies: among examples assigned a 0.7 score, roughly 70% should be positive. Formally, $P(Y = 1 \mid \hat{p}(X) = p) = p$ for all $p \in [0, 1]$.

Calibration is distinct from discrimination. A model with high AUC can still produce miscalibrated probabilities, and a perfectly calibrated model can still rank examples poorly. Calibration matters whenever raw scores are used in downstream models or threshold-based decisions.

## When calibration matters

- Probability thresholds, where the threshold's intended meaning depends on calibration (a fraud rule firing at $p > 0.8$ is only meaningful if 0.8 reflects an 80% positive rate).
- Multi-score arithmetic, such as the $\text{pCTR} \times \text{pConv} \times \text{value}$ pattern in ad ranking. Miscalibrated inputs distort the product even when each model has a good AUC.
- Score combination across models: blending scores from two or more models (a weighted average or sum) only respects the intended weighting if the scores are on the same probability scale. A 50/50 blend of a calibrated and an over-confident model is dominated by the over-confident one even though the weights say otherwise.
- Expected-utility decisions: any cost-benefit calculation that multiplies a predicted probability by a reward or loss.
- Resource allocation: manual review queues, fraud investigation, credit limits, moderation actions, and notification budgets often gate on calibrated risk.
- Modern deep classifiers, which are typically over-confident compared to shallow models or earlier architectures ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)).

Calibration is not the right objective for pure ranking systems. AUC, [[NDCG]], and mean average precision (MAP) depend only on score order, so any monotonic transformation of model outputs leaves them unchanged. In [[Recommendation system]] pipelines, retrieval is rank-only, while ranking-stage scores feed downstream arithmetic and need calibration.

## Sources of miscalibration

### Miscalibration by model family

Different model families miscalibrate in characteristic ways:

- Logistic regression: well-calibrated when the log-odds-linear assumption holds and training prevalence matches deployment, since log loss is a strictly proper scoring rule. Miscalibrates under model misspecification, regularization shrinkage, class weighting, or prevalence shift.
- Gradient boosting trained with log loss (modern LightGBM, XGBoost defaults): generally well-calibrated, since the loss is proper. [Niculescu-Mizil & Caruana (2005)](https://doi.org/10.1145/1102351.1102430) document this empirically.
- Random forests and bagging ensembles: under-confident at the extremes. Leaf-level averaging and ensemble averaging both pull predictions away from 0 and 1.
- AdaBoost and other margin-driven boosters: characteristically sigmoidal miscalibration. Platt scaling was originally designed for this shape.
- Deep neural networks: typically over-confident, especially with high capacity, batch normalization, and weight decay ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)).
- Naive Bayes: over-confident because the conditional-independence assumption double-counts correlated features.

### Miscalibration from training and data choices

These act on top of any model-family tendency:

- Distribution shift across slices: country, device, season, traffic source. Global ECE (Expected Calibration Error) can hide severe slice-level miscalibration.
- Sampled negatives or class-balanced training: the raw output reflects training prevalence, not deployment prevalence. See [[Negative sampling]] and the prior-correction section below.
- Training-time choices: focal loss down-weights easy examples and can produce under-confident outputs; label smoothing caps confidence at $1 - \varepsilon$; mixup and cutmix train on interpolated labels and need recalibration; class weighting changes the effective training prevalence.
- Delayed and censored labels: conversions, churn, refunds, and chargebacks arrive late. Calibration measured before labels mature is biased; the calibration target specifies a label window (1-day click, 7-day conversion, 30-day churn).
- Selection and exposure bias: in [[Recommendation system]] settings, labels exist only for items that were shown. Calibration is for the exposed distribution and is conditional on the logging/serving policy. A policy change can move the exposure distribution and break calibration even when feature distributions look stable.
- [[Training-serving skew]]: feature transformations, missing-value defaults, normalization, or freshness differing between training and serving create skew that calibration can mask but not fix.

## Measuring calibration

### Reliability diagram

A reliability diagram bins predicted probabilities into $M$ buckets (typically 10 or 20), then plots the bucket midpoint against the empirical positive rate within the bucket. Perfect calibration lies on the $y = x$ diagonal. Bins above the diagonal indicate under-confidence; bins below indicate over-confidence. Reliability plots should overlay bin counts or confidence intervals: sparse high-score bins are noisy and should not be over-interpreted.

![[calibration_reliability_diagram.excalidraw.light.svg]]

For highly imbalanced problems (fraud, conversion), the linear-axis reliability diagram hides errors at small $p$, where the relevant decisions are actually made. Log-axis variants and bin-count overlays help. Production monitoring should track calibration in the decision region (around thresholds, top-k buckets, or auction-relevant ranges) in addition to the global view; a low global ECE can coexist with badly miscalibrated 0.8–1.0 buckets.

### Expected Calibration Error (ECE)

ECE summarizes a reliability diagram into a single number:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$

where $M$ is the number of bins, $B_m$ is the $m$-th bin, $|B_m|$ its size, $n$ the total sample size, $\text{acc}(B_m)$ the empirical positive rate inside the bin, and $\text{conf}(B_m)$ the mean predicted probability inside the bin.

ECE is binning-sensitive. Equal-width and equal-frequency schemes give different values on the same predictions. Too few bins underestimate miscalibration by averaging errors inside wide bins; too many inflate variance when per-bin sample size is small.

[Nixon et al. (2019)](https://arxiv.org/abs/1904.01685) propose adaptive binning. [Roelofs et al. (2022)](https://arxiv.org/abs/2012.08668) document the estimation bias and recommend equal-mass binning with bias correction. [Popordanoska et al. (2022)](https://arxiv.org/abs/2210.07810) sidestep binning entirely with a KDE-based unbinned estimator.

### Brier score

The Brier score is the mean squared error between predicted probability and outcome:

$$\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

It is a strictly proper scoring rule: its expected value is minimized only when reported probabilities equal the true probabilities. It decomposes into reliability, resolution, and uncertainty terms ([Murphy 1973](https://doi.org/10.1175/1520-0450%281973%29012%3C0595%3AANVPOA%3E2.0.CO%3B2)). A discriminating but miscalibrated model can have a low Brier score because high resolution offsets poor reliability, so the components have to be read separately.

### Log loss

Log loss on held-out data is also a strictly proper scoring rule. It is more sensitive than Brier to confident-wrong predictions because the per-example contribution diverges as $p \to 0$ or $p \to 1$ on a misclassification. Useful when the application penalizes confident-wrong predictions disproportionately.

## Recalibration methods

Recalibration fits a one-dimensional mapping from raw scores to calibrated probabilities on a held-out set. The three commonly-used methods differ mainly in flexibility and parametric form.

![[calibration_recalibration_methods.excalidraw.light.svg]]

### Platt scaling

Platt scaling fits a logistic regression on raw model scores ([Platt 1999](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods)):

$$p_{\text{calibrated}} = \frac{1}{1 + \exp\!\big(-(\alpha s + \beta)\big)}$$

where $s$ is the raw model score and $\alpha, \beta$ are scalar parameters fit on a held-out set by minimizing log loss. Stable on small validation sets, but assumes a sigmoidal miscalibration shape. Originally targeted [[SVM]] decision values; also used for shallow models with smooth score distributions and for AdaBoost-style classifiers whose miscalibration is sigmoidal by construction.

### Isotonic regression

Isotonic regression fits a non-parametric monotonic mapping by minimizing squared error subject to the monotonicity constraint ([Zadrozny & Elkan 2002](https://doi.org/10.1145/775047.775151)). It is more flexible than Platt scaling, more accurate on non-sigmoidal miscalibration, and more prone to overfitting on small validation sets. The fitted mapping is piecewise constant: extreme scores can collapse onto the same calibrated probability, and the mapping does not extrapolate beyond the training-score range.

### Temperature scaling

Temperature scaling rescales softmax logits of a neural-net classifier by a single learned scalar $T$ ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)):

$$\mathbf{p} = \text{softmax}(\mathbf{z} / T)$$

where $\mathbf{z}$ is the pre-softmax logit vector and $\mathbf{p}$ the calibrated probability vector. $T < 1$ sharpens the distribution; $T > 1$ flattens it. Modern deep classifiers are typically over-confident, so the fit yields $T > 1$. $T$ is fit on a held-out set by minimizing log loss. The single-parameter form overfits less than multi-parameter alternatives on modern deep classifiers.

Within an example, temperature scaling preserves the order of logits, so the predicted class does not change. Across examples, it does not preserve confidence ranking: top-1 confidence depends on the full logit vector (not just the gap to the runner-up), so two examples can switch order in confidence ranking as $T$ changes. For binary classification, this does not arise, since $p = \sigma(z/T)$ is monotonic in the single logit.

### Beta calibration

Beta calibration ([Kull et al. 2017](https://proceedings.mlr.press/v54/kull17a.html)) generalizes Platt scaling to non-sigmoidal shapes. The standard fit is a logistic regression on $[\log s, \log(1 - s)]$, giving 2–3 parameters depending on the chosen restriction. It sits between Platt (too rigid for some reliability-diagram shapes) and isotonic (prone to overfitting on small validation sets).

## Prior correction for sampled training data

Many production models train on data with prevalence different from production: recsys with negative downsampling, fraud models with positive oversampling, ad-CTR pipelines with aggressive negative subsampling. The raw model output is calibrated to the training sample, not the production base rate. Recalibrating with Platt or isotonic on a validation set with the same artificial sampling carries the bias forward; recalibrating on an unbiased, production-like validation set can learn the base-rate correction directly.

The general correction shifts the logit by the difference between deployment and training log-odds:

$$\text{logit}(p_{\text{cal}}) = \text{logit}(p_{\text{model}}) + \log\frac{\pi_{\text{prod}}}{1 - \pi_{\text{prod}}} - \log\frac{\pi_{\text{train}}}{1 - \pi_{\text{train}}}$$

where $\pi_{\text{train}}$ is the positive-class prevalence in training data, $\pi_{\text{prod}}$ the production prevalence, $p_{\text{model}}$ the raw model output, and $p_{\text{cal}}$ the corrected probability.

For the special case of negative downsampling at rate $w$ (the fraction of negatives kept in training), the correction simplifies to a constant logit shift:

$$\text{logit}(p_{\text{cal}}) = \text{logit}(p_{\text{model}}) + \log w$$

or equivalently in probability form:

$$p_{\text{cal}} = \frac{p_{\text{model}}}{p_{\text{model}} + (1 - p_{\text{model}}) / w}$$

This correction is exact only under the prior shift, where sampling depends on the class label alone. If sampling probability depends on features, item or user segments, candidate source, or time, the shift is biased; importance weighting during training or calibration on an unbiased production-like validation set is needed instead.

Prior correction is applied before or together with Platt/isotonic/temperature recalibration. If skipped, the recalibrator absorbs the prevalence shift implicitly. That works on validation data drawn from the training distribution but fails as soon as production prevalence drifts.

## Multi-class calibration

Approaches to multi-class calibration are different from the binary one:

- Top-label calibration: among predictions where the top class has confidence $p$, the top class is correct with frequency $p$. The most commonly reported metric.
- Class-wise calibration: each class probability is calibrated separately, treated as a one-vs-rest binary problem. Stronger.
- Joint calibration: the full probability vector matches the conditional distribution of $Y$ given $\hat{p}(X)$. Strongest, hardest to estimate.

Methods generalize accordingly. Temperature scaling extends with a single $T$ shared across classes. Vector scaling allows a per-class temperature and bias. Matrix scaling applies a full linear transformation to the logit vector, more flexible but more prone to overfitting. Dirichlet calibration ([Kull et al. 2019](https://arxiv.org/abs/1910.12656)) parameterizes a Dirichlet family on the probability simplex. Per-class one-vs-rest with Platt or isotonic is a common pragmatic approximation: each class gets its own calibrator, then the outputs are renormalized to sum to one. The renormalization step can change top-1 predictions, because per-class calibrators are monotonic individually but not jointly.

## Per-segment recalibration

Calibration drifts across slices (country, device, cohort, time of day) and within each slice over time. Global recalibration averages segment-level errors, which can leave individual segments badly miscalibrated even when the global ECE is small. Per-slice recalibration trades higher variance (each slice needs enough validation data) for lower bias.

When per-slice positive counts are small, hierarchical or partial-pooling approaches that shrink slice estimates toward the global recalibrator keep the variance manageable. Stratified sampling of the calibration set with a minimum positive count per slice is the simpler operational version.

Production monitoring tracks ECE and reliability diagrams per slice in addition to the global value. Sudden drift in a single slice often signals an upstream feature pipeline issue or a population shift that the global metric averages out. See [[AB Tests]] for calibration as a guardrail metric during launches.

Per-segment calibrators can change cross-segment ordering even when each per-segment mapping is monotonic. This matters when downstream systems compare scores across segments: a global threshold, a unified review queue, or an auction that ranks across user types. In those settings, the per-segment calibrators have to be kept consistent at decision boundaries, or the ranking step has to operate on pre-calibration scores.

## Practical considerations

- Calibration set independence: the calibration set is held out from training and from any model selection that touched it. Use a true three-way split (train, calibrate, test); reusing the validation set for both selection and calibration produces optimistic calibration estimates. For time-dependent systems, split by time.
- Distribution match: the calibration set should match production. Same population filters, same feature path, same label window, same negative-sampling correction (or none).
- Refresh cadence: refit on a recent window (daily, weekly, or per release) when the production distribution drifts.
- Online ECE: streaming binning with windowed or exponentially decayed counts catches drift as it happens; offline ECE on historical batches catches it after the fact.
- Continuous training: the calibration window slides with the training window. Tracking $T$ (or the Platt slope) over time exposes drift early. A sudden swing in fitted parameters usually signals data or pipeline issues before downstream metrics move.
- Multi-task heads: a multi-task ranker (pCTR + pConv + dwell) typically has each head trained on differently sampled data and emits uncalibrated probabilities. Per-head recalibration is needed before scores are combined.
- Ensembles of calibrated models: averaging two already-calibrated models produces an output that is no longer calibrated. The ensemble needs its own recalibration step on a held-out set.

## Limitations

- Recalibration is fit on a held-out set; if the production distribution drifts, the calibration map drifts too, so it needs periodic refitting on a recent window.
- Calibration cannot fix a poorly discriminating model. A constant classifier predicting the base rate is perfectly calibrated and useless; recalibration only operates on the ordering the model already produces.
- Pure ranking systems are unaffected by monotonic recalibration.
- Isotonic regression can clip extreme scores flat; the fitted mapping does not extrapolate to scores outside the calibration range.
- Calibration provides marginal-frequency guarantees, not per-prediction coverage. When uncertainty intervals matter on individual predictions, it is necessary to use conformal prediction.
- Calibration reports how often predictions at confidence $p$ are correct. Deciding whether to abstain on low-confidence inputs (selective prediction) is a separate problem: it combines calibration with the cost of errors and the value of correct predictions.
- LLM token probabilities post-RLHF (reinforcement learning from human feedback) or DPO (direct preference optimization) are systematically distorted: over-confident on the policy distribution. Any system using LLM logprobs as confidence scores needs explicit recalibration; reliability diagrams on token-level distributions typically show consistent over-confidence.

## Code example

> [!example]- sklearn CalibratedClassifierCV
> ```python
> import matplotlib.pyplot as plt
> from sklearn.calibration import CalibratedClassifierCV, calibration_curve
> from sklearn.metrics import brier_score_loss
>
> # sklearn >= 1.2 uses `estimator=`; older versions used `base_estimator=`.
> calibrated = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=5)
> calibrated.fit(X_train, y_train)
> probs = calibrated.predict_proba(X_test)[:, 1]
>
> brier = brier_score_loss(y_test, probs)
> prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
>
> plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
> plt.plot(prob_pred, prob_true, marker="o")
> plt.xlabel("Mean predicted probability")
> plt.ylabel("Empirical positive rate")
> plt.title(f"Reliability diagram (Brier = {brier:.3f})")
> plt.show()
> ```
> `method="sigmoid"` switches to Platt scaling. `cv="prefit"` calibrates an already-trained model on a separate validation set; in newer sklearn it is being phased out in favor of wrapping the base estimator with `FrozenEstimator`.

## Links

- [[Recommendation system]]: auctions and multi-score combination, a frequent production case for calibration.
- [[Negative sampling]]: causes the base-rate shift that prior correction addresses.
- [[Training-serving skew]]: a frequent driver of calibration drift in production.
- [[AB Tests]]: calibration drift as a guardrail metric during ramps and launches.
- [[SVM]]: the original Platt-scaling target.
- [Platt — *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods* (1999)](https://www.researchgate.net/publication/2594015_Probabilistic_Outputs_for_Support_Vector_Machines_and_Comparisons_to_Regularized_Likelihood_Methods)
- [Zadrozny & Elkan — *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* (KDD 2002)](https://doi.org/10.1145/775047.775151)
- [Niculescu-Mizil & Caruana — *Predicting Good Probabilities With Supervised Learning* (ICML 2005)](https://doi.org/10.1145/1102351.1102430)
- [Guo, Pleiss, Sun, Weinberger — *On Calibration of Modern Neural Networks* (ICML 2017)](https://arxiv.org/abs/1706.04599)
- [Kull, Filho, Flach — *Beta Calibration* (AISTATS 2017)](https://proceedings.mlr.press/v54/kull17a.html)
- [Kull et al. — *Beyond Temperature Scaling: Dirichlet Calibration* (NeurIPS 2019)](https://arxiv.org/abs/1910.12656)
- [Nixon et al. — *Measuring Calibration in Deep Learning* (CVPR Workshops 2019)](https://arxiv.org/abs/1904.01685)
- [Popordanoska, Sayer, Blaschko — *A Consistent and Differentiable Lp Canonical Calibration Error Estimator* (NeurIPS 2022)](https://arxiv.org/abs/2210.07810)
- [Roelofs et al. — *Mitigating Bias in Calibration Error Estimation* (AISTATS 2022)](https://arxiv.org/abs/2012.08668)
- [scikit-learn — Probability calibration user guide](https://scikit-learn.org/stable/modules/calibration.html)
