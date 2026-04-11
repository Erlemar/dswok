---
tags:
  - metrics
  - loss
aliases:
  - metrics
---

# Metrics and losses

Notes on evaluation metrics and loss functions used across ML domains.

## Loss functions
How models learn — the optimization signal during training.

- [[General losses]] — Cross-Entropy, MSE/MAE, KL Divergence, L1/L2/Elastic Net regularization.
- [[NLP losses]] — NLL, Perplexity, CTC, Triplet, Contrastive, RLHF (PPO/DPO).
- [[Computer vision losses]] — Focal, Dice, IoU/Jaccard, Perceptual, Adversarial, SSIM.

## Evaluation metrics
How models are measured — performance assessment after training.

- [[f1 score]] — harmonic mean of precision and recall, with macro/micro/weighted variants.
- [[Confusion matrix]] — true/false positives and negatives for classification.
- [[Regression metrics]] — MSE, RMSE, MAE, R-squared and variants.
- [[Recommendation system metrics]] — NDCG, MAP, MRR, Hit Rate, coverage, diversity.
- [[Computer vision metrics]] — IoU, mAP, pixel accuracy, FID, SSIM.
- [[NLP metrics]] — BLEU, ROUGE, METEOR, perplexity, BERTScore.
- [[Clustering metrics]] — silhouette score, Davies-Bouldin, Calinski-Harabasz, purity.
