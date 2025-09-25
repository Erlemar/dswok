---
tags:
  - recsys
  - bias-correction
  - sampling
---
LogQ correction is a bias correction technique used in recommendation systems to account for non-uniform sampling during training. When negative samples are not drawn uniformly at random (in case of popularity-based sampling, in-batch sampling), the model learns a biased representation that favors frequently sampled items.

LogQ correction adjusts the training objective by subtracting the log probability of sampling each item as a negative:

**Original Loss (Contrastive):** $$\mathcal{L} = -\log \frac{\exp(s_{u,i^+})}{\exp(s_{u,i^+}) + \sum_{j \in \text{negatives}} \exp(s_{u,j})}$$

**Corrected Loss:** $$\mathcal{L}_{\text{corrected}} = -\log \frac{\exp(s_{u,i^+})}{\exp(s_{u,i^+}) + \sum_{j \in \text{negatives}} \exp(s_{u,j} - \log Q(j))}$$

Where:

- $s_{u,i}$ is the similarity score between user $u$ and item $i$
- $Q(j)$ is the probability of sampling item $j$ as a negative


LogQ correction effectively "discounts" the similarity scores of frequently sampled items. If an item is sampled with high probability $Q(j)$, then $\log Q(j)$ is less negative, making $s_{u,j} - \log Q(j)$ smaller and reducing the item's influence in the loss.

**For uniform sampling:** $Q(j) = \frac{1}{N}$ for all items, so $\log Q(j)$ is constant and cancels out.

**For popularity-based sampling:** $Q(j) \propto \text{popularity}(j)$, so popular items get larger corrections.
### Computing Q(j)

**In-batch Sampling:** If using other items in the batch as negatives: $$Q(j) = \frac{\text{frequency of item } j \text{ in training data}}{\text{total interactions}}$$

**Popularity-based Sampling:** If sampling negatives proportional to popularity: $$Q(j) = \frac{\text{interaction\_count}(j)}{\sum_k \text{interaction\_count}(k)}$$

## Links

- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)
- [On Sampling Strategies for Neural Network-based Collaborative Filtering](https://arxiv.org/abs/1706.07881)