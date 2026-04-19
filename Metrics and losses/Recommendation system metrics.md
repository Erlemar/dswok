---
aliases: []
tags:
  - recsys
  - evaluation
cssclasses:
  - term-table
---
Recommendation system metrics are quantitative measures used to evaluate the performance and effectiveness of recommendation algorithms. These metrics help assess how well a system can predict user preferences, rank items, and provide valuable recommendations.

## When to use which metric

| Metric | When to use |
|---|---|
| Precision@k | User sees only a few recs — those few must be highly relevant. |
| Recall@k | Retrieve as many relevant items as possible from a large catalog. |
| Hit Rate@k | Overall system effectiveness — does any rec hit? |
| MAP@k | Ranked recs where order matters and you want average precision across users. |
| NDCG@k | Ranked recs with graded relevance — higher positions matter more. |
| MRR@k | The first good rec matters most (search, top-of-feed). |
| ILD@k | Guard against filter-bubble — diversity inside a single list. |
| Novelty@k | Push users toward non-popular items. |
| Serendipity@k | Unexpected *and* relevant — delight discoveries. |
| Coverage | Long-tail health — what fraction of the catalog ever gets recommended. |
| CTR | Online click behaviour on recommended items. |
| Conversion Rate | Downstream action (purchase, signup) per recommendation. |
| User Satisfaction | Survey / explicit feedback. |

## Precision@k

Proportion of relevant items among the top-k recommendations. Useful when the user sees only a few recommendations and we want those few to be highly relevant.

$$\text{Precision@k} = \frac{\text{Number of relevant items in top-k recommendations}}{\text{k}}$$

## Recall@k

Proportion of relevant items that are present in the top-k recommendations. Useful when we want to retrieve as many relevant items as possible from a large catalog.

$$\text{Recall@k} = \frac{\text{Number of relevant items in top-k recommendations}}{\text{Total number of relevant items}}$$

## Hit Rate@k

Proportion of users for whom at least one relevant item appears in their top-k recommendations. Good for overall system effectiveness. Does not differentiate between one and multiple relevant recommendations.

$$\text{Hit Rate@k} = \frac{\text{Number of users with at least one hit in top-k}}{\text{Total number of users}}$$

## Mean Average Precision (MAP@k)

Mean of Average Precision (AP) across all users, where AP is the average of precision values at each relevant position in the ranked recommendations. Useful for ranked recommendations where order matters.

$$\text{AP@k} = \frac{1}{\min(m, k)} \sum_{i=1}^{k} \text{Precision@i} \cdot \text{rel}(i)$$

Where:
- $m$ is the number of relevant items for the user
- $\text{rel}(i)$ is an indicator function (1 if the item at position $i$ is relevant, 0 otherwise)

$$\text{MAP@k} = \frac{1}{|U|} \sum_{u \in U} \text{AP@k}(u)$$

## Normalized Discounted Cumulative Gain (NDCG@k)

Measures the quality of ranking by assigning higher weights to relevant items appearing higher in the list and normalizing by the ideal ranking. Penalizes relevant items appearing lower in the list.

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}(i)} - 1}{\log_2(i+1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

## Mean Reciprocal Rank (MRR@k)

Average of reciprocal ranks of the first relevant item across all users. Useful when the first good recommendation is most important (search engines).

$$\text{MRR@k} = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u}$$

## Diversity

Measures how diverse the recommended items are across various dimensions. Helps prevent the "filter bubble" phenomenon.

**Intra-List Diversity (ILD@k)** — the average pairwise dissimilarity between items in a recommendation list.

$$\text{ILD@k} = \frac{2}{k(k-1)} \sum_{i=1}^{k-1} \sum_{j=i+1}^{k} \text{dist}(i, j)$$

Where $\text{dist}(i, j)$ is the distance or dissimilarity between items $i$ and $j$.

## Novelty

Measures how unusual or unfamiliar the recommended items are to users. Helps users discover new content beyond popular items.

$$\text{Novelty@k} = \frac{1}{k} \sum_{i=1}^{k} -\log_2 \frac{|\text{Users who interacted with item } i|}{|\text{Total users}|}$$

## Serendipity

Measures how unexpected yet relevant the recommendations are. Aims to delight users with discoveries they wouldn't have found on their own.

$$\text{Serendipity@k} = \frac{1}{k} \sum_{i=1}^{k} \text{unexp}(i) \cdot \text{rel}(i)$$

Where:
- $\text{unexp}(i)$ is the unexpectedness of item $i$ (often calculated as dissimilarity from user's profile)
- $\text{rel}(i)$ is the relevance of item $i$

## Coverage

**Item Coverage** — the proportion of all available items that are recommended to at least one user. Helps prevent the "long-tail" problem where many items are never recommended.

$$\text{Item Coverage} = \frac{|\text{Items recommended to at least one user}|}{|\text{All available items}|}$$

**User Coverage** — the proportion of users who receive at least one recommendation.

$$\text{User Coverage} = \frac{|\text{Users receiving at least one recommendation}|}{|\text{All users}|}$$

## Conversion Rate

Percentage of recommendations that lead to a desired action (e.g., click, purchase).

$$\text{Conversion Rate} = \frac{\text{Number of recommendations resulting in conversion}}{\text{Total number of recommendations}}$$

## Click-Through Rate (CTR)

Ratio of clicks to impressions for recommended items.

$$\text{CTR} = \frac{\text{Number of clicks on recommendations}}{\text{Number of recommendation impressions}}$$

## User Satisfaction

Direct measurement of user satisfaction with recommendations, often collected through surveys or feedback mechanisms.

## Links
- [Evaluating Recommendation Systems: A Guide](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)
- [Ranking Metrics for Information Retrieval Systems](https://weaviate.io/blog/retrieval-evaluation-metrics)
