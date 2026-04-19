---
tags:
  - nlp
  - loss
cssclasses:
  - term-table
---
Loss functions commonly used in NLP tasks. For general losses (Cross-Entropy, MSE, KL Divergence), see [[General losses]]. For NLP evaluation metrics, see [[NLP metrics]].

## When to use which loss

| Loss | When to use |
|---|---|
| Negative Log-Likelihood (NLL) | Language modeling, sequence prediction. |
| Perplexity | Evaluation form of NLL — geometric mean of inverse probabilities. |
| CTC | Sequence-to-sequence without explicit alignment (speech, OCR). |
| Triplet | Metric learning; similar items close, dissimilar far. |
| Contrastive | Siamese networks; discriminative feature learning. |
| PPO | Policy-gradient RL with clipped updates for stability. |
| DPO | Preference-based LLM fine-tuning without a reward model. |

## Negative Log-Likelihood (NLL) Loss

Commonly used in language modeling and sequence prediction.

$$L_{\text{NLL}} = -\frac{1}{N} \sum_{i=1}^{N} \log(p(y_i | x_i))$$

Where $p(y_i | x_i)$ is the predicted probability of the true token/class.

**Applications:** Language modeling, machine translation, text generation, sequence prediction.

## Perplexity

Exponential transformation of the average negative log-likelihood, making it interpretable as the weighted average number of choices the model is uncertain about. Perplexity is an evaluation metric, but minimizing NLL is equivalent to minimizing perplexity.

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(y_i | y_{<i})\right)$$

Where $p(y_i | y_{<i})$ is the probability of the $i$-th token given previous tokens.

**Applications:** Language modeling, text generation evaluation, speech recognition.

## Connectionist Temporal Classification (CTC) Loss

Aligns sequence-to-sequence data without requiring pre-segmented training data or explicit alignments.

$$L_{\text{CTC}} = -\log\left(\sum_{\pi \in \mathcal{A}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t | x)\right)$$

Where:
- $\mathcal{A}^{-1}(y)$ is the set of all possible alignments that correspond to the target sequence $y$.
- $p(\pi_t | x)$ is the probability of alignment $\pi$ at time $t$ given input $x$.

**Applications:** Speech recognition, handwriting recognition, protein sequence alignment.

## Triplet Loss

Learns embeddings where similar items are closer together and dissimilar items are farther apart.

$$L_{\text{triplet}} = \max(d(a, p) - d(a, n) + \text{margin}, 0)$$

Where:
- $a$ is the anchor sample.
- $p$ is a positive sample similar to the anchor.
- $n$ is a negative sample dissimilar to the anchor.
- $d$ is a distance function (typically Euclidean or cosine).
- margin is a hyperparameter.

**Applications:** Sentence embeddings, document similarity, face recognition, image retrieval.

## Contrastive Loss

Used to learn discriminative features by pushing similar samples closer and dissimilar samples further apart.

$$L_{\text{contrastive}} = (1-Y) \cdot \frac{1}{2} \cdot D^2 + Y \cdot \frac{1}{2} \cdot \max(0, \text{margin} - D)^2$$

Where:
- $Y$ is 0 for dissimilar pairs and 1 for similar pairs.
- $D$ is the distance between samples.

**Applications:** Sentence similarity, learning text embeddings, Siamese networks for document comparison.

## Reinforcement Learning from Human Feedback (RLHF) Losses

### PPO (Proximal Policy Optimization) Loss

$$L_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

Where:
- $r_t(\theta)$ is the ratio of new policy probability to old policy probability.
- $A_t$ is the advantage estimate.
- $\epsilon$ is a hyperparameter that constrains policy updates.

### Direct Preference Optimization (DPO) Loss

$$L_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Where:
- $\pi_\theta$ is the policy being trained.
- $\pi_{\text{ref}}$ is the reference policy.
- $(x, y_w, y_l)$ are input, preferred output, and dispreferred output.
- $\beta$ is a hyperparameter.

**Applications:** Fine-tuning language models based on human preferences, aligning large language models with human values, improving language model outputs for specific criteria.

## Links

- [PyTorch Loss Functions Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [TensorFlow Loss Functions Guide](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
