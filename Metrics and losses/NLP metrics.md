---
tags:
  - nlp
  - evaluation
cssclasses:
  - term-table
---
NLP metrics evaluate tasks from machine translation and summarization to question answering and sequence labeling. Most boil down to n-gram overlap, embedding similarity, or classification-style precision/recall tailored to the output type.

## When to use which metric

| Metric | When to use |
|---|---|
| BLEU | Translation / generation; n-gram precision. |
| ROUGE-N / L / S | Summarization; n-gram, LCS, or skip-bigram recall. |
| METEOR | Translation with synonym and stemming awareness. |
| chrF | Character-level n-gram F-score; robust for morphologically rich languages. |
| BERTScore | Semantic similarity via BERT embeddings. |
| MoverScore | Semantic distance via Earth Mover's on contextualized embeddings. |
| Perplexity | Language modeling — how well a model predicts a sequence. |
| Bits per Character (BPC) | Same idea as perplexity, character-level. |
| Exact Match (EM) | QA — does the predicted answer match exactly? |
| QA F1 | QA — word-level bag-of-words F1. |
| Answer Accuracy | Multiple-choice QA. |
| MRR | Rank of the first correct answer (QA / IR). |
| Entity-level F1 | NER at the entity level. |
| Span-based F1 | NER with exact span match. |
| CoNLL Score | NER averaged across entity types. |

## Machine Translation & Text Generation Metrics

Used for translation, summarization, image captioning, and dialogue.

### BLEU (Bilingual Evaluation Understudy)

Measures the precision of n-gram matches between the candidate translation and reference translations. Correlates moderately with human judgment; doesn't capture fluency or semantic meaning well. Prefers shorter sentences.

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:
- $p_n$ is the modified n-gram precision.
- $w_n$ is the weight for n-gram precision (typically uniform weights).
- BP is the brevity penalty to penalize short translations.
- $N$ is the maximum n-gram size (typically 4).

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

F-score based on word matches, considering synonyms, stemming, and word order.

$$\text{METEOR} = F_{mean} \cdot (1 - \text{Penalty})$$

Where:
- $F_{mean}$ is a weighted harmonic mean of precision and recall.
- Penalty accounts for fragmentation (poor word order).

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures n-gram recall (and sometimes precision / [[f1 score|F1]]) between generated and reference texts.

- **ROUGE-N** — n-gram overlap.

$$\text{ROUGE-N} = \frac{\sum_{S \in {\text{References}}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in {\text{References}}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

- **ROUGE-L** — longest common subsequence (LCS). Captures sentence-level structure similarity.
- **ROUGE-S** — skip-bigram overlap (pairs of words in sentence order with gaps allowed). ROUGE-SU extends ROUGE-S by adding unigrams.

### chrF

Character n-gram F-score.

$$\text{chrF} = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

Where:
- chrP is character n-gram precision.
- chrR is character n-gram recall.
- $\beta$ determines recall importance (typically $\beta = 2$).

### BERTScore

Uses BERT embeddings to compute similarity between candidate and reference translations at the token level.

### MoverScore

Uses contextualized embeddings and Earth Mover's Distance to measure semantic distance between generated and reference texts.

### Perplexity

Measures how well a model predicts a sample. Lower perplexity indicates better prediction. The exponentiated average negative log-likelihood of a sequence.

$$\text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, ..., w_{i-1})}$$

Where:
- $P(w_i | w_1, ..., w_{i-1})$ is the conditional probability of word $w_i$ given previous words.
- $N$ is the number of words.

### Bits Per Character (BPC)

Similar to perplexity but measured at the character level.

$$\text{BPC} = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(c_i | c_1, ..., c_{i-1})$$

Where:
- $P(c_i | c_1, ..., c_{i-1})$ is the conditional probability of character $c_i$ given previous characters.
- $N$ is the total number of characters.

## Question Answering Metrics

### Exact Match (EM)

Binary measure indicating whether the predicted answer exactly matches the ground truth. Can also be calculated as a percentage of predictions that match one of the ground truth answers exactly.

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\text{prediction}_i = \text{groundtruth}_i)$$

### F1 Score

Word-level F1 between prediction and ground truth, treating both as bags of words.

### Answer Accuracy

For multiple-choice QA, the proportion of questions answered correctly.

### Mean Reciprocal Rank (MRR)

For QA models that return a ranked list of answers, the average of the reciprocal rank of the correct answer.

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

## Sequence Labeling Metrics

Named Entity Recognition (NER) and Part-of-Speech (POS) tagging involve identifying and classifying named entities or parts of speech in text.

### Entity-Level F1 Score

F1 score calculated at the entity level rather than the token level.

### Span-Based F1 Score

F1 score based on the exact match of entity spans.

### Partial Matching Metrics

- **Partial Precision / Recall** — give credit for partial overlap between predicted and true entities.
- **Type-Based Evaluation** — separate evaluation for entity type classification and entity boundary detection.

### CoNLL Score

Average of F1 scores across all entity types.

## Links

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [HuggingFace Evaluate Library](https://huggingface.co/docs/evaluate/index)
