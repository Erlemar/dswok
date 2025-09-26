---
tags:
  - nlp
  - metric
  - evaluation
---
### Machine Translation & Text Generation Metrics (Summarization, Image Captioning, Dialogue)

1. **BLEU (Bilingual Evaluation Understudy)**: Measures the precision of n-gram matches between the candidate translation and reference translations. Correlates moderately with human judgment, doesn't capture fluency or semantic meaning well. Prefers shorter sentences.

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Where:

- $p_n$ is the modified n-gram precision
- $w_n$ is the weight for n-gram precision (typically uniform weights)
- BP is the brevity penalty to penalize short translations
- $N$ is the maximum n-gram size (typically 4)

2. **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: Calculates an F-score based on word matches, considering synonyms, stemming, and word order.

$$\text{METEOR} = F_{mean} \cdot (1 - \text{Penalty})$$

Where:

- $F_{mean}$ is a weighted harmonic mean of precision and recall
- Penalty accounts for fragmentation (poor word order)

3. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures n-gram recall (and sometimes precision/[[f1 score|F1]]) between generated and reference texts.
    
    - **ROUGE-N**: Measures n-gram overlap.
    
    $$\text{ROUGE-N} = \frac{\sum_{S \in {\text{References}}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in {\text{References}}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$
    
    - **ROUGE-L**: Measures the longest common subsequence (LCS). Captures sentence-level structure similarity.
    - **ROUGE-S**: Measures skip-bigram overlap (pairs of words in sentence order with gaps allowed). ROUGE-SU extends ROUGE-S by adding unigrams to the evaluation.

4. **chrF**: Character n-gram F-score.

$$\text{chrF} = (1 + \beta^2) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \cdot \text{chrP} + \text{chrR}}$$

Where:

- chrP is character n-gram precision
- chrR is character n-gram recall
- β determines the recall importance (typically β=2)

5. **BERTScore**: Uses BERT embeddings to compute similarity between candidate and reference translations at the token level.
6. **MoverScore**: Uses contextualized embeddings and Earth Mover's Distance to measure semantic distance between generated and reference texts.
7. **Perplexity**: Measures how well a model predicts a sample. Lower perplexity indicates better prediction. The exponentiated average negative log-likelihood of a sequence.

$$\text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, ..., w_{i-1})}$$

Where:

- $P(w_i | w_1, ..., w_{i-1})$ is the conditional probability of word $w_i$ given previous words
- $N$ is the number of words

8. **Bits Per Character (BPC)**: Similar to perplexity but measured at the character level.
    

$$\text{BPC} = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(c_i | c_1, ..., c_{i-1})$$

Where:

- $P(c_i | c_1, ..., c_{i-1})$ is the conditional probability of character $c_i$ given previous characters
- $N$ is the total number of characters

### Question Answering Metrics

1. **Exact Match (EM)**: Binary measure indicating whether the predicted answer exactly matches the ground truth answer. Can be also calculated as a percentage of predictions that match one of the ground truth answers exactly.

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\text{prediction}_i = \text{groundtruth}_i)$$

2. **F1 Score**: Word-level F1 score between prediction and ground truth, treating both as bags of words.
    
3. **Answer Accuracy**: For multiple-choice QA, the proportion of questions answered correctly.
    
4. **Mean Reciprocal Rank (MRR)**: For QA models that return a ranked list of answers, the average of the reciprocal of the rank of the correct answer.
    

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
### Sequence Labeling Metrics (Named Entity Recognition - NER, Part-of-Speech - POS Tagging)

Named Entity Recognition (NER) involves identifying and classifying named entities in text into predefined categories.

1. **Entity-Level F1 Score**: F1 score calculated at the entity level rather than the token level.
    
2. **Span-Based F1 Score**: F1 score based on the exact match of entity spans.
    
3. **Partial Matching Metrics**:
    
    - **Partial Precision/Recall**: Give credit for partial overlap between predicted and true entities.
    - **Type-Based Evaluation**: Separate evaluation for entity type classification and entity boundary detection.
4. **CoNLL Score**: The average of F1 scores across all entity types.

## Links

- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- [HuggingFace Evaluate Library](https://huggingface.co/docs/evaluate/index)
