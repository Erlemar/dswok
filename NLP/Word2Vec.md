---
aliases:
  - word2vec
  - w2v
tags:
  - algorithm
  - model
  - nlp
  - embeddings
---
Word2Vec is a approach for learning word embeddings. It became the first popular approach to representing words as dense vectors in a continuous vector space where semantically similar words are positioned closer together.

Word2Vec uses shallow neural networks with a single hidden layer to learn vector representations by predicting words from their context (Continuous Bag-of-Words) or predicting context words from a target word (Skip-gram).

Skip-gram Model takes a center word as input and predicts surrounding context words. This is better for rare words and larger datasets.
CBOW Model takes context words as input and predicts the center word. This leads to faster training and is better for frequent words.
### Architecture
1. Convert input word to one-hot vector
2. Multiply by weight matrix W of size $V \times D$ (equivalent to lookup)
3. For CBOW: average the resulting embeddings
4. Multiply by context matrix W' of size $D \times V$
5. Apply softmax to obtain probability distribution
### Training Optimizations
- Replace full softmax with hierarchical softmax
- Don't update all weights at each iteration, instead use negative sampling to update the positive and negative samples only. 
- Randomly discard frequent words during training

> [!example]- PyTorch implementation
> 
> ```python
> class Word2Vec(nn.Module):
>     def __init__(self, vocab_size, embedding_dim, model_type):
>         super(Word2Vec, self).__init__()
>         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
>         self.linear = nn.Linear(embedding_dim, vocab_size)
>         self.model_type = model_type
> 
>     def forward(self, context_words):
>         # context_words: (batch_size, context_size)
>         embeds = self.embeddings(context_words)
>         if self.model_type == 'CBOW':
>             embeds = embeds.mean(dim=1)
>         out = self.linear(embeds)
>         return out
> ```

## Links

- [Original Paper: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Second Paper: Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)