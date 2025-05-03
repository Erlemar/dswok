---
aliases:
  - glove
tags:
  - algorithm
  - model
  - nlp
  - embeddings
---
GloVe (Global Vectors for Word Representation) is a word embedding technique developed in 2014. While [[Word2Vec]] learns word co-occurrence via a sliding window (local statistics), GloVe learns via a co-occurrence matrix (global statistics). GloVe then trains word vectors so their differences predict co-occurrence ratios. Even though Word2Vec and GloVe have different starting points, their word representations turn out to be similar.

The first step is to build word-word co-occurrence matrix based on the corpus. The training is done on non-zero co-occurences. The training objective optimizes word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence.

![[Pasted image 20250316083135.png]]
## Links

- [Original Paper: GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- [Stanford NLP GloVe Project Page](https://nlp.stanford.edu/projects/glove/)
- [NLP Course | For You: GloVe](https://lena-voita.github.io/nlp_course/word_embeddings.html#glove)
