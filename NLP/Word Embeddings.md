Word embedding is a representation of a word, usually with a vector of values.

The simplest way to represent words is by using one-hot encoding - i-th word in the vocabulary will have a vector with 1 in i-th position and 0 in all the others. But this isn't efficient: such vectors are too sparse, too long and provide too little information.

Slightly better approaches are count-based encoding and [[TF-IDF]].

The first paper suggesting using embeddings was [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (2003) - they authors trained a basic language model to learn a distributed representation for words. Another early paper was [A unified architecture for natural language processing: deep neural networks with multitask learning](https://dl.acm.org/doi/10.1145/1390156.1390177) (2008), where the words were "embedded into a d-dimensional space using a lookup table" - basically, a first modern mention of word embeddings.

But the [[Word2Vec]] paper [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) was the start of the wide-spread use of word embeddings. Word2Vec provided an efficient way to train shallow neural networks on large corpora to produce high-dimensional vectors, such that words appearing in similar contexts have similar vectors​.

[[GloVe]] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) used global word co-occurrence statistics to produce embeddings, using a different approach (leveraging matrix factorization and corpus-wide statistics)​

[[fastText]] [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) extended Word2Vec by representing each word as a bag of character n-grams, enabling the model to generate vectors for rare or misspelled words by composing subword information​

Later, [[BERT]]-like models started being used for word embeddings. [[ELMo]] produces contextual embeddings.
