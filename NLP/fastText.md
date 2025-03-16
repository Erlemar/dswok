---

aliases:

- FastText tags:
- algorithm
- model
- nlp
- embeddings

---
fastText is a library for efficient text classification and word representation learning by Facebook. It extends [[Word2Vec]] by incorporating subword information, which allows it to generate meaningful embeddings for rare and out-of-vocabulary words.

For example, the word "apple" could be represented as: the word itself: `apple` and character n-grams: `ap, app, appl, apple, ppl, pple, ple, le`. Each of these n-grams has its own vector representation, and the final word vector is the sum of these subword vectors and the vector of the word itself.

## Links

- [Official fastText Website](https://fasttext.cc/)
- [Original Paper: Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- [Text Classification Paper: Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
