---
aliases:
  - Natural Language Processing
  - natural language processing
  - nlp
---
Natural Language Processing (NLP) is a multidisciplinary field that combines linguistics, computer science, and artificial intelligence to enable computers to understand, interpret, and generate human language.

Before 2010s most approaches relied on hand-crafted features (word frequencies, n-grams, linguistic annotations) extracted from text. At the early stages of NLP, the main approaches to solving NLP tasks consisted of if-else rules and templates, using rule-based grammars, ontologies, and heuristics.
Some examples:
- checking if a word (or it's stem/lemma) are present in a dictionary
- checking if a part of the text follows a pattern of pre-defined grammatical rules
- applying Hidden Markov Models to part-of-speech tagging and Conditional Random Fields to NER
- applying Naïve Bayes and SVM to word counts (including [[TF-IDF]]) and n-grams

In 2010s, [[Word Embeddings]] were introduced - [[Word2Vec]], [[GloVe]], [[fastText]] and [[recurrent models]] became more widely used. While [[RNN]] and LSTM [[were]] proposed earlier, people started using them more together with word embeddings. Later, [[GRU]] variation became popular.

In 2017, the paper [Attention is all you need](https://arxiv.org/abs/1706.03762) appeared and started the era of [[Transformers]]. [[BERT]]- and [[GPT]]-style transformers became the core of the further approaches.

There was another pivotal paper in 2017 - [[Universal Language Model Fine-tuning for Text Classification]], which introduced fine-tuning as we know it: general pre-training, target-task fine-tuning and target-task classifier fine-tuning.

There were some other papers which explored similar ideas. In [Large Language Models in Machine Translation](https://aclanthology.org/D07-1090/) the authors trained a model on up to 2T tokens with up to 300B n-grams (up to 5-gram). And [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) suggested pretraining a sequence autoencoder and then fine-tuning it for classification.

The appearance of [[Large Language Models]] was another significant step. The criteria of LLM can be vague, so even [[BERT]] can be called an LLM, but if by LLM we mean large models that can do a variety of tasks through prompting and without fine-tuning (or minimal fine-tuning), then GPT-2 or GPT-3 can be called the first LLMs.

The rest of this note will serve as an index for the other notes related to NLP.
## Text processing
Raw texts need to be processed in order to be used in ML models. This processing includes:
- Stemming and lemmatization - converting words to their basic forms and finding word roots, respectively
- Stop word removal - getting rid of the words that don't add much information
- [[Tokenization]] - splitting text into elements (character, word fragments, words) and encoding them

## Architectures
- [[RNN|Recurrent Neural Net]], [[LSTM]], [[GRU]]
- [[Transformer]]
- [[T5]]

## Training and Fine-tuning Techniques
- Pre-training objectives
- Fine-tuning strategies
- Instruction Learning
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
## NLP Tasks

- Named Entity Recognition
- Machine Translation
- Text Classification
	- Topic modeling - a special type of text classification, when text gets assigned certain topics
- Machine Translation
- Text Generation
- Chat-bots
- Information Retrieval and Extraction
	- Search
	- Question answering
	- Summarization

## Metrics
- BLUE
- ROUGE
- METEOR
- Perplexity

## Links
* [NLP for Supervised Learning - A Brief Survey](https://eugeneyan.com/writing/nlp-supervised-learning-survey/)
* 