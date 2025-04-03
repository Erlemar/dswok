---
aliases:
  - BERT paper
tags:
  - nlp
---
[Paper](https://arxiv.org/abs/1810.04805)
BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

### The approach
![[Pasted image 20250328183623.png]]
BERT is trained in two phases: pre-training and fine-tuning. During pre-training, BERT learns from large amounts of unlabeled text using language modeling tasks. In the fine-tuning phase, the model is initialized with these pre-trained parameters and then further trained on labeled data specific to a downstream task, such as question answering.

A key characteristic of BERT is its unified architecture across various tasks. BERT is based on the Transformer encoder and uses bidirectional self-attention, which allows each token to attend to both its left and right context.

The authors create two model sizes: $BERT_{BASE}$ and $BERT_{LARGE}$. $BERT_{BASE}$ has 12 Transformer layers, a hidden size of 768, and 12 self-attention heads, totaling about 110 million parameters. $BERT_{LARGE}$ is a deeper and wider version with 24 layers, a hidden size of 1024, 16 attention heads, and approximately 340 million parameters. $BERT_{BASE}$ was designed to match the size of OpenAI's GPT for comparison purposes.

![[Pasted image 20250329113516.png]]
To handle a variety of natural language processing tasks, BERT uses a flexible input representation. It supports both single sentences and pairs of sentences by concatenating them into a single sequence. Special tokens are used to mark structure: every input starts with a `[CLS]` token used for classification outputs, and sentence pairs are separated by a `[SEP]` token. Additionally, each token in the sequence receives embeddings that indicate the token identity, its position in the sequence, and which sentence (the first or the second) it belongs to. The model uses WordPiece embeddings with a vocabulary of 30,000 tokens. The final hidden state of the `[CLS]` token serves as a fixed-length representation of the whole sequence for classification tasks, while the hidden states of other tokens can be used for token-level tasks.

#### Pre-training
In **Masked Language Modeling (MLM)**, 15% of input tokens are randomly selected and masked. The model is trained to predict these masked tokens based on their surrounding context. To prevent a mismatch between pre-training and fine-tuning (since the `[MASK]` token doesn’t appear during fine-tuning), the masked token is replaced with `[MASK]` 80% of the time, a random token 10% of the time, and kept unchanged 10% of the time. Unlike traditional denoising autoencoders, MLM predicts only the masked tokens, not the full input sequence.

**Next Sentence Prediction (NSP)** helps BERT learn the relationship between pairs of sentences - important for tasks like question answering and natural language inference. During training, 50% of sentence pairs consist of consecutive sentences from the corpus and 50% are randomly paired sentences. The model uses the `[CLS]` token’s final hidden state to predict whether the second sentence follows the first.

BERT uses two datasets for pre-training: **BooksCorpus** (800 million words) and **English Wikipedia** (2.5 billion words), keeping only continuous text passages and excluding non-narrative content.

#### Fine-tuning
Unlike earlier approaches that separately encoded each sentence before applying cross-attention, BERT concatenates the pair and processes them together using self-attention, effectively achieving bidirectional cross-attention within a unified architecture.

To fine-tune BERT for a specific task, task-specific inputs and outputs are added, and all model parameters are trained end-to-end. The pre-training sentence pair format maps directly to different tasks: paraphrasing, entailment, question answering, classification (with an empty second sentence). For token-level tasks like question answering or tagging, the token outputs are used, while for classification tasks, the final hidden state of the `[CLS]` token is used.
