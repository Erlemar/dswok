---
tags:
  - nlp
  - architecture
  - transformer
  - pre-trained
---
 Most of the information is available in the paper: [[BERT. Pre-training of Deep Bidirectional Transformers forLanguage Understanding|BERT paper]].
 
Key details:
- Multi-head attention. [[Transformer]] encoder. Two model sizes: BASE and LARGE
- This 768-dimensional representation is used across all layers in BERT Base.
- WordPiece tokenization, embeddings with 30,000 token vocabulary. Position embeddings.
- Bidirectional model
- Masked Language Modeling (randomly mask 15% tokens: 80% are replaced with `[MASK]` token, 10% with random token, 10% unchanged) and Next Sentence Prediction (predict if the next sentence is random or following) pretraining

### Variants and Extensions
- RoBERTa (Robustly Optimized BERT Approach): no NSP task, dynamic masking, Byte-Pair Encoding
- DistilBERT: Distilled version of BERT with 40% fewer parameters, 60% faster while retaining 97% of BERT's performance
- ALBERT (A Lite BERT): factorized embedding parameterization and cross-layer parameter sharing, inter-sentence coherence task instead of NSP
- ELECTRA: Replaced Token Detection instead of MLM, generator-discriminator architecture for more efficient pre-training
- ModernBERT: Updates BERT with the modern improvements to Transformer architecture

## Links

- [Original BERT Paper](https://arxiv.org/abs/1810.04805)
- [BERT Explained - Visual Guide](https://jalammar.github.io/illustrated-bert/)
- [Google AI Blog on BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- [BERT GitHub Repository](https://github.com/google-research/bert)