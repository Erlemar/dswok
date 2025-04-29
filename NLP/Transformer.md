---
tags:
  - attention
  - architecture
  - nlp
---
The first Transformer was introduced in the [[Attention Is All You Need|Attention_paper]], soon after that [[BERT]] was published.
This note will cover different ways of building Transformers and how they evolved over time.

### Vanilla Transformer
The encoder consists of 6 identical layers, each with two sub-layers: a multi-head self-attention and a position-wise feed-forward network, both followed by residual connections and layer normalization. All outputs are of dimension 512.

The decoder also has 6 similar layers but adds a third sub-layer for multi-head attention over the encoder's output. It uses the same residual and normalization approach. Additionally, it masks future positions in self-attention to ensure that predictions for the next position only depend on earlier positions.

### Variations
#### Tokenizers
- Byte-Pair Encoding (BPE) - the most frequent pairs of characters/subwords are merged iteratively, words - are split into the longest known subword pieces. Examples: GPT-2, RoBERTa.
- WordPiece - optimized BPE. Tries to match longest substrings starting from the beginning. Examples: BERT.
- SentencePiece - used by T5, ALBERT
#### Positional embeddings
- Positional embeddings are added to the token embeddings. Optionally, segment embeddings can be added too - in BERT, segment embeddings separate pairs of sentences.
- Fixed (sinusoidal) embeddings can be easily extended for longer sequence length, but usually works worse than learnable embeddings.
- Learned Absolute Positional Embeddings - fixed max length during training. Each position has its own vector. These embeddings are added inside the attention mechanism.
- Relative Positional Embeddings - encodes distance between tokens (like token B is N steps ahead of token A).
- T5 Bias - learns a simple scalar bias term that is added directly to the attention logits based on the relative distance between the query and key positions.
- Rotary Positional Embeddings (RoPE) - encodes absolute position information into the query and key vectors by applying position-dependent rotations to chunks of their embedding dimensions before the dot product is calculated.
![[Pasted image 20250405180744.png]]
### Decoding Strategies

- Greedy Search: At each step, select the single token with the highest probability according to the model's output distribution. Simple, deterministic, prone to repetition and loops. Often produces bland, predictable, and unnatural-sounding text. It might make locally optimal choices that lead to a poor overall sequence.
- Beam Search: Instead of just keeping the single best token at each step, beam search maintains k (the "beam width") parallel candidate sequences (called "beams"). First, it selects the top k most likely tokens. Then, for each of the k beams, it considers all possible next tokens and calculates the cumulative probability or log-probability of the resulting sequences. After this, it selects the top k sequences from all possibilities generated across all beams. This continues until the sequences end or reach maximum length. The final output is usually the sequence with the highest overall probability. Produces more coherent text, but can be repetitive.
- Temperature Sampling: Temperature is a hyperparameter used to rescale the logits before calculating the probabilities. The formula becomes $softmax(logits / T)$. T > 1 increases randomness, T < 1 decreases randomness. T close to 0 is similar to greedy search. High temperature is good for creativity.
- Top-K Sampling: At each step, consider only the K most probable tokens predicted by the model. Prevents unrealistic generation.
- Top-P (Nucleus) Sampling: select the smallest set of tokens whose cumulative probability exceeds a threshold P (the "nucleus"). For example, if P=0.9, sort tokens by probability and keep adding them to the set until their combined probability reaches 0.9. More adaptive than Top-K.

### Notable architectures
- ALBERT (A Lite BERT for Self-supervised Learning of Language Representations): factorized embedding parameterization (splits the large vocabulary embedding matrix into two smaller matrices), cross-layer parameter sharing, inter-sentence coherence loss.
- Alpaca: LLaMA fine-tuned on ~52k instruction-following synthetical examples.
- BART (Bidirectional and Auto-Regressive Transformers): combines BERT's bidirectional encoder with GPT's autoregressive decoder.
- CLIP (Contrastive Language-Image Pre-training): dual-encoder - an image encoder (ViT or ResNet) and a text encoder (Transformer). Pre-trained with contrastive learning, learns to predict which text caption corresponds to which image within a large batch, maximizing similarity for correct pairs and minimizing it for incorrect pairs.
- DALL-E - models generating images directly from text. DALL-E 1: a discrete VAE to tokenize images and an autoregressive Transformer to model the joint distribution of text and image tokens.
- DeBERTa (Decoding-enhanced BERT with disentangled attention): disentangled attention and mask decoder enhanced with absolute word position information
- ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately): uses two models - a small generator (like a small BERT) that replaces some input tokens with plausible alternatives using MLM, and a larger discriminator (the main ELECTRA model) trained to predict whether each token is original or was replaced by the generator.
- LLaMA (Large Language Model Meta AI): decoder only with pre-normalization (RMSNorm), SwiGLU and RoPE.
- Mistral: efficient decoder-only model. Uses Grouped-Query Attention (GQA) and Sliding Window Attention (SWA).
- RoBERTa (Robustly Optimized BERT Pretraining Approach): optimized BERT with more data, no NSP, dynamic masking and larger byte-level BPE vocabulary.
- T5 (Text-to-Text Transfer Transformer): encoder-decoder, pre-trained on span corruption objective.
- XLNet: permutation language modeling.
- RWKV: Linear time like RNNs; performs like Transformers.
- Mamba: state-space model.

### Various information
- Encoder-only, encoder-decoder, and decoder-only models:
	- Encoder-only: are good for encoding input into a rich representation; often used for sentence classification, NER, QA classification, embedding generation. They are often trained with MLM objective. Examples: BERT.
	- Decoder-only: are good for generation; often used for text/code completion, dialogues, language modeling. They are often trained with Causal Language Modeling. Examples: GPT, LLaMA, Mistral.
	- Encoder-decoder (or seq2seq): are good for text transformation; often used for machine translation, summarization, QA generation, image captioning. They are often trained with seq2seq learning. Examples: T5, BART.
- Layer Normalization (LayerNorm): normalizes samples across the feature dimension instead of normalizing features across batches (like in Batch Normalization). It normalizes by mean and variance with two learnable parameters: $y = γ * (x - μ) / sqrt(σ² + ε) + β$. In the standard Transformer is was applied after multi-head attention and dense sub-layer. Currently, pre-normalization is more widely used - before multi-head attention and dense sub-layer.
- [[Attention]]
- 
