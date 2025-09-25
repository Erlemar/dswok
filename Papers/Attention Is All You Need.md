---
aliases:
  - Attention_paper
tags:
  - nlp
  - attention
  - architecture
  - transformer
---
[Paper](https://arxiv.org/abs/1706.03762)
The Transformer is a neural network architecture for sequence transduction tasks that relies entirely on [[attention]] mechanisms, removing the need for recurrence and convolutions. It outperforms previous models in machine translation tasks, while being more parallelizable and faster to train. The Transformer also generalizes well to other tasks like English constituency parsing.

### The architecture
![[Pasted image 20250401084839.png]]
The Transformer encoder consists of 6 identical layers, each with two sub-layers: a multi-head self-attention and a position-wise feed-forward network, both followed by residual connections and layer normalization. All outputs are of dimension 512.

The decoder also has 6 similar layers but adds a third sub-layer for multi-head attention over the encoder's output. It uses the same residual and normalization approach. Additionally, it masks future positions in self-attention to ensure that predictions for the next position only depend on earlier positions.

![[Pasted image 20250401085506.png]]

The Transformer uses Scaled Dot-Product Attention, which computes attention scores by taking the dot product of queries and keys, scaling by $√d_k$ to avoid large gradients, applying softmax, and weighting the values accordingly.

To enhance the model's ability to focus on different information, Transformer uses Multi-Head Attention: queries, keys, and values are linearly projected into smaller subspaces, attention is computed in parallel across 8 heads, and the outputs are concatenated and projected again. This allows the model to attend to information from multiple representation subspaces simultaneously.

The Transformer applies attention in three ways:
* Encoder-decoder attention: Decoder queries attend to encoder outputs.
* Encoder self-attention: Each encoder position attends to all others.
* Decoder self-attention: Each position attends to earlier positions only (using masking) to preserve autoregressive behavior.

Each encoder and decoder layer in the Transformer has FFN applied independently to each position. It consists of two linear layers with a ReLU activation in between: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$. Input and output dimension are 512, hidden layer dimension is 2048, parameters are unique to each layer but shared across positions. This is conceptually similar to using two convolutions with kernel size 1.

The Transformer uses learned embeddings to convert input and output tokens into vectors of size 512. It also uses a shared weight matrix for input embeddings, output embeddings, and the final linear layer before softmax.

Since the model has no recurrence or convolution, it adds positional encodings to the embeddings to inject information about token order. These encodings have the same dimension as the embeddings and are based on sine and cosine functions of different frequencies:
$\text{PE(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
$\text{PE(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

This allows the model to represent relative positions effectively and generalize to longer sequences.

