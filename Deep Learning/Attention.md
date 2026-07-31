---
tags:
  - nlp
  - architecture
  - attention
  - transformer
prereqs:
  - "[[Word Embeddings]]"
---
[The original Transformer paper](https://arxiv.org/abs/1706.03762). Neural sequence attention predates it: [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) introduced soft alignment for machine translation.

Attention is a mechanism that lets neural networks combine information selectively from different input positions. Some architectures approximate standard attention for efficiency, such as [Linformer](https://andlukyane.com/blog/paper-review-linformer), while others replace it with a different sequence-mixing mechanism, such as [Hyena](https://andlukyane.com/blog/paper-review-hyena). Attention is also widely used in computer vision, as seen in [Swin Transformers](https://andlukyane.com/blog/paper-review-swin_v2) and object detection models like [DETR](https://andlukyane.com/blog/paper-review-detr).

A fundamental type is Scaled Dot-Product Attention (used in [[Transformer]]). It has three inputs:
- Query (Q): The current token trying to gather information.
- Key (K): A representation of each token in the sequence that's available to be attended to.
- Value (V): What each token provides if selected by the attention mechanism.

Attention calculation step-by-step:
1. Each key $K_i$ is scored against the query $Q$ with a dot product: $\text{scores} = Q \times K^T$
2. The scores are divided by $\sqrt{d_k}$, where $d_k$ is the dimensionality of the key vectors: $\text{scaled\_scores} = \frac{Q \times K^T}{\sqrt{d_k}}$. When $d_k$ is large, the dot product can grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. For dot products, the variance grows with $d_k$. The square root is what keeps the scores on a roughly dimension-independent scale.
3. Convert the scores into a probability distribution to see how much attention should be given to each element: $\alpha = \text{softmax}(\text{scaled\_scores})$.
4. Multiply each value $V_i$ by its attention weight $\alpha_i$ and sum to get the final output: $\text{Attention}(Q, K, V) = \alpha \times V$
This yields a context vector that highlights the most relevant information from $V$ for the query $Q$.

In short: attention computes a weighted sum of input elements (values) where the weights are determined by a compatibility function between a query and corresponding keys: $Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

>[!hint]
> Imagine you're at a large party trying to focus on a specific conversation. You're asking yourself about each person: "How relevant is what this person is saying to what I want to know?" (computing attention scores). Then you focus more on people providing useful information (applying the attention weights) while still maintaining some awareness of everyone else. Your brain combines all this information, giving more weight to important sources (weighted sum of values).

>[!hint]
> A simple explanation: attention is just a dictionary with approximation. In a usual dictionary we have a pair of key-value and we pass a query to get a result. We either get the value of the key or nothing. In attention we get the answer even if we can't find the exact key.

### Self-Attention
- Keys, queries, and values all come from the same source sequence
- Allows each position to attend to all positions in the sequence

### Cross-Attention
- The queries come from one sequence (e.g., the decoder in a seq2seq model), while the keys and values come from another (e.g., the encoder).
- Often used in machine translation and generative tasks where one sequence attends to another.

### Multi-Head Attention (MHA)
- Runs multiple attention mechanisms in parallel
- Allows the model to jointly attend to information from different representation subspaces at different positions. Each head can potentially learn to focus on different types of relationships or features.

### Multi-Query Attention (MQA)
- All query heads share the same key and value matrices, only query matrices are different
- Significantly reduces memory requirements and inference time
- Can lead to quality degradation compared to MHA

### Grouped-Query Attention (GQA)
- Introduced to balance the efficiency of MQA and the quality of MHA
- In MHA, each query head has its own key and value heads (higher quality but high memory usage). In MQA, all query heads share one key head and one value head (most efficient, and typically some quality loss). GQA divides the $H$ query heads into $G$ groups, and each group shares one key head and one value head.
- The number of groups $G$ is a hyperparameter: $G = H$ is MHA, $G = 1$ is MQA, and intermediate values trade between them
- Used in models like Llama 2-70B, Mistral 7B, and Falcon 40B. Particularly useful in multi-GPU environments with tensor parallelism

### Multi-head Latent Attention (MLA)
- Introduced in [DeepSeek-V2](https://arxiv.org/abs/2405.04434). MQA and GQA shrink the key-value (KV) cache by sharing key-value heads across queries; MLA compresses them instead.
- A down-projection maps the hidden state to a latent vector $c^{KV}$ of dimension $d_c \ll n_h d_h$, where $n_h$ is the number of heads and $d_h$ is the per-head dimension. Setting position encoding aside, only $c^{KV}$ needs to be cached; keys and values are reconstructed by up-projections $W^{UK}$ and $W^{UV}$.
- During autoregressive decoding those up-projections do not have to run at all: $W^{UK}$ can be absorbed into the query projection $W^Q$ and $W^{UV}$ into the output projection $W^O$, so full keys and values are never materialized. Prefill commonly uses the unabsorbed form instead.
- Rotary position embeddings (RoPE) break the absorption. The rotation matrix for the token being generated sits between $W^Q$ and $W^{UK}$, and matrix multiplication does not commute, so keys would have to be recomputed for every prefix token. The workaround is decoupled RoPE: a separate set of query dimensions plus one key $k^R$ shared across heads, each of size $d_h^R$, carry the positional signal and are concatenated with the compressed parts. That shared key is cached alongside the latent.
- KV cache per token is $(d_c + d_h^R)l$ across $l$ layers, against $2 n_h d_h l$ for MHA. DeepSeek-V2 reports a 93.3% KV cache reduction relative to their 67B MHA model, at roughly the cache cost of GQA with 2.25 groups.
- Used in DeepSeek-V2 and V3 and in Kimi K2. Kimi K3 pairs Gated MLA with linear-attention layers rather than using it throughout: 69 KDA layers to 24 Gated MLA across 93 layers, per the [model card](https://huggingface.co/moonshotai/Kimi-K3).
- MLA is more involved to train and serve than GQA. The compression does not require large scale, though: DeepSeek-V2-Lite applies it at 15.7B total parameters.

### Global vs. Local Attention
- Global Attention attends to all unmasked positions in the sequence (standard approach). It helps maintain long-range dependencies that local attention might miss.
- Local Attention attends only to a window of $w$ positions around the current position. It reduces computational complexity from $O(n^2)$ to $O(nw)$, which is linear in $n$ when $w$ is held fixed.
- Architectures like Longformer and BigBird use hybrid approaches combining both: local attention for most tokens, augmented with some form of global attention (specific tokens attending globally, or sparse global attention patterns) to retain the ability to capture long-range dependencies where needed.

### Linear and hybrid attention
- Kernelized linear attention replaces the softmax similarity with a decomposable feature map $\phi$. The product can then be re-associated to accumulate $\phi(K)^T V$ first rather than forming the $n \times n$ attention matrix (the normalizing denominator re-associates the same way), making cost linear in sequence length ([Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)). The name now covers a wider set of recurrent, gated, and delta-rule formulations beyond this kernel construction.
- Re-association means the model carries a fixed-size state rather than a KV cache that grows with the sequence, so per-token decoding cost is constant. This is the trade-off an [[RNN]] makes: bounded state, but the state is lossy.
- The weak point is exact retrieval, since recovering a specific token from far back in the context is what a fixed-size state compresses away.
- The delta-rule line addresses this through the state update rule. DeltaNet partially replaces the value stored at the current key instead of adding to it. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) adds a decay gate so stale entries fade. Kimi Delta Attention (KDA) makes that gate channel-wise, so one feature of the state can be held while another decays ([Kimi Linear](https://arxiv.org/abs/2510.26692), scaled up in [Kimi K3](https://arxiv.org/abs/2607.24653)).
- Hybrid stacks interleave a small number of full-attention layers among many linear ones, recovering much of the retrieval quality at a fraction of the KV cache. A [systematic comparison](https://arxiv.org/abs/2507.06457) recommends linear-to-full ratios between 3:1 and 6:1 as the efficiency compromise: recall keeps improving as the ratio drops below 3:1 and more full-attention layers are added, but the KV cache savings shrink with it.
- Linear-attention hybrids include MiniMax-01 (lightning attention), Qwen3-Next (Gated DeltaNet), and Kimi K3 (KDA with Gated MLA). Jamba is a related attention-SSM hybrid that interleaves Transformer and Mamba layers.

### Multi-token attention
- Addresses limitations of single-token attention, where each raw attention logit is determined by the similarity of just one query-key pair
- Applies convolution operations over queries, keys, and heads so that neighboring positions and other heads influence those logits before normalization ([Golovneva et al., 2025](https://arxiv.org/abs/2504.00927))


> [!example]- Scaled dot-product attention code
> ```python
> import torch
> import torch.nn.functional as F
> 
> def scaled_dot_product_attention(Q, K, V, mask=None):
>     """
>     Q: (batch_size, query_len, d_k)
>     K: (batch_size, key_len, d_k)
>     V: (batch_size, key_len, d_v)
>     mask: broadcastable to (batch_size, query_len, key_len);
>           0 marks positions that must not be attended to
>     """
>     d_k = Q.size(-1)  # dimensionality
>     scores = torch.matmul(Q, K.transpose(-2, -1)) # (batch_size, seq_len, seq_len)
>     scores = scores / (d_k ** 0.5) # scale
> 
>     if mask is not None:
>         scores = scores.masked_fill(mask == 0, float('-inf'))
> 
>     # softmax along the last dimension
>     attn_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
> 
>     # multiply by values
>     output = torch.matmul(attn_weights, V) # (batch_size, seq_len, dim)
> 
>     return output, attn_weights
> ```


> [!example]- Multi-head self-attention code
> ```python
> import torch
> from torch import nn
> 
> class SelfAttention(nn.Module):
>     """
>     Multi-head attention. Passing the same tensor as q, k, and v gives
>     self-attention; passing a different source for k and v gives cross-attention.
>     """
>     def __init__(self, embed_size, heads):
>         super(SelfAttention, self).__init__()
>         self.embed_size = embed_size
>         self.heads = heads
>         self.head_dim = embed_size // heads
> 
>         assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
> 
>         # Linear projections
>         self.q_linear = nn.Linear(embed_size, embed_size)
>         self.k_linear = nn.Linear(embed_size, embed_size)
>         self.v_linear = nn.Linear(embed_size, embed_size)
>         self.out_linear = nn.Linear(embed_size, embed_size)
> 
>     def forward(self, q, k, v, mask=None):
>         """
>         q, k, v: (batch_size, seq_len, embed_size)
>         mask: (seq_len, seq_len) or (batch_size, seq_len, seq_len) or
>               (batch_size, 1, seq_len, seq_len); 0 marks blocked positions
>         """
>         batch_size = q.size(0)
> 
>         # Linear projections and reshape for multi-head
>         q = self.q_linear(q).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
>         k = self.k_linear(k).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
>         v = self.v_linear(v).view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
> 
>         # Compute attention scores
>         scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
> 
>         # Apply mask (if provided). scores is (batch, heads, seq_len, seq_len),
>         # so a per-batch mask needs a head axis before it will broadcast.
>         if mask is not None:
>             if mask.dim() == 3:
>                 mask = mask.unsqueeze(1)
>             scores = scores.masked_fill(mask == 0, float('-inf'))
> 
>         # Normalize scores to probabilities
>         attention_weights = torch.softmax(scores, dim=-1)
> 
>         # Compute weighted sum
>         out = torch.matmul(attention_weights, v)
> 
>         # Reshape and apply final linear projection
>         out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_size)
>         out = self.out_linear(out)
> 
>         return out, attention_weights
> ```_
