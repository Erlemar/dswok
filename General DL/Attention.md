---
tags:
  - architecture
  - attention
  - nlp
---
[The original paper](https://arxiv.org/abs/1706.03762)

Attention is a mechanism that lets neural networks focus on specific parts of an input sequence. This mechanism has been explored in various architectures like [[Hyena Hierarchy. Towards Larger Convolutional Language Models|Hyena]] and [[Linformer. Self-Attention with Linear Complexity|Linformer]] for efficiency improvements. Attention is also widely used in computer vision, as seen in [[Swin Transformer V2. Scaling Up Capacity and Resolution|Swin Transformers]] and object detection models like [[End-to-End Object Detection with Transformers|DETR]]. 

A fundamental type is Scaled Dot-Product Attention (used in [[Transformer]]). It has three inputs:
- Query (Q): The current token trying to gather information.
- Key (K): A representation of each token in the sequence that’s available to be attended to.
- Value (V): What each token provides if selected by the attention mechanism.

Attention calculation step-by-step:
1. We measure how relevant each key $K_i$ is to our query $Q$ using a dot product: $\text{scores} = Q \times K^T$
2. To keep the values stable for large embeddings, we divide by $\sqrt{d_k}$, where $d_k$ is the dimensionality of the key vectors: $\text{scaled\_scores} = \frac{Q \times K^T}{\sqrt{d_k}}$. When the $d_k$ is large, the dot product can grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. For dot products, the variance grows with $d_k$. That's why the square root is used instead - we normalize the scale.
3. Convert the scores into a probability distribution to see how much attention should be given to each element: $\alpha = \text{softmax}(\text{scaled\_scores})$.
4. Multiply each value $V_i$ by its attention weight $alpha_i$ and sum to get the final output: $\text{Attention}(Q, K, V) = \alpha \times V$
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
### Multi-Head Attention
- Runs multiple attention mechanisms in parallel
- Allows the model to jointly attend to information from different representation subspaces at different positions. Each head can potentially learn to focus on different types of relationships or features.
### MultiQuery Attention (MQA)
- All query heads share the same key and value matrices, only query matrices are different
- Significantly reduces memory requirements and inference time
- Can lead to quality degradation compared to MHA
### Grouped-Query Attention (GQA)
- Introduced to balance the efficiency of MQA and the quality in MHA
- In MHA, each query head has its own key-value heads (maximum quality but high memory usage). In MQA, all query heads share just one key-value head (maximum efficiency but lower quality). GQA divides query heads into groups, where each group shares a set of key-value heads.
- The number of groups (G) is a hyperparameter - more groups is closer to MHA, fewer is closer to MQA
- Used in models like Llama 2-70B, Mistral 7B, and Falcon 40B. Particularly useful in multi-GPU environments with tensor parallelism
### Global vs. Local Attention
- Global Attention attends to all positions in the sequence (standard approach). It helps maintain long-range dependencies that local attention might miss.
- Local Attention attends only to a window of positions around the current position. It reduces computational complexity from O(n²) to O(n).
- Architectures like Longformer and BigBird use hybrid approaches combining both: local attention for most tokens, augmented with some form of global attention (specific tokens attending globally, or sparse global attention patterns) to retain the ability to capture long-range dependencies where needed.

### Multi-token attention
- Addresses limitations of single-token attention where individual weights are determined by similarity of just one query-key pair
- Applies convolution operations over queries, keys, and heads to allow neighboring tokens to influence each other's attention weights    


> [!example]- Scaled dot attention code
> ```python
> import torch
> import torch.nn.functional as F
> 
> def scaled_dot_product_attention(Q, K, V, mask=None):
>     """
>     Q, K, V: (batch_size, seq_len, dim)
>     mask (batch_size, seq_len, seq_len) to prevent attention on certain positions
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
> class SelfAttention(nn.Module):
>     """
>     This class implements multi-head self-attention.
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
>         # Apply mask (if provided)
>         if mask is not None:
>             scores = scores.masked_fill(mask == 0, -1e9)
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
> ```
