---
tags:
  - interview
---
Q: What is the number of the parameters of convolution?
A: $(3*3*3 + 1) *3$ 

Q: We have a multiclass neural net, what should we change in it, if we want to make it multilabel?
A: Change the output activation function from softmax to sigmoid and the [[Losses|loss function]] from categorical cross-entropy to binary cross-entropy.

Q: Here is a pseudocode for a neural net. Explain how it works, point out mistakes or inefficiencies in the architecture:
```
net = Sequential() 
net.add(InputLayer([100, 100, 3])) 
net.add(Conv2D(filters=512, kernel_size=(3, 3),  
               kernel_initializer=init.zeros())) 
net.add(Conv2D(filters=128, kernel_size=(1, 1),  
               kernel_initializer=init.zeros())) 
net.add(Activation('relu')) 
net.add(Conv2D(filters=32, kernel_size=(3, 3),  
               kernel_initializer=init.zeros())) 
net.add(Conv2D(filters=32, kernel_size=(1, 1),  
               kernel_initializer=init.zeros())) 
net.add(Activation('relu')) 
net.add(MaxPool2D(pool_size=(6, 6))) 
net.add(Conv2D(filters=8, kernel_size=(10, 10),  
               kernel_initializer=init.normal())) 
net.add(Activation('relu')) 
net.add(Conv2D(filters=8, kernel_size=(10, 10),  
               kernel_initializer=init.normal())) 
net.add(Activation('relu')) 
net.add(MaxPool2D(pool_size=(3, 3))) 
net.add(Flatten()) # convert 3d tensor to a vector of features 
net.add(Dense(units=512)) 
net.add(Activation('softmax')) 
net.add(Dropout(rate=0.5)) 
net.add(Dense(units=512)) 
net.add(Activation('softmax')) 
net.add(Dense(units=10)) 
net.add(Activation('sigmoid')) 
net.add(Dropout(rate=0.5))
```
A:
* Initializing weights to zero is generally a bad practice as it can lead to symmetric weights and prevent the network from learning effectively.
* The first layer has 512 filters immediately, which might be too much for the initial layer. The drop from 512 to 128 to 32 filters can be too drastic.
* ReLU activations are missing after some convolutional layers, which could limit the network's ability to learn non-linear patterns.
* The first MaxPool2D layer has a large pool size of (6,6), which might reduce spatial information too aggressively.
* The 10x10 kernel size in later convolutional layers is too large.
* Using softmax activation for intermediate dense layers is unusual.
* Dropout is applied after the activation functions, which is less common. It's usually applied before the activation.
* It could be a good idea to have pooling after each CNN layer
* We could optionally replace the last max pooling with adaptive pooling so that we could work with images of varied size
* 3 dense layers at the end is too much, usually 1 or 2 are used.

Q: We have a linear layer with 30 neurons. How can we get/hack the weights if we don't have a direct access to them?
A: When we input an identity matrix to a linear layer, the output will effectively be the weight matrix itself.
```python
# Create identity matrix
identity = torch.eye(num_inputs)
# Pass through the layer
output = linear_layer(identity)

# To get biases, we can pass a zero vector
zero_input = torch.zeros((1, num_inputs))
bias_output = linear_layer(zero_input)
```

Q: The same with 3x3x3 convolution.
A: For each position in the kernel and each input channel, create an input where only one pixel is 1 and the rest are 0. This is a pseudocode:
```python

def extract_cnn_weights(cnn_layer, input_shape=(32, 32, 3), kernel_size=(3, 3), num_filters=3):
    height, width, in_channels = input_shape
    k_h, k_w = kernel_size

    weights = []
    for h in range(height):
        for w in range(width):
            for c in range(in_channels):
                input_tensor = np.zeros((1, height, width, in_channels))
                input_tensor[0, h, w, c] = 1
                output = cnn_layer(input_tensor)
                weights.append(output[0, 0, 0, :])

    return np.array(weights).reshape(k_h, k_w, in_channels, num_filters)

def extract_cnn_biases(cnn_layer, input_shape=(32, 32, 3)):
    zero_input = np.zeros((1,) + input_shape)
    return cnn_layer(zero_input)[0, 0, 0, :]

```

Q: We have a CNN model pre-trained on a certain domain. We want to fine-tune it on a different domain. At the beginning of the fine-tuning, the loss is huge. What to do?
A: Lower learning rate, freeze earlier layers and unfreeze gradually, use gradient clipping/label smoothing/gradient norm.

Q: We have a stream of 50 frames of the same person. We want to make a prediction with a face recognition model, but we have to select a single frame for it. What could be the criteria for it?
A: Select frame with best face visibility/image quality. Possibly train a small model to predict the best suitable frame.

Q: Write a basic PyTorch code for implementing sampling for a simple LLM decoder. Implement the following variations: greedy, top-k, top-p.
A: Inspired by https://medium.com/@pashashaik/natural-language-generation-from-scratch-in-large-language-models-with-pytorch-4d9379635316
> [!info]- Code
>```python
>import torch
>import torch.nn as nn
>import torch.nn.functional as F
>from einops import rearrange
>
>class SimpleDecoder(nn.Module):
>    def __init__(self):
>        super().__init__()
>        # Assume the model architecture is already defined here
>
>    def forward(self, input_ids):
>        # Assume this method is already implemented
>        # It should return logits of shape (batch_size, sequence_length, vocab_size)
>        pass
>
>	def greedy_search(self, input_ids, max_length, tokenizer):  
>	    with torch.inference_mode():  
>	        for _ in range(max_length):  
>	            outputs = self(input_ids) # (batch_size, sequence_length, vocab_size)
>	            next_token_logits = outputs[:, -1, :] # (batch_size, vocab_size) logits for the next token for each sequence in the batch.
>
>	            next_token = torch.argmax(next_token_logits, dim=-1)  
>
>	            if next_token.item() == self.eos_token_id:  
>	                break  
>				# the following two lines give the same results, select the one which syntax you prefer
>	            # input_ids = torch.cat([input_ids, rearrange(next_token, 'c -> 1 c')], dim=-1)
>	            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1), dim=-1)  
>
>	        generated_text = tokenizer.decode(input_ids[0])  
>
>	    return generated_text
>
>	def top_k_sampling(self, input_ids, max_length, tokenizer, k=50, temperature=1.0):  
>	    with torch.inference_mode():  
>	        for _ in range(max_length):  
>	            outputs = self(input_ids)  
>	            next_token_logits = outputs[:, -1, :] / temperature  
>
>	            # Get top k tokens  
>	            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)  
>
>	            # Apply softmax to convert logits to probabilities  
>	            probs = F.softmax(top_k_logits, dim=-1)  
>
>	            # Sample from the top k  
>	            next_token_index = torch.multinomial(probs, num_samples=1)  
>
>	            # Convert back to vocabulary space  
>	            next_token = torch.gather(top_k_indices, -1, next_token_index)  
>
>	            # Check for EOS token  
>	            if next_token.item() == self.eos_token_id:  
>	                break  
>
>	            # Concatenate the next token to the input  
>	            input_ids = torch.cat([input_ids, next_token], dim=-1)  
>
>	        generated_text = tokenizer.decode(input_ids[0])  
>
>	    return generated_text
>
>
>def top_p_sampling(self, input_ids, max_length, tokenizer, p=0.9, temperature=1.0): 
>	    """
>	    Top-p (nucleus sampling) controls the cumulative probability of the generated tokens.
>	    """ 
>    with torch.inference_mode():  
>        for _ in range(max_length):  
>            outputs = self(input_ids)  
>            next_token_logits = outputs[:, -1, :] / temperature  
>
>            # Sort logits in descending order  
>            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)  
>            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  
>
>            # Remove tokens with cumulative probability above the threshold  
>            sorted_indices_to_remove = cumulative_probs > p  
>            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  
>            sorted_indices_to_remove[..., 0] = 0  
>
>            # Scatter sorted tensors to original indexing  
>            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)  
>            next_token_logits[indices_to_remove] = float('-inf')  
>
>            # Sample from the filtered distribution  
>            probs = F.softmax(next_token_logits, dim=-1)  
>            next_token = torch.multinomial(probs, num_samples=1)  
>
>            # Check for EOS token  
>            if next_token.item() == self.eos_token_id:  
>                break  
>
>            # Concatenate the next token to the input  
>            input_ids = torch.cat([input_ids, next_token], dim=-1)  
>
>        generated_text = tokenizer.decode(input_ids[0])  
>
>    return generated_text
>
>def beam_search(self, input_ids, max_length, tokenizer, beam_size=5):  
>    beam_scores = torch.zeros(beam_size, device=device)  
>    beam_sequences = input_ids.repeat(beam_size, 1)  
>    active_beams = torch.ones(beam_size, dtype=torch.bool, device=device)  
>
>    for _ in range(max_length):  
>        with torch.no_grad():  
>            outputs = self(beam_sequences)  
>            next_token_logits = outputs[:, -1, :]  
>
>            # Calculate log probabilities  
>            log_probs = F.softmax(next_token_logits, dim=-1)  
>
>            # Calculate scores for all possible next tokens  
>            vocab_size = log_probs.size(-1)  
>            next_scores = beam_scores.unsqueeze(-1) + log_probs  
>            next_scores = next_scores.view(-1)  
>
>            # Select top-k best scores  
>            top_scores, top_indices = next_scores.topk(beam_size, sorted=True)  
>
>            # Convert flat indices to beam indices and token indices  
>            beam_indices = top_indices // vocab_size  
>            token_indices = top_indices % vocab_size  
>
>            # Update sequences  
>            beam_sequences = torch.cat([  
>                beam_sequences[beam_indices],  
>                token_indices.unsqueeze(-1)  
>            ], dim=-1)  
>
>            # Update scores  
>            beam_scores = top_scores  
>
>            # Update active beams  
>            active_beams = token_indices != self.eos_token_id  
>
>            if not active_beams.any():  
>                break  
>
>    # Select the sequence with the highest score  
>    best_sequence = beam_sequences[beam_scores.argmax()]  
>    generated_text = tokenizer.decode(best_sequence)  
>
>    return generated_text
>```

Q: What are the trade-offs between using weights in Cross-Entropy Loss vs using Focal Loss?
A: The weights in Cross-Entropy Loss weight classes statically based on the number of samples in them, this approach focuses on minority classes. Focal loss focuses on hard-to-classify examples and requires tuning a hyperparameter.