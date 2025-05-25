---
aliases:
  - CLIP
tags:
  - nlp
  - cv
  - multimodal
  - embeddings
  - contrastive
---
CLIP (Contrastive Language-Image Pre-training) is a neural net model developed by OpenAI that efficiently learns visual concepts from natural language supervision. It's a dual-encoder model that jointly trains an image encoder and a text encoder to produce similar embeddings for corresponding pairs. It has good zero-shot learning capabilities, allowing it to work for many tasks without being explicitly trained for them.

### Architecture

* Image Encoder: Vision Transformer (ViT) or ResNet takes images and outputs fixed-dimensional image embeddings (512 or 768 dimensionality)
* Text Encoder: Transformer processes text and outputs fixed-dimensional text embeddings. Maximum context length is 77 tokens
* Both encoders project their outputs into a shared embedding space, embeddings are L2-normalized to unit length, similarity is computed using dot product (equivalent to cosine similarity for normalized vectors)

### Training

CLIP uses contrastive learning on large-scale image-text pairs:
- A batch contains N image-text pairs
- Both images and texts are encoded into the shared embedding space
- Compute similarity matrix between all N×N possible image-text combinations
- Maximize similarity for correct pairs (diagonal) and minimize for incorrect pairs (off-diagonal) using contrastive loss

### Zero-shot classification

- For a given image classification task create descriptive text prompts for each class.
- Pass the input image through CLIP's Image Encoder to get its [[Word Embeddings]].
- Pass each class prompt through CLIP's Text Encoder to get text embeddings for each class.
- Compute the cosine similarity between the image embedding and each class prompt embedding.
- The class corresponding to the prompt with the highest similarity to the image embedding is chosen as the prediction.

### Fine-tuning CLIP
- Full fine-tuning - unfreeze and update all parameters of both encoders. Requires significant computational resources, but provides the best performance.
- Linear Probing - freeze both encoders and train only a linear classifier on top. Works well, but worse than full fine-tuning.
- Parameter-Efficient Fine-tuning: [[LoRA]], Adapter layers, prompt/prefix tuning.
- Few-short learning with task-specific prompts

### Practical considerations
- Handling text with more than 77 tokens: truncation, chunking and averaging/pooling embeddings, hierarchical encoding (use a model to process the sequence of embeddings), sliding window with attention, compress the text.

### Links

- [Original CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub Repository](https://github.com/openai/CLIP)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)