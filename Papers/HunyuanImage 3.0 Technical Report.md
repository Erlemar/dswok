---
tags:
- llm
- imagegeneration
---
[Paper](https://arxiv.org/abs/2509.23951)

[Code](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)

![Main image](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_19-00-26.jpg)

HunyuanImage 3.0 is a new multimodal, autoregressive model that handles both multimodal understanding and generation. Its performance comes from several major components: carefully curated training data, a modern architecture with a Mixture-of-Experts design (~80B total parameters, 13B active per token), a built-in Chain-of-Thought approach, staged pre-training, heavy post-training, and efficient large-scale infrastructure. Experiments show it reaches or matches SOTA levels in text-image alignment and visual quality.

### Data Preparation

#### Data Filtering

The authors take an initial pool of 10+ billion images, then apply a strict three-stage filtering pipeline that keeps under 45% of the data.

Stage 1 – Basic quality cleanup: Removed low-resolution images (<512px), broken files, poorly exposed or over-saturated images, and exact duplicates.

Stage 2 – Primary data curation:
* Objective filtering for watermarks, logos, excessive text (through OCR), collages, borders, and AI-generated content. AIGC sources were aggressively filtered to avoid distribution shifts. 
* Subject-scoring models for clarity (sharpness, noise, dynamic range) and aesthetics, with criteria defined by artists (color, light/shadow, composition). Images below thresholds were removed.

Stage 3 – Embedding-based deduplication & enrichment:
Deduplicated clusters and added specialized datasets (knowledge-augmented, stylized, text-heavy, graphic design) to boost semantic diversity.

Additionally, the authors build a 100M+ multi-image dataset (image pairs and multi-image samples) for learning interleaved relationships.

#### Image Captioning

![Image Captioning Pipeline](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-00-36.jpg)

The authors propose a pipeline for producing rich and accurate image descriptions using three ideas:
* a hierarchical bilingual schema that breaks each description into levels of detail, stylistic attributes, and factual entities;
* a compositional synthesis method that mixes different parts of the schema to create diverse captions of varying length;
* specialized agents for OCR and named-entity recognition, combined with a bidirectional verification step to ensure factual correctness.

For paired images, an additional model generates detailed captions explaining the differences between them, using both images, their captions, and supporting video frames.

#### Reasoning Dataset Construction

In order to enable an internal Chain-of-Thought reasoning process for image generation, the model is trained on two kinds of data. First, text-to-text reasoning data collected from a wide variety of real image-generation prompts strengthens instruction following, interpretation, and step-by-step textual reasoning. Second, a text-to-text&image dataset pairs high-quality images with captions and explicit reasoning traces, teaching the model to translate abstract intent into detailed visual specifications.

### Model Design

![The model](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-22-04.jpg)

The model is built on the Hunyuan-A13B decoder-only architecture, a mixture-of-experts system with 64 experts, 8 active per token, giving about 13B active parameters and a good balance of power and efficiency. Text is processed with the Hunyuan tokenizer extended with new special tokens for image tasks.

For image generation, the system uses a VAE that maps images into a 32-dimensional latent space with 16x downsampling, which the authors found simpler and higher-quality than the usual 8x VAE combined with patchification. For image-conditioned tasks, latent features from the VAE are concatenated with features from a vision encoder, creating a unified multimodal sequence that supports both image understanding and image generation without switching pipelines.

Two projector modules align these features to the transformer's latent space: VAE latents go through a timestep-modulated residual block, while vision-encoder features pass through a two-layer MLP. Timestep embeddings are injected into the sequence to improve conditioning during the diffusion process.

### Generalized Causal Attention

![Attention](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-26-27.jpg)

The authors combine causal [[attention]] for text with full attention for images through a Generalized Causal Attention scheme. Text tokens only look to earlier multimodal tokens, preserving autoregressive generation, while image tokens can look both to previous tokens and to all other tokens within the same image to capture global spatial structure.

When training sequences contain zero or one generated image, the mask follows this pattern directly. When several generated images appear in the same sequence, earlier generated-image regions are masked out so later tokens cannot attend to them, creating intentional gaps in the attention mask.

During inference, only one generated image appears at a time, because once it is produced, it becomes a conditional input. The model, therefore, always uses the standard Generalized Causal Attention structure, maintaining autoregressive behavior while still enabling coherent multimodal reasoning.

### Position Embedding

![RoPE](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-30-22.jpg)

The model extends standard 1D rotary position embeddings into a generalized 2D version so image tokens can receive spatially meaningful coordinates while keeping full compatibility with the pretrained language model. Text tokens continue using 1D RoPE, which is equivalent to treating them as lying on a diagonal in 2D space, while image tokens use separate x- and y-based rotations.

Because training sequences with multiple generated images place tokens in positions that differ from inference, token positions are shifted during training so they match the positions used at inference time.

### Model Training

![Model Training](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-33-15.jpg)

#### Pre-training

The training setup supports many multimodal tasks: text-to-image generation, language modeling, multimodal understanding, interleaved text–image modeling, and Chain-of-Thought reasoning. It is organized into four stages that move from coarse, low-resolution data to smaller, higher-quality datasets.

First, the transformer is trained (with the ViT frozen) on huge amounts of mixed text and image data at low resolution to align text and image representations. Next, the transformer is frozen and the ViT plus its aligner are fine-tuned on multimodal-understanding data to strengthen visual comprehension. In the third stage, both components are trained together on higher-resolution images and interleaved text–image tasks, improving multimodal modeling and editing capabilities. The final stage uses only high-resolution images, adds the reasoning data used for Chain-of-Thought image generation, and trains reasoning tokens autoregressively.

After these stages, an additional instruction-tuning phase formats T2I, language-modeling, and reasoning data into instruction templates and further fine-tunes the model specifically for text-to-image generation.

#### Post-training

The model is refined through several post-training stages that progressively improve image quality, alignment, and realism. It begins with supervised fine-tuning on a carefully curated, high-quality dataset covering a wide range of categories. Next, DPO is applied using pairs of good and bad model outputs to reduce structural distortions. MixGRPO then performs large-scale online reinforcement learning with multiple reward models to improve style, composition, lighting, and overall text–image alignment while reducing artifacts.

SRPO further boosts realism by injecting noise into latent features and optimizing the early part of the denoising trajectory using differentiable reward signals, helping fix issues like oversaturation, inconsistent lighting, and poor textures. Finally, a new method called ReDA aligns the model's output distribution with a high-reward set of diverse, high-quality images.

### Results

![SSAE](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-48-44.jpg)

Standard T2I benchmarks like T2ICompBench and GenEval are limited in comprehensiveness and reliability because their prompts are too simple and their automated metrics often disagree with human judgment, especially on relational accuracy and attribute binding. To overcome these weaknesses, the authors introduce SSAE, a new evaluation method built on structured semantic alignment.

They gather 500 diverse prompts, extract 3500 detailed semantic points using an LLM parser, organize these points into 12 fine-grained categories, and refine them with another LLM and human checking. These fixed points serve as stable references for evaluating any model.An advanced multimodal LLM with Chain-of-Thought reasoning then compares each generated image against its prompt’s semantic points using 0–1 matching, producing field-level accuracy as well as two global metrics: Mean Image Accuracy and Global Accuracy.

Compared with newer human-like benchmarks such as DreamBench++, this approach offers a more thorough semantic taxonomy and more flexible evaluation modes. Under SSAE, HunyuanImage 3.0 performs on par with top existing models across all categories.

### GSB

![GSB](https://andlukyane.com/images/paper_reviews/hunyuanimage3/2025-11-16_18-51-12.jpg)

The model is evaluated using the GSB (Good/Same/Bad) method, where 1000 balanced prompts are created and each model generates one image per prompt with no cherry-picking. Over 100 professional evaluators compare pairs of results. HunyuanImage 3.0 shows a 14.10% win rate over HunyuanImage 2.1, making it the strongest open-source model so far. It also achieves small but consistent win rates over Seedream 4.0, Nano Banana, and GPT-Image, indicating that its image quality is now competitive with leading closed-source commercial systems.