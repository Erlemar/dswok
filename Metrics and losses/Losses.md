---
tags:
  - loss
---
Loss functions (also called objective functions or cost functions) are mathematical measures of the error between predicted and actual values. They quantify how well a model is performing and provide the optimization signal for training.

## General Loss Functions
### 1. Cross-Entropy Loss

Cross-entropy loss (or log loss) measures the performance of a classification model whose output is a probability value between 0 and 1.

**Binary Cross-Entropy**

$$L_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

**Categorical Cross-Entropy** is used for multi-class problems, where each sample belongs to a single class.

$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Where:

- $y$ is the ground truth label
- $\hat{y}$ is the predicted probability
- $N$ is the number of samples
- $C$ is the number of classes

- **Label Smoothing Cross-Entropy**: Helps prevent overconfidence by replacing one-hot encoded ground truth with a mixture of the original labels and a uniform distribution.

$$\tilde{y}_{i,c} = (1 - \alpha) \cdot y_{i,c} + \alpha \cdot \frac{1}{C}$$

### 2. Mean Squared Error (MSE)

MSE measures the average of the squares of the errors between predicted and actual values.

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Variants:**

- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a measure in the same units as the target variable.
- **Mean Absolute Error (MAE)**: Uses absolute differences instead of squared differences, making it less sensitive to outliers.

$$L_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### 3. Kullback-Leibler Divergence (KL Divergence)

KL divergence measures how one probability distribution diverges from a second, expected probability distribution.

$$L_{\text{KL}} = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)$$

Where:

- $P$ is the true distribution
- $Q$ is the approximated distribution

**Applications:**

- Variational autoencoders (VAEs)
- Knowledge distillation
- Distribution matching

**Variants:**

- **Reverse KL Divergence**: Swaps the order of the distributions, yielding different behavior.
- **Jensen-Shannon Divergence**: A symmetrized and smoothed version of KL divergence.

## NLP-Specific Loss Functions

### 1. Negative Log-Likelihood (NLL) Loss

NLL loss is commonly used in language modeling and sequence prediction.

$$L_{\text{NLL}} = -\frac{1}{N} \sum_{i=1}^{N} \log(p(y_i | x_i))$$

Where:

- $p(y_i | x_i)$ is the predicted probability of the true token/class

**Applications:**

- Language modeling
- Machine translation
- Text generation
- Sequence prediction

### 2. Perplexity

Perplexity is an exponential transformation of the average negative log-likelihood, making it interpretable as the weighted average number of choices the model is uncertain about. Perplexity is an evaluation metric, but minimizing NLL is equivalent to minimizing perplexity.

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log p(y_i | y_{<i})\right)$$

Where:

- $p(y_i | y_{<i})$ is the probability of the $i$-th token given previous tokens

**Applications:**

- Language modeling
- Text generation evaluation
- Speech recognition

### 3. Connectionist Temporal Classification (CTC) Loss

CTC loss aligns sequence-to-sequence data without requiring pre-segmented training data or explicit alignments.

$$L_{\text{CTC}} = -\log\left(\sum_{\pi \in \mathcal{A}^{-1}(y)} \prod_{t=1}^{T} p(\pi_t | x)\right)$$

Where:

- $\mathcal{A}^{-1}(y)$ is the set of all possible alignments that correspond to the target sequence $y$
- $p(\pi_t | x)$ is the probability of alignment $\pi$ at time $t$ given input $x$

**Applications:**

- Speech recognition
- Handwriting recognition
- Protein sequence alignment

### 4. Triplet Loss

Triplet loss learns embeddings where similar items are closer together and dissimilar items are farther apart.

$$L_{\text{triplet}} = \max(d(a, p) - d(a, n) + \text{margin}, 0)$$

Where:

- $a$ is the anchor sample
- $p$ is a positive sample similar to the anchor
- $n$ is a negative sample dissimilar to the anchor
- $d$ is a distance function (typically Euclidean or cosine)
- margin is a hyperparameter

**Applications:**

- Sentence embeddings
- Document similarity
- Face recognition
- Image retrieval

### 5. Contrastive Loss

Contrastive loss is used to learn discriminative features by pushing similar samples closer and dissimilar samples further apart.

$$L_{\text{contrastive}} = (1-Y) \cdot \frac{1}{2} \cdot D^2 + Y \cdot \frac{1}{2} \cdot \max(0, \text{margin} - D)^2$$

Where:

- $Y$ is 0 for dissimilar pairs and 1 for similar pairs
- $D$ is the distance between samples

**Applications:**

- Sentence similarity
- Learning text embeddings
- Siamese networks for document comparison

### 6. Reinforcement Learning from Human Feedback (RLHF) Losses

#### A. PPO (Proximal Policy Optimization) Loss

$$L_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

Where:

- $r_t(\theta)$ is the ratio of new policy probability to old policy probability
- $A_t$ is the advantage estimate
- $\epsilon$ is a hyperparameter that constrains policy updates

#### B. Direct Preference Optimization (DPO) Loss

$$L_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

Where:

- $\pi_\theta$ is the policy being trained
- $\pi_{\text{ref}}$ is the reference policy
- $(x,y_w,y_l)$ are input, preferred output, and dispreferred output
- $\beta$ is a hyperparameter

**Applications:**

- Fine-tuning language models based on human preferences
- Aligning large language models with human values
- Improving language model outputs for specific criteria

## Computer Vision-Specific Loss Functions

### 1. Focal Loss

Focal loss addresses class imbalance by down-weighting the contribution of easy examples.

$$L_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:

- $p_t$ is the probability of the correct class
- $\alpha_t$ is a balancing factor
- $\gamma$ is a focusing parameter

**Applications:**

- Object detection (RetinaNet)
- Segmentation with imbalanced classes
- Medical image analysis

### 2. Dice Loss

Dice loss is based on the Dice coefficient, which measures the overlap between predicted and ground truth segmentation.

$$L_{\text{Dice}} = 1 - \frac{2 \sum_{i}^{N} p_i g_i}{\sum_{i}^{N} p_i^2 + \sum_{i}^{N} g_i^2}$$

Where:

- $p_i$ is the predicted probability
- $g_i$ is the ground truth binary mask

**Applications:**

- Medical image segmentation
- Semantic segmentation
- Instance segmentation

**Variants:**

- **Tversky Loss**: Generalization of Dice loss that allows for tuning precision and recall.
- **Combo Loss**: Combination of Dice loss and weighted cross-entropy.

### 3. IoU (Intersection over Union)/Jaccad Loss

IoU loss is based on the IoU metric and helps directly optimize the quality of bounding box predictions.

$$L_{\text{IoU}} = 1 - \frac{\text{area of overlap}}{\text{area of union}}$$

**Applications:**

- Object detection
- Instance segmentation
- Bounding box regression

### 4. Perceptual Loss

Perceptual loss compares high-level feature representations extracted by a pre-trained CNN instead of pixel-wise differences.

$$L_{\text{perceptual}} = \sum_{j} \lambda_j \frac{1}{C_j H_j W_j} \sum_{c,h,w} (\Phi_j(I)_{c,h,w} - \Phi_j(\hat{I})_{c,h,w})^2$$

Where:

- $\Phi_j$ is the feature map from the $j$-th layer of a pre-trained network
- $I$ is the ground truth image
- $\hat{I}$ is the generated image
- $C_j, H_j, W_j$ are the dimensions of the feature map

**Applications:**

- Super-resolution
- Style transfer
- Image-to-image translation
- Image generation

### 5. Adversarial Loss

Adversarial loss comes from Generative Adversarial Networks (GANs) and involves a minimax game between a generator and discriminator.

$$L_{\text{adv}} = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Where:

- $D$ is the discriminator
- $G$ is the generator
- $p_{\text{data}}$ is the real data distribution
- $p_z$ is the noise distribution

**Applications:**

- Image generation
- Image-to-image translation
- Domain adaptation
- Text-to-image generation

**Variants:**

- **WGAN Loss**: Uses Wasserstein distance to provide more stable gradients.
- **LSGAN Loss**: Uses least squares instead of log-likelihood for more stable training.
- **Hinge Loss**: Alternative formulation that has shown good results for image generation.

### 6. SSIM (Structural Similarity Index) Loss

SSIM loss measures the structural similarity between images, focusing on structural information, luminance, and contrast.

$$L_{\text{SSIM}} = 1 - \text{SSIM}(x, y)$$

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Where:

- $\mu_x$, $\mu_y$ are the average pixel values
- $\sigma_x^2$, $\sigma_y^2$ are the variances
- $\sigma_{xy}$ is the covariance
- $C_1$, $C_2$ are constants to avoid division by zero

**Applications:**

- Image restoration
- Super-resolution
- Image compression
- Image quality assessment

## Regularization Losses

These are typically added to the main loss function to prevent overfitting and improve generalization.

### 1. L1 Regularization (Lasso)

$$L_{\text{L1}} = \lambda \sum_{i=1}^{n} |w_i|$$

**Properties:**

- Encourages sparse weights (many weights become exactly zero)
- Less sensitive to outliers than L2
- Can be used for feature selection

### 2. L2 Regularization (Ridge)

$$L_{\text{L2}} = \lambda \sum_{i=1}^{n} w_i^2$$

**Properties:**

- Penalizes large weights more heavily
- Rarely sets weights to exactly zero
- More stable solutions than L1 regularization

### 3. Elastic Net

$$L_{\text{ElasticNet}} = \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$

**Properties:**

- Combines L1 and L2 regularization
- Can select groups of correlated features
- More robust than either L1 or L2 alone

## Links

- [PyTorch Loss Functions Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [TensorFlow Loss Functions Guide](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
- [Understanding Dice Loss for Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
- [A Gentle Introduction to Cross-Entropy for Machine Learning](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [GAN Loss Functions: A Comparison](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)