---
tags:
  - cv
  - loss
---
Loss functions commonly used in computer vision tasks. For general losses (Cross-Entropy, MSE, KL Divergence), see [[General losses]]. For CV evaluation metrics, see [[Computer vision metrics]].

## Focal Loss

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

## Dice Loss

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

## IoU (Intersection over Union)/Jaccard Loss

IoU loss is based on the IoU metric and helps directly optimize the quality of bounding box predictions.

$$L_{\text{IoU}} = 1 - \frac{\text{area of overlap}}{\text{area of union}}$$

**Applications:**

- Object detection
- Instance segmentation
- Bounding box regression

## Perceptual Loss

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

## Adversarial Loss

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

## SSIM (Structural Similarity Index) Loss

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

## Links

- [Understanding Dice Loss for Segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [GAN Loss Functions: A Comparison](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f)
