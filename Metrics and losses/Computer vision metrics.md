---
tags:
  - cv
  - evaluation
cssclasses:
  - term-table
---
Computer vision metrics evaluate models across detection, segmentation, generation, 3D, and pose estimation tasks. Most reduce to a precision/recall-style overlap measure, tailored to the prediction type (box, mask, keypoint, point cloud, image).

## When to use which metric

| Metric | When to use |
|---|---|
| IoU | Box or mask overlap with ground truth. |
| AP | Area under precision-recall for a class at a given IoU. |
| mAP | Mean AP across classes (and often across IoU thresholds). |
| Pixel Accuracy | Fraction of correctly classified pixels — dominated by large classes. |
| mIoU | Mean IoU across classes (standard semantic-segmentation metric). |
| FWIoU | mIoU weighted by class frequency. |
| Dice Coefficient | F1 equivalent for segmentation masks. |
| Mask AP | AP computed over masks instead of boxes. |
| Panoptic Quality (PQ) | Panoptic segmentation — segmentation quality × recognition quality. |
| FID | Generative image quality/diversity vs real distribution. |
| Inception Score (IS) | Generative image sharpness and diversity. |
| SSIM | Perceptual similarity via luminance/contrast/structure. |
| PSNR | Reconstruction quality (denoising, super-resolution). |
| LPIPS | Deep-feature perceptual similarity. |
| Chamfer / EMD | Point-cloud distance. |
| PCK / MPJPE / OKS | Pose-keypoint accuracy. |

## Object Detection Metrics

Object detection models predict bounding boxes around objects and classify them.

### Intersection over Union (IoU)

Measures the overlap between predicted and ground truth bounding boxes.

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

### Average Precision (AP)

Area under the Precision-Recall curve for a specific class, calculated at a specific IoU threshold.

$$\text{AP} = \int_{0}^{1} p(r) dr$$

Where $p(r)$ is the precision at recall level $r$.

### Mean Average Precision (mAP)

Mean of AP values across all object classes, often calculated at multiple IoU thresholds.

$$\text{mAP} = \frac{1}{n} \sum_{i=1}^{n} \text{AP}_i$$

## Semantic Segmentation Metrics

In semantic segmentation, each pixel belongs to one class.

### Pixel Accuracy

Proportion of correctly classified pixels among all pixels. Can be dominated by large classes (e.g., background).

$$\text{Pixel Accuracy} = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}$$

### Mean Intersection over Union (mIoU)

Average IoU across all classes.

$$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

$$\text{mIoU} = \frac{1}{n_c} \sum_{c=1}^{n_c} \text{IoU}_c$$

Where:
- $\text{TP}_c$ is the number of true positive pixels for class $c$.
- $\text{FP}_c$ is the number of false positive pixels for class $c$.
- $\text{FN}_c$ is the number of false negative pixels for class $c$.
- $n_c$ is the number of classes.

### Frequency Weighted IoU (FWIoU)

Weighted version of mIoU that accounts for class imbalance.

$$\text{FWIoU} = \frac{1}{\sum_{k=1}^{n_c} t_k} \sum_{c=1}^{n_c} t_c \cdot \text{IoU}_c$$

Where $t_c$ is the total number of pixels that truly belong to class $c$.

### Dice Coefficient

F1 equivalent for segmentation masks.

$$Dice_{class} = \frac{2 \cdot TP_{class}}{(2 \cdot TP_{class} + FP_{class} + FN_{class})} = \frac{2 \cdot IoU} {(IoU + 1)}$$

## Instance Segmentation Metrics

Instance segmentation involves both semantic segmentation and instance differentiation (separating individual objects).

### Mask AP

Average Precision calculated based on IoU between predicted and ground truth masks instead of bounding boxes.

### Panoptic Quality (PQ)

Combines recognition and segmentation quality for panoptic segmentation tasks.

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_\text{segmentation quality (SQ)} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_\text{recognition quality (RQ)}$$

Where:
- $p$ is a predicted segment.
- $g$ is a ground truth segment.
- $TP$, $FP$, $FN$ are true positives, false positives, and false negatives.

## Image Generation and Synthesis Metrics

These metrics evaluate the quality, diversity, and realism of generated images.

### Fréchet Inception Distance (FID)

Measures the distance between the distribution of features from generated images and real images, extracted using a pre-trained Inception network. Compares the mean and covariance of these feature distributions. Lower values indicate more realistic generated images.

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

Where:
- $\mu_r$ and $\mu_g$ are the mean feature representations of real and generated images.
- $\Sigma_r$ and $\Sigma_g$ are the covariance matrices of the feature representations.

### Inception Score (IS)

Measures the quality (sharpness, recognizability by a pre-trained Inception network) and diversity of generated images.

$$\text{IS} = \exp\left( \mathbb{E}_x [ \text{KL}(p(y|x) || p(y)) ] \right)$$

Where:
- $p(y|x)$ is the conditional class distribution for image $x$.
- $p(y)$ is the marginal class distribution.

### Structural Similarity Index (SSIM)

Measures perceptual difference between two images based on luminance, contrast, and structure. Ranges from −1 to 1 (or 0 to 1); 1 = perfect similarity. More consistent with human perception than PSNR/MSE.

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Where:
- $\mu_x$ and $\mu_y$ are the average pixel values.
- $\sigma_x^2$ and $\sigma_y^2$ are the variances.
- $\sigma_{xy}$ is the covariance.
- $c_1$ and $c_2$ are constants to avoid division by zero.

### Peak Signal-to-Noise Ratio (PSNR)

Measures the quality of reconstructed images in tasks like denoising or super-resolution. Ratio between the maximum possible power of a signal and the power of corrupting noise that affects its fidelity. Based on MSE.

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

Where:
- $\text{MAX}_I$ is the maximum possible pixel value.
- $\text{MSE}$ is the mean squared error between images.

### Learned Perceptual Image Patch Similarity (LPIPS)

Measures perceptual similarity using deep features from pre-trained networks (VGG, AlexNet). Aligns better with human perception than pixel-wise metrics like MSE.

## 3D Vision Metrics

Metrics for evaluating 3D reconstruction, depth estimation, and point cloud processing.

### Depth Estimation

- **Mean Absolute Error (MAE)** — average absolute difference between predicted and ground truth depths.
- **Root Mean Squared Error (RMSE)** — square root of the average squared differences.
- **Threshold Accuracy** — percentage of pixels with ratio of predicted to ground truth depth within threshold $t$ (commonly $t \in {1.25, 1.25^2, 1.25^3}$).

### Point Cloud

- **Chamfer Distance** — average distance from each point in one cloud to its nearest neighbor in another.

$$\text{CD}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} ||x-y||_2^2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} ||y-x||_2^2$$

- **Earth Mover's Distance (EMD)** — minimum "cost" to transform one point cloud into another.
- **F-Score** — harmonic mean of precision and recall at a specific distance threshold.

### 3D Reconstruction

- **Volumetric IoU** — intersection over union of 3D volumes.
- **Surface-to-Surface Distance** — average distance between reconstructed and ground truth surfaces.

## Human Pose Estimation Metrics

### Percentage of Correct Keypoints (PCK)

Percentage of predicted keypoints that fall within a distance threshold of the ground truth keypoints.

### Mean Per Joint Position Error (MPJPE)

Average Euclidean distance between predicted and ground truth joint positions.

### Object Keypoint Similarity (OKS)

Similar to IoU but for keypoints — accounts for keypoint visibility and scale.

$$\text{OKS} = \frac{\sum_i \exp(-d_i^2 / (2s^2k_i^2)) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$$

Where:
- $d_i$ is the Euclidean distance between predicted and ground truth keypoint $i$.
- $s$ is the object scale.
- $k_i$ is the per-keypoint constant.
- $v_i$ is the visibility flag for keypoint $i$.

## Links

- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)
- [Scikit-image Comparison Metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
- [PyTorch Vision Metrics](https://pytorch.org/vision/stable/reference.html)
