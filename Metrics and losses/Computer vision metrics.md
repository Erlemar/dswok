---
tags:
  - cv
  - metric
  - evaluation
---
### Object Detection Metrics

Object detection models predict bounding boxes around objects and classify them.

1. **Intersection over Union (IoU)**: Measures the overlap between predicted and ground truth bounding boxes.

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

2. **Average Precision (AP)**: The area under the Precision-Recall curve for a specific class, calculated at a specific IoU threshold.

$$\text{AP} = \int_{0}^{1} p(r) dr$$

Where $p(r)$ is the precision at recall level $r$.

3. **Mean Average Precision (mAP)**: The mean of AP values across all object classes, often calculated at multiple IoU thresholds.

$$\text{mAP} = \frac{1}{n} \sum_{i=1}^{n} \text{AP}_i$$
### Semantic Segmentation Metrics

In semantic segmentation, each pixel belongs to one class.

1. **Pixel Accuracy**: The proportion of correctly classified pixels among all pixels. Can be dominated by large classes (background).

$$\text{Pixel Accuracy} = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}$$
2. **Mean Intersection over Union (mIoU)**: The average IoU across all classes.

$$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

$$\text{mIoU} = \frac{1}{n_c} \sum_{c=1}^{n_c} \text{IoU}_c$$

Where:

- $\text{TP}_c$ is the number of true positive pixels for class $c$
- $\text{FP}_c$ is the number of false positive pixels for class $c$
- $\text{FN}_c$ is the number of false negative pixels for class $c$
- $n_c$ is the number of classes

3. **Frequency Weighted IoU (FWIoU)**: A weighted version of mIoU that accounts for class imbalance.

$$\text{FWIoU} = \frac{1}{\sum_{k=1}^{n_c} t_k} \sum_{c=1}^{n_c} t_c \cdot \text{IoU}_c$$

Where $t_c$ is the total number of pixels that truly belong to class $c$.

4. **Dice Coefficient**: (F1 Score equivalent for segmentation).
$$Dice_{class} = \frac{2 * TP_{class}}{(2 * TP_{class} + FP_{class} + FN_{class})} = \frac{2 * IoU} {(IoU + 1)}$$


## Instance Segmentation Metrics

Instance segmentation involves both semantic segmentation and instance differentiation (separating individual objects).

1. **Mask AP**: Average Precision calculated based on IoU between predicted and ground truth masks instead of bounding boxes.
    
2. **Panoptic Quality (PQ)**: Combines recognition and segmentation quality for panoptic segmentation tasks.
    

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_\text{segmentation quality (SQ)} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_\text{recognition quality (RQ)}$$

Where:

- $p$ is a predicted segment
- $g$ is a ground truth segment
- $TP$, $FP$, $FN$ are true positives, false positives, and false negatives

## Image Generation and Synthesis Metrics

These metrics evaluate the quality, diversity, and realism of generated images.

1. **Fréchet Inception Distance (FID)**: Measures the distance between the distribution of features from generated images and real images, extracted using a pre-trained Inception network. Compares mean and covariance of these feature distributions. Lower values indicate that generated images are more similar to real images in terms of deep features.

$$\text{FID} = ||\mu_r - \mu_g||^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

Where:

- $\mu_r$ and $\mu_g$ are the mean feature representations of real and generated images
- $\Sigma_r$ and $\Sigma_g$ are the covariance matrices of the feature representations

2. **Inception Score (IS)**: Measures the quality (sharpness, recognizability by a pre-trained Inception network) and diversity of generated images.

$$\text{IS} = \exp\left( \mathbb{E}_x [ \text{KL}(p(y|x) || p(y)) ] \right)$$

Where:

- $p(y|x)$ is the conditional class distribution for image $x$
- $p(y)$ is the marginal class distribution

3. **Structural Similarity Index (SSIM)**: Measures the perceptual difference between two images based on luminance, contrast, and structure. Ranges from -1 to 1 (or 0 to 1). `1` indicates perfect similarity. Aims to be more consistent with human perception than PSNR/MSE.

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Where:

- $\mu_x$ and $\mu_y$ are the average pixel values
- $\sigma_x^2$ and $\sigma_y^2$ are the variances
- $\sigma_{xy}$ is the covariance
- $c_1$ and $c_2$ are constants to avoid division by zero

4. **Peak Signal-to-Noise Ratio (PSNR)**: Measures the quality of reconstructed images in tasks like denoising or super-resolution. Ratio between the maximum possible power of a signal and the power of corrupting noise that affects its fidelity. Based on MSE.

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$$

Where:

- $\text{MAX}_I$ is the maximum possible pixel value
- $\text{MSE}$ is the mean squared error between images

5. **Learned Perceptual Image Patch Similarity (LPIPS)**: Measures perceptual similarity using deep features from pre-trained networks (VGG, AlexNet). Aims to align better with human perception of similarity than pixel-wise metrics like MSE.
### 3D Vision Metrics

Metrics for evaluating 3D reconstruction, depth estimation, and point cloud processing.

1. **Depth Estimation Metrics**:
    - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and ground truth depths.
    - **Root Mean Squared Error (RMSE)**: Square root of the average squared differences.
    - **Threshold Accuracy**: Percentage of pixels with ratio of predicted to ground truth depth within threshold $t$ (commonly $t \in {1.25, 1.25^2, 1.25^3}$).
2. **Point Cloud Metrics**:
    - **Chamfer Distance**: Measures the average distance from each point in one point cloud to its nearest neighbor in another point cloud.
    
    $$\text{CD}(S_1, S_2) = \frac{1}{|S_1|} \sum_{x \in S_1} \min_{y \in S_2} ||x-y||_2^2 + \frac{1}{|S_2|} \sum_{y \in S_2} \min_{x \in S_1} ||y-x||_2^2$$
    
    - **Earth Mover's Distance (EMD)**: The minimum "cost" to transform one point cloud into another.
    - **F-Score**: The harmonic mean of precision and recall at a specific distance threshold.
3. **3D Reconstruction Metrics**:
    
    - **Volumetric IoU**: The intersection over union of 3D volumes.
    - **Surface-to-Surface Distance**: The average distance between reconstructed and ground truth surfaces.

## Human Pose Estimation Metrics

1. **Percentage of Correct Keypoints (PCK)**: The percentage of predicted keypoints that fall within a distance threshold of the ground truth keypoints.
    
2. **Mean Per Joint Position Error (MPJPE)**: The average Euclidean distance between predicted and ground truth joint positions.
    
3. **Object Keypoint Similarity (OKS)**: Similar to IoU but for keypoints, accounting for keypoint visibility and scale.
    

$$\text{OKS} = \frac{\sum_i \exp(-d_i^2 / (2s^2k_i^2)) \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$$

Where:

- $d_i$ is the Euclidean distance between predicted and ground truth keypoint $i$
- $s$ is the object scale
- $k_i$ is the per-keypoint constant
- $v_i$ is the visibility flag for keypoint $i$

## Links

- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)
- [Scikit-image Comparison Metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html)
- [PyTorch Vision Metrics](https://pytorch.org/vision/stable/reference.html)
