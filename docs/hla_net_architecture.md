# HLA-Net: Hierarchical Lesion-Aware Multiscale Attention Network

## 1. Motivation & Architectural Novelty

Standard computer vision backbones (ResNet, EfficientNet, ViT) were originally conceived for natural scene classification on ImageNet (e.g., distinguishing dogs, cars, and landscapes). When applied to agricultural plant pathology, these architectures suffer from three fundamental limitations:

1. **Receptive Field Mismatch:** Generic CNNs downsample images rapidly in the stem ($7\times7$, stride 2 followed by max-pooling), destroying early-stage fungal pustules, bacterial specks, and fine boundary margins.
2. **Background Shortcut Susceptibility:** Agricultural datasets (such as ABCGMP and PlantVillage) often contain background artifacts (tabletops, studio paper, soil). As demonstrated by shortcut probing, models frequently attend to background noise rather than leaf pathology.
3. **Severe Intra-Class Lesion Scale Disparity:** Agricultural symptoms vary from microscopic pinpricks ($1\text{–}2\text{ mm}$) to large necrotic blights ($5\text{–}10\text{ cm}$). Standard fixed receptive fields fail to represent both simultaneously.

**HLA-Net** is designed from first principles in PyTorch to address these domain-specific challenges through four architectural innovations:

---

## 2. Core Modules & Mathematical Formulation

### 2.1 Multi-Scale Receptive Stem (MSR-Stem)
Instead of aggressive striding, the MSR-Stem captures high-frequency symptom boundaries using dual parallel receptive fields:

$$\mathbf{X}_{3\times3} = \text{SiLU}\Big(\text{BN}\big(\text{Conv}_{3\times3}^{s=2}(\mathbf{X})\big)\Big)$$
$$\mathbf{X}_{5\times5} = \text{SiLU}\Big(\text{BN}\big(\text{Conv}_{5\times5}^{s=2}(\mathbf{X})\big)\Big)$$
$$\mathbf{X}_{\text{stem}} = \text{MaxPool}_{3\times3}^{s=2}\Big(\text{SiLU}\big(\text{BN}(\text{Conv}_{1\times1}([\mathbf{X}_{3\times3}, \mathbf{X}_{5\times5}]))\big)\Big)$$

This retains high-frequency edge information while reducing spatial dimensions from $224 \times 224$ to $56 \times 56$.

### 2.2 Dilated Multi-Scale Lesion Block (D-MSLB)
Each residual block splits expanded features into three parallel depthwise paths operating across varying atrous dilation rates ($d=1, 2, 4$):

$$\mathbf{Y}_{d=1} = \text{DConv}_{3\times3}^{d=1}(\mathbf{X}_1) \quad \text{[Micro-lesions: pinpricks, cankers]}$$
$$\mathbf{Y}_{d=2} = \text{DConv}_{3\times3}^{d=2}(\mathbf{X}_2) \quad \text{[Meso-lesions: frogeye margins, halos]}$$
$$\mathbf{Y}_{d=4} = \text{DConv}_{3\times3}^{d=4}(\mathbf{X}_3) \quad \text{[Macro-lesions: necrotic blotches, blights]}$$

The paths are concatenated and projected with a $1\times 1$ pointwise convolution.

### 2.3 Coordinate Lesion Attention (CLA)
To suppress background shortcut learning and preserve precise lesion spatial location, CLA decomposes global pooling into horizontal and vertical 1D pooling:

$$z_c^h(h) = \frac{1}{W} \sum_{0 \le i < W} x_c(h, i), \quad z_c^w(w) = \frac{1}{H} \sum_{0 \le j < H} x_c(j, w)$$

These directional feature vectors are concatenated, passed through a shared dimension-reduction convolution, and split back into attention vectors $\mathbf{a}^h \in \mathbb{R}^{H \times 1}$ and $\mathbf{a}^w \in \mathbb{R}^{1 \times W}$:

$$\mathbf{Y}_{\text{attn}} = \mathbf{X} \odot \sigma\big(\text{Conv}_h(\mathbf{f}^h)\big) \odot \sigma\big(\text{Conv}_w(\mathbf{f}^w)\big)$$

---

## 3. Explicit Safeguards Against Overfitting and Underfitting

| Challenge | Risk | HLA-Net Technical Countermeasure |
|---|---|---|
| **Overfitting** | Model memorizes the 6,248 training photos and fails on held-out test data. | 1. **Parameter Efficiency:** ~2.79M parameters (vs. ResNet-50's 25M) via depthwise separable convolutions.<br>2. **Stochastic Depth (DropPath):** Linearly scheduled drop rate ($0.0 \to 0.1$) across blocks.<br>3. **Head Dropout:** $p = 0.3$ before final classification linear layer.<br>4. **Weight Decay:** $L_2$ penalty of $1 \times 10^{-4}$ in AdamW.<br>5. **Data Augmentation:** Random rotations ($\pm 15^\circ$), vertical/horizontal flips, and mild color jitter.<br>6. **De-leaked Splits:** Group-aware manifest splitting ensuring no rotated/flipped copies cross partitions. |
| **Underfitting** | Model fails to achieve high training loss convergence or misses rare classes. | 1. **MSR-Stem & Multi-Dilation:** Parallel multi-rate filters ensure all lesion scales are captured.<br>2. **Residual Shortcuts:** Gradient highway across every block prevents vanishing gradients.<br>3. **Class-Weighted Cross-Entropy:** Inverse-frequency class weighting ensures rare classes (e.g. classes with $<20$ samples) receive proportional gradient updates.<br>4. **Plateau Learning Rate Scheduling:** Drops learning rate by $0.5\times$ upon validation metric plateaus. |

---

## 4. Verification and Compatibility Suite

HLA-Net has been designed for end-to-end compatibility with all diagnostic and explainability tools in the repository:
* **Grad-CAM (`src/explain.py`):** Fully hooks `model.backbone.layer4` to generate lesion visual attention heatmaps.
* **OOD Mahalanobis Detector (`src/assistant/ood.py`):** `model.backbone(x)` provides clean 512-dimensional penultimate representations for class-conditional distance computation.
* **Experiment Tracker & Metrics:** Integrates directly with `src/train.py`, `src/evaluate.py`, and `logs/experiments.csv`.
