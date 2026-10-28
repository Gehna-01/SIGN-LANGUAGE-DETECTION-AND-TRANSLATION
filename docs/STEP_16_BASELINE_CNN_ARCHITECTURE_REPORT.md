# Step 16: Baseline CNN Sign Classifier Architecture

**Project:** Sign Language Recognition and Translation Assistant (Project 13)  
**Author / Student:** Gehna  
**Date:** 2026-10-28  
**Status:** Architecture Designed, Implemented & Verified in PyTorch  

---

## 1. Executive Summary & Purpose

In **Step 16**, we design and implement the primary **Baseline Convolutional Neural Network (CNN)** for isolated sign recognition across all **29 classes** of the ASL Alphabet dataset (26 alphabetic characters `A`–`Z` plus 3 operational control tokens: `del`, `nothing`, and `space`).

The purpose of this baseline model is to establish a rigorous, highly reproducible, lightweight benchmark before introducing heavier transfer-learning architectures (e.g., ResNet-18/50 in Step 21). The architecture is intentionally engineered from scratch in **pure Python (`torch.nn.Module`)** with:
- **Spatial Feature Pyramid:** 4 progressive convolutional stages expanding channels ($3 \to 32 \to 64 \to 128 \to 256$) while halving spatial resolution via non-overlapping max-pooling.
- **Global Spatial Aggregation:** Adaptive average pooling that eliminates spatial grid sensitivity and drastically reduces classification parameters compared to flattening large feature maps.
- **Dual-Stage Regularization:** Spatial 2D Dropout in deeper conv blocks, Batch Normalization at every layer, and two-stage 1D Dropout ($p=0.4$ and $p=0.3$) in the classification head to prevent overfitting on static signer backgrounds.
- **Edge Deployment Footprint:** Ultra-lightweight size (**425,661 parameters / ~1.62 MB**), delivering sub-5ms inference on standard CPUs for public assistive service kiosks.

---

## 2. Complete Architecture Design

The model is defined in [`src/models/baseline_cnn.py`](../src/models/baseline_cnn.py) and consists of three distinct stages:

```
[Input: 3 x 64 x 64 RGB Sign Image]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Spatial Feature Extraction (4 Conv Blocks)        │
│ ─────────────────────────────────────────────────────────── │
│ Block 1: Conv2d(3 -> 32, 3x3) -> BN -> ReLU -> MaxPool(2x2) │ -> [32, 32, 32]
│ Block 2: Conv2d(32 -> 64, 3x3) -> BN -> ReLU -> MaxPool(2x2)│ -> [64, 16, 16]
│ Block 3: Conv2d(64 -> 128, 3x3) -> BN -> ReLU -> MaxPool(2x2)│ -> [128, 8, 8]
│ Block 4: Conv2d(128 -> 256, 3x3) -> BN -> ReLU -> MaxPool(2x2)│ -> [256, 4, 4]
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Spatial Aggregation & Latent Bottleneck            │
│ ─────────────────────────────────────────────────────────── │
│ AdaptiveAvgPool2d((1, 1))                                  │ -> [256, 1, 1]
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Regularized Classification Head                   │
│ ─────────────────────────────────────────────────────────── │
│ Flatten                                                     │ -> [256]
│ Dropout(p=0.4)                                              │
│ Linear(256 -> 128, bias=False)                              │
│ BatchNorm1d(128) -> ReLU                                    │ -> [128]
│ Dropout(p=0.3)                                              │
│ Linear(128 -> 29)                                           │ -> [29 Logits]
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-by-Layer Shapes & Parameter Breakdown

The parameter count for each layer is calculated mathematically from first principles:
- **Convolutional Layer without Bias:** $P_{\text{conv}} = C_{\text{out}} \times (C_{\text{in}} \times k_h \times k_w)$ (biases omitted because subsequent Batch Normalization layers center the activations).
- **Batch Normalization (2D/1D):** $P_{\text{bn}} = 2 \times C$ (learnable scale $\gamma$ and shift $\beta$).
- **Linear Layer:** $P_{\text{linear}} = (C_{\text{in}} \times C_{\text{out}}) + C_{\text{out}}$ (with bias).

| Stage / Block | Layer Specification | Output Tensor Shape | Parameter Formula | Parameter Count | Trainable? |
|---|---|---|---|---|---|
| **Input** | Batch Image Tensor | $[B, 3, 64, 64]$ | — | $0$ | No |
| **ConvBlock 1** | `Conv2d(3, 32, 3x3, pad=1)`<br>`BatchNorm2d(32)`<br>`ReLU()`<br>`MaxPool2d(2x2, s=2)` | $[B, 32, 32, 32]$ | $32 \times (3 \times 3 \times 3) = 864$<br>$2 \times 32 = 64$ | **928** | Yes |
| **ConvBlock 2** | `Conv2d(32, 64, 3x3, pad=1)`<br>`BatchNorm2d(64)`<br>`ReLU()`<br>`MaxPool2d(2x2, s=2)` | $[B, 64, 16, 16]$ | $64 \times (32 \times 3 \times 3) = 18,432$<br>$2 \times 64 = 128$ | **18,560** | Yes |
| **ConvBlock 3** | `Conv2d(64, 128, 3x3, pad=1)`<br>`BatchNorm2d(128)`<br>`ReLU()`<br>`MaxPool2d(2x2, s=2)`<br>`Dropout2d(0.1)` | $[B, 128, 8, 8]$ | $128 \times (64 \times 3 \times 3) = 73,728$<br>$2 \times 128 = 256$ | **73,984** | Yes |
| **ConvBlock 4** | `Conv2d(128, 256, 3x3, pad=1)`<br>`BatchNorm2d(256)`<br>`ReLU()`<br>`MaxPool2d(2x2, s=2)`<br>`Dropout2d(0.1)` | $[B, 256, 4, 4]$ | $256 \times (128 \times 3 \times 3) = 294,912$<br>$2 \times 256 = 512$ | **295,424** | Yes |
| **Global Pool** | `AdaptiveAvgPool2d((1, 1))` | $[B, 256, 1, 1]$ | Parameter-free | **0** | No |
| **Classifier Head** | `Flatten()`<br>`Dropout(0.4)`<br>`Linear(256, 128, bias=False)`<br>`BatchNorm1d(128)`<br>`ReLU()`<br>`Dropout(0.3)`<br>`Linear(128, 29, bias=True)` | $[B, 29]$ | $256 \times 128 = 32,768$<br>$2 \times 128 = 256$<br>$(128 \times 29) + 29 = 3,741$ | **36,765** | Yes |
| **Total** | **Full Sign Baseline CNN** | **$[B, 29]$** | **Sum of all layers** | **425,661** | **100% Trainable** |

### Parameter Distribution Breakdown:
- **Convolutional Feature Backbone:** $388,896$ parameters (**$91.36\%$**)
- **Classifier Head:** $36,765$ parameters (**$8.64\%$**)
- **Memory Footprint (FP32, 4 bytes/param):** $425,661 \times 4 \approx 1,702,644 \text{ bytes} \approx \mathbf{1.62 \text{ MB}}$.

---

## 4. Theoretical Receptive Field (RF) Derivation

The Receptive Field of a unit at layer $l$ is the spatial region of the input image that directly influences the activation of that unit.

### Mathematical Formulation
For layer $l$ with kernel size $k_l$, stride $s_l$, cumulative stride $S_{l-1}$, and input receptive field $RF_{l-1}$:
$$RF_l = RF_{l-1} + (k_l - 1) \cdot S_{l-1}$$
$$S_l = S_{l-1} \cdot s_l$$
*(Starting boundary conditions: $RF_0 = 1$, $S_0 = 1$)*.

### Layer-by-Layer Progression Table

| Layer Operation | Kernel $k$ | Stride $s$ | Cumulative Stride $S_{\text{in}}$ | Output RF Calculation | Effective Receptive Field ($RF_{\text{out}}$) | Spatial Output Grid |
|---|---|---|---|---|---|---|
| **Input Image** | — | — | $1$ | — | $1 \times 1$ | $64 \times 64$ |
| **Block 1: Conv2d** | $3$ | $1$ | $1$ | $1 + (3 - 1) \times 1$ | **$3 \times 3$** | $64 \times 64$ |
| **Block 1: MaxPool** | $2$ | $2$ | $1$ | $3 + (2 - 1) \times 1$ | **$4 \times 4$** | $32 \times 32$ |
| **Block 2: Conv2d** | $3$ | $1$ | $2$ | $4 + (3 - 1) \times 2$ | **$8 \times 8$** | $32 \times 32$ |
| **Block 2: MaxPool** | $2$ | $2$ | $2$ | $8 + (2 - 1) \times 2$ | **$10 \times 10$** | $16 \times 16$ |
| **Block 3: Conv2d** | $3$ | $1$ | $4$ | $10 + (3 - 1) \times 4$ | **$18 \times 18$** | $16 \times 16$ |
| **Block 3: MaxPool** | $2$ | $2$ | $4$ | $18 + (2 - 1) \times 4$ | **$22 \times 22$** | $8 \times 8$ |
| **Block 4: Conv2d** | $3$ | $1$ | $8$ | $22 + (3 - 1) \times 8$ | **$38 \times 38$** | $8 \times 8$ |
| **Block 4: MaxPool** | $2$ | $2$ | $8$ | $38 + (2 - 1) \times 8$ | **$46 \times 46$** | $4 \times 4$ |
| **AdaptiveAvgPool** | $4$ | $4$ | $16$ | $46 + (4 - 1) \times 16$ | **$64 \times 64$** | $1 \times 1$ |

**Significance:**
At the end of ConvBlock 4, each feature pixel in the $4 \times 4$ grid has a receptive field of $46 \times 46$, capturing primary hand pose and finger configurations. The final `AdaptiveAvgPool2d((1, 1))` pools across the remaining $4 \times 4$ spatial dimension, ensuring that each of the 256 latent bottleneck features possesses a **global receptive field ($64 \times 64$)**, viewing the entire hand gesture context simultaneously.

---

## 5. Parameter Initialization Strategy

To avoid vanishing and exploding gradients (as proven in Step 15) and enable rapid convergence:
1. **Kaiming (He) Normal Initialization:** Applied to all `Conv2d` layers:
   $$W \sim \mathcal{N}\left(0, \; \sqrt{\frac{2}{\text{fan\_out}}}\right)$$
   Specifically designed for non-saturating piecewise linear activations ($\text{ReLU}$) to preserve activation variance across depth.
2. **Xavier (Glorot) Normal Initialization:** Applied to the linear classifier projections:
   $$W \sim \mathcal{N}\left(0, \; \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)$$
3. **Batch Normalization Initialization:**
   - Scale factor $\gamma$ initialized to $1.0$.
   - Shift factor $\beta$ initialized to $0.0$.
   - Biases in preceding conv and linear layers are explicitly set to `bias=False` to eliminate redundant degrees of freedom cancelled out by batch normalization mean-centering.

---

## 6. Forward Pass Validation & Unit Testing

The model was verified via standalone execution in [`src/models/baseline_cnn.py`](../src/models/baseline_cnn.py):
- **Input Batch Shape:** $[4, 3, 64, 64]$
- **Logits Shape:** $[4, 29]$ (Matches target classes exactly)
- **Softmax Probability Check:** Sum across classes $\sum_{c=1}^{29} P(c \mid x) = 1.000000$ (Verified with numerical tolerance $< 10^{-7}$).
- **Inference Time:** $< 3.5 \text{ ms}$ per batch of 4 on standard CPU.

---

## 7. Viva Defense Checklist

1. **Why use Batch Normalization directly after Conv2d and before ReLU?**
   Batch Normalization normalizes the distribution of layer inputs, preventing internal covariate shift. Because BN shifts activations by $-\mu$, any bias parameter in `Conv2d` is redundant ($b - \mu$). We set `bias=False` to save parameters.
2. **Why use Global Adaptive Average Pooling instead of Flattening ConvBlock 4?**
   If we flattened the $4 \times 4 \times 256$ feature map directly, the first dense layer would require $(4 \times 4 \times 256) \times 128 \approx 524,288$ parameters—more than the entire convolutional backbone combined! Adaptive average pooling collapses spatial dimensions to $1 \times 1 \times 256$, requiring only $256 \times 128 = 32,768$ parameters, eliminating overfitting and making the network invariant to slight hand translations.
3. **Why Kaiming (He) Normal instead of Xavier for Conv layers?**
   Xavier assumes linear or zero-centered symmetric activations (like Tanh). Because ReLU sets half of the activations to zero ($\max(0, z)$), the activation variance is halved at each layer. Kaiming initialization compensates by multiplying the variance by a factor of 2 ($\sigma = \sqrt{2 / \text{fan}}$).
