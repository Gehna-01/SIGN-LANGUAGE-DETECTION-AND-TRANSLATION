# Source and Contribution Log

This log documents all external resources, tools, code fragments, libraries, datasets, and AI contributions used in the project to guarantee complete academic integrity and provenance tracking.

---

## 1. Project Metadata
- **Project Title:** Sign Language Recognition and Translation Assistant
- **Project Number:** Project 13
- **Industry Domain:** Assistive Technology
- **Author / Student:** Gehna
- **GitHub Repository:** [https://github.com/Gehna-01/SIGN-LANGUAGE-DETECTION-AND-TRANSLATION.git](https://github.com/Gehna-01/SIGN-LANGUAGE-DETECTION-AND-TRANSLATION.git)

---

## 2. Resource Log Table

| Entry ID | Resource Name | Resource Type | Source / Link | Licence | Purpose in Project | What Was Used / Adapted | What Was Implemented by Student |
|---|---|---|---|---|---|---|---|
| RES-001 | Antigravity AI Assistant | Generative AI Coding Assistant | Internal / Google DeepMind | Proprietary Developer Tool | Pair programming, step-by-step guidance, documentation drafting, conceptual review | AI structured the initial Step 1 & Step 2 formulation drafts and comparison tables | Problem scoping, candidate review, verification, viva preparation |
| RES-002 | ASL Alphabet Dataset | Candidate Public Dataset | https://www.kaggle.com/datasets/grassknoted/asl-alphabet | GPL 2.0 | Proposed Candidate 1 for isolated sign classification baseline | Profiled and compared against hackathon criteria | Researched and evaluated for Phase I suitability |
| RES-003 | Sign Language MNIST | Candidate Public Dataset | https://www.kaggle.com/datasets/datamunge/sign-language-mnist | CC0 / Public Domain | Proposed Candidate 2 for toy numerical experiments | Profiled and compared against hackathon criteria | Researched and evaluated for Phase I suitability |
| RES-004 | WLASL Dataset | Candidate Public Dataset | https://github.com/dxli94/WLASL | C-UDA-1.0 | Proposed Candidate 3 for video word-level sign recognition | Profiled for Phase I/II suitability | Researched and evaluated for Phase I/II roadmap |
| RES-005 | INCLUDE Dataset | Candidate Public Dataset | https://github.com/AI4Bharat/INCLUDE | CC BY 4.0 | Proposed Candidate 4 for Indian Sign Language recognition | Profiled for Phase I/II suitability | Researched and evaluated for Phase I/II roadmap |
| RES-006 | RWTH-PHOENIX-Weather 2014T | Candidate Public Dataset | https://www-i6.informatik.rwth-aachen.de/~phoenix/ | CC BY-NC-SA | Proposed Candidate 5 for continuous sign translation | Profiled for Phase I/II suitability | Researched and evaluated for Phase I/II roadmap |
| RES-007 | Tensor Mathematics Sandbox | Mathematical Foundation | From Scratch (Pure Python) | Academic / Project Code | Core linear algebra & multi-dimensional tensor indexing | Row-major layout, 4D reshaping, manual broadcasting engine | Implemented and numerically verified against NumPy |
| RES-008 | McCulloch-Pitts Neuron | Neural Architecture Foundation | From Scratch (Pure Python) | Academic / Project Code | Mathematical logic threshold gate proof | Step activation and logical boolean gates (AND, OR, AND-NOT) | Implemented and verified linear separability bounds |
| RES-009 | Sigmoid Neuron & Binary CE | Neural Architecture Foundation | From Scratch (Pure Python) | Academic / Project Code | Differentiable unit & continuous gradient descent | Smooth non-linearity and analytical backpropagation | Implemented and evaluated gradient dynamics |
| RES-010 | Multilayer Perceptron (MLP) | Neural Architecture Foundation | From Scratch (Pure Python) | Academic / Project Code | Non-linear classification engine solving XOR | 2-4-1 hidden layer architecture and full backpropagation | Implemented and trained to 100% accuracy |
| RES-011 | Finite-Difference Gradient Checking | Diagnostic & Verification Tool | From Scratch (Pure Python) | Academic / Project Code | Numerical verification of analytical backprop gradients | Two-sided central differences (eps=1e-7) | Verified analytical gradients with error < 1e-8 |
| RES-012 | Optimization Algorithms Suite | Optimization Suite | From Scratch (Pure Python) | Academic / Project Code | Comparative training dynamics (BGD, SGD, Momentum, NAG) | Momentum & Nesterov Accelerated Gradient state tracking | Benchmark convergence rates and oscillations |
| RES-013 | Biological Learning Rules Suite | Unsupervised & Neurobiological | From Scratch (Pure Python) | Academic / Project Code | Demonstration of Hebbian, Oja's PCA, WTA, Delta, Boltzmann | Unsupervised plasticity and synaptic weight dynamics | Implemented and verified mathematical stability bounds |
| RES-014 | Activation Functions & Vanishing Gradients | Architectural Foundation | From Scratch (Pure Python) | Academic / Project Code | Non-linearities (Step, Sigmoid, Tanh, ReLU, LeakyReLU) | Analytical derivatives and 10-layer gradient simulation | Implemented and justified CNN vs LSTM activations |
| RES-015 | Baseline CNN Sign Classifier | Deep Learning Model | From Scratch (PyTorch `nn.Module`) | Academic / Project Code | 29-class isolated ASL fingerspelling and control token classification | 4 Conv-BN-ReLU-Pool stages, AdaptiveAvgPool, regularized head | Implemented, mathematically derived receptive fields, verified forward pass |
| RES-016 | Baseline Training Pipeline | Deep Learning Training System | From Scratch (PyTorch & Scikit-Learn) | Academic / Project Code | Multi-class cross-entropy training, AdamW, ReduceLROnPlateau, checkpointing | Model checkpoints (`best_baseline_cnn.pt`), macro-F1 tracking, early stopping | Implemented and empirically validated convergence on 29 classes |
| RES-017 | Baseline Evaluation & Confusion Matrix | Evaluation & Error Diagnostics | From Scratch (Scikit-Learn, Seaborn, Matplotlib) | Academic / Project Code | 29-class classification report, normalized confusion matrix, error pair diagnostics | Evaluated on validation set, rendered publication heatmap, ranked top 10 error pairs | Implemented and analyzed morphological ambiguities (B vs V, J vs H, space vs nothing) |
| RES-018 | Learning Curve Diagnostics | Training Diagnostics Suite | From Scratch (Matplotlib & Pandas) | Academic / Project Code | Multi-panel convergence visualization, loss trajectories, generalization gap | Ingested epoch logs, plotted 4-panel diagnostic curves | Analyzed underfitting vs overfitting and bias-variance dynamics |
| RES-019 | Controlled Regularization Ablations | Scientific Experimentation Engine | From Scratch (PyTorch & Scikit-Learn) | Academic / Project Code | Ceteris paribus ablation study isolating Dropout, Weight Decay, and Augmentation | Automated 4-condition benchmarking harness and comparative bar charts | Isolated causal regularizer contributions to generalization gap |
| RES-020 | Pretrained ResNet-18 Vision Backbone | Transfer Learning Architecture | PyTorch / Torchvision Models | BSD 3-Clause Licence | Advanced vision model benchmark for 29-class sign recognition | Pretrained ImageNet-1K residual backbone with custom regularized head | Implemented transfer wrapper, layer freezing strategies, and fine-tuning regimes |

---

## 3. Dataset Attribution
- **Dataset Name:** ASL Alphabet Dataset
- **Official Author / Contributor:** Akash Nagaraj
- **Hosting Platform:** Kaggle (`grassknoted/asl-alphabet`)
- **Direct Link:** [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Licence:** GNU General Public License v2.0 (GPL 2.0)
- **Status:** **APPROVED BY USER on 2026-10-28**
- **Intended Purpose:** Training and benchmarking foundational CNN baseline for isolated sign gesture and fingerspelling classification with operational control tokens (`SPACE`, `DELETE`, `NOTHING`).

---

## 4. Models and Pretrained Weights Attribution
- **Model Architecture:** ResNet-18 (Deep Residual Learning for Image Recognition)
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research, CVPR 2016)
- **Pretrained Weights Source:** Torchvision Hub (`ResNet18_Weights.DEFAULT` / `resnet18-f37072fd.pth`)
- **Pretrained Dataset:** ImageNet-1K ($1.28$M images across 1,000 classes)
- **Licence:** BSD 3-Clause Licence
- **Purpose in Project:** Advanced transfer learning feature extractor to resolve fine-grained knuckle and finger-count ambiguities identified in the baseline evaluation.
