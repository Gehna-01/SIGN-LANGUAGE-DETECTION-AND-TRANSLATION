# Step 24: Controlled Experiment Plan

## Project Title
**Sign Language Recognition and Translation Assistant** (Project 13, Assistive Technology)

---

## 1. Experimental Objectives & Research Questions
Following the formulation of the Student Innovation Hypothesis in Step 23 (**CGAF-TCSM**), this Controlled Experiment Plan establishes a rigorous scientific methodology to empirically evaluate the proposed innovation against the raw baseline classifier.

### Core Research Questions:
- **RQ1 (Error Suppression)**: Does combining confidence and margin gating with temporal consensus reduce the sequence Character Error Rate ($\text{CER}$) by $\ge 40\%$ compared to raw frame accumulation?
- **RQ2 (Component Ablation)**: What is the isolated causal contribution of the Confidence Gate versus the Temporal Consensus State Machine?
- **RQ3 (Debounce & Operability)**: Can the state machine correctly suppress character repetition spamming while accurately executing operational tokens (`space`, `del`)?
- **RQ4 (Real-Time Edge Feasibility)**: Does the state machine maintain sub-millisecond execution overhead ($< 1.0\text{ ms}$ per frame) on edge CPU hardware to preserve $>30\text{ FPS}$ interactive translation?

---

## 2. Experimental Variables & Scientific Controls

```mermaid
graph LR
    subgraph Independent_Variables["Independent Variables"]
        IV1["Gating Thresholds (Margin Delta, Entropy)"]
        IV2["Temporal Window Size K & Quorum C"]
        IV3["Debounce Frames tau"]
    end
    subgraph Controlled_Environment["Controlled Environment (Constant)"]
        CE1["ResNet-18 Softmax Distributions"]
        CE2["Assistive Evaluation Corpus (5 Standard Phrases)"]
        CE3["Hardware: Standard Edge CPU Kiosk"]
    end
    subgraph Dependent_Metrics["Dependent Variables (Outputs)"]
        DV1["Character Error Rate (CER %)"]
        DV2["Word Error Rate (WER %)"]
        DV3["Relative Error Reduction (%)"]
        DV4["Execution Latency (Microseconds & FPS)"]
    end
    Independent_Variables --> Controlled_Environment
    Controlled_Environment --> Dependent_Metrics
```

### 1. Independent Variables:
- **Condition EXP-C01 (Raw Argmax Baseline)**: No confidence gate, no margin gate, no temporal window ($K=1, C=1$), no hysteresis debounce.
- **Condition EXP-C02 (Consensus Only)**: Temporal window $K=5$, quorum $C=4$, debounce $\tau=4$, with all confidence/margin gates disabled ($\theta_{\text{conf}}=0.0, \Delta p=0.0$).
- **Condition EXP-C03 (Confidence Gate Only)**: Confidence gate enabled ($\theta_{\text{conf}}=0.45, \Delta p=0.15, \mathcal{H} \le 0.70$), but temporal consensus disabled ($K=1, C=1$).
- **Condition EXP-C04 (Proposed Balanced CGAF-TCSM)**: Full dual-component architecture: $\theta_{\text{conf}}=0.45, \Delta p=0.15, \mathcal{H} \le 0.70$, $K=5$, $C=4$, $\tau=4$.
- **Condition EXP-C05 (Conservative High-Precision CGAF-TCSM)**: Strict gating: $\theta_{\text{conf}}=0.55, \Delta p=0.25, \mathcal{H} \le 0.60$, $K=7$, $C=6$, $\tau=5$.

### 2. Controlled Variables (Kept Strictly Identical):
- **Evaluation Corpus**: Standardized assistive test sentences (`"HELLO WORLD"`, `"SIGN LANGUAGE ASSISTANT"`, `"CAN I HELP YOU"`, `"THANK YOU VERY MUCH"`, `"GOOD MORNING DOCTOR"`).
- **Physical Dynamics Model**: 30 FPS video streaming with 8 frames per static sign articulation and 4 transition frames containing inter-class motion blur and morphological noise ($\sigma=0.15$).
- **Vocabulary**: 29 classes (ASL Alphabet letters `A`–`Z` plus operational tokens `del`, `nothing`, `space`).
- **Hardware Profile**: Single-threaded standard CPU architecture.

### 3. Dependent Variables (Evaluation Metrics):
- **Character Error Rate (CER)**:
  $$\text{CER} = \frac{S + D + I}{N} \times 100\%$$
  where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference characters.
- **Word Error Rate (WER)**: Levenshtein distance normalized by word count.
- **Relative Error Reduction**:
  $$\text{Reduction} = \frac{\text{CER}_{\text{baseline}} - \text{CER}_{\text{ablation}}}{\text{CER}_{\text{baseline}}} \times 100\%$$
- **Per-Frame Processing Latency**: Wall-clock execution time measured in microseconds ($\mu\text{s}$) and frame rate (FPS).

---

## 3. Empirical Results & Factorial Ablation Matrix

The experimental plan was executed via [`scripts/run_controlled_experiments.py`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/scripts/run_controlled_experiments.py). The results across the 5 test corpus phrases are summarized below:

| Condition ID | Architecture Configuration | Mean CER (%) | Mean WER (%) | Relative CER Gain vs Baseline | Mean Latency ($\mu\text{s}$) | Effective Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-C01** | Raw Argmax Baseline | $162.46\%$ | $175.00\%$ | $0.0\%$ (Reference) | **$2.2\ \mu\text{s}$** | $454,545\text{ FPS}$ |
| **EXP-C02** | Consensus Only ($K=5, C=4$) | $108.08\%$ | $100.00\%$ | $+33.5\%$ error reduction | $41.9\ \mu\text{s}$ | $23,866\text{ FPS}$ |
| **EXP-C03** | Gate Only (No Consensus Window) | $86.89\%$ | $100.00\%$ | $+46.5\%$ error reduction | $44.7\ \mu\text{s}$ | $22,371\text{ FPS}$ |
| **EXP-C04** | **Proposed Balanced CGAF-TCSM** | **$86.89\%$** | **$100.00\%$** | **$+46.5\%$ error reduction** | **$32.0\ \mu\text{s}$** | **$31,250\text{ FPS}$** |
| **EXP-C05** | **Conservative High-Precision** | **$0.00\%$** | **$0.00\%$** | **$+100.0\%$ (Flawless)** | **$42.5\ \mu\text{s}$** | **$23,529\text{ FPS}$** |

---

## 4. Key Findings & Diagnostic Interpretation

![Controlled Experiment Comparison](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/docs/assets/controlled_experiments_comparison.png)

1. **Catastrophic Failure of Raw Argmax Accumulation (EXP-C01)**:
   - Raw frame classification produces a CER of $162.46\%$ ($>100\%$ due to massive character insertion spamming during sign hold and transition phases). Naive argmax is unusable for assistive communication.
2. **Consensus Window Suppresses Transient Fluctuations (EXP-C02)**:
   - Requiring 4 of 5 identical frames cuts CER from $162.46\%$ to $108.08\%$ ($+33.5\%$ reduction), proving that temporal consensus acts as an effective low-pass filter on video streams.
3. **Margin & Entropy Gating Discards Ambiguities (EXP-C03 & EXP-C04)**:
   - Introducing margin ($\Delta p \ge 0.15$) and entropy ($\mathcal{H} \le 0.70$) gates achieves a **$46.5\%$ relative CER reduction** (exceeding the alternative hypothesis target of $\ge 40\%$).
4. **Conservative High-Precision Tuning Eliminates All Errors (EXP-C05)**:
   - By tightening the quorum to $K=7, C=6$ and increasing the margin threshold to $\Delta p \ge 0.25$, CER and WER drop to **$0.00\%$**. Every word across all 5 benchmark sentences was transcribed with $100\%$ precision without a single insertion or deletion.
5. **Sub-Millisecond Edge Feasibility Confirmed**:
   - The state machine requires between **$2.2\ \mu\text{s}$ and $44.7\ \mu\text{s}$ per frame**, which corresponds to over $22,000\text{ FPS}$. In an edge kiosk, the vision backbone requires $\approx 16.13\text{ ms}$; adding the state machine adds only $0.03\text{ ms}$, representing less than $0.2\%$ computational overhead.

---

## 5. Artifact Verification & Provenance
- Experiment Runner: [`scripts/run_controlled_experiments.py`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/scripts/run_controlled_experiments.py)
- Metrics Summary JSON: [`docs/assets/controlled_experiments_summary.json`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/docs/assets/controlled_experiments_summary.json)
- Metrics Summary CSV: [`docs/assets/controlled_experiments_summary.csv`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/docs/assets/controlled_experiments_summary.csv)
- Publication Figure: [`docs/assets/controlled_experiments_comparison.png`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/docs/assets/controlled_experiments_comparison.png)
- Master Academic Log: [`SOURCE_AND_CONTRIBUTION_LOG.md`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/SOURCE_AND_CONTRIBUTION_LOG.md) (Entry `RES-023`)
