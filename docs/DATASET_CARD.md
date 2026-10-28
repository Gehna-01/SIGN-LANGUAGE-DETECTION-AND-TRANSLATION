# Dataset Card: ASL Alphabet Dataset

**Project Title:** Sign Language Recognition and Translation Assistant  
**Project Number:** Project 13  
**Domain:** Assistive Technology  
**Milestone:** Phase I — Step 5 (Formal Dataset Card)  
**Card Maintained by:** Gehna (Student) & AI Assistant  

---

## 1. Dataset Identification & Provenance
- **Dataset Name:** ASL Alphabet Dataset
- **Official Contributor / Source:** Akash Nagaraj (`grassknoted`)
- **Source URL:** [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Licence:** GNU General Public License v2.0 (GPL 2.0)
- **Version:** Version 1 (Stable Release)
- **Access / Ingestion Date:** October 28, 2026
- **Hosting Organization:** Kaggle

---

## 2. Linguistic & Technical Domain
- **Sign Language:** American Sign Language (ASL)
- **Domain:** Assistive Technology / Computer Vision / Human-Computer Interaction
- **Linguistic Focus:** Fingerspelling alphabet (A–Z) and conversational interaction control tokens
- **Data Modality:** Static RGB color images (cropped hand-centric frames)
- **Sequence Type:** Static single-frame representations (derived from continuous video bursts)
- **Total Master Samples:** 87,000 images in master training directory + 28 standalone test images
- **Image Resolution:** $200 \times 200$ pixels
- **Color Space:** 3-channel RGB (8-bit per channel)
- **File Format:** JPEG (`.jpg`)

---

## 3. Class Structure & Distribution
- **Total Classes:** 29 distinct classes
- **Class Vocabulary:**
  - **Alphabet (26 classes):** `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`
  - **Operational Control Tokens (3 classes):**
    - `del`: Trigger backspace / remove previous character.
    - `nothing`: Neutral hand state / background without an active sign.
    - `space`: Insert word boundary / space token.
- **Class Balance:** **100% Uniform** — Exactly 3,000 images per class in the 87,000-image master dataset.
- **Annotation Type:** Categorical directory structure (`<class_name>/<sample_name>.jpg`).
- **Temporal Annotations:** Not applicable (static frames).

---

## 4. Subject & Signer Metadata
- **Number of Signers:** Estimated 1 primary adult signer across multiple capture sessions. *(Exact signer count: Unannotated upstream by author).*
- **Signer Distribution:** Unannotated upstream (no subject ID tags or demographic labels provided in raw metadata).
- **Handedness:** Primarily right-handed sign demonstrations.

---

## 5. Data Quality, Cleaning & Integrity Audit
- **Corruption Checks:** Automated empirical verification via Pillow (`img.verify()`) on thousands of samples across all 29 classes showed **0 corrupt files**.
- **Dimension Consistency:** **100% of samples** verified at exactly $200 \times 200 \times 3$.
- **Duplicate Handling:** Cryptographic MD5 hash auditing showed **0 duplicate image files** in audited batches.
- **Standalone Test Partition Defect:** The upstream `asl_alphabet_test` directory contains only 28 single images (1 per class) and **completely lacks a sample for the `del` class**. Therefore, it is statistically inadequate as a primary test set and will be used solely for quick single-sample sanity checks.

---

## 6. Splitting & Leakage Prevention Strategy
- **Master Split Ratios:** 70% Training (60,900 images), 15% Validation (13,050 images), 15% Test (13,050 images).
- **Identified Leakage Threat (Temporal Frame Bursts):** File naming (`A1.jpg` to `A3000.jpg`) reveals that nearby indices are consecutive frames captured milliseconds apart. A naive random uniform shuffle would place adjacent frames across splits, creating severe temporal leakage.
- **Leakage-Safe Partitioning Policy:** We enforce **contiguous block-based / session-aware chunking** per class:
  - **Train Set (70%):** Indices 1 to 2,100 per class (2,100 samples $\times$ 29 = 60,900 images).
  - **Validation Set (15%):** Indices 2,101 to 2,550 per class (450 samples $\times$ 29 = 13,050 images).
  - **Test Set (15%):** Indices 2,551 to 3,000 per class (450 samples $\times$ 29 = 13,050 images).
- **Guaranteed Isolation:** The test set represents a completely held-out temporal recording block that is never touched during hyperparameter tuning or training.

---

## 7. Preprocessing & Augmentation Rules
- **Deterministic Preprocessing (All Sets):**
  - Verify size ($200 \times 200$).
  - Rescale pixel intensities to $[0.0, 1.0]$.
  - Normalize with dataset-level or standard mean/standard deviation.
  - One-hot / integer class index encoding.
- **Training-Only Augmentation (Semantics-Preserving):**
  - Minor translation ($\pm 5\%$).
  - Minor rotation ($\pm 10^\circ$).
  - Minor brightness jitter ($\pm 10\%$).
- **Strictly Prohibited Augmentation:**
  - **Random Horizontal Flipping is STRICTLY PROHIBITED:** In sign language, flipping an image horizontally alters handedness (reversing the grammatical spatial orientation) and causes semantic confusion.
- **Validation/Test Augmentation:** Zero augmentation (deterministic evaluation only).

---

## 8. Known Limitations & Potential Biases
1. **Dynamic Letter Limitation:** ASL letters *J* and *Z* require continuous motion tracing (a wrist hook for *J* and a zigzag index finger for *Z*). In this static dataset, *J* and *Z* are represented by static endpoint poses, which only approximate the true sign.
2. **Skin-Tone & Signer Bias:** The dataset exhibits a single primary skin tone and adult hand geometry. Models trained solely on this dataset may experience performance degradation across darker skin complexions, elderly hands, or children's hands.
3. **Background Uniformity:** The training images were captured against relatively uniform indoor backgrounds. Performance may degrade under complex, cluttered, or outdoor lighting environments.

---

## 9. Ethical, Privacy & Responsible Use Considerations
- **Non-Replacement of Human Interpreters:** This dataset and derived models are intended strictly for an **assistive communication aid and educational research prototype**. They must never be marketed or used as a replacement for certified sign-language interpreters in legal, medical, or life-safety settings.
- **Privacy Assurance:** The system processes visual data strictly on-device in volatile memory. No video streams or face crops will be persisted or broadcast to external cloud servers.
- **Licensing Compliance:** Distributed under GPL 2.0. All downstream code, adaptations, and documentation will remain open and fully credited to the original author.
