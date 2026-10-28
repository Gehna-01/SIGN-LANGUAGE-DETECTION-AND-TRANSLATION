# Step 2: Research & Proposed Candidate Datasets

**Project:** Sign Language Recognition and Translation Assistant (Project 13)  
**Milestone:** Phase I — Step 2 (Candidate Dataset Evaluation)  
**Status:** PROPOSALS ONLY — Awaiting explicit user approval before downloading or finalizing.

---

## 1. Candidate Dataset Evaluation Profiles

### Candidate 1: ASL Alphabet Dataset (Grassknoted / Akash Nagaraj)
- **Dataset Name:** ASL Alphabet
- **Direct Source Link:** [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Official Hosting Organization:** Kaggle / Contributed by Akash Nagaraj
- **Licence:** GNU General Public License v2.0 (GPL 2.0)
- **Sign Language:** American Sign Language (ASL)
- **Dataset Purpose:** Educational and computer vision benchmark for static sign classification and fingerspelling recognition.
- **Static Images or Video:** Static RGB color images.
- **Isolated Signs or Continuous Signing:** Isolated signs (single hand pose per image).
- **Total Samples:** 87,000 images in training set + 29 standalone test images.
- **Total Images / Videos:** 87,029 images total.
- **Approximate Download Size:** ~1.1 GB (compressed zip) / ~1.2 GB uncompressed.
- **Image Resolution:** $200 \times 200$ pixels, 3 channels (RGB).
- **Number of Classes / Signs:** 29 classes.
- **Class Names:** 26 Latin letters (`A` through `Z`) + 3 functional conversational control tokens: `SPACE`, `DELETE`, and `NOTHING` (background/neutral hand).
- **Number of Signers:** Estimated 1 primary recorded signer across varying lighting, camera angles, and small translational shifts.
- **Signer Information:** Unannotated signer metadata (individual signer IDs not separated into distinct subfolders).
- **Class Distribution:** Perfectly balanced: exactly 3,000 images per class in the training partition.
- **Annotation Type:** Folder-level categorical classification labels (image classification directory structure).
- **Temporal Annotations:** None (static frames).
- **Classification Suitability:** **Very High** — Standard format for CNN, ResNet, and transfer learning image classification.
- **Detection Annotation Availability:** None (full image contains cropped hand; no bounding boxes provided).
- **Segmentation-Mask Availability:** None.
- **Sequence-Recognition Suitability:** Low for raw dynamic video, but functional tokens (`SPACE`, `DELETE`) allow building synthetic character sequences.
- **Translation Suitability:** Low for grammatical sentences; High for word-level fingerspelled translation at inquiry desks.
- **Possible Phase II Usefulness:** Can serve as the visual backbone for fingerspelling classification, hand bounding-box detector pretraining, and VAE/GAN augmentation experiments.
- **Advantages:**
  1. Clean, high-resolution RGB ($200 \times 200$) realistic webcam-like hand crops.
  2. Perfectly balanced (3,000 samples/class), eliminating artificial class-frequency bias.
  3. Includes realistic control tokens (`SPACE`, `DELETE`, `NOTHING`) essential for an interactive typing assistant.
  4. Manageable size (~1.1 GB), easy to download, inspect, and train on student hardware.
- **Limitations:**
  1. Lack of explicit multi-signer IDs means signer-independent splitting must rely on camera/session clustering rather than explicit subject tags.
  2. Static images cannot model dynamic gestures for letters *J* and *Z* (which have motion paths in true ASL, though static approximations are included).
- **Licensing Concerns:** GPL 2.0 allows free academic, educational, and open-source usage.
- **Annotation Concerns:** None; clean folder-per-class structure.
- **Signer Leakage Concerns:** Because images come from uniform collection sessions, naive random train/val/test splitting could cause session/background correlation. Must use stratified grouping or distinct session-block splitting to test true generalization.
- **Sequence Leakage Concerns:** Not applicable (static frames).

---

### Candidate 2: Sign Language MNIST
- **Dataset Name:** Sign Language MNIST
- **Direct Source Link:** [https://www.kaggle.com/datasets/datamunge/sign-language-mnist](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Official Hosting Organization:** Kaggle / Derived from academic American Sign Language hand gesture clips
- **Licence:** CC0: Public Domain / Open Research
- **Sign Language:** American Sign Language (ASL)
- **Dataset Purpose:** Educational drop-in replacement for the classic MNIST digit dataset.
- **Static Images or Video:** Static grayscale pixel matrices (provided in CSV format).
- **Isolated Signs or Continuous Signing:** Isolated signs.
- **Total Samples:** 34,627 samples (27,455 train + 7,172 test).
- **Total Images / Videos:** 34,627 rows in CSV format (784 pixel columns + 1 label column).
- **Approximate Download Size:** ~30 MB (extremely lightweight).
- **Image Resolution:** $28 \times 28$ pixels, 1 channel (grayscale).
- **Number of Classes / Signs:** 24 classes.
- **Class Names:** Letters `A` through `Y` (excluding `J` and `Z` due to gesture motion requirements). Numeric labels 0–25 (indices 9 and 25 omitted).
- **Number of Signers:** Multiple volunteer signers from the original source dataset.
- **Signer Information:** Not preserved in the CSV; anonymous pixel rows.
- **Class Distribution:** Slightly unbalanced (roughly 1,000 to 1,400 samples per letter).
- **Annotation Type:** Integer class labels (0–25).
- **Temporal Annotations:** None.
- **Classification Suitability:** High for basic numerical MLPs and shallow educational CNNs.
- **Detection Annotation Availability:** None.
- **Segmentation-Mask Availability:** None.
- **Sequence-Recognition Suitability:** None.
- **Translation Suitability:** Very low (lacks control tokens, missing letters J and Z).
- **Possible Phase II Usefulness:** Very limited beyond toy comparisons.
- **Advantages:**
  1. Extremely small download (~30 MB) and rapid loading via standard CSV.
  2. Zero setup overhead; directly compatible with from-scratch mathematical modules (Step 8–11).
- **Limitations:**
  1. Very low resolution ($28 \times 28$ grayscale), making it unrealistic for real-world webcam deployment.
  2. Missing letters *J* and *Z*.
  3. No RGB color information, preventing skin-tone or realistic lighting robustness analysis.
- **Licensing Concerns:** None (Public Domain / CC0).
- **Annotation Concerns:** Missing class indices (9 and 25) can confuse standard loss functions if not re-indexed.
- **Signer Leakage Concerns:** Moderate; train and test sets were pre-separated, but individual signer identities cannot be verified from the CSV.
- **Sequence Leakage Concerns:** Not applicable.

---

### Candidate 3: WLASL (Word-Level American Sign Language)
- **Dataset Name:** WLASL
- **Direct Source Link:** [https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL) (also hosted via arXiv:1910.11006 / WACV 2020)
- **Official Hosting Organization:** Australian National University / Data61-CSIRO
- **Licence:** Computational Use of Data Agreement (C-UDA-1.0) — Strict Academic/Research Use Only.
- **Sign Language:** American Sign Language (ASL)
- **Dataset Purpose:** Large-scale video benchmark for word-level deep sign language recognition.
- **Static Images or Video:** Video clips (.mp4 format).
- **Isolated Signs or Continuous Signing:** Isolated word-level signs (glosses).
- **Total Samples:** 21,083 video instances across subsets (WLASL100, WLASL300, WLASL1000, WLASL2000).
- **Total Images / Videos:** ~12,000+ active raw video clips.
- **Approximate Download Size:** ~12 GB to 25 GB depending on subset video quality.
- **Image / Video Resolution:** Variable video resolutions (360p to 720p), variable frame rates (25–30 FPS).
- **Number of Classes / Signs:** Up to 2,000 distinct word classes (with standard subsets of 100, 300, 1000, and 2000 words).
- **Class Names:** Broad English gloss vocabulary (e.g., *book, drink, computer, help, family, hospital*).
- **Number of Signers:** >100 distinct signers.
- **Signer Information:** High diversity; multi-signer recordings from public educational web portals.
- **Class Distribution:** Long-tailed distribution (some words have 15–20 video instances, others only 4–5).
- **Annotation Type:** JSON metadata containing bounding boxes, signer IDs, split assignments, and gloss labels.
- **Temporal Annotations:** Video frame start/end annotations for each sign gloss.
- **Classification Suitability:** Moderate for static Phase I (requires video frame extraction and aggregation).
- **Detection Annotation Availability:** Yes (hand and face bounding boxes included in metadata).
- **Segmentation-Mask Availability:** None.
- **Sequence-Recognition Suitability:** **Excellent** — Specifically created for temporal deep learning (CNN+LSTM, 3D-CNN, Video Transformer).
- **Translation Suitability:** High for word glosses; intermediate step towards sentence-level translation.
- **Possible Phase II Usefulness:** **Outstanding** for Phase II (temporal sequence recognition, LSTM, Transformer, and YOLO hand detection).
- **Advantages:**
  1. True multi-signer diversity (>100 signers) enabling rigorous signer-independent benchmarking.
  2. Real word vocabulary rather than just single-letter fingerspelling.
  3. Officially provided train/val/test splits designed by original researchers.
- **Limitations:**
  1. Very heavy download (>12 GB); downloading from YouTube/external URLs can suffer from broken links.
  2. High video processing overhead for a simple Phase I static CNN baseline.
- **Licensing Concerns:** C-UDA 1.0 restricts to non-commercial academic research.
- **Annotation Concerns:** Complex JSON parsing required to crop frames and synchronize timestamps.
- **Signer Leakage Concerns:** Low, because official splits are signer-partitioned.
- **Sequence Leakage Concerns:** Must ensure frame sequences are never split across train/test.

---

### Candidate 4: INCLUDE (Indian Sign Language Dataset)
- **Dataset Name:** INCLUDE / INCLUDE-50
- **Direct Source Link:** [https://github.com/AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) / [HuggingFace Mirror](https://huggingface.co/datasets/ai4bharat/INCLUDE)
- **Official Hosting Organization:** AI4Bharat / IIT Madras
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Sign Language:** Indian Sign Language (ISL)
- **Dataset Purpose:** Large-scale benchmark for Indian Sign Language recognition across educational and daily domains.
- **Static Images or Video:** Video clips (.mp4).
- **Isolated Signs or Continuous Signing:** Isolated signs (words and phrases).
- **Total Samples:** 4,287 videos (spanning 263 classes) / INCLUDE-50 subset contains ~1,500 videos across 50 classes.
- **Total Images / Videos:** 4,287 raw video clips (~0.27 million frames).
- **Approximate Download Size:** ~5.5 GB (INCLUDE-50) to ~35 GB (Full INCLUDE).
- **Image / Video Resolution:** Full HD ($1920 \times 1080$), 30 FPS.
- **Number of Classes / Signs:** 263 classes (Full) or 50 classes (INCLUDE-50 subset).
- **Class Names:** Everyday categories: Greetings, Numbers, Family, Actions, Emergency, Questions.
- **Number of Signers:** 7 fluent Deaf signers.
- **Signer Information:** Well-documented signer IDs for each recording session.
- **Class Distribution:** Roughly 15–20 video clips per sign class.
- **Annotation Type:** Tabular metadata mapping video files to sign glosses, signer IDs, and split partitions.
- **Temporal Annotations:** Video-level isolated sign boundaries.
- **Classification Suitability:** Requires frame extraction and keyframe selection for static Phase I classification.
- **Detection Annotation Availability:** None (requires automated hand detector like MediaPipe/YOLO to crop hands).
- **Segmentation-Mask Availability:** None.
- **Sequence-Recognition Suitability:** **Very High** — Realistic temporal dynamics for Indian Sign Language.
- **Translation Suitability:** High for isolated ISL word recognition.
- **Possible Phase II Usefulness:** High for multi-frame sequence modeling.
- **Advantages:**
  1. High cultural and regional relevance for Indian Sign Language (ISL).
  2. High definition video ($1920 \times 1080$) recorded in controlled environments.
  3. Clearly documented signer IDs enabling genuine signer-independent evaluation.
- **Limitations:**
  1. Large download footprint (~5.5 GB even for the 50-class subset).
  2. Needs substantial video decoding and frame preprocessing before a simple 2D CNN can consume it in Phase I.
- **Licensing Concerns:** Clean, permissive academic licence (CC BY 4.0).
- **Signer Leakage Concerns:** Can be strictly eliminated because signer IDs are known and can be held out in testing.

---

### Candidate 5: RWTH-PHOENIX-Weather 2014T
- **Dataset Name:** RWTH-PHOENIX-Weather 2014T
- **Direct Source Link:** [https://www-i6.informatik.rwth-aachen.de/~phoenix/](https://www-i6.informatik.rwth-aachen.de/~phoenix/)
- **Official Hosting Organization:** RWTH Aachen University (Human Language Technology Group)
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA)
- **Sign Language:** German Sign Language (Deutsche Gebärdensprache - DGS)
- **Dataset Purpose:** Continuous sign language recognition and neural sign-to-text translation benchmark.
- **Static Images or Video:** Continuous video frame sequences + parallel text translations.
- **Isolated Signs or Continuous Signing:** Continuous signing (complete broadcast weather forecast sentences).
- **Total Samples:** 8,257 continuous sentence videos / 1,061 distinct vocabulary glosses.
- **Total Images / Videos:** >1 million extracted PNG video frames.
- **Approximate Download Size:** ~8 GB to 15 GB.
- **Image / Video Resolution:** $210 \times 260$ pixels, 25 FPS.
- **Number of Classes / Signs:** 1,061 glosses with full German spoken text sentences.
- **Number of Signers:** 9 professional broadcast signers.
- **Signer Information:** Signer identities strictly documented.
- **Annotation Type:** Continuous sentence gloss alignments and parallel target text translations.
- **Temporal Annotations:** Full continuous sentence alignments.
- **Classification Suitability:** Very Poor for static single-frame baseline; requires continuous temporal alignment (CTC loss or encoder-decoder attention).
- **Sequence-Recognition & Translation Suitability:** **State-of-the-art gold standard** for continuous sign translation.
- **Advantages:** Real continuous sentences with ground-truth German translation.
- **Limitations:** Far too complex for Phase I isolated baseline; non-English spoken target (German); massive sequence decoding overhead.

---

## 2. Comprehensive Comparative Matrix

| Criteria | Candidate 1: ASL Alphabet | Candidate 2: Sign Language MNIST | Candidate 3: WLASL (Top 100/300) | Candidate 4: INCLUDE-50 | Candidate 5: RWTH-PHOENIX-2014T |
|---|---|---|---|---|---|
| **Direct URL** | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | [Kaggle Sign MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) | [GitHub WLASL](https://github.com/dxli94/WLASL) | [AI4Bharat INCLUDE](https://github.com/AI4Bharat/INCLUDE) | [RWTH-PHOENIX Page](https://www-i6.informatik.rwth-aachen.de/~phoenix/) |
| **Licence** | GPL 2.0 | CC0 / Public Domain | C-UDA 1.0 (Academic) | CC BY 4.0 | CC BY-NC-SA |
| **Language** | American Sign Language (ASL) | American Sign Language (ASL) | American Sign Language (ASL) | Indian Sign Language (ISL) | German Sign Language (DGS) |
| **Data Format** | RGB Images ($200 \times 200$) | Grayscale CSV ($28 \times 28$) | MP4 Videos (variable) | Full HD Videos ($1080\text{p}$) | PNG Frame Sequences |
| **Total Size** | ~1.1 GB | ~30 MB | ~12 GB | ~5.5 GB | ~8–15 GB |
| **Classes** | 29 (A–Z + Space, Del, Nothing) | 24 (A–Y, no J/Z) | 100 to 2,000 words | 50 to 263 signs | 1,061 glosses |
| **Sign Type** | Isolated static | Isolated static | Isolated dynamic words | Isolated dynamic words | Continuous sentences |
| **Phase I Suitability** | **Highest (Ideal)** | Good for toy math | Moderate (video overhead) | Moderate (video overhead) | Infeasible for Phase I |
| **Phase II Expansion** | Can expand to typing words | Very limited | **Highest** | **High** | Highest for continuous |
| **Signer Metadata** | Single / limited signer tags | Anonymized | >100 signers | 7 signers | 9 signers |
| **Webcam Realism** | High (true color hand crops) | Low (tiny grayscale) | High (real humans) | High (real humans) | High (broadcast video) |

---

## 3. Recommended Dataset for Phase I

### **Recommendation: Candidate 1 — ASL Alphabet Dataset (Akash Nagaraj / Kaggle)**
**Direct Link:** [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

### Why Candidate 1 is the Superior Choice for Phase I:
1. **Perfect Alignment with Phase I Objective:** 
   Our Phase I objective is to establish a rigorous, clean, and reproducible **baseline sign classifier** with an interpretable convolutional architecture. The ASL Alphabet dataset provides clean $200 \times 200$ RGB color images of hands, providing realistic visual features (edges, palm orientation, finger contours) without overwhelming student hardware with heavy multi-gigabyte video decoding pipelines.
2. **Conversational Control Tokens for Assistive Use:** 
   Unlike textbook alphabet sets, Candidate 1 includes three vital operational tokens: `SPACE`, `DELETE`, and `NOTHING` (neutral state). This directly serves our assistive desk use case, enabling an interactive assistant that can distinguish between idle states, space between words, and character deletions.
3. **Manageable & Reproducible:** 
   At ~1.1 GB, it can be downloaded and audited efficiently, ensuring complete audit reproducibility without risking corrupted video streams or external YouTube link decay.
4. **Permissive Academic Licence:** 
   Distributed under GPL 2.0, ensuring full legal clarity for educational hackathon development.
5. **Bridge to Phase II:** 
   Once the static baseline classifier and fundamentals are proven, we can seamlessly advance into Phase II by incorporating video datasets (such as WLASL or INCLUDE) for temporal sequence recognition and LSTM/Transformer decoding.

---

## 4. Academic Integrity & Dataset Control Reminder
> [!IMPORTANT]
> **No dataset has been downloaded or finalized.**  
> As required by Section 3 of the working guidelines:
> 1. You must personally review the provided links above.
> 2. You must provide your **explicit approval** (e.g., *"Use Candidate 1: ASL Alphabet"* or *"I approve the ASL Alphabet dataset"*) before any download, inspection, or code implementation begins.
