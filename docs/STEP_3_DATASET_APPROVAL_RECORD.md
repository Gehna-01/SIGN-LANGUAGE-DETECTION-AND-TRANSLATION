# Step 3: Dataset Approval & Provenance Record

**Project Title:** Sign Language Recognition and Translation Assistant  
**Project Number:** Project 13  
**Domain:** Assistive Technology  
**Milestone:** Phase I — Step 3 (Approved Dataset Formal Record)  

---

## 1. Formal Approval Declaration
- **Approved by:** Project Student / User
- **Decision Timestamp:** 2026-10-28
- **Selected Dataset:** Candidate 1 — **ASL Alphabet Dataset (Akash Nagaraj / Kaggle)**
- **Approval Statement:** User explicitly confirmed: *"use ASL Alphabet Dataset (Akash Nagaraj / Kaggle).. i approve"*

---

## 2. Dataset Provenance & Metadata Record

| Metadata Attribute | Approved Specification |
|---|---|
| **Exact Dataset Name** | ASL Alphabet Dataset |
| **Source URL** | [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Hosting Organization** | Kaggle / Community contribution by Akash Nagaraj |
| **Version** | Version 1 (Stable Release) |
| **Licence** | GNU General Public License v2.0 (GPL 2.0) |
| **Access / Ingestion Date** | October 28, 2026 |
| **Sign Language** | American Sign Language (ASL) fingerspelling and control gestures |
| **Total Samples** | 87,000 images in training set (3,000 per class) + 29 sample test images |
| **Image Resolution** | $200 \times 200$ pixels, 3 channels (RGB) |
| **Number of Classes** | 29 classes |
| **Class Vocabulary** | 26 alphabet letters (`A`–`Z`) + 3 functional tokens: `SPACE`, `DELETE`, `NOTHING` |
| **Data Format** | Categorical directory hierarchy: `asl_alphabet_train/asl_alphabet_train/<class_name>/*.jpg` |
| **Intended Project Use** | Foundational vision classifier for static sign gestures, fingerspelling recognition, and interactive campus service-desk typing assistant. |

---

## 3. Storage and Reproducibility Policy
- **Local Ingestion:** Acquired directly via Kaggle API / kagglehub distribution package ensuring exact byte-for-byte reproducibility from the official source.
- **Data Protection:** The raw ingested files are marked read-only and preserved in their authentic structure. No raw images will be altered in-place.
- **Licensing Compliance:** GPL 2.0 terms are respected; the dataset is used strictly for non-commercial educational and academic research.

---

## 4. Next Step Transition
With the dataset officially approved and recorded, the project proceeds to **Step 4: Comprehensive Dataset Audit and Leakage Checks** (inspecting actual image files, dimensions, channels, duplicate detection, class balance verification, and session/signer leakage analysis).
