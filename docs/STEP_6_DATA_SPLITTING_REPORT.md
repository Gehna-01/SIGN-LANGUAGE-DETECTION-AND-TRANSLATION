# Step 6: Leakage-Safe Data Splitting Report

**Project Title:** Sign Language Recognition and Translation Assistant  
**Project Number:** Project 13  
**Domain:** Assistive Technology  
**Milestone:** Phase I — Step 6 (Leakage-Safe Partitioning)  
**Execution Date:** October 28, 2026  
**Auditor / Engineer:** Gehna & AI Assistant  

---

## 1. Rationale: Preventing Temporal & Session Leakage
In Step 4, empirical audit revealed that within each sign class folder, filenames are strictly indexed from `1` to `3000` (e.g., `A1.jpg` through `A3000.jpg`). These images were recorded during continuous camera video bursts, where adjacent frames (e.g., `A1001.jpg` and `A1002.jpg`) differ by only fractions of a second, sharing identical wrist positions, background shadows, and lighting.

### The Pitfall of Naive Random Splitting
If standard random shuffling (`train_test_split(shuffle=True)`) is applied to individual images, consecutive frames are scattered across both the training set and the test set. The model can then achieve artificially inflated $>99\%$ test accuracy by simply memorizing background pixels and wrist positions from adjacent frames rather than learning generalized sign language semantics.

### The Engineering Solution: Contiguous Block Partitioning
To eliminate temporal burst leakage, we applied **contiguous block-based session partitioning**. The numerical sequence of each class was sorted, and distinct non-overlapping temporal blocks were allocated to each partition. The test set represents a completely held-out recording segment (frames 2,551 to 3,000) that shares no temporal adjacency with the training frames.

---

## 2. Partition Specifications & Distribution

| Partition | Allocation Ratio | Samples per Class | Total Samples | Frame Index Range | Manifest File | Manifest SHA-256 / MD5 |
|---|---|---|---|---|---|---|
| **Training Set** | **70.0%** | **2,100** | **60,900 images** | `1` to `2100` | [`data/splits/train_manifest.csv`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/data/splits/train_manifest.csv) | Verified & Logged |
| **Validation Set** | **15.0%** | **450** | **13,050 images** | `2101` to `2550` | [`data/splits/val_manifest.csv`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/data/splits/val_manifest.csv) | Verified & Logged |
| **Test Set** | **15.0%** | **450** | **13,050 images** | `2551` to `3000` | [`data/splits/test_manifest.csv`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/data/splits/test_manifest.csv) | Verified & Logged |
| **Total Master** | **100.0%** | **3,000** | **87,000 images** | `1` to `3000` | [`data/splits/split_summary.json`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/data/splits/split_summary.json) | Complete |

---

## 3. Automated Disjointness & Integrity Verifications

The script [`scripts/create_splits.py`](file:///c:/Users/16476/OneDrive/Desktop/SIGN%20LANGUAGE%20DETECTION%20AND%20TRANSLATION/scripts/create_splits.py) executed automated cryptographic set-intersection assertions to mathematically verify partition independence:

1. **Train $\cap$ Validation Overlap:**
   $$\text{Set}(\text{Train}) \cap \text{Set}(\text{Val}) = \emptyset \quad (\text{Count} = 0)$$
2. **Train $\cap$ Test Overlap:**
   $$\text{Set}(\text{Train}) \cap \text{Set}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$
3. **Validation $\cap$ Test Overlap:**
   $$\text{Set}(\text{Val}) \cap \text{Set}(\text{Test}) = \emptyset \quad (\text{Count} = 0)$$
4. **Conservation of Total Data:**
   $$60,900 + 13,050 + 13,050 = 87,000 \quad (100.0\% \text{ accounted for})$$
5. **Class Balance Invariance:**
   Every single class in the training set has exactly 2,100 samples; every single class in validation and test has exactly 450 samples.

---

## 4. Test Set Quarantine Protocol
To preserve scientific validity throughout Phase I:
- The **Test Set (13,050 images)** is quarantined and sealed.
- The test set must **never** be used to guide model architecture decisions, adjust learning rates, trigger early stopping, or tune hyperparameters.
- The **Validation Set (13,050 images)** will be used exclusively for epoch-by-epoch loss monitoring, learning curves, and model checkpointing.
- The Test Set will be evaluated only once on the finalized baseline and comparison models.
