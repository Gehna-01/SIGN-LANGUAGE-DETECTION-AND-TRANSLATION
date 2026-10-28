# Step 1: Assistive-Technology Use Case Definition

**Project Title:** Sign Language Recognition and Translation Assistant  
**Project Number:** Project 13  
**Domain:** Assistive Technology  
**Milestone:** Phase I  

---

## 1. Executive Summary & Working Philosophy
This project develops an assistive-technology deep learning system designed to bridge interpersonal communication barriers between Deaf / Hard-of-Hearing (D/HH) individuals and non-signing hearing persons. 

To maintain scientific integrity and realistic engineering goals, **we do not claim to build a universal, unconstrained sign-language translator**. Real-world sign languages (such as ASL, BSL, or ISL) possess rich, nuanced linguistic structures involving 3D spatial grammar, simultaneous facial expressions (non-manual features), rapid temporal transitions, and dialectal variations. 

Instead, this project establishes a **bounded, practical assistive communication tool** focused on structured, high-frequency interaction scenarios (e.g., service desks, academic assistance counters, or self-service kiosks). 

---

## 2. Exact Problem Statement
In daily administrative, educational, and service encounters (such as university helpdesks, public transit booths, and library information counters), Deaf and Hard-of-Hearing individuals who communicate primarily through sign language face immediate communication barriers with hearing personnel who lack sign language knowledge. 

Human sign-language interpreters are scarce, costly, and unavailable for quick, spontaneous 2-minute daily queries. Relying on clumsy handwriting or smartphone typing is slow, disruptive, and unnatural for primary sign-language users. 

**Core Problem:** There is a need for a lightweight, accessible vision-based assistant that captures sign gestures via standard consumer cameras, recognizes them reliably with high confidence, and translates them into real-time textual and audible representations for non-signing interlocutors without requiring expensive specialized hardware (e.g., sensory gloves).

---

## 3. Target User and Key Stakeholders

### A. Primary Target Users
1. **The Signer (Deaf or Hard-of-Hearing individual):** Uses natural sign gestures (starting with isolated signs / fingerspelled terms in Phase I, progressing to key phrase sequences) to express requests without needing to type or search for pen and paper.
2. **The Receiver (Hearing non-signer):** Front-desk attendant, teacher, or fellow student who reads the translated text and confidence indicators on a screen.

### B. Relevant Stakeholders (Filtered for Scope)
- **Educational Institutions & University Helpdesks:** Seeking to meet accessibility and inclusion standards by deploying automated assistive kiosks.
- **Accessibility Service Coordinators:** Professionals who oversee assistive accommodations and require transparent, privacy-respecting technology.
- **Sign Language Educators / Learners:** Students and practitioners who benefit from visual feedback and verification of sign correctness.

*(Excluded from Phase I scope: Emergency dispatchers and acute clinical healthcare, where false negatives carry life-critical consequences).*

---

## 4. Assistive-Technology Setting & Operating Context
- **Physical Environment:** Semi-controlled indoor settings (e.g., university inquiry desk, library checkout station, quiet office counter).
- **Lighting Conditions:** Standard indoor ambient lighting (avoiding extreme backlighting or complete darkness).
- **Camera Setup:** Front-facing standard consumer RGB webcam (720p/1080p, 30 FPS) positioned 0.5m to 1.5m in front of the signer.
- **Signer Framing:** Upper body / torso, neck, and hands clearly visible in the camera field of view; single active signer in frame.
- **Background:** Moderate indoor background (e.g., office wall or desk partition).

---

## 5. Domain Scope: Sign Language Specification
- **Target Language:** Standardized Sign Language (e.g., American Sign Language (ASL) or Indian Sign Language (ISL) vocabulary, to be formally approved in Step 3 based on public dataset auditing).
- **Phase I Vocabulary Scope:** A clearly defined, manageable set of static and isolated signs (e.g., standard alphabet fingerspelling A–Z, numbers 0–9, or high-priority conversational tokens such as *Hello, Thank You, Help, Yes, No, Please*).
- **Phase II Expansion:** Temporal dynamic gestures and phrase-level sequence combinations decoded into grammatical sentences.

---

## 6. Input and Expected Output Definitions

### Phase I (Baseline Formulation)
- **Input:** Single static RGB hand-pose image or isolated video frame ($H \times W \times 3$, normalized).
- **Processing:** Custom Deep Convolutional Neural Network (CNN) feature extractor and classifier.
- **Output:**
  1. Discrete predicted sign class label (e.g., `"Sign: B"` or `"Token: THANK YOU"`).
  2. Softmax prediction probability / confidence score (e.g., `Confidence: 94.2%`).
  3. Real-time text display on screen.

### Eventual Complete System (Phase II Target)
- **Input:** Continuous real-time video stream (sequence of frames $T \times H \times W \times C$).
- **Processing:** Hand/pose localization (YOLO/RetinaNet) $\rightarrow$ Spatiotemporal feature extraction (CNN/ResNet) $\rightarrow$ Sequence decoding (LSTM / Transformer) with language post-processing.
- **Output:** Decoded text sentence with temporal smoothing, audible speech synthesis, and error-handling indicators.

---

## 7. Critical Distinction: Sign Recognition vs. Sign Translation

| Dimension | Isolated Sign Recognition (Phase I) | Full Sign-Language Translation (Future Scope) |
|---|---|---|
| **Linguistic Unit** | Isolated sign / static hand configuration | Continuous, grammatical sentences |
| **Temporal Context** | Single frame or isolated motion window | Long-range spatiotemporal dependencies |
| **Grammar** | No grammatical syntax required | Spatial grammar, word-order reordering |
| **Non-Manual Features** | Hand shape, orientation, position | Facial expressions, head tilts, shoulder shifts |
| **Output** | Single token / class label | Coherent natural language text |
| **Evaluation Metrics** | Accuracy, Precision, Recall, Macro-F1 | Word Error Rate (WER), BLEU, ROUGE |

*Academic Honesty Principle: In Phase I, we explicitly state that we are performing **isolated sign classification**, which serves as the foundational visual alphabet/vocabulary module for downstream sequence translation.*

---

## 8. Measurable Success Criteria for Phase I

1. **Classification Accuracy & Macro F1:** Baseline CNN must achieve $\ge 85\%$ Macro-F1 on a clean, strictly isolated test set.
2. **Confusion Matrix Transparency:** Clear separation of visually distinct signs and transparent documentation of error rates for visually similar sign pairs (e.g., ASL *'M'* vs. *'N'* vs. *'T'*).
3. **Inference Latency:** $\le 50\text{ ms}$ per sample on standard CPU hardware ($\ge 20\text{ FPS}$ throughput capability), ensuring viability for real-time assistive feedback.
4. **Leakage-Safe Validation:** $0\%$ sample, duplicate, or signer leakage between training, validation, and testing partitions.
5. **From-Scratch Deep Learning Rigor:** Flawless manual mathematical verification of core fundamentals (tensor math, McCulloch-Pitts neuron, sigmoid neuron, backpropagation, and gradient checking with relative error $\le 10^{-6}$).

---

## 9. Practical and Deployment Constraints
- **Hardware Limitations:** Must operate efficiently on consumer-grade laptop / desktop processors without mandating expensive enterprise GPU clusters for inference.
- **Native Language Requirement:** The university guidelines mandate eventual C, C++, or Java deployment.
  - *Strategy:* Use Python for rapid, transparent scientific exploration, dataset auditing, PyTorch/Keras training, and ONNX model export. Simultaneously build the fundamentals module and inference pipeline in **C++ (using CMake)** or **Java**, ensuring full compliance with university expectations.
- **Camera Compatibility:** Must rely on ordinary webcams without depth sensors (no LiDAR / Kinect required).

---

## 10. Possible Risks, Ethical Considerations & Responsible Use

1. **Over-reliance & Misinterpretation:** Users must not depend on the system in critical medical or legal situations where misinterpretation could cause harm. Clear UI disclaimers must state that the system is an assistive aid.
2. **Signer & Skin Tone Bias:** Deep vision models trained on homogenous datasets can degrade severely across diverse skin complexions, hand shapes, or background contrasts. Signer diversity and background auditing are mandatory in Step 4.
3. **Privacy & Video Surveillance Concerns:** Real-time assistive cameras must process frames locally in memory without silently persisting or broadcasting unauthorized video feeds of individuals.
4. **Handedness Variability:** Left-handed signers execute mirrored hand configurations. Naive horizontal flipping during training data augmentation can alter or corrupt linguistic meaning and must be explicitly controlled.
5. **Cultural Respect for the Deaf Community:** Sign language is a complete, natural language and culture, not a mechanical code. Assistive tools should be designed with humility, prioritizing Deaf-centered dignity and clear functional boundaries.

---

## 11. Expected Benefits
- Provides immediate, low-friction interaction for routine campus/desk transactions.
- Reduces social friction and communication anxiety for D/HH students and staff.
- Demonstrates an end-to-end, scientifically grounded, reproducible deep learning system from raw tensor mathematics to convolutional computer vision.
