# 🎬 NoCap: AI-Powered Deepfake Video Detection System

![Deployment](https://img.shields.io/badge/Deployed_on-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Metrics](https://img.shields.io/badge/Celeb--DF_v2_AUC-0.9637-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

[cite_start]**NoCap** (No Cap on Fakes) is a complete, end-to-end deepfake video detection system designed to combat the rising threat of synthetic media[cite: 6, 16]. [cite_start]By combining state-of-the-art deep learning, visual explainability, and an accessible web interface, NoCap allows anyone to upload a video and receive a forensic-grade authenticity verdict in seconds[cite: 16].

[cite_start]Live Web Application: [Streamlit Cloud](https://github.com/bitmorphic/nocap-deepfake) [cite: 9]

---

## 🔴 The Problem
[cite_start]Deepfake technology has evolved to produce hyper-realistic synthetic media, posing critical threats to public trust, electoral processes, and personal privacy[cite: 14, 15]. [cite_start]While the volume of manipulated content has grown exponentially, accessible detection tools for ordinary users remain scarce[cite: 39, 62]. [cite_start]NoCap bridges this gap by providing an explainable, robust, and zero-installation detection platform[cite: 41, 42].

---

## ⚙️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

[cite_start]**Libraries Used:** PyTorch, torchvision, facenet-pytorch (MTCNN), OpenCV (headless), NumPy, Matplotlib, Pandas, Streamlit, ReportLab, and gdown.

---

## 🧠 System Architecture & Pipeline

[cite_start]The inference pipeline takes raw video and outputs a calibrated forensic verdict[cite: 44]:

1. [cite_start]**Frame Extraction:** OpenCV extracts 20 evenly spaced frames from the uploaded video[cite: 19].
2. [cite_start]**Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks) detects and crops faces via a 3-stage cascade[cite: 19, 115].
3. [cite_start]**Feature Extraction & Scoring:** An EfficientNet-B4 model (19M parameters, compound-scaled) scores each cropped face[cite: 17, 92, 93].
4. [cite_start]**Score Normalization:** Corrects the 79% fake class imbalance present in the pretraining dataset by mapping scores using empirically measured anchors[cite: 101, 102].
5. [cite_start]**Dual-Condition Verdict:** Eliminates false positives by requiring a normalized average score > 0.50 AND at least 60% of individual frames to have a raw score > 0.92[cite: 104, 105].
6. [cite_start]**Explainability (Grad-CAM):** Generates spatial heatmaps indicating which facial regions most influenced the model's prediction[cite: 20, 108].
7. [cite_start]**Forensic Report:** Generates a downloadable in-memory PDF via ReportLab detailing the 5-level risk assessment (Authentic to Likely Deepfake)[cite: 20, 113, 115].

---

## 📊 Methodology & Training Protocols

[cite_start]NoCap overcomes severe domain shift issues via a **two-stage training protocol**[cite: 90, 95]:
* [cite_start]**Stage 1 (Pretraining):** Trained on the large DeepFake Detection Challenge (DFDC) dataset (93,853 training crops) for robust feature extraction[cite: 17, 97].
* [cite_start]**Stage 2 (Domain Adaptation):** Fine-tuned on the Celeb-DF v2 dataset to handle real-world compression and unseen generation artifacts[cite: 17, 99].

### Performance Results (Celeb-DF v2 Test Split)
[cite_start]NoCap surpasses all published baseline methods on the Celeb-DF v2 official test split[cite: 18].

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **AUC** | **0.9637** | [cite_start]Threshold-independent best metric [cite: 123] |
| **F1 Score** | 0.9580 | [cite_start]Harmonic mean of precision and recall [cite: 123] |
| **Accuracy** | 93.53% | [cite_start]At optimal threshold 0.740 [cite: 123] |
| **Fake Detection Rate** | 97.35% | [cite_start]Correctly identifies actual fakes [cite: 123] |

[cite_start]*(For comparison, the best prior baseline, Face X-ray, achieved an AUC of 0.747)*[cite: 133].

---

## 💻 Local Setup & Installation

To run NoCap locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bitmorphic/nocap-deepfake.git](https://github.com/bitmorphic/nocap-deepfake.git)
   cd nocap-deepfake
