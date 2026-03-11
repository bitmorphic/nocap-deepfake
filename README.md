# NoCap — Deepfake Video Detector

AI-powered deepfake detection using EfficientNet-B4 fine-tuned on the DFDC dataset. Analyses videos frame-by-frame, detects manipulated faces, visualises suspicious regions with Grad-CAM, and generates a downloadable forensic PDF report.

---

## Features

- **Video analysis** — Extracts 20 evenly spaced frames, detects faces via MTCNN, scores each face with EfficientNet-B4
- **5-level risk meter** — Authentic / Low Risk / Suspicious / High Risk / Likely Deepfake
- **Score normalisation** — Calibrated against real-world anchors to reduce false positives from training bias
- **Frame-by-frame chart** — Visual breakdown of fake probability across all analysed frames
- **Grad-CAM heatmap** — Highlights which facial regions influenced the model's decision
- **PDF forensic report** — Downloadable report with verdict, chart, Grad-CAM, and interpretation

---

## Model

| Property | Value |
|---|---|
| Architecture | EfficientNet-B4 |
| Pretrained on | ImageNet |
| Fine-tuned on | DFDC (DeepFake Detection Challenge) |
| Training set | 93,853 face crops |
| Validation set | 30,794 face crops |
| Val AUC | **0.9507** |
| Val F1 | **0.9188** |
| Val Accuracy | **87.44%** |
| Fake detection rate | 91% |
| Real detection rate | 78% |

---

## Pipeline

```
Video file
    └── OpenCV          → extract 20 evenly spaced frames
        └── MTCNN       → detect and crop faces to 224×224
            └── EfficientNet-B4  → score each face (raw 0–1)
                └── Score normalisation  → calibrated fake probability (0–1)
                    └── Dual condition   → avg > 0.90 AND 60% frames > 0.92
                        └── Verdict      → FAKE / REAL + 5-level risk label
                            └── Grad-CAM → activation heatmap on most suspicious face
                                └── PDF  → forensic report
```

---

## Score Normalisation

The DFDC training set is 79% fake, which compresses raw model scores toward 1.0 even for real inputs. Raw scores are normalised using empirically measured anchors:

```python
REAL_ANCHOR    = 0.88   # typical avg score for a real video
FAKE_ANCHOR    = 0.97   # typical avg score for a fake video
NORM_THRESHOLD = 0.50   # decision threshold on normalised score

normalised = (raw_score - REAL_ANCHOR) / (FAKE_ANCHOR - REAL_ANCHOR)
```

A video is classified FAKE only when **both** conditions are met:
1. Normalised score > 0.50
2. At least 60% of frames score above 0.92 (raw)

---

## Risk Levels

| Level | Normalised Score | Color |
|---|---|---|
| AUTHENTIC | 0.00 – 0.20 | Green |
| LOW RISK | 0.20 – 0.40 | Light green |
| SUSPICIOUS | 0.40 – 0.60 | Orange |
| HIGH RISK | 0.60 – 0.80 | Dark orange |
| LIKELY DEEPFAKE | 0.80 – 1.00 | Red |

---

## Stack

- **PyTorch** — model inference and Grad-CAM backpropagation
- **torchvision** — EfficientNet-B4 architecture and image transforms
- **facenet-pytorch** — MTCNN face detection
- **OpenCV** — video frame extraction
- **Streamlit** — web interface and deployment
- **ReportLab** — PDF report generation
- **Matplotlib** — frame score chart and heatmap colourmap
- **gdown** — model weight download from Google Drive

---

## Project Structure

```
nocap-deepfake/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (libgl1)
└── models/
    └── efficientnet_b4_dfdc.pth   # Downloaded at runtime from Google Drive
```

## Dependencies

**requirements.txt**
```
streamlit
torch
torchvision
facenet-pytorch
opencv-python-headless
numpy
Pillow
matplotlib
gdown
scikit-learn
reportlab
pandas
```

**packages.txt**
```
libgl1
```

---

## Deployment

The app is deployed on Streamlit Cloud directly from this repository. Any push to `main` triggers an automatic redeploy.

Live app: [nocap-deepfake app](https://nocap-deepfake.streamlit.app/)

---

## Known Limitations

- Model trained exclusively on DFDC dataset — may not generalise to all manipulation types
- Real detection rate (78%) is lower than fake detection rate (91%) due to class imbalance in training data
- Score normalisation anchors were calibrated on a small set of real-world test videos and may need tuning for specific video types
- Inference runs on CPU on Streamlit Cloud — analysis takes 15–30 seconds per video depending on length
- Single-frame webcam detection is unreliable due to domain gap between webcam captures and DFDC face crops

---

## License

MIT License. See `LICENSE` for details.
