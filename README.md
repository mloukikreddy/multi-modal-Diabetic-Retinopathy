# 🧠 Multi-Modal Diabetic Retinopathy Classification
### DenseNet121 (Fundus) + DenseNet121 (OCT) · Fusion Layer · LightGBM · SHAP

> **Major Project** — AI-powered 3-class DR grading using paired retinal imaging modalities with full explainability.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cicYcsZg32RakChF-KBpIsVJRpRTVKcM?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-brightgreen)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)

---

## 📌 Overview

This project presents a **multi-modal deep learning pipeline** for Diabetic Retinopathy (DR) stage classification. Unlike single-modality systems, it fuses features from **two independent retinal imaging types** — Fundus photography and OCT scans — to achieve richer, more robust DR grading.

The pipeline uses two **DenseNet121** branches (one per modality), concatenates their feature vectors into a **2048-dim fused representation**, and feeds it into a **LightGBM classifier**. SHAP explainability reveals *which modality and which feature level* drives each prediction.

**DR Classes:**

| Class | Label | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | NPDR | Non-Proliferative Diabetic Retinopathy |
| 2 | PDR | Proliferative Diabetic Retinopathy |

---

## 🏗️ Architecture

```
Fundus Image ──► DenseNet121 (frozen, ImageNet) ──► 1024-dim features ──┐
                                                                          ├──► Concatenate (2048-dim)
OCT Image    ──► DenseNet121 (frozen, ImageNet) ──► 1024-dim features ──┘
                                                                          │
                                                                   StandardScaler
                                                                          │
                                                               LightGBM Classifier
                                                                          │
                                                          No DR  /  NPDR  /  PDR
```

**Why DenseNet121 for both branches?**
- Dense connectivity: each layer receives input from **all** previous layers → superior feature reuse
- Mitigates vanishing gradient → better gradient flow during training
- Symmetric 2048-dim fusion (1024 + 1024) enables **fair SHAP modality comparison**
- Smaller fused vector vs EfficientNet variants (2560-dim) → lower overfitting risk

---

## ✨ Key Features

- **Dual-modality fusion** — Fundus + OCT images analyzed in parallel
- **Patient-wise data split** — `GroupShuffleSplit` ensures zero patient overlap between train/test (no data leakage)
- **Paired augmentation** — Same random transforms applied to both modalities simultaneously
- **LightGBM classifier** — Fast, gradient-boosted tree ensemble with `class_weight=balanced`
- **SHAP explainability** — Summary plots, waterfall plots, and per-class modality contribution charts
- **Manual upload prediction** — Upload any Fundus + OCT pair and get instant DR stage + confidence

---

## 🔬 Pipeline Steps

| Step | Description |
|------|-------------|
| 1–2 | Install deps, mount Drive, import libraries |
| 3 | Configuration (IMG_SIZE=224, LGBM params, paths) |
| 4 | Load CSV labels for Fundus and OCT datasets |
| 5 | Pair Fundus + OCT by Patient ID + Eye key |
| 6 | Preprocess images (BGR→RGB, resize, normalize [0,1]) |
| 7 | Patient-wise GroupShuffleSplit (80/20, no leakage) |
| 8 | Paired augmentation (Rotate, Flip, Brightness, GaussNoise, Blur) |
| 9–12 | Build DenseNet121 feature extractors (GlobalAveragePooling2D) |
| 13 | Extract features for train + test sets |
| 14 | Fuse features → StandardScaler → LightGBM fit |
| 15 | Evaluate (Accuracy, Precision, Recall, F1, Classification Report) |
| 16 | Confusion matrix visualization |
| 17–18 | SHAP summary, modality contribution bar charts, waterfall plot |
| 19 | Manual upload prediction with confidence + per-class probabilities |
| 20 | Save all models to Google Drive |

---

## ⚙️ Configuration

```python
IMG_SIZE        = 224
BATCH_SIZE      = 8
NUM_CLASSES     = 3       # No DR, NPDR, PDR
TEST_SIZE       = 0.2
NUM_AUGMENTS    = 4       # augmented copies per training sample

LGBM_ESTIMATORS = 500
LGBM_LR         = 0.05
LGBM_LEAVES     = 63
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow / Keras, DenseNet121 (ImageNet pretrained) |
| ML Classifier | LightGBM (`class_weight=balanced`) |
| Explainability | SHAP (`TreeExplainer`) |
| Augmentation | Albumentations |
| Preprocessing | OpenCV, NumPy, Pandas |
| Visualization | Matplotlib |
| Environment | Google Colab / Jupyter Notebook |
| Model Persistence | Joblib (LightGBM, Scaler), Keras `.h5` (DenseNet) |

---

## 📊 Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision, Recall, F1 (Macro)** — Per-class and averaged
- **Confusion Matrix** — Visual breakdown across No DR / NPDR / PDR
- **SHAP Summary Plot** — Global feature importance per DR stage
- **Modality Contribution Chart** — Fundus vs OCT SHAP contribution (%) per class
- **Waterfall Plot** — Per-sample prediction explanation

---

## 🚀 Quick Start

**Run on Google Colab (recommended):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cicYcsZg32RakChF-KBpIsVJRpRTVKcM?usp=sharing)

1. Open the Colab link above
2. Mount your Google Drive
3. Set dataset paths in **Step 3** to match your Drive structure:
   ```
   /content/drive/MyDrive/octdataset/
   ├── eye fundus (2)/eye fundus/   ← Fundus images
   ├── OCT/OCT_NEW/                 ← OCT images
   ├── EYE FUNDUS.csv               ← Fundus labels
   └── OCT.csv                      ← OCT labels
   ```
4. Run all cells top-to-bottom

**Expected CSV format:**

| Name | DR |
|------|----|
| 1333_OI_f_2.jpg | NPDR |
| 1334_OD_f_1.jpg | 0 |
| 1335_OI_f_3.jpg | PDR |

DR label mapping: `'0'` → No DR · `'NPDR'` → Class 1 · `'PDR'` → Class 2

---

## 📦 Saved Model Outputs

After running Step 20, models are saved to Google Drive:

```
saved_models/
├── lgbm_dense_dense_ml.pkl          # LightGBM classifier
├── scaler_dense_dense_ml.pkl        # StandardScaler
├── densenet121_fundus.h5            # Fundus DenseNet121
└── densenet121_oct.h5               # OCT DenseNet121
```

---

## 🔗 Related Project

> **Mini Project (Foundation):** Single-modal DR detection using VGG16 + LightGBM  
> 🔗 [github.com/mloukikreddy/Diabetic-Retinopathy](https://github.com/mloukikreddy/Diabetic-Retinopathy)

This major project upgrades the mini project by:
- Replacing VGG16 with DenseNet121 (better gradient flow, feature reuse)
- Adding a second imaging modality (OCT) alongside Fundus
- Implementing patient-wise splitting to prevent data leakage
- Adding SHAP-based modality contribution analysis

---

## 👤 Author

**Loukik Reddy Mekala**  
📌 [github.com/mloukikreddy](https://github.com/mloukikreddy)

**Domain:** Artificial Intelligence · Medical Image Analysis · Multi-Modal Deep Learning

---

## 📄 License

This project is intended for academic and research purposes.
