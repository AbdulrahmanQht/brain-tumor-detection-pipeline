# Brain Tumor Detection Pipeline

A decoupled multi-stage deep learning pipeline for automated brain tumor detection and classification from MRI scans. The system physically isolates the tumor region before classification, eliminating the background noise bottleneck present in monolithic single-pass architectures.

---

## Pipeline Overview

```
MRI Scan (640×640 RGB)
        │
        ▼
┌───────────────────┐
│  YOLOv11 + CBAM   │  ← Tumor localization
│  (Localization)   │     CBAM suppresses healthy tissue noise
└───────────────────┘
        │
        ├── No detection → Label: "No Tumor"
        │
        ▼
┌───────────────────┐
│   Padded ROI      │  ← 10% padding around bbox
│   Cropping        │     Crop resized to 224×224
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  ResNet50 │ MobileNetV2 │ EfficientB0 │  ← Transfer learning
│           Classification              │     Two-stage fine-tuning
└───────────────────────────────────────┘
        │
        ▼
  Glioma │ Meningioma │ Pituitary │ No Tumor
```

---

## Key Features

- **Decoupled architecture**: localization and classification are separate, independently optimized stages
- **CBAM attention**: inserted into YOLOv11 neck to suppress healthy brain tissue and skull noise
- **Padded ROI cropping**: 10% bbox padding preserves tumor boundary context before classification
- **Three classifiers**: ResNet50 (primary), MobileNetV2, EfficientNetB0 trained on identical ROI crops for fair comparison
- **Two-stage transfer learning**: frozen head training followed by selective layer unfreezing at a lower learning rate
- **Class imbalance handling**: weighted cross-entropy loss compensating for the minority No Tumor class (592 vs 1116–1378 samples)
- **Bayesian hyperparameter optimization**: Optuna search over learning rate, dropout, batch size, and YOLO confidence threshold
- **Grad-CAM explainability**: heatmap visualizations confirming classifiers attend to the tumor region
- **Direct benchmarking**: results compared against 20 studies from the literature survey

---

## Dataset

**Source:** [Brain Tumor MRI — Roboflow Universe](https://universe.roboflow.com/eksperiment/brain-tumor-mri-ycidy)

| Split | Images |
|---|---|
| Train | 2,444 (62.6%) |
| Validation | 1,074 (27.5%) |
| Test | 385 (9.9%) |
| **Total** | **3,903** |

| Class | Count |
|---|---|
| Glioma | 1,321 |
| Meningioma | 1,378 |
| Pituitary | 1,116 |
| No Tumor | 592 |

All images are pre-resized to **640×640 RGB** by Roboflow. YOLO annotation format (normalized `.txt` labels per image).

---

## Project Structure

```
brain-tumor-detection-pipeline/
│
├── main.py                  # Global config, Optuna, pipeline execution controller
├── data.py                  # Dataset classes, preprocessing, augmentation, class weights
├── pipeline_logic.py        # All model classes and pipeline orchestrator
├── results_manager.py       # Metrics, plots, Grad-CAM, benchmarking
│
├── dataset/                 # Original Roboflow dataset (YOLO format)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── roi_dataset/             # Auto-generated — split-aligned 224×224 crops
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   ├── valid/
│   └── test/
│
├── weights/                 # Saved model weights
└── results/                 # Output plots, metrics, Grad-CAM images
```

> **Note:** `roi_dataset/` is generated once from the final trained YOLO model and then frozen. It is not rebuilt on subsequent runs unless `force_rebuild_roi: True` is set in CONFIG. This ensures classifier training is always reproducible.

---

## Installation

```bash
git clone https://github.com/your-username/brain-tumor-detection-pipeline.git
cd brain-tumor-detection-pipeline

pip install ultralytics torch torchvision optuna opencv-python pillow \
            matplotlib seaborn scikit-learn pytorch-grad-cam
```

---

## Usage

### 1. Configure
Edit the `CONFIG` dictionary in `main.py` to set your dataset path and adjust hyperparameters.

### 2. Run full pipeline
```bash
python main.py
```

This executes all stages in order:
1. Load and preprocess dataset
2. Train YOLOv11 + CBAM (stops at mAP@0.5:0.95 plateau)
3. Build ROI dataset from final YOLO weights (split-aligned, one-time)
4. Train all three classifiers using two-stage transfer learning
5. Run Optuna hyperparameter search
6. Generate all evaluation metrics and plots
7. Generate Grad-CAM visualizations
8. Output benchmark comparison table

### 3. Key CONFIG flags

| Flag | Default | Description |
|---|---|---|
| `run_optuna` | `True` | Enable Bayesian hyperparameter search |
| `optuna_trials` | `30` | Number of Optuna trials |
| `force_rebuild_roi` | `False` | Rebuild ROI dataset from scratch |
| `run_gradcam` | `True` | Generate Grad-CAM heatmaps after evaluation |
| `no_tumor_fallback` | `True` | No YOLO detection → classify as No Tumor |

---

## Evaluation Metrics

**Localization (YOLO):**
- Intersection over Union (IoU)
- mAP@0.5 and mAP@0.5:0.95 ← primary stopping criterion

**Classification (per classifier, per class):**
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix

**Outputs saved to `results/`:**
- Confusion matrices (one per classifier)
- Training loss and accuracy curves (Stage 1 and Stage 2 separately)
- Before/After sample images (original 640×640 scan → 224×224 ROI crop)
- Comparative table: ResNet50 vs MobileNetV2 vs EfficientNetB0
- Grad-CAM heatmaps on sample test ROIs
- Benchmark table vs literature

---

## Technical Details

| Component | Detail |
|---|---|
| Detector | YOLOv11n with CBAM in PAN neck (C2f insertion points) |
| YOLO stopping criterion | mAP@0.5:0.95 plateau (10 epochs patience) |
| ROI padding | 10% of bbox width/height on each side |
| Classifiers | ResNet50, MobileNetV2, EfficientNetB0 |
| Transfer learning | ImageNet pretrained → frozen head training (10 ep) → top-2 block fine-tuning (20 ep) |
| Optimizer | Adam |
| Loss | Weighted cross-entropy (class weights computed from training split only) |
| Hyperparameter search | Optuna — Bayesian optimization, 30 trials |
| XAI | Grad-CAM on final conv layer of each classifier |
| Input size (YOLO) | 640×640 RGB |
| Input size (classifiers) | 224×224 RGB, ImageNet normalized |

---

## Course

CS 522 — Selected Topics in Computer Science
Imam Abdulrahman Bin Faisal University, Department of Computer Science