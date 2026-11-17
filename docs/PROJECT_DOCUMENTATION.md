# ExplainX-Ray v2: Complete Project Documentation (A to Z)

*Status as of 14 November 2026. Covers everything from the original v1 project up to the uncommitted work currently in the repository.*

---

## Table of contents

1. [What the project is](#1-what-the-project-is)
2. [Background: why v2 exists (the v1 failures)](#2-background-why-v2-exists-the-v1-failures)
3. [Timeline: start to now](#3-timeline-start-to-now)
4. [System architecture](#4-system-architecture)
5. [Technology stack](#5-technology-stack)
6. [Datasets](#6-datasets)
7. [Data preparation](#7-data-preparation)
8. [Model](#8-model)
9. [Training](#9-training)
10. [Training results so far](#10-training-results-so-far)
11. [Inference pipeline (what happens when an X-ray is uploaded)](#11-inference-pipeline-what-happens-when-an-x-ray-is-uploaded)
12. [Explainability: Grad-CAM and zone-wise analysis](#12-explainability-grad-cam-and-zone-wise-analysis)
13. [Severity scoring: ASFI and ZCI](#13-severity-scoring-asfi-and-zci)
14. [Out-of-distribution (OOD) detection](#14-out-of-distribution-ood-detection)
15. [Decision thresholds (Youden's J)](#15-decision-thresholds-youdens-j)
16. [Clinical report generator](#16-clinical-report-generator)
17. [Backend API](#17-backend-api)
18. [Frontend](#18-frontend)
19. [Evaluation scripts](#19-evaluation-scripts)
20. [Configuration reference](#20-configuration-reference)
21. [File-by-file reference](#21-file-by-file-reference)
22. [How to set up and run](#22-how-to-set-up-and-run)
23. [Current state of the repository](#23-current-state-of-the-repository)
24. [Known issues and gaps](#24-known-issues-and-gaps)
25. [Recommended next steps](#25-recommended-next-steps)
26. [Glossary](#26-glossary)

---

## 1. What the project is

**ExplainX-Ray v2** is an explainable chest X-ray disease classifier.

- A user uploads a chest radiograph through a web interface.
- A **DenseNet121** convolutional neural network, trained on **NIH ChestX-ray14** (and optionally fine-tuned on **CheXpert**), predicts the probability of **8 pathologies**.
- For each detected finding, the system produces a **Grad-CAM heatmap** showing which parts of the image drove the prediction.
- The heatmap is split into a **6-zone lung grid** (Right/Left × Upper/Mid/Lower) with a per-zone activation percentage.
- A custom **ASFI severity score**, a **ZCI distribution pattern** (Focal / Multifocal / Diffuse), and an auto-generated **clinical-style report** are produced.
- Non-chest-X-ray inputs are rejected by an **out-of-distribution (OOD) gate**.

The 8 pathologies screened:

| # | Pathology | Anatomical region label used in the UI |
|---|-----------|----------------------------------------|
| 1 | Atelectasis | Basal lung fields |
| 2 | Cardiomegaly | Cardiac silhouette |
| 3 | Effusion | Costophrenic angle / pleural base |
| 4 | Infiltrate | Bilateral parenchyma |
| 5 | Mass | Focal parenchymal region |
| 6 | Nodule | Focal pulmonary parenchyma |
| 7 | Pneumonia | Lobar / bronchovascular parenchyma |
| 8 | Pneumothorax | Pleural space / lung margin |

These 8 classes were chosen deliberately: they are exactly the classes for which NIH provides **984 radiologist-drawn bounding boxes** (`BBox_List_2017.csv`). That makes it possible to measure quantitatively whether Grad-CAM points at the right place.

> **Disclaimer built into the product:** all output is for academic and demonstration purposes. It is not a diagnosis and has not been reviewed by a radiologist.

---

## 2. Background: why v2 exists (the v1 failures)

The earlier version (v1) had a **Node.js server that spawned a new Python subprocess for each request** to run the model. It broke in three separate ways:

1. **Wrong Python interpreter on Windows.** The spawned process did not use the virtual environment's Python.
2. **`require()` in an ES-module context.** The Node code crashed on module loading.
3. **Unhandled `stdin` EOF crash.** The subprocess communication broke.

There was also a **silent class-mismatch bug**: the training config's pathology list was edited from 8 to 14 classes *after* the model was trained. The saved checkpoint had 8 outputs, but the code expected 14. This only surfaced as a raw PyTorch shape-mismatch error once the subprocess crashes were fixed.

v1's best result was **0.82 validation AUROC on the 14-class set**.

### How v2 fixes these structurally

| v1 problem | v2 fix |
|------------|--------|
| Node spawning Python per request | **Flask is the inference server.** The model loads once into memory at startup. No subprocesses. |
| Checkpoint and class list drifting apart | `train.py` **always writes `best_model.pth` and `labels.json` together** in `save_checkpoint()`. |
| Cryptic shape-mismatch errors | `model_service.py` **refuses to start** if either file is missing or if the checkpoint's output count differs from `labels.json`, with a specific error message. |
| Risk of fake/mock results | When the model is not ready, `/api/analyze` returns **HTTP 503** with instructions. It never returns fake data. |
| Client-side canvas compositing bugs | The heatmap overlay is **rendered on the server** and sent as a ready-made base64 PNG. |

---

## 3. Timeline: start to now

Dates come from git commits and file modification times.

| Date (2026) | What happened |
|-------------|---------------|
| Before Aug | **v1** built (Node + Python subprocess, 14 classes, 0.82 AUROC). Failed in the ways described in Section 2. |
| **19 Aug** | v2 core written: `config.py`, `model.py`, `gradcam_utils.py`. Python virtual environment `cxr-env` created. |
| **20 Aug, ~04:18** | **First v2 training run finished** on NIH ChestX-ray14, 8 classes. `best_model.pth` + `labels.json` saved (this checkpoint now lives in `trained_model_old/`). |
| **20 Aug, ~10:09** | Training curves (`loss_curve.png`, `auroc_curve.png`) and `history.json` produced. |
| **20 Aug, ~19:26** | `severity.py` (ASFI, ZCI) and `report_generator.py` (clinical report) added. `requirements.txt` updated with `scikit-image`. |
| **20–21 Aug** | Backend run and tested end-to-end. Server log shows the model loaded on CUDA with 8 classes and several successful `/api/analyze` calls. |
| **9 Sep** | **Commit 1** `81682cfb`: "Initial commit: ExplainX-Ray v2 - Flask + React chest X-ray classifier" (28 files, 3,897 lines). |
| **29 Sep** | `ood_detection.py` written (grayscale check + Mahalanobis distance). |
| **30 Sep** | `fit_ood_detector.py` and `prepare_chexpert.py` written. CheXpert-v1.0-small extracted to `backend/data/chexpert`, manifests generated. |
| **30 Sep, ~15:34** | **CheXpert fine-tuning experiment** produced a checkpoint (fine-tuned from the NIH model). |
| **1 Oct** | Experiment curves/history saved. `config.py` and `app.py` updated: model path and port can be overridden by environment variables, so two backends can run side by side. |
| **16 Oct** | `dataset.py`, `prepare_data.py`, `evaluate_test.py`, `evaluate_localization.py`, `evaluate_chexpert.py` revised (observation masks, `Infiltration`→`Infiltrate` mapping fixes, test evaluation with Youden thresholds). Old models moved to `trained_model_old/`; a new empty `trained_model/chexpert_experiment/` was created. |
| **28 Oct, 15:48** | **Commit 2** `f19724df`: "Add CheXpert evaluation and OOD detection" (16 files, +814/−54). |
| **28 Oct, 15:50** | **Uncommitted work**: per-class Youden's J thresholds in training and inference, an "all probabilities and thresholds" panel in the UI, and README additions on scope. |
| **14 Nov** | This document. |

---

## 4. System architecture

```
┌─────────────────────────────┐        /api/*  (Vite proxy)       ┌──────────────────────────────────────┐
│  React frontend (Vite)      │  ───────────────────────────────▶ │  Flask backend  (app.py, port 5000)  │
│  localhost:5173             │                                   │                                      │
│  - UploadModal              │  POST /api/analyze {imageBase64}  │  model_service.py                    │
│  - DiagnosticStudio         │ ◀───────────────────────────────  │   ├─ DenseNet121 (loaded once, GPU)  │
│  - FindingsPanel            │   JSON: probabilities, findings,  │   ├─ ood_detection.py                │
│  - ClinicalReport           │   heatmap PNG, zones, report      │   ├─ gradcam_utils.py                │
└─────────────────────────────┘                                   │   ├─ severity.py (ASFI, ZCI)         │
                                                                  │   └─ report_generator.py             │
                                                                  └──────────────────────────────────────┘
Offline (command line):
  prepare_data.py → prepare_chexpert.py → train.py → fit_ood_detector.py
  evaluate_test.py · evaluate_chexpert.py · evaluate_localization.py
```

**Key design decisions**

- **Single process, model loaded once.** `debug=False` in Flask so the auto-reloader does not reload the model onto the GPU on every file save.
- **Vite dev proxy** forwards `/api/*` to Flask, so there is no hard-coded backend host in the frontend and no CORS setup during development (Flask also enables CORS as a fallback).
- **Experimental models are kept separate.** Training writes to `trained_model/chexpert_experiment/`, never over the served model. A model is promoted only after evaluation, by hand.

---

## 5. Technology stack

**Backend (Python)**

| Package | Version | Use |
|---------|---------|-----|
| Flask | 3.0.3 | HTTP API |
| flask-cors | 4.0.1 | CORS |
| torch | 2.13.0+cu130 | Deep learning (CUDA 13.0 build) |
| torchvision | 0.28.0+cu130 | DenseNet121, transforms |
| numpy | ≥1.26 | Arrays |
| pandas | ≥2.2 | CSV handling |
| Pillow | ≥10.2 | Image I/O |
| matplotlib | ≥3.8 | Curves, jet colormap |
| scikit-learn | ≥1.4 | AUROC, ROC, splits, metrics |
| scikit-image | ≥0.22 | Otsu thresholding |

**Frontend (JavaScript)**

| Package | Version |
|---------|---------|
| React / React-DOM | ^18.3.1 |
| Vite | ^5.4.1 |
| @vitejs/plugin-react | ^4.3.1 |

Fonts: Space Grotesk, IBM Plex Mono, Inter (Google Fonts).

**Hardware:** NVIDIA RTX 4050 laptop GPU (CUDA), Windows 11.

---

## 6. Datasets

### 6.1 NIH ChestX-ray14 (primary)

- About 45 GB, **112,120 frontal chest X-ray PNGs** from 30,805 patients.
- Labels are text-mined from radiology reports (weak labels). Multi-label: one image can have several findings (`Finding Labels` separated by `|`).
- Location: `backend/data/images/` and `backend/data/Data_Entry_2017.csv`.
- Optional: `backend/data/BBox_List_2017.csv`, the 984 radiologist bounding boxes used for localization evaluation.
- NIH calls one finding **"Infiltration"**. The project uses the display name **"Infiltrate"** and maps between the two.

### 6.2 CheXpert-v1.0-small (fine-tuning experiment)

- Stanford dataset, downscaled version. Extracted to `backend/data/chexpert/` (64,540 patient folders under `train/`).
- Labels are `1` (positive), `0` (negative), `-1` (uncertain), or blank (not mentioned).
- Only **frontal** images are used.
- Only **5 labels overlap** with the project's 8 classes:

| Project class | CheXpert column |
|---------------|-----------------|
| Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly |
| Effusion | Pleural Effusion |
| Pneumonia | Pneumonia |
| Pneumothorax | Pneumothorax |
| Infiltrate, Mass, Nodule | *not available in CheXpert* |

---

## 7. Data preparation

### 7.1 `prepare_data.py` (NIH)

1. Reads `Data_Entry_2017.csv`.
2. Creates one 0/1 column for each of the 8 pathologies (mapping `Infiltration` → `Infiltrate`).
3. **Splits by patient, never by image**, so follow-up scans of the same patient cannot leak between train and test:
   - 70% train, 15% validation, 15% test (`random_state=42`).
4. Writes `train.csv`, `val.csv`, `test.csv` to `backend/data/` and prints image counts, patient counts and positive counts per split.

Current split sizes (from the CSVs on disk):

| Split | Images |
|-------|--------|
| Train | 78,566 |
| Validation | 17,063 |
| Test | 16,491 |
| **Total** | **112,120** |

### 7.2 `prepare_chexpert.py` (CheXpert)

1. Reads `train.csv` and `valid.csv` from the CheXpert folder (`--dataset-root`).
2. Keeps frontal images only; normalizes image paths; extracts the patient ID from the path.
3. For each of the 8 classes, writes a label column **and an observation mask column** (`<name>__observed`):
   - `1.0` or `0.0` → observed.
   - Uncertain (`-1`) or blank → **unobserved** (ignored in the loss).
   - NIH-only classes (Infiltrate, Mass, Nodule) → always unobserved for CheXpert rows.
4. Splits CheXpert *train* by patient into 90% train / 10% validation (`GroupShuffleSplit`, `random_state=42`).
5. **Keeps the official CheXpert `valid` split untouched** as a held-out external test set.

Output in `backend/data/chexpert_manifests/`:

| File | Frontal images | Purpose |
|------|----------------|---------|
| `train.csv` | 171,927 | Fine-tuning |
| `validation.csv` | 19,100 | Checkpoint selection, LR scheduling, threshold tuning |
| `official_valid.csv` | 202 | Final external evaluation only |

### 7.3 `dataset.py` (PyTorch loading)

- `ChestXrayDataset` (NIH): returns `(image, labels, observed)` where `observed` is all ones, because NIH treats absent findings as negatives.
- `CheXpertDataset`: returns `(image, labels, observed)` using the mask columns.
- **Training augmentation:** resize to 224×224, random horizontal flip (p=0.5), random rotation ±10°, ImageNet normalization.
- **Evaluation transform:** resize to 224×224 and ImageNet normalization only.
- **Mixed loader (when CheXpert manifests exist):**
  - NIH train and CheXpert train are concatenated.
  - A `WeightedRandomSampler` draws **80% CheXpert / 20% NIH** per epoch. The NIH share is a "replay set" that keeps signal for the three NIH-only classes and reduces catastrophic forgetting.
  - Epoch length equals the CheXpert train size (171,927 samples).
- Without CheXpert manifests, it trains on NIH only with normal shuffling.
- `num_workers=0` (safe default on Windows).
- Helper loaders: `get_chexpert_validation_loader()` and `get_chexpert_official_valid_loader()`.

---

## 8. Model

`model.py` → `build_model(num_classes, pretrained=True)`

- **DenseNet121** from torchvision, initialized with **ImageNet weights** (`IMAGENET1K_V1`).
- The final classifier is replaced with `nn.Linear(1024, num_classes)` (8 outputs).
- Each output goes through a **sigmoid** (multi-label: classes are independent, not softmax).
- Checkpoint size: about 28 MB.
- Architecture is unchanged from v1. The v1 problem was never the architecture.

---

## 9. Training

`train.py`

| Setting | Value |
|---------|-------|
| Image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 15 |
| Optimizer | Adam, learning rate 1e-4 |
| Loss | `BCEWithLogitsLoss(reduction="none")`, **masked by the observation mask** and averaged over observed labels only |
| LR schedule | `ReduceLROnPlateau` (factor 0.5, patience 2) on the selection loss |
| Initialization | Fine-tunes from `trained_model/best_model.pth` if it exists; otherwise starts from ImageNet |
| Checkpoint selection | Best mean AUROC. With CheXpert manifests present, selection uses the **CheXpert-only patient-held-out validation set**, not the mixed set. |

**Each epoch:**

1. Train one epoch with the masked loss.
2. Evaluate on the validation loader: loss, mean AUROC, per-class AUROC (only on observed labels).
3. If CheXpert is present, also evaluate on the CheXpert validation set and use that for selection and LR scheduling.
4. If the selection AUROC improves:
   - Save `best_model.pth` + `labels.json` to `trained_model/chexpert_experiment/` (never overwriting the served model).
   - *(Uncommitted)* Compute per-class **Youden's J thresholds** on the selection validation set and save `thresholds.json` beside the checkpoint.

**After training:** saves `history.json`, `loss_curve.png` and `auroc_curve.png` into the experiment folder.

### How training evolved between commit 1 and commit 2

| Commit 1 (initial) | Commit 2 (CheXpert) |
|--------------------|---------------------|
| Plain BCE loss over all labels | Masked BCE: uncertain/missing labels ignored |
| Saved directly over the served model | Saves to a separate experiment folder |
| Always started from ImageNet | Fine-tunes from the existing checkpoint if present |
| Selection on NIH validation AUROC | Selection on CheXpert-only validation AUROC when available |
| Printed mean AUROC only | Also prints per-class AUROC |

---

## 10. Training results so far

### 10.1 Run 1: NIH only, 8 classes (20 Aug)

Source: `backend/history.json`, `backend/loss_curve.png`, `backend/auroc_curve.png`.

| Epoch | Train loss | Val loss | Val mean AUROC |
|-------|-----------|----------|----------------|
| 1 | 0.1606 | 0.1487 | 0.8093 |
| 3 | 0.1444 | 0.1434 | 0.8263 |
| 5 | 0.1387 | 0.1449 | 0.8290 |
| **6** | **0.1360** | **0.1429** | **0.8321 ← best** |
| 8 | 0.1306 | 0.1452 | 0.8231 |
| 10 | 0.1178 | 0.1504 | 0.8211 |
| 12 | 0.1067 | 0.1591 | 0.8102 |
| 15 | 0.0860 | 0.1771 | 0.7925 |

- **Best validation mean AUROC: 0.832 at epoch 6.** This is above v1's 0.82 (which was on 14 classes, so not a strict comparison).
- After epoch 6 the model **overfits**: train loss keeps falling while validation loss rises and AUROC drops. Only the epoch-6 weights were saved, so the deployed model is the good one.
- This checkpoint is now at `backend/trained_model_old/best_model.pth`.

### 10.2 Run 2: CheXpert fine-tuning experiment (30 Sep – 1 Oct)

Source: `backend/trained_model_old/chexpert_experiment/history.json`.

Fine-tuned from the Run 1 checkpoint with the 80/20 CheXpert/NIH mix.

| Epoch | Train loss | Mixed val loss | Mixed val mean AUROC |
|-------|-----------|----------------|----------------------|
| 1 | 0.1625 | 0.1693 | 0.8918 |
| **2** | **0.1560** | **0.1672** | **0.8956 ← best** |
| 5 | 0.1422 | 0.1733 | 0.8854 |
| 10 | 0.0940 | 0.2183 | 0.8676 |
| 15 | 0.0685 | 0.2483 | 0.8580 |

- The recorded AUROC is on the **mixed NIH + CheXpert validation set**, so it is **not directly comparable** with Run 1's NIH-only 0.832.
- The checkpoint itself was selected on the CheXpert-only validation AUROC, which is printed during training but not stored in `history.json`.
- Overfitting starts very early (after epoch 2).
- Whether this checkpoint beats the baseline on unseen data has to be confirmed with `evaluate_chexpert.py` (official CheXpert valid) and `evaluate_test.py` (NIH test). **No saved result files from these evaluations exist in the repository yet.**

---

## 11. Inference pipeline (what happens when an X-ray is uploaded)

`model_service.analyze_image_bytes()`:

1. **Model check.** If the model failed to load, raise `ModelNotReadyError` → HTTP 503.
2. **Decode** the image and convert it to RGB. Make a 224×224 copy for the overlay.
3. **Preprocess:** resize to 224×224, tensor, ImageNet normalization.
4. **Single forward pass** with a hook on the classifier that captures the 1024-dimensional penultimate feature vector (so OOD checking adds no second inference).
5. **Sigmoid** → 8 probabilities.
6. **OOD check** (Section 14). If the image is out of distribution, return immediately with `outOfDistribution: true` and no findings.
7. **Thresholding** (Section 15): keep classes whose probability ≥ that class's threshold, sorted by probability, **maximum 3 findings**.
8. **For each finding:**
   - Compute Grad-CAM for that class.
   - Compute 6-zone breakdown and the peak zone.
   - Compute ASFI score, severity tag, ZCI and distribution pattern.
9. **Overlay:** blend the top finding's heatmap (jet colormap, alpha 0.45) onto the image → base64 PNG.
10. **Clinical report** generation.
11. Return JSON (schema in Section 17).

### How finding selection evolved

| Version | Rule |
|---------|------|
| Commit 1 | Threshold = max(0.25, 0.45 × top probability), top 3. **Always forced at least one finding**, even if every class was low. |
| Commit 2 | Same rule, but **no longer forces a finding**. Reason: unrelated images (for example a leg X-ray) were being reported as "6% Effusion". |
| Uncommitted | **Per-class calibrated Youden's J thresholds** from `thresholds.json`; 0.5 fallback when no calibration file exists. |

---

## 12. Explainability: Grad-CAM and zone-wise analysis

### 12.1 Grad-CAM (`gradcam_utils.compute_gradcam`)

1. Hook DenseNet121's final feature block (`model.features`) during a forward pass.
2. Take the gradient of the target class score with respect to those feature maps.
3. Weight each channel by its mean gradient, sum the weighted maps.
4. Apply ReLU, normalize to [0, 1], bilinearly upsample to 224×224.
5. If the hook captures nothing, it **raises an error** rather than returning a placeholder.

### 12.2 Zone breakdown (`zone_breakdown`)

- The 224×224 CAM is divided into a **3 rows × 2 columns** grid: Upper/Mid/Lower × Right/Left.
- Each zone's score is the **mean activation × 100**.
- Zones are returned row by row, which matches the frontend's 2-column CSS grid exactly.
- The zone numbers come from the **same array** that draws the heatmap, so the image and the percentages always agree.

### 12.3 Overlay (`make_overlay_base64`)

- Jet colormap applied to the CAM, blended at 45% over the 224×224 image, encoded as a `data:image/png;base64,...` string for direct use in an `<img>` tag.

---

## 13. Severity scoring: ASFI and ZCI

Defined in `severity.py`. These are **project-specific model-derived indicators, not validated clinical scores.**

### 13.1 ASFI (Activation-based Severity Index)

1. Compute **Otsu's threshold** on that scan's CAM (adapts per image).
2. **AF** (activated fraction) = share of pixels above the threshold.
3. **MI** (mean intensity) = mean CAM value of those pixels.
4. **P** = model probability for the class.

```
ASFI = 100 × (0.40 × AF + 0.30 × MI + 0.30 × P)
```

| ASFI | Severity tag |
|------|--------------|
| ≥ 75 | Critical |
| ≥ 55 | Severe |
| ≥ 35 | Moderate |
| ≥ 15 | Mild |
| < 15 | Normal |

Empty or constant CAMs return 0.

### 13.2 ZCI (Zone Concentration Index)

The **Herfindahl–Hirschman index** over the 6 zone shares:

```
ZCI = Σ (zone_activation / total_activation)²
```

Ranges from 1/6 ≈ 0.167 (spread evenly) to 1.0 (all in one zone).

| ZCI | Pattern |
|-----|---------|
| ≥ 0.55 | Focal |
| ≥ 0.30 | Multifocal |
| < 0.30 | Diffuse |

---

## 14. Out-of-distribution (OOD) detection

Added in commit 2 (`ood_detection.py`, `fit_ood_detector.py`). Two gates, cheapest first:

### Gate 1: grayscale check (always active)

- Computes the mean absolute difference between the R, G and B channels.
- Real radiographs are grayscale, so the score should be near 0.
- If the score is **> 0.04**, the image is rejected: *"Image does not appear to be a grayscale radiograph (significant colour content detected)."*

### Gate 2: Mahalanobis feature distance (active only after calibration)

- `fit_ood_detector.py` runs the training images through the model, collects the 1024-d penultimate features, and saves the **mean**, the **pseudo-inverse covariance**, and a **threshold = 99th percentile** of training distances to `trained_model/ood_stats.npz`.
- At inference, the uploaded image's feature distance is compared with that threshold. Above it → rejected: *"Image features fall well outside the distribution of the model's training data."*
- If `ood_stats.npz` does not exist, this gate is skipped and the response says `featureCheckAvailable: false`.

**Frontend behaviour:** if `outOfDistribution` is true, all three panels (studio, findings, report) are replaced with an "Analysis not performed" warning showing the reason.

---

## 15. Decision thresholds (Youden's J)

*(This is the uncommitted work from 28 Oct.)*

**Problem:** a single fixed cutoff does not suit every class. Some pathologies are rare, so their predicted probabilities are naturally low.

**Solution:** for each class, choose the cutoff that maximizes **Youden's J = sensitivity + specificity − 1 = TPR − FPR** on the ROC curve, using **validation data only** (never test data).

- **Training (`train.py`, `find_youden_thresholds`)**: whenever a new best checkpoint is saved, thresholds are computed on the selection validation set (CheXpert validation if present, otherwise NIH validation) using observed labels only. Saved as:
  ```json
  { "method": "youden_j", "thresholds": { "Atelectasis": 0.xx, ... } }
  ```
  to `trained_model/chexpert_experiment/thresholds.json`. If a class has only one label value in validation, it falls back to 0.5.
- **Inference (`model_service._load_thresholds`)**: reads `thresholds.json` from **the same folder as the active `labels.json`**. Missing file, unreadable file, or values outside [0, 1] → 0.5 for that class.
- **API:** the response now includes `classThresholds`.
- **UI:** a collapsible section "All class probabilities and decision thresholds" lists every class with its probability, "Above/Below threshold" and the cutoff. When no class passes, the findings panel says *"No class cleared its decision threshold for this scan."*

`evaluate_test.py` uses the same method independently: thresholds are fitted on the NIH validation split and then applied to the test split.

---

## 16. Clinical report generator

`report_generator.py` builds the report **only from computed values** (no free text from a language model).

- **Summary:** "Primary finding: *X* (*p*% probability, *severity* severity)." plus the count of additional findings; or "No findings met the model's reporting threshold for this scan."
- **Per-finding explanation:** probability, percentage of the field activated, distribution pattern phrase (focal / multifocal / diffuse with the peak zone), ZCI, ASFI and the severity category.
- **Methodology paragraph:** DenseNet121, NIH ChestX-ray14 transfer learning, Grad-CAM on the last convolutional block, Otsu thresholding, 6-region grid.
- **Disclaimer:** academic/demonstration use, not reviewed by a radiologist, not a diagnosis.

---

## 17. Backend API

Base URL: `http://localhost:5000` (port configurable with the `PORT` environment variable).

### `GET /api/health`

```json
{
  "ready": true,
  "numClasses": 8,
  "pathologies": ["Atelectasis", "..."],
  "device": "cuda",
  "error": null
}
```

### `POST /api/analyze`

**Request**
```json
{ "imageBase64": "data:image/png;base64,iVBORw0..." }
```
The `data:...;base64,` prefix is optional.

**Response (normal case)**
```json
{
  "modelReady": true,
  "numClasses": 8,
  "pathologiesScreened": ["..."],
  "allProbabilities": { "Effusion": 0.6123, "...": 0.0 },
  "classThresholds": { "Effusion": 0.5, "...": 0.5 },
  "findings": [
    {
      "id": "f-1",
      "name": "Effusion",
      "probability": 0.6123,
      "severity": "Moderate",
      "anatomicalRegion": "Costophrenic angle / pleural base",
      "primaryZone": "Right Lower",
      "zones": [ { "zone": "Right Upper", "row": "Upper", "col": "Right", "activation": 12.3 }, "..." ],
      "asfi": { "score": 48.2, "activatedFraction": 0.31, "meanIntensity": 0.62, "otsuThreshold": 0.41 },
      "distribution": { "zci": 0.34, "pattern": "Multifocal" }
    }
  ],
  "heatmapOverlay": "data:image/png;base64,...",
  "primaryZones": [ "...6 zones of the top finding..." ],
  "clinicalReport": { "summary": "...", "findings": [ "..." ], "methodology": "...", "disclaimer": "..." }
}
```

**Response (OOD case)**
```json
{
  "modelReady": true,
  "outOfDistribution": true,
  "outOfDistributionReason": "Image does not appear to be a grayscale radiograph ...",
  "outOfDistributionDetails": { "channelSimilarityScore": 0.12 },
  "findings": [], "heatmapOverlay": null, "primaryZones": [], "clinicalReport": null
}
```

**Errors**

| Code | When |
|------|------|
| 400 | `imageBase64` missing or not decodable |
| 503 | Model not trained / files missing / class-count mismatch |
| 500 | Any other inference failure (traceback printed on the server) |

---

## 18. Frontend

React 18 + Vite, dark clinical-style theme (`index.css`, about 690 lines).

| Component | Role |
|-----------|------|
| `App.jsx` | Header with brand and model status pill ("Model ready · 8 classes · cuda" / "Model not trained"), "Load X-ray scan" button, tabs (Diagnostic studio / Clinical report), API calls, OOD warning panels. |
| `UploadModal.jsx` | Drag-and-drop or click-to-browse image upload (PNG/JPG/WEBP), preview with file name and size, "Analyze X-ray" button. |
| `DiagnosticStudio.jsx` | Shows the uploaded image, replaced by the Grad-CAM overlay after analysis; a 2×3 **zone grid overlay** with percentages (the hottest zone highlighted); a **zone strip** of bars below; a scan-line loading animation; error messages. |
| `FindingsPanel.jsx` | Up to 3 finding cards: name, probability bar coloured by severity, severity tag, anatomical region, peak zone, ASFI score, pattern. *(Uncommitted)* collapsible table of all probabilities vs thresholds. Footer: "Screened for: ..." list. |
| `ClinicalReport.jsx` | Summary, per-finding explanations, methodology, disclaimer. |

**Dev proxy** (`vite.config.js`): `/api` → `VITE_API_TARGET` or `http://127.0.0.1:5000`. Dev server port 5173.

---

## 19. Evaluation scripts

| Script | What it measures | Output |
|--------|------------------|--------|
| `evaluate_test.py` | Served model on the **NIH test split**. Youden thresholds fitted on validation only, then test AUROC, precision, recall, F1 per class and mean AUROC. | `test_evaluation.csv`, `test_evaluation.json` |
| `evaluate_chexpert.py` | **Baseline vs fine-tuned** model on the untouched **official CheXpert valid** (202 images). AUROC, AUPRC, precision/recall/F1 at 0.5, loss, and the AUROC change. | Console |
| `evaluate_localization.py` | **Grad-CAM localization quality**: for each of the 984 NIH bounding boxes, computes the Otsu-thresholded CAM mask and its **IoU** with the box (boxes scaled from original size to 224). Mean IoU per pathology and overall. | `localization_results.csv`, `localization_iou.png` |
| `fit_ood_detector.py` | Not an evaluation, but a calibration step for the Mahalanobis OOD gate. | `trained_model/ood_stats.npz` |

---

## 20. Configuration reference

All in `backend/config.py` (single source of truth).

| Name | Default | Notes |
|------|---------|-------|
| `DATA_DIR` | `backend/data` | |
| `IMAGE_DIR` | `data/images` | NIH PNGs |
| `CSV_PATH` | `data/Data_Entry_2017.csv` | |
| `BBOX_CSV_PATH` | `data/BBox_List_2017.csv` | Localization eval only |
| `TRAIN_CSV` / `VAL_CSV` / `TEST_CSV` | `data/*.csv` | From `prepare_data.py` |
| `CHEXPERT_DIR` | env `CHEXPERT_DIR` or `data/chexpert` | |
| `CHEXPERT_MANIFEST_DIR` | `data/chexpert_manifests` | |
| `MODEL_DIR` | `backend/trained_model` | |
| `MODEL_PATH` | env `XRAY_MODEL_PATH` or `trained_model/best_model.pth` | Served model |
| `LABELS_PATH` | env `XRAY_LABELS_PATH` or `trained_model/labels.json` | |
| `EXPERIMENT_MODEL_DIR` | `trained_model/chexpert_experiment` | Training output |
| `DEFAULT_PATHOLOGIES` | 8 classes | |
| `IMAGE_SIZE` | 224 | |
| `BATCH_SIZE` | 16 | |
| `NUM_EPOCHS` | 15 | |
| `LEARNING_RATE` | 1e-4 | |
| `DEVICE` | cuda if available else cpu | |
| `ZONE_ROWS` / `ZONE_COLS` | Upper/Mid/Lower, Right/Left | |

Other environment variables: `PORT` (Flask port), `VITE_API_TARGET` (frontend proxy target).

---

## 21. File-by-file reference

```
explainx-ray-v2/
├── README.md                     Setup guide and design rationale
├── .gitignore                    Ignores cxr-env, data, trained_model, node_modules, logs
├── docs/
│   └── PROJECT_DOCUMENTATION.md  This document
├── backend/
│   ├── app.py                    Flask API: /api/health, /api/analyze
│   ├── config.py                 Paths, classes, hyperparameters, zone grid
│   ├── model.py                  DenseNet121 builder
│   ├── dataset.py                NIH + CheXpert datasets, mixed sampler, loaders
│   ├── prepare_data.py           NIH patient-level 70/15/15 split
│   ├── prepare_chexpert.py       CheXpert manifests with observation masks
│   ├── train.py                  Training loop, masked loss, Youden thresholds
│   ├── model_service.py          Model loading/validation, full inference pipeline
│   ├── gradcam_utils.py          Grad-CAM, zone breakdown, overlay PNG
│   ├── severity.py               ASFI, severity tags, ZCI, distribution pattern
│   ├── report_generator.py       Template-based clinical report
│   ├── ood_detection.py          Grayscale + Mahalanobis OOD gates
│   ├── fit_ood_detector.py       Fits OOD statistics
│   ├── evaluate_test.py          NIH test evaluation
│   ├── evaluate_chexpert.py      Baseline vs fine-tuned on CheXpert valid
│   ├── evaluate_localization.py  Grad-CAM IoU vs NIH bounding boxes
│   ├── requirements.txt          Python dependencies (CUDA 13.0 PyTorch)
│   ├── history.json              Run 1 (NIH) training history
│   ├── loss_curve.png            Run 1 loss curve
│   ├── auroc_curve.png           Run 1 AUROC curve
│   ├── server.stdout.log / server.stderr.log   Logs from 20–21 Aug test run (ignored by git)
│   ├── cxr-env/                  Python virtual environment (ignored)
│   ├── data/                     NIH + CheXpert data and split CSVs (ignored)
│   ├── trained_model/            Served model location; currently EMPTY (ignored)
│   └── trained_model_old/        Run 1 model + Run 2 experiment (NOT ignored, untracked)
└── frontend/
    ├── index.html, package.json, vite.config.js
    └── src/
        ├── main.jsx, App.jsx, index.css
        └── components/ UploadModal, DiagnosticStudio, FindingsPanel, ClinicalReport
```

---

## 22. How to set up and run

All commands are PowerShell, run from the project folder.

### 22.1 Backend environment
```powershell
cd backend
python -m venv cxr-env
.\cxr-env\Scripts\activate
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 22.2 Data
Place `backend/data/images/` and `backend/data/Data_Entry_2017.csv` (and optionally `BBox_List_2017.csv`). Then:
```powershell
python prepare_data.py
```

### 22.3 Optional CheXpert
```powershell
$env:CHEXPERT_DIR = (Resolve-Path ".\data\chexpert").Path
python prepare_chexpert.py --dataset-root $env:CHEXPERT_DIR
```

### 22.4 Train
```powershell
python train.py
```
Output goes to `trained_model/chexpert_experiment/`. To **serve** it, copy `best_model.pth`, `labels.json` and `thresholds.json` into `trained_model/` (or use the environment variables below).

### 22.5 Calibrate OOD (optional but recommended)
```powershell
python fit_ood_detector.py
```

### 22.6 Evaluate
```powershell
python evaluate_test.py
python evaluate_chexpert.py
python evaluate_localization.py
```

### 22.7 Run the app
```powershell
# terminal 1
cd backend; .\cxr-env\Scripts\activate; python app.py
# terminal 2
cd frontend; npm install; npm run dev
```
Open `http://localhost:5173` → **Load X-ray scan** → choose image → **Analyze X-ray**.

### 22.8 Run an experimental model side by side
```powershell
# backend on 5001
$env:XRAY_MODEL_PATH = (Resolve-Path ".\trained_model\chexpert_experiment\best_model.pth").Path
$env:XRAY_LABELS_PATH = (Resolve-Path ".\trained_model\chexpert_experiment\labels.json").Path
$env:PORT = "5001"
python app.py
# frontend on 5174
$env:VITE_API_TARGET = "http://127.0.0.1:5001"
npm run dev -- --port 5174
```

---

## 23. Current state of the repository

**Branch:** `main`. **Commits:** 2 (`81682cfb`, `f19724df`).

**Uncommitted modifications**

| File | Change |
|------|--------|
| `backend/train.py` | `find_youden_thresholds()`; writes `thresholds.json` with each best checkpoint |
| `backend/model_service.py` | `_load_thresholds()`; per-class thresholds replace the relative 0.25/0.45 rule; `classThresholds` in the response |
| `frontend/src/components/FindingsPanel.jsx` | All-probabilities/thresholds table; new empty-state message |
| `frontend/src/index.css` | Styles for that table (with a narrow-screen layout) |
| `README.md` | Threshold explanation; "Scope of the displayed analysis" section |

**Untracked:** `backend/trained_model_old/` (two checkpoints, about 57 MB total).

**Model folders on disk**

| Folder | Contents |
|--------|----------|
| `trained_model/` | Only an **empty** `chexpert_experiment/` subfolder. **No served model.** |
| `trained_model_old/` | Run 1 NIH `best_model.pth` + `labels.json`; `chexpert_experiment/` with Run 2 checkpoint, labels, history and curves. No `thresholds.json` and no `ood_stats.npz` anywhere. |

**Data on disk:** NIH (112,120 images + split CSVs) and CheXpert (images + 3 manifests) are both present.

**Evaluation output files:** none present yet (`test_evaluation.*`, `localization_results.csv` were not found).

---

## 24. Known issues and gaps

1. **The app currently has no model to serve.** `trained_model/` is empty, so the header will show "Model not trained" and `/api/analyze` will return 503. Copy a checkpoint and its `labels.json` back from `trained_model_old/` (or train again) to restore it.
2. **Running `train.py` now starts from scratch.** Because `trained_model/best_model.pth` is missing, training starts from ImageNet weights, not from the NIH model. Since CheXpert manifests exist, it will train on the 80/20 mixed data.
3. **No calibrated thresholds exist yet for any saved model.** `thresholds.json` is only produced by the new (uncommitted) training code. Both existing checkpoints will use the **0.5 fallback** for every class. With the class imbalance in NIH, most scans will probably show no findings above 0.5. Running training again, or computing thresholds once for an existing checkpoint, is needed.
4. **`thresholds.json` is written only to the experiment folder.** When promoting a model to `trained_model/`, `thresholds.json` must be copied too, or the 0.5 fallback is used silently.
5. **`evaluate_test.py` ignores the observation mask.** When CheXpert manifests exist, its validation loader is the mixed NIH + CheXpert set, and CheXpert's unobserved labels (Infiltrate, Mass, Nodule, uncertain values) are treated as negatives. That biases the validation-fitted thresholds. Test AUROC is unaffected because the test loader is NIH-only.
6. **OOD Mahalanobis gate is not calibrated.** `ood_stats.npz` does not exist, so only the grayscale check is active. A grayscale non-chest image (for example a leg X-ray) can still pass.
7. **Overfitting.** Both runs peak early (epoch 6 and epoch 2) and keep training for 15 epochs. Early stopping would save time.
8. **No random seeds for PyTorch**, so training runs are not exactly reproducible (the data splits are seeded).
9. **`trained_model_old/` is not in `.gitignore`.** A careless `git add .` would commit about 57 MB of weights.
10. **Documentation drift.** The README's "Project layout" does not list the newer files (OOD, CheXpert, evaluation scripts). `config.py` mentions `scripts/inspect_bboxes.py`, which does not exist. The report's methodology paragraph mentions only NIH, not the CheXpert fine-tuning.
11. **Scope limits.** Only 8 of the 14 NIH labels; at most 3 findings displayed; ASFI/ZCI are not clinically validated; Grad-CAM is a model explanation, not lesion segmentation.
12. **Development server only.** Flask's built-in server is used; it is not meant for production deployment.
13. **No automated tests.**

---

## 25. Recommended next steps

1. **Restore a served model** (copy from `trained_model_old/`) so the app runs again.
2. **Generate thresholds** for the chosen checkpoint (retrain with the new code, or write a small one-off script using `find_youden_thresholds`).
3. **Run all evaluations** and save the numbers for the dissertation:
   - `evaluate_test.py` (NIH test AUROC/F1),
   - `evaluate_chexpert.py` (baseline vs fine-tuned),
   - `evaluate_localization.py` (Grad-CAM IoU).
4. **Decide which model to promote** based on those results.
5. **Run `fit_ood_detector.py`** for the promoted model.
6. **Fix `evaluate_test.py`** to respect the observation mask (or to use the NIH-only validation set).
7. **Add early stopping and seeds** to training.
8. **Commit** the uncommitted threshold work, and add `backend/trained_model_old/` to `.gitignore`.
9. **Update the README** layout and the report methodology text.

---

## 26. Glossary

| Term | Meaning |
|------|---------|
| **AUROC** | Area under the ROC curve. 0.5 = random, 1.0 = perfect. Threshold-independent. |
| **AUPRC** | Area under the precision–recall curve. More informative than AUROC for rare classes. |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping: a heatmap of the image regions that most increased a class score. |
| **Youden's J** | Sensitivity + specificity − 1. The cutoff that maximizes it balances missed cases and false alarms. |
| **Otsu's method** | Automatic threshold that best separates an image's values into two groups. |
| **ASFI** | This project's activation-based severity index (Section 13.1). |
| **ZCI** | This project's zone concentration index, a Herfindahl–Hirschman index over 6 zones (Section 13.2). |
| **OOD** | Out-of-distribution: an input unlike the training data. |
| **Mahalanobis distance** | Distance from a point to a distribution that accounts for feature correlations. |
| **Observation mask** | Per-label flag saying whether the label is known; unknown labels are excluded from the loss. |
| **Patient-level split** | All images of a patient go to the same split, preventing data leakage. |
| **Replay set** | A share of the old dataset mixed into fine-tuning to reduce forgetting. |
| **Catastrophic forgetting** | A model losing earlier skills when trained on new data. |
| **Transfer learning** | Starting from weights trained on another task (ImageNet) instead of random weights. |
