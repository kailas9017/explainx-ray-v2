# ExplainX-Ray v2 — Project Record (A–Z)

**Prepared:** 14 November 2026  
**Repository:** `explainx-ray-v2`  
**Purpose:** A consolidated, evidence-based account of the project from its initial repository commit through the current working-tree state.

> This document reflects the files and Git history available in this workspace. It separates verified implementation and recorded experiment results from proposals and missing outputs. It is a project/academic prototype record, not clinical validation.

## 1. Executive overview

ExplainX-Ray v2 is a local web application for screening chest radiographs with an eight-label DenseNet121 model. The backend is Python/Flask and loads the model once when the process starts. The React/Vite frontend accepts an image, calls the API, and presents model probabilities, Grad-CAM heatmaps, six-zone activation values, a derived severity estimate, and a generated explanatory report. The current branch also contains CheXpert fine-tuning and evaluation workflows, NIH test/localization evaluation scripts, an out-of-distribution (OOD) detector, and class-specific operating thresholds.

The project has three Git commits visible in the repository: an initial application commit, a subsequent baseline checkpoint commit, and a feature commit for CheXpert evaluation and OOD detection. There are also uncommitted changes in README, inference, training, UI/CSS, and a local archive of model files. Consequently, this report describes the *current working tree* as well as its committed history and makes that distinction explicit.

## 2. Origin, problem, and design direction

The README describes problems in an earlier build that used a Node server to spawn a Python process for every request. The reported failure modes were a wrong Python interpreter on Windows, `require()` in an ECMAScript-module context, and an unhandled standard-input EOF crash. It also describes a model/checkpoint mismatch: code expected 14 labels although the stored classifier had eight outputs. The v2 design removes per-request subprocesses: Flask is the inference server and loads one model at startup. Training saves model weights and a label manifest together; inference checks that both exist and that classifier output count agrees with the manifest.

The repository history starts with commit `81682cfb` (“Initial commit: ExplainX-Ray v2 - Flask + React chest X-ray classifier”). Commit `06bd891e` records an agent-host baseline snapshot with additional files and archived checkpoints. Commit `f19724df` (“Add CheXpert evaluation and OOD detection”) adds the CheXpert data/training/evaluation pathway, NIH test evaluation, OOD detector/fitting, and related inference/UI changes. The present working tree additionally has uncommitted threshold calibration/display updates and scope/limitations documentation. The README itself is modified, so its latest claims are not all part of the latest commit.

## 3. Current scope and labels

The configured model predicts these eight NIH ChestX-ray14 labels:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltrate (the source NIH label is `Infiltration`)
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax

This is not a 14-class model. The eight-label scope corresponds to the NIH bounding-box annotation subset. CheXpert supplies overlapping labels for Atelectasis, Cardiomegaly, Effusion (mapped from Pleural Effusion), Pneumonia, and Pneumothorax; NIH-only labels remain unobserved for CheXpert examples. The model is a multi-label classifier, so more than one class can clear its decision threshold.

## 4. Architecture and tools

- **Model:** TorchVision DenseNet121, ImageNet V1 initialization for new training, replaced final linear classifier with eight outputs; sigmoid probabilities at inference.
- **Backend:** Python, PyTorch/TorchVision, Flask, Flask-CORS, NumPy, pandas, Pillow, scikit-learn, scikit-image, Matplotlib.
- **Frontend:** React 18, Vite 5, JavaScript/JSX, CSS.
- **Device:** CUDA when available; otherwise CPU, selected by `backend/config.py`.
- **Input dimensions:** 224×224 after resize; ImageNet normalization is applied.
- **Development routing:** Vite proxies `/api` to Flask at port 5000 by default; `VITE_API_TARGET` can override the proxy target.

The backend files divide responsibilities among configuration, dataset/manifests, training, model architecture, Grad-CAM, severity/report generation, model service, API, and evaluation/OOD scripts. The frontend has upload, diagnostic studio, findings, and clinical-report components.

## 5. Data preparation and training

### NIH ChestX-ray14

`prepare_data.py` derives one binary target column per configured pathology from `Finding Labels`. It splits by patient ID, not image, using a fixed random seed; 70% of patients are assigned to training and the remainder divided evenly between validation and test (15% each, approximately). This reduces patient overlap across splits. It writes `train.csv`, `val.csv`, and `test.csv` under `backend/data/`. The data loader applies image transforms and returns labels with observation masks; training uses masked binary cross-entropy with logits. Training augments images with horizontal flips and rotations.

Training is configured for 15 epochs, batch size 16, and learning rate 1e-4. Adam is used with a ReduceLROnPlateau scheduler. Training can initialize from ImageNet or fine-tune an existing configured checkpoint. Checkpoint selection is by validation AUROC (or CheXpert held-out validation AUROC when CheXpert manifests exist). The training script saves loss/AUROC history and curve images.

### CheXpert experiment

`prepare_chexpert.py` expects an extracted CheXpert-v1.0-small directory with train/valid manifests. It filters to frontal radiographs, maps the five shared labels, represents uncertain/missing labels with observation masks, creates a patient-separated 10% validation subset from CheXpert train, and preserves official `valid` for final evaluation. During fine-tuning the sampler targets 80% CheXpert and 20% NIH replay samples, to emphasize the target domain while retaining NIH-only-label signal. Experimental outputs are isolated under `trained_model/chexpert_experiment/`; code does not automatically promote them to the served model.

## 6. Recorded experiment results

### NIH run recorded in `backend/history.json`

The committed history contains 15 epochs. Training loss fell from 0.1606 to 0.0860. Validation loss reached its minimum, 0.1429, at epoch 6, then rose to 0.1771 by epoch 15. Mean validation AUROC peaked at **0.8321 in epoch 6** and ended at 0.7925. This pattern is consistent with overfitting after the early best epoch. The history does not include per-class scores or a held-out test result. README references an earlier 14-class result of approximately 0.82 AUROC, but that is historical context, not a separately verifiable v2 test result.

### Archived CheXpert experiment

`backend/trained_model_old/chexpert_experiment/history.json` records a 15-epoch run: training loss fell from 0.1625 to 0.0685; validation loss ended at 0.2483. Mean validation AUROC peaked at **0.8956 in epoch 2**, then declined to 0.8580 at epoch 15. This is the patient-held-out validation trajectory stored with the archived experiment. The repository contains `evaluate_chexpert.py` to compare a baseline and experimental checkpoint on the untouched official valid split, including AUROC/AUPRC and precision/recall/F1 at 0.5, but no output from that official-split evaluation is present. Do not interpret the validation score as the official CheXpert result.

The local archived checkpoint files are approximately 28.5 MB each and have matching eight-label manifests. They reside in `backend/trained_model_old/`, not the configured default `backend/trained_model/` path. Their existence alone does not establish that the current API loads a checkpoint.

## 7. Inference API and lifecycle

`backend/app.py` exposes:

- `GET /api/health`: reports readiness, class count, label names, device, and loading error.
- `POST /api/analyze`: accepts JSON with `imageBase64`, decodes the image, and returns analysis. Missing/invalid data yields a client error; unavailable model yields HTTP 503; unexpected inference errors yield HTTP 500.

At startup, `model_service.py` checks for the configured weights and labels, reconstructs DenseNet121 with the matching number of outputs, checks the stored classifier shape, loads the state dict, and switches to eval mode. It returns an explicit “not ready” state rather than inventing outputs if the checkpoint is absent or inconsistent. Environment variables `XRAY_MODEL_PATH`, `XRAY_LABELS_PATH`, and `PORT` allow separate model paths/ports.

For a valid in-distribution request, the service computes probabilities, retains findings above per-class thresholds (up to three), computes Grad-CAM for reported findings, creates a heatmap overlay and zone breakdown for the highest ranked finding, derives ASFI/ZCI descriptors, and generates a report. The current uncommitted implementation uses saved Youden-J validation thresholds where available and 0.5 fallback thresholds otherwise. Training writes thresholds alongside its experimental checkpoint; a file-location/configuration mismatch should be checked before treating these as deployed calibration.

## 8. Grad-CAM and six-zone explanation

`gradcam_utils.py` hooks DenseNet121’s final feature block, differentiates the target class score with respect to feature maps, forms a weighted activation map, applies ReLU, normalizes it, and resizes it to 224×224. The overlay blends a heatmap with the resized input. The same CAM array is split into a 3×2 grid: Right/Left × Upper/Mid/Lower. Each zone’s percentage is its mean CAM activation multiplied by 100; the frontend grid/strip uses those values. Anatomical-side labels follow the project’s stated radiological convention.

The code includes `evaluate_localization.py`, which compares thresholded Grad-CAM masks to NIH bounding boxes and computes IoU by pathology. Although NIH has 984 bounding boxes described by the README, the workspace contains no localization output CSV/chart, so localization performance is not currently established. Grad-CAM saliency and zone values are model explanations and are not radiologist-confirmed lesion locations.

## 9. Derived severity and report

`severity.py` computes ASFI from activated fraction, mean activation intensity, and class probability (weights 0.40/0.30/0.30, scaled to 100). Otsu’s threshold determines the active CAM region. Severity bands are Normal (<15), Mild (15–34), Moderate (35–54), Severe (55–74), and Critical (75+). ZCI is a six-zone concentration measure; the code labels distributions Focal, Multifocal, or Diffuse using fixed cutoffs. `report_generator.py` turns these computed values into rule-based sentences, provides a methodology statement, and includes an explicit research/demo disclaimer.

These are heuristic, model-derived values. The project README explicitly says ASFI is not a validated clinical severity score; generated reports are not diagnoses and have not been reviewed by a radiologist.

## 10. OOD safeguards

`ood_detection.py` has two gates. First, it measures mean RGB-channel difference and flags images above a fixed color-content cutoff. Second, if fitted statistics exist, it compares DenseNet penultimate features against a Mahalanobis-distance threshold. `fit_ood_detector.py` extracts training features and stores mean, pseudo-inverse covariance, and the 99th percentile training-distance threshold as `ood_stats.npz`. Inference captures features during the same forward pass, avoiding a second classifier pass. If statistics are unavailable, the grayscale heuristic remains active and the response states the feature check was unavailable; this is not a comprehensive OOD guarantee. OOD responses suppress findings and reports. No OOD validation results or saved `ood_stats.npz` are present in the tracked/current archive listing.

## 11. User interface

The application has a model/device status indicator, an image upload modal with drag-and-drop and file selection, a Diagnostic Studio with original/heatmap image and six-zone overlay, a Findings panel with ranked pathology predictions/severity/region/peak zone/ASFI/pattern, and a Clinical Report tab. The current uncommitted UI additionally shows all class probabilities with whether each clears its decision threshold. It displays explicit warnings when the OOD gate rejects an image. Vite/React source is present; no frontend build artifact is included.

## 12. Current repository state

At inspection, `git status` showed modified files: `README.md`, `backend/model_service.py`, `backend/train.py`, `frontend/src/components/FindingsPanel.jsx`, and `frontend/src/index.css`; it also showed untracked `backend/trained_model_old/`. These are local working-tree changes and archive artifacts, not all committed to `main`. Do not overwrite or assume they are disposable. The tracked code includes the Flask/React app, training/data scripts, reports, localization/test evaluation scripts, CheXpert workflow, and OOD workflow.

The workspace does not show `backend/data/`, a default `backend/trained_model/`, prepared NIH split CSVs, CheXpert manifests, test-evaluation outputs, localization outputs, or OOD statistics. Those paths are excluded/ignored or absent here. Therefore, current model readiness and reproducible inference cannot be confirmed from this repository snapshot alone. Also absent are deployment configuration, authentication, persistent image storage, automated tests, and a production clinical validation package.

## 13. How to run (as specified by the project)

1. Create/activate a Python environment in `backend` and install `backend/requirements.txt`.
2. Place NIH images under `backend/data/images/` and `Data_Entry_2017.csv` under `backend/data/`.
3. Run `python prepare_data.py`, then `python train.py` (unless using an existing compatible checkpoint).
4. Run `python app.py` from `backend`; check `http://localhost:5000/api/health`.
5. In another terminal, install frontend dependencies in `frontend` and run `npm run dev`; open the Vite URL, normally `http://localhost:5173`.
6. Upload a chest X-ray and choose Analyze. For CheXpert experiments, prepare manifests and explicitly select experimental checkpoint paths as documented in README.

These are documented steps, not steps executed as part of preparing this report.

## 14. Evaluation coverage and open work

Implemented evaluation tooling:

- Training/validation loss and AUROC histories and plots.
- NIH test AUROC and thresholded metrics using Youden-J cutoffs fitted on validation only (`evaluate_test.py`).
- CheXpert official-valid AUROC/AUPRC and fixed-0.5 precision/recall/F1 (`evaluate_chexpert.py`).
- Grad-CAM/bounding-box localization IoU (`evaluate_localization.py`).
- OOD feature-statistics fitting (`fit_ood_detector.py`).

Important evidence still needed to call these evaluated: run the scripts against the correctly prepared datasets and compatible checkpoints, retain their outputs, report class-wise results and sample counts, and assess calibration, subgroup performance, and localization thresholds. Confirm which checkpoint should be served; verify its matching labels and thresholds. The latest recorded validation curves show a clear overfitting signal, so assess early stopping/regularization before claiming generalization. These are data/evaluation next steps, not implemented result claims.

## 15. Key limitations and careful interpretation

- Eight NIH labels only; it does not screen all 14 NIH labels or other diseases.
- Training/validation AUROC is not a substitute for untouched test/external validation.
- CheXpert official-valid results are not available in the repository snapshot.
- Grad-CAM provides a saliency explanation; it is not a validated bounding-box detector.
- ASFI severity is a heuristic, not a clinical scale.
- OOD checks are lightweight safeguards and have no recorded sensitivity/specificity evaluation.
- No default live checkpoint/data artifacts are present in the inspected workspace, so running prediction may report “Model not trained” until artifacts are supplied/configured.
- The README’s CUDA installation recipe and library versions are environment-specific; verify compatibility with the machine before installation.

## 16. File guide

- `README.md` — setup, project rationale, usage, CheXpert and scope notes.
- `backend/config.py` — paths, label list, training parameters, device, zone names.
- `backend/model.py`, `dataset.py`, `prepare_data.py`, `prepare_chexpert.py`, `train.py` — architecture/data/training.
- `backend/app.py`, `model_service.py` — HTTP endpoints and inference lifecycle.
- `backend/gradcam_utils.py`, `severity.py`, `report_generator.py` — explanation and derived report values.
- `backend/ood_detection.py`, `fit_ood_detector.py` — OOD checks and calibration fitting.
- `backend/evaluate_test.py`, `evaluate_chexpert.py`, `evaluate_localization.py` — offline evaluation.
- `backend/history.json` — recorded NIH run curves.
- `backend/trained_model_old/` — local archived baseline and CheXpert experiment weights, manifests, history, and plot images.
- `frontend/src/App.jsx` and `frontend/src/components/` — app flow and views.
- `frontend/src/index.css` — visual styling; currently modified locally.
- `frontend/vite.config.js`, `frontend/package.json` — frontend development configuration/dependencies.

## 17. Final status

The project has moved from a fragile subprocess-based prototype design to a structured Flask + React application with a consistent eight-class model contract, real Grad-CAM zone explanations, derived report generation, patient-separated NIH data preparation, an experimental CheXpert fine-tuning path, and evaluation/OOD tooling. The current work adds per-class calibration display and clearer scope/limitations. There is recorded evidence of promising validation AUROC (NIH 0.8321; CheXpert experiment 0.8956 on its held-out validation), alongside late-epoch overfitting. The key remaining gap is not feature code but reproducible, retained evaluation against the untouched datasets and confirmation of the exact checkpoint/configuration intended for use.
