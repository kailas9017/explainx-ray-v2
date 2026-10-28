# ExplainX-Ray v2

Chest X-ray disease classifier with real Grad-CAM zone-wise analysis.
Self-trained DenseNet121 on NIH ChestX-ray14 (8-pathology subset), served
directly by Flask, with a React frontend.

## Why this version is structured differently from before

The previous build broke in three separate ways, all traced back to one
design choice: a Node.js server spawning a Python subprocess per request
(wrong interpreter on Windows, `require()` in an ESM context, an unhandled
`stdin` EOF crash). It also had a silent, undetected bug — the training
config's pathology list was edited *after* the model was trained, so the
saved checkpoint (8 output classes) no longer matched the class list the
code expected (14), and PyTorch's shape check only surfaced this by chance
once the subprocess crashes were fixed.

This version removes both problems structurally:

- **No subprocess spawning.** Flask *is* the inference server — the model is
  loaded once into memory when Flask starts, not launched fresh per request.
- **The checkpoint and its class list are saved together, always.**
  `train.py` writes `best_model.pth` and `labels.json` in the same step
  (`save_checkpoint()` in `train.py`). `model_service.py` refuses to start if
  either file is missing, or if their class counts disagree — loudly, with a
  specific message, instead of a bare PyTorch shape-mismatch traceback.

## Project layout

```
backend/
  config.py          # all paths + the 8-pathology list, single source of truth
  model.py            # DenseNet121 architecture (unchanged from before)
  dataset.py           # PyTorch Dataset / DataLoaders
  prepare_data.py     # builds train/val/test CSVs from Data_Entry_2017.csv
  train.py            # training loop; saves checkpoint + labels.json together
  gradcam_utils.py    # Grad-CAM, zone-wise aggregation, heatmap overlay image
  model_service.py    # loads model once, runs inference + Grad-CAM per request
  app.py              # Flask API (/api/health, /api/analyze)
  requirements.txt
frontend/
  src/
    App.jsx
    components/
      DiagnosticStudio.jsx   # the image viewer + zone grid overlay
      FindingsPanel.jsx      # ranked pathology list
      UploadModal.jsx        # drag & drop upload
    index.css
  package.json
  vite.config.js       # proxies /api/* to Flask on :5000 in dev
```

## 1. Backend setup

```powershell
cd backend
python -m venv cxr-env
.\cxr-env\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the CUDA-enabled PyTorch build, so an NVIDIA GPU is
used automatically. Verify it after installation:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Copy your existing NIH ChestX-ray14 data into `backend/data/`:

```
backend/data/images/                 (all the .png files)
backend/data/Data_Entry_2017.csv
```

(You already have this 45GB dataset downloaded from the previous project —
just point/copy it here, no re-download needed.)

## 2. Prepare data and train

```powershell
python prepare_data.py
python train.py
```

### Optional: fine-tune with CheXpert-v1.0-small

Keep the extracted CheXpert folder separate from `backend/data`. In PowerShell,
from `backend`, point to the extracted folder and prepare patient-separated
manifests:

```powershell
$env:CHEXPERT_DIR = (Resolve-Path ".\data\chexpert").Path
python prepare_chexpert.py --dataset-root $env:CHEXPERT_DIR
python train.py
python evaluate_chexpert.py
```

The training loader samples 80% CheXpert and 20% NIH per epoch. It maps
only shared labels (`Atelectasis`, `Cardiomegaly`, `Effusion` ← `Pleural
Effusion`, `Pneumonia`, and `Pneumothorax`). NIH-only labels are masked for
CheXpert samples, and uncertain/missing CheXpert labels are ignored in the
loss. The patient-held-out CheXpert validation subset drives checkpoint
selection and learning-rate reduction when CheXpert manifests exist. The
untouched official `valid` split is used only by `evaluate_chexpert.py`, which
reports AUROC, AUPRC, and precision/recall/F1 at 0.5 per class. The
experimental checkpoint is saved under
`trained_model/chexpert_experiment/`; it does not replace the currently served
model automatically.

To try the experimental checkpoint beside the current app, start a second
backend from `backend`:

```powershell
$env:XRAY_MODEL_PATH = (Resolve-Path ".\trained_model\chexpert_experiment\best_model.pth").Path
$env:XRAY_LABELS_PATH = (Resolve-Path ".\trained_model\chexpert_experiment\labels.json").Path
$env:PORT = "5001"
python app.py
```

Then start a second frontend from `frontend` in another terminal:

```powershell
$env:VITE_API_TARGET = "http://127.0.0.1:5001"
npm run dev -- --port 5174
```

Open `http://localhost:5174` to try the fine-tuned model. The original app can
continue using its original backend on port 5000.

This trains DenseNet121 on the 8-pathology subset (same approach as before —
transfer-learned from ImageNet, ~15 epochs). On an RTX 4050 this should take
roughly the same time as your earlier successful run (which reached 0.82
AUROC on the 14-class set — expect a similar or slightly better range on the
smaller 8-class set, since each class now has more relative training signal).

When it finishes, you'll have:
```
backend/trained_model/best_model.pth
backend/trained_model/labels.json
```

Both are required — `model_service.py` checks for both on startup.

## 3. Run the backend

```powershell
python app.py
```

Should print `Loaded model with 8 classes on cuda: [...]` and start serving
on `http://0.0.0.0:5000`. Check it directly:

```powershell
curl http://localhost:5000/api/health
```

## 4. Run the frontend (separate terminal)

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). The dev server
proxies `/api/*` to Flask on `:5000` automatically — no CORS setup needed,
no hardcoded hostnames to get wrong.

## 5. Use it

Click **Load X-ray scan** → drop or select an image → **Analyze X-ray**.
The Diagnostic Studio panel shows the uploaded image with the Grad-CAM
heatmap blended in, a 6-zone grid (Right/Left × Upper/Mid/Lower) with each
zone's activation percentage, and the Findings panel lists the top predicted
pathologies ranked by probability, each with a severity tag and its peak
activation zone.

If the model isn't trained yet, the header shows **"Model not trained"** and
`/api/analyze` returns a clear 503 with instructions — it will never fall
back to fake or mocked results.

## Notes on the zone-wise analysis

Zone scores come directly from the same Grad-CAM array used to render the
heatmap (`gradcam_utils.zone_breakdown`) — there's no separate/fabricated
number generator behind them, so the heatmap image and the zone percentages
will always agree with each other.

NIH's `BBox_List_2017.csv` contains 984 radiologist-drawn bounding boxes for
exactly these 8 pathologies. It isn't wired in yet, but if you want a
quantitative "does our Grad-CAM actually localize where a radiologist marked
the finding" section for your dissertation, that file is the ground truth to
compare against (IoU between the Grad-CAM's high-activation region and the
annotated box, per pathology).
