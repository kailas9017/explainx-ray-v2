import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Data locations (point these at your existing 45GB NIH ChestX-ray14 download) ----
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
BBOX_CSV_PATH = os.path.join(DATA_DIR, "BBox_List_2017.csv")  # optional, used only by scripts/inspect_bboxes.py

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# ---- Trained model artifacts ----
# IMPORTANT: train.py saves BOTH of these together, every time it saves a new best
# checkpoint. model_service.py refuses to start if they don't both exist or if the
# class count in labels.json doesn't match the checkpoint's classifier shape.
# This is the fix for the "checkpoint trained on 8 classes but config says 14"
# bug from the previous build.
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ---- The 8-pathology subset ----
# This matches NIH's BBox_List_2017.csv exactly (the only 8 classes NIH radiologists
# hand-annotated with bounding boxes). Using this subset means we can validate our
# Grad-CAM zone localization against real ground-truth boxes later, and it's the
# same 8-class scope the original project's lost checkpoint appears to have used.
DEFAULT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Zone grid for zone-wise analysis (rows x cols over the 224x224 CAM) ----
ZONE_ROWS = ["Upper", "Mid", "Lower"]
ZONE_COLS = ["Right", "Left"]  # radiological convention: patient's right is image-left,
                                 # but we label zones by anatomical side for the report
