"""Prepare patient-separated CheXpert manifests without altering NIH data.

Example (PowerShell, from backend):
    python prepare_chexpert.py --dataset-root "C:\\Users\\Kaila\\Downloads\\archive (11)"

The official CheXpert valid split is preserved for external evaluation and is
not split into training or validation data here. Uncertain/missing labels are
marked as unobserved; NIH-only classes stay unobserved for CheXpert rows.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import CHEXPERT_DIR, DATA_DIR, DEFAULT_PATHOLOGIES


CHEXPERT_LABELS = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


def _image_relative_path(csv_path: str) -> str:
    """Trim any archive wrapper from CSV paths, keeping train/valid/... onward."""
    parts = Path(csv_path.replace("\\", "/")).parts
    for index, part in enumerate(parts):
        if part in {"train", "valid"}:
            return Path(*parts[index:]).as_posix()
    raise ValueError(f"Could not find train/valid in CheXpert path: {csv_path}")


def _patient_id(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) < 2 or not parts[1].startswith("patient"):
        raise ValueError(f"Could not determine patient ID from path: {relative_path}")
    return parts[1]


def _convert_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    frontal = frame[frame["Frontal/Lateral"].eq("Frontal")].copy()
    output = pd.DataFrame(index=frontal.index)
    output["image_path"] = frontal["Path"].map(_image_relative_path)
    output["patient_id"] = output["image_path"].map(_patient_id)
    output["source_split"] = split

    for pathology in DEFAULT_PATHOLOGIES:
        source_label = CHEXPERT_LABELS.get(pathology)
        if source_label is None:
            output[pathology] = 0.0
            output[f"{pathology}__observed"] = 0.0
            continue

        values = pd.to_numeric(frontal[source_label], errors="coerce")
        observed = values.isin([0.0, 1.0])
        output[pathology] = values.eq(1.0).astype(float)
        output[f"{pathology}__observed"] = observed.astype(float)

    return output.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=CHEXPERT_DIR, help="Extracted CheXpert-v1.0-small folder")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(DATA_DIR, "chexpert_manifests"),
        help="Where to write manifests (default: backend/data/chexpert_manifests)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    train_csv = root / "train.csv"
    valid_csv = root / "valid.csv"
    if not train_csv.is_file() or not valid_csv.is_file():
        raise FileNotFoundError(
            f"Expected train.csv and valid.csv directly inside {root}. "
            "Pass the extracted CheXpert-v1.0-small folder with --dataset-root."
        )

    train = _convert_split(pd.read_csv(train_csv), "train")
    valid = _convert_split(pd.read_csv(valid_csv), "official_valid")

    # Create a patient-level validation subset from CheXpert train. The official
    # valid split remains untouched for a final external evaluation.
    split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_indices, val_indices = next(split.split(train, groups=train["patient_id"]))
    train_manifest = train.iloc[train_indices].reset_index(drop=True)
    val_manifest = train.iloc[val_indices].reset_index(drop=True)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest.to_csv(output_dir / "train.csv", index=False)
    val_manifest.to_csv(output_dir / "validation.csv", index=False)
    valid.to_csv(output_dir / "official_valid.csv", index=False)

    print(f"CheXpert root: {root}")
    print(f"Frontal train images: {len(train_manifest):,}")
    print(f"Frontal validation images: {len(val_manifest):,}")
    print(f"Frontal official valid images (reserved): {len(valid):,}")
    print(f"Manifests saved to: {output_dir}")
    print("Uncertain/missing labels have observed=0; NIH-only labels are unobserved.")


if __name__ == "__main__":
    main()
