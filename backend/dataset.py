import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
import torchvision.transforms as transforms

from config import (
    IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, DEFAULT_PATHOLOGIES,
    BATCH_SIZE, IMAGE_SIZE, CHEXPERT_DIR, CHEXPERT_MANIFEST_DIR,
)


class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, pathologies, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.pathologies = pathologies
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        image = Image.open(img_path).convert("RGB")
        labels = row[self.pathologies].values.astype(np.float32)
        labels = torch.tensor(labels)
        if self.transform:
            image = self.transform(image)
        # NIH's labels are weakly mined; absent findings are treated as negatives
        # in its source annotations, so all configured targets are observed.
        return image, labels, torch.ones_like(labels)


class CheXpertDataset(Dataset):
    """CheXpert rows with per-label observation masks for uncertain/missing labels."""

    def __init__(self, manifest_path, pathologies, transform=None):
        self.df = pd.read_csv(manifest_path).reset_index(drop=True)
        self.pathologies = pathologies
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(CHEXPERT_DIR, *row["image_path"].split("/"))
        image = Image.open(image_path).convert("RGB")
        labels = torch.tensor(row[self.pathologies].to_numpy(dtype=np.float32))
        mask_columns = [f"{name}__observed" for name in self.pathologies]
        observed = torch.tensor(row[mask_columns].to_numpy(dtype=np.float32))
        if self.transform:
            image = self.transform(image)
        return image, labels, observed


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(pathologies=None):
    pathologies = pathologies or DEFAULT_PATHOLOGIES
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    nih_train = ChestXrayDataset(train_df, IMAGE_DIR, pathologies, train_transform)
    nih_val = ChestXrayDataset(val_df, IMAGE_DIR, pathologies, eval_transform)
    test_dataset = ChestXrayDataset(test_df, IMAGE_DIR, pathologies, eval_transform)

    train_manifest = os.path.join(CHEXPERT_MANIFEST_DIR, "train.csv")
    val_manifest = os.path.join(CHEXPERT_MANIFEST_DIR, "validation.csv")
    has_chexpert_manifest = os.path.isfile(train_manifest) or os.path.isfile(val_manifest)
    if has_chexpert_manifest and not (os.path.isfile(train_manifest) and os.path.isfile(val_manifest)):
        raise FileNotFoundError(
            f"CheXpert manifests are incomplete in {CHEXPERT_MANIFEST_DIR}. "
            "Run prepare_chexpert.py again."
        )
    if has_chexpert_manifest:
        if not os.path.isdir(CHEXPERT_DIR):
            raise FileNotFoundError(
                f"CheXpert image folder not found at {CHEXPERT_DIR}. "
                "Set CHEXPERT_DIR to the extracted dataset folder before training."
            )
        chexpert_train = CheXpertDataset(train_manifest, pathologies, train_transform)
        chexpert_val = CheXpertDataset(val_manifest, pathologies, eval_transform)
        train_dataset = ConcatDataset([nih_train, chexpert_train])
        val_dataset = ConcatDataset([nih_val, chexpert_val])

        # Use CheXpert as the primary training source for the CheXpert experiment.
        # NIH remains a small replay set to retain useful signal for the three
        # NIH-only classes and reduce catastrophic forgetting on shared labels.
        nih_share = 0.2
        chexpert_share = 1.0 - nih_share
        weights = np.concatenate([
            np.full(len(nih_train), nih_share / len(nih_train), dtype=np.float64),
            np.full(len(chexpert_train), chexpert_share / len(chexpert_train), dtype=np.float64),
        ])
        # Keep epoch length in the same range as the CheXpert set; otherwise
        # replay sampling can inflate training time without adding target data.
        sampler = WeightedRandomSampler(
            weights, num_samples=len(chexpert_train), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_dataset, val_dataset = nih_train, nih_val
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def get_chexpert_official_valid_loader(pathologies=None):
    """Return the untouched official CheXpert valid split for final evaluation."""
    pathologies = pathologies or DEFAULT_PATHOLOGIES
    manifest = os.path.join(CHEXPERT_MANIFEST_DIR, "official_valid.csv")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"CheXpert official validation manifest not found at {manifest}. "
            "Run prepare_chexpert.py first."
        )
    dataset = CheXpertDataset(manifest, pathologies, eval_transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def get_chexpert_validation_loader(pathologies=None):
    """Return the patient-held-out validation subset made from CheXpert train."""
    pathologies = pathologies or DEFAULT_PATHOLOGIES
    manifest = os.path.join(CHEXPERT_MANIFEST_DIR, "validation.csv")
    if not os.path.isfile(manifest):
        raise FileNotFoundError(
            f"CheXpert validation manifest not found at {manifest}. "
            "Run prepare_chexpert.py first."
        )
    dataset = CheXpertDataset(manifest, pathologies, eval_transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels, observed = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("Observed label shape:", observed.shape)
