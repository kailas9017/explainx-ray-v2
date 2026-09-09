import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import IMAGE_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, DEFAULT_PATHOLOGIES, BATCH_SIZE, IMAGE_SIZE


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
        return image, labels


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

    train_dataset = ChestXrayDataset(train_df, IMAGE_DIR, pathologies, train_transform)
    val_dataset = ChestXrayDataset(val_df, IMAGE_DIR, pathologies, eval_transform)
    test_dataset = ChestXrayDataset(test_df, IMAGE_DIR, pathologies, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
