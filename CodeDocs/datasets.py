"""
datasets.py
───────────
PyTorch Dataset classes for all four models.

  FaceImageDataset    – full RGB image + label          → M1
  FaceLandmarkDataset – pre-computed landmark vector + label → M2, M3
  FaceFusionDataset   – image + landmark vector + label  → M4
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config as C


# ─── Standard ImageNet transforms ─────────────────────────────────────────────

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ═══════════════════════════════════════════════════════════════════════════════
#  M1 Dataset – full image
# ═══════════════════════════════════════════════════════════════════════════════

class FaceImageDataset(Dataset):
    """
    Returns (image_tensor, score) for M1.
    Normalises score to [0,1] (original range 1-5).
    """
    def __init__(self, df: pd.DataFrame, train: bool = True):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(train)
        # Normalise score: (s - 1) / 4  → [0, 1]
        self.scores = torch.tensor(
            ((df["score"].values - 1.0) / 4.0), dtype=torch.float32
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(row["filepath"]).convert("RGB")
        img  = self.transform(img)
        lbl  = self.scores[idx]
        return img, lbl


# ═══════════════════════════════════════════════════════════════════════════════
#  M2 / M3 Dataset – landmark vector only
# ═══════════════════════════════════════════════════════════════════════════════

class FaceLandmarkDataset(Dataset):
    """
    Returns (landmark_vector [1404], score) for M2 (or M3).
    `landmarks_dict` maps filename → np.ndarray [1404].
    """
    def __init__(self, df: pd.DataFrame, landmarks_dict: dict):
        # Keep only rows present in landmarks_dict
        mask   = df["filename"].isin(landmarks_dict)
        self.df = df[mask].reset_index(drop=True)
        self.landmarks_dict = landmarks_dict
        self.scores = torch.tensor(
            ((self.df["score"].values - 1.0) / 4.0), dtype=torch.float32
        )
        if C.VERBOSE:
            print(f"[LandmarkDataset] kept {len(self.df)} / {len(df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn  = self.df.iloc[idx]["filename"]
        lm  = torch.tensor(self.landmarks_dict[fn], dtype=torch.float32)
        lbl = self.scores[idx]
        return lm, lbl


# ═══════════════════════════════════════════════════════════════════════════════
#  M4 Dataset – image + landmark
# ═══════════════════════════════════════════════════════════════════════════════

class FaceFusionDataset(Dataset):
    """
    Returns (image_tensor, landmark_vector [1404], score) for M4.
    Requires both a filepath and an entry in landmarks_dict.
    """
    def __init__(self, df: pd.DataFrame, landmarks_dict: dict, train: bool = True):
        mask    = df["filename"].isin(landmarks_dict)
        self.df = df[mask].reset_index(drop=True)
        self.landmarks_dict = landmarks_dict
        self.transform = get_transforms(train)
        self.scores = torch.tensor(
            ((self.df["score"].values - 1.0) / 4.0), dtype=torch.float32
        )
        if C.VERBOSE:
            print(f"[FusionDataset] kept {len(self.df)} / {len(df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        lm  = torch.tensor(self.landmarks_dict[row["filename"]], dtype=torch.float32)
        lbl = self.scores[idx]
        return img, lm, lbl


# ═══════════════════════════════════════════════════════════════════════════════
#  DataLoader helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_image_loaders(train_df, test_df):
    """M1 dataloaders."""
    train_ds = FaceImageDataset(train_df, train=True)
    test_ds  = FaceImageDataset(test_df,  train=False)
    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE,
                          shuffle=True,  num_workers=C.NUM_WORKERS,
                          pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE,
                          shuffle=False, num_workers=C.NUM_WORKERS,
                          pin_memory=True)
    return train_dl, test_dl


def get_landmark_loaders(train_df, test_df, landmarks_dict):
    """M2 / M3 dataloaders."""
    train_ds = FaceLandmarkDataset(train_df, landmarks_dict)
    test_ds  = FaceLandmarkDataset(test_df,  landmarks_dict)
    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE,
                          shuffle=True,  num_workers=C.NUM_WORKERS)
    test_dl  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE,
                          shuffle=False, num_workers=C.NUM_WORKERS)
    return train_dl, test_dl


def get_fusion_loaders(train_df, test_df, landmarks_dict):
    """M4 dataloaders."""
    train_ds = FaceFusionDataset(train_df, landmarks_dict, train=True)
    test_ds  = FaceFusionDataset(test_df,  landmarks_dict, train=False)
    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE,
                          shuffle=True,  num_workers=C.NUM_WORKERS,
                          pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=C.BATCH_SIZE,
                          shuffle=False, num_workers=C.NUM_WORKERS,
                          pin_memory=True)
    return train_dl, test_dl
