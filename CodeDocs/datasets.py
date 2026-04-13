"""
datasets.py
───────────
PyTorch Dataset implementations for the Facial Beauty Prediction pipeline.

Coordinates the loading of raw images and pre-calculated facial landmarks 
from the SCUT-FBP5500 dataset.

Datasets Provided:
1. FaceImageDataset: Returns (image, score) for the texture-only CNN (M1).
2. FaceLandmarkDataset: Returns (landmark_vector, score) for the geometry-only MLPs (M2, M3).
3. FaceFusionDataset: Returns (image, landmark_vector, score) for the Adaptive Fusion model (M4).

Key Operation:
- All datasets normalize the 1-5 beauty scores into a [0, 1] range.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config as C


# ─── Data Augmentation & Preprocessing ────────────────────────────────────────

def get_transforms(train: bool):
    """
    Define image transformation pipeline.
    
    Args:
        train (bool): If True, applies random augmentations for training.
    
    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
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
#  M1 Dataset – Full RGB Image
# ═══════════════════════════════════════════════════════════════════════════════

class FaceImageDataset(Dataset):
    """
    Dataset for Model M1. Handles raw RGB images.
    Normalizes beauty scores from [1, 5] to [0, 1].
    """
    def __init__(self, df: pd.DataFrame, train: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'filepath' and 'score'.
            train (bool): Toggle for training augmentations.
        """
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(train)
        
        # Min-Max normalization of labels: (s - 1) / 4  → [0, 1]
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
#  M2 / M3 Dataset – Landmark Vectors
# ═══════════════════════════════════════════════════════════════════════════════

class FaceLandmarkDataset(Dataset):
    """
    Dataset for Models M2 and M3. Uses flattened landmark vectors.
    """
    def __init__(self, df: pd.DataFrame, landmarks_dict: dict):
        """
        Args:
            df (pd.DataFrame): Dataset partition.
            landmarks_dict (dict): Map of {filename: landmark_flattened_array}.
        """
        # Intersection of available landmarks and the split dataframe
        mask   = df["filename"].isin(landmarks_dict)
        self.df = df[mask].reset_index(drop=True)
        self.landmarks_dict = landmarks_dict
        
        self.scores = torch.tensor(
            ((self.df["score"].values - 1.0) / 4.0), dtype=torch.float32
        )
        if C.VERBOSE:
            print(f"[LandmarkDataset] initialized with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn  = self.df.iloc[idx]["filename"]
        lm  = torch.tensor(self.landmarks_dict[fn], dtype=torch.float32)
        lbl = self.scores[idx]
        return lm, lbl


# ═══════════════════════════════════════════════════════════════════════════════
#  M4 Dataset – Image + Landmark Fusion
# ═══════════════════════════════════════════════════════════════════════════════

class FaceFusionDataset(Dataset):
    """
    Multimodal Dataset for Model M4. 
    Returns pairs of (image, landmarks) for the Adaptive Fusion branch.
    """
    def __init__(self, df: pd.DataFrame, landmarks_dict: dict, train: bool = True):
        """
        Args:
            df (pd.DataFrame): Dataset partition.
            landmarks_dict (dict): Map of {filename: landmark_flattened_array}.
            train (bool): Toggle for training augmentations on the image branch.
        """
        mask    = df["filename"].isin(landmarks_dict)
        self.df = df[mask].reset_index(drop=True)
        self.landmarks_dict = landmarks_dict
        self.transform = get_transforms(train)
        
        self.scores = torch.tensor(
            ((self.df["score"].values - 1.0) / 4.0), dtype=torch.float32
        )
        if C.VERBOSE:
            print(f"[FusionDataset] initialized with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.transform(img)
        lm  = torch.tensor(self.landmarks_dict[row["filename"]], dtype=torch.float32)
        lbl = self.scores[idx]
        return img, lm, lbl


# ─── DataLoader Factory Functions ──────────────────────────────────────────────

def get_image_loaders(train_df, test_df):
    """Construct DataLoader pair for the M1 CNN model."""
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
