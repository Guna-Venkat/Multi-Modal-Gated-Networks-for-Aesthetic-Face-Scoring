"""
config.py
─────────
Central configuration and global settings for the Facial Beauty Prediction project.

This module handles:
1. Environment detection (Local vs. Kaggle).
2. Path resolution for datasets, checkpoints, and results.
3. Architecture-specific hyperparameters (LR, epochs, hidden dims).
4. Global training constants (IMG_SIZE, BATCH_SIZE).

Usage:
    Import this module in other scripts: `import config as C`
"""

# ──────────────────────────────────────────────────────────────────────────────
#  ★  ENVIRONMENT SETTINGS  ★
# ──────────────────────────────────────────────────────────────────────────────
# Change ENV to "kaggle" when running on Kaggle kernels to use cloud paths.
ENV = "local"   # Options: "kaggle" | "local"
# ──────────────────────────────────────────────────────────────────────────────

import os
import torch

# ─── Directory & File Paths ───────────────────────────────────────────────────
if ENV == "kaggle":
    # Kaggle-specific paths for dataset inputs and working outputs
    BASE_DIR        = "/kaggle/working"
    DATASET_DIR     = "/kaggle/input/datasets/gunavenkatdoddi/cv-project-data/dataset"
    IMAGE_DIR       = os.path.join(DATASET_DIR, "Images")
    LABEL_CSV       = os.path.join(DATASET_DIR, "train_test_files/All_Ratings.xlsx")
    TRAIN_SPLIT_CSV = os.path.join(DATASET_DIR, "train_test_files/split_of_60%training and 40%testing/train.txt")
    TEST_SPLIT_CSV  = os.path.join(DATASET_DIR, "train_test_files/split_of_60%training and 40%testing/test.txt")
else:
    # Local development paths relative to the project root
    BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
    IMAGE_DIR       = os.path.join(DATASET_DIR, "Images", "Images")
    LABEL_CSV       = os.path.join(DATASET_DIR, "labels.txt")
    TRAIN_SPLIT_CSV = os.path.join(DATASET_DIR, "train.txt")
    TEST_SPLIT_CSV  = os.path.join(DATASET_DIR, "test.txt")

# Writable directories for processed assets
CACHE_DIR        = os.path.join(BASE_DIR, "cache")           # Pre-calculated landmarks & splits
LANDMARKS_2D_DIR = os.path.join(CACHE_DIR, "landmarks_2d")
LANDMARKS_3D_DIR = os.path.join(CACHE_DIR, "landmarks_3d")
CHECKPOINT_DIR   = os.path.join(BASE_DIR, "checkpoints")     # Model weights (.pt)
RESULTS_DIR      = os.path.join(BASE_DIR, "results")         # Final plots & evaluation CSVs

# Ensure all directories exist
for d in [CACHE_DIR, LANDMARKS_2D_DIR, LANDMARKS_3D_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Image Pre-processing ─────────────────────────────────────────────────────
IMG_SIZE        = 224          # Dimensions to which face crops are resized
FACE_MARGIN     = 0.2          # Extra crop margin (20%) around detected face mesh

# ─── Data Split Fallbacks ─────────────────────────────────────────────────────
# Used only if TRAIN_SPLIT_CSV / TEST_SPLIT_CSV are not found
TRAIN_RATIO     = 0.80
RANDOM_SEED     = 42

# ─── Landmark Configuration ───────────────────────────────────────────────────
N_LANDMARKS     = 468                # Standard MediaPipe Face Mesh landmark count
LANDMARK_DIM_2D = N_LANDMARKS * 2    # Flattened (x,y) coordinates
LANDMARK_DIM_3D = N_LANDMARKS * 3    # Flattened (x,y,z) coordinates

# ─── Global Training Constants ────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 2 if ENV == "kaggle" else 4
PATIENCE        = 5            # Epochs to wait for improvement before Early Stopping

# ─── M1 Configuration (Texture Branch - CNN) ─────────────────────────────────
M1_LR           = 1e-3
M1_EPOCHS       = 30
M1_WEIGHT_DECAY = 1e-4
M1_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m1_cnn.pt")
M1_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m1_preds.npy")

# ─── M2 Configuration (Geometry Branch - 2D MLP) ──────────────────────────────
M2_HIDDEN       = [256, 128]
M2_LR           = 1e-3
M2_EPOCHS       = 50
M2_WEIGHT_DECAY = 1e-4
M2_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m2_mlp2d.pt")
M2_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m2_preds.npy")

# ─── M3 Configuration (Geometry Branch - 3D Procrustes + MLP) ─────────────────
M3_HIDDEN       = [256, 128]
M3_LR           = 1e-3
M3_EPOCHS       = 50
M3_WEIGHT_DECAY = 1e-4
M3_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m3_mlp3d.pt")
M3_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m3_preds.npy")
# Reference shape used for Procrustes 3D alignment
PROCRUSTES_REF  = os.path.join(CACHE_DIR, "procrustes_reference.npy")

# ─── M4 Configuration (Adaptive Gated Fusion) ────────────────────────────────
M4_LR           = 1e-4
M4_EPOCHS       = 40
M4_WEIGHT_DECAY = 1e-4
M4_ENTROPY_LAMBDA = 0.01       # Weighted regularisation for gating entropy
M4_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m4_fusion.pt")
M4_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m4_preds.npy")
M4_GATES_PATH   = os.path.join(RESULTS_DIR,    "m4_gates.npy")

# ─── Execution Environment ─────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = True

def print_config():
    """Print the active configuration for debug traceability."""
    print("=" * 60)
    print(f"  PROJECT ENVIRONMENT CONFIG  ")
    print("-" * 60)
    print(f"  ENV            : {ENV}")
    print(f"  DEVICE         : {DEVICE}")
    print(f"  IMAGE_DIR      : {IMAGE_DIR}")
    print(f"  LABEL_CSV      : {LABEL_CSV}")
    print(f"  CACHE_DIR      : {CACHE_DIR}")
    print(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"  RESULTS_DIR    : {RESULTS_DIR}")
    print("=" * 60)
n(RESULTS_DIR,    "m4_preds.npy")
M4_GATES_PATH   = os.path.join(RESULTS_DIR,    "m4_gates.npy")  # (alpha, beta) per sample

# ── Device ───────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Verbosity ────────────────────────────────
VERBOSE = True

def print_config():
    print("=" * 55)
    print(f"  ENV            : {ENV}")
    print(f"  DEVICE         : {DEVICE}")
    print(f"  IMAGE_DIR      : {IMAGE_DIR}")
    print(f"  LABEL_CSV      : {LABEL_CSV}")
    print(f"  CACHE_DIR      : {CACHE_DIR}")
    print(f"  CHECKPOINT_DIR : {CHECKPOINT_DIR}")
    print(f"  RESULTS_DIR    : {RESULTS_DIR}")
    print("=" * 55)
