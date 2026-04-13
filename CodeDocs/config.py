"""
config.py – Central configuration for Facial Beauty Prediction project.
Switch between 'kaggle' and 'local' by changing ENV below.
"""

# ─────────────────────────────────────────────
#  ★  SET YOUR ENVIRONMENT HERE  ★
# ─────────────────────────────────────────────
ENV = "local"   # "kaggle" | "local"
# ─────────────────────────────────────────────

import os

# ── Paths ────────────────────────────────────
if ENV == "kaggle":
    BASE_DIR        = "/kaggle/working"
    DATASET_DIR     = "/kaggle/input/datasets/gunavenkatdoddi/cv-project-data/dataset"   # adjust to your dataset slug
    IMAGE_DIR       = os.path.join(DATASET_DIR, "Images")
    LABEL_CSV       = os.path.join(DATASET_DIR, "train_test_files/All_Ratings.xlsx")
    # Kaggle keeps pre-split CSVs in some versions; fall back to auto-split
    TRAIN_SPLIT_CSV = os.path.join(DATASET_DIR, "train_test_files/split_of_60%training and 40%testing/train.txt")
    TEST_SPLIT_CSV  = os.path.join(DATASET_DIR, "train_test_files/split_of_60%training and 40%testing/test.txt")
else:  # local
    # Move one folder up since config.py is inside CodeDocs/
    BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR     = os.path.join(BASE_DIR, "dataset")
    IMAGE_DIR       = os.path.join(DATASET_DIR, "Images", "Images")
    LABEL_CSV       = os.path.join(DATASET_DIR, "labels.txt")
    TRAIN_SPLIT_CSV = os.path.join(DATASET_DIR, "train.txt")
    TEST_SPLIT_CSV  = os.path.join(DATASET_DIR, "test.txt")

# ── Processed / cache dirs (always writable) ─
CACHE_DIR        = os.path.join(BASE_DIR, "cache")
LANDMARKS_2D_DIR = os.path.join(CACHE_DIR, "landmarks_2d")
LANDMARKS_3D_DIR = os.path.join(CACHE_DIR, "landmarks_3d")
CHECKPOINT_DIR   = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

for d in [CACHE_DIR, LANDMARKS_2D_DIR, LANDMARKS_3D_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Image pre-processing ──────────────────────
IMG_SIZE        = 224          # resize target (H x W)
FACE_MARGIN     = 0.2          # fractional margin around detected face

# ── Data split (used only when split txt not found) ─
TRAIN_RATIO     = 0.80
RANDOM_SEED     = 42

# ── Landmarks ────────────────────────────────
N_LANDMARKS     = 468          # MediaPipe Face Mesh count
LANDMARK_DIM_2D = N_LANDMARKS * 2    # (x,y)  → 936
LANDMARK_DIM_3D = N_LANDMARKS * 3    # (x,y,z) → 1404

# ── Training – shared ─────────────────────────
BATCH_SIZE      = 32
NUM_WORKERS     = 2 if ENV == "kaggle" else 4
PATIENCE        = 5            # early stopping patience

# ── M1 – Full-image CNN ───────────────────────
M1_LR           = 1e-3
M1_EPOCHS       = 30
M1_WEIGHT_DECAY = 1e-4
M1_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m1_cnn.pt")
M1_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m1_preds.npy")

# ── M2 – 2D Landmarks + MLP ───────────────────
M2_HIDDEN       = [256, 128]
M2_LR           = 1e-3
M2_EPOCHS       = 50
M2_WEIGHT_DECAY = 1e-4
M2_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m2_mlp2d.pt")
M2_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m2_preds.npy")

# ── M3 – 3D Landmarks + Procrustes + MLP ─────
M3_HIDDEN       = [256, 128]
M3_LR           = 1e-3
M3_EPOCHS       = 50
M3_WEIGHT_DECAY = 1e-4
M3_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m3_mlp3d.pt")
M3_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m3_preds.npy")
PROCRUSTES_REF  = os.path.join(CACHE_DIR, "procrustes_reference.npy")

# ── M4 – Adaptive Fusion ─────────────────────
M4_LR           = 1e-4
M4_EPOCHS       = 40
M4_WEIGHT_DECAY = 1e-4
M4_ENTROPY_LAMBDA = 0.01       # λ for entropy regularisation
M4_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "m4_fusion.pt")
M4_PREDS_PATH   = os.path.join(RESULTS_DIR,    "m4_preds.npy")
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
