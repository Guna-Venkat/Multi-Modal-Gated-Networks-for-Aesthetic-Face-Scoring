"""
phase3_m2_landmarks.py  &  phase4_m3_3d.py  (combined)
────────────────────────────────────────────────────────
Phase 3 – M2: 2D Landmarks + Anchor Normalisation + MLP   (Days 6–7)
Phase 4 – M3: 3D Landmarks + Procrustes + MLP             (Days 8–9)  [optional]

Both models share the same LandmarkMLP architecture (input 1404).
Only the feature vector differs:
  M2 → anchor-normalised 2D distances  [1404]
  M3 → Procrustes-aligned 3D coords   [1404]

Usage:
    python phase3_m2_landmarks.py          # runs M2 only
    python phase3_m2_landmarks.py --m3     # runs M2 + M3
    python phase3_m2_landmarks.py --only-m3  # runs M3 only (if 3D available)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import config as C
from datasets import get_landmark_loaders
from models   import LandmarkMLP
from trainer  import fit, compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper – load pre-computed landmark dicts
# ═══════════════════════════════════════════════════════════════════════════════

def _load_lm2d_norm() -> dict:
    """Load anchor-normalised 2D landmark vectors {filename: np.ndarray[1404]}."""
    path = os.path.join(C.CACHE_DIR, "norm_2d.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalised 2D landmarks not found at {path}.\n"
            "Run phase1_data_prep.py first."
        )
    data = np.load(path, allow_pickle=True).item()
    print(f"[M2] Loaded 2D landmarks: {len(data)} samples")
    return data


def _load_lm3d_aligned() -> dict:
    """Load Procrustes-aligned 3D landmark vectors {filename: np.ndarray[1404]}."""
    path = os.path.join(C.CACHE_DIR, "aligned_3d.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Aligned 3D landmarks not found at {path}.\n"
            "Run phase1_data_prep.py with 3D enabled first."
        )
    data = np.load(path, allow_pickle=True).item()
    print(f"[M3] Loaded 3D landmarks: {len(data)} samples")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  run_m2() – Phase 3
# ═══════════════════════════════════════════════════════════════════════════════

def run_m2(train_df: pd.DataFrame,
           test_df:  pd.DataFrame,
           lm2d_norm: dict = None,
           epochs: int   = C.M2_EPOCHS,
           lr:     float = C.M2_LR,
           wd:     float = C.M2_WEIGHT_DECAY):
    """
    Train M2: 2D Landmarks + Anchor Normalisation + MLP.

    Returns:
        preds, targets, metrics
    """
    print("\n" + "═"*55)
    print("  PHASE 3 – M2: 2D Landmarks + Anchor Norm + MLP")
    print("═"*55)

    if lm2d_norm is None:
        lm2d_norm = _load_lm2d_norm()

    train_loader, test_loader = get_landmark_loaders(train_df, test_df, lm2d_norm)
    print(f"  Batches – train: {len(train_loader)}  test: {len(test_loader)}")

    model     = LandmarkMLP(input_dim=C.LANDMARK_DIM_3D,   # 1404 (3 dists × 468)
                             hidden_dims=C.M2_HIDDEN)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=C.VERBOSE
    )

    preds, targets, history = fit(
        model           = model,
        train_loader    = train_loader,
        val_loader      = test_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        epochs          = epochs,
        checkpoint_path = C.M2_CHECKPOINT,
        model_name      = "M2-MLP-2D",
        is_image_model  = False,
    )

    np.save(C.M2_PREDS_PATH, preds)
    print(f"  Predictions saved → {C.M2_PREDS_PATH}")

    metrics = compute_metrics(preds, targets)
    print(f"\n  M2 Final Metrics (test set):")
    print(f"    Pearson ρ : {metrics['pearson_r']:.4f}")
    print(f"    MAE       : {metrics['mae']:.4f}  (1-5 scale)")
    print(f"    RMSE      : {metrics['rmse']:.4f}")

    return preds, targets, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  run_m3() – Phase 4  [OPTIONAL]
# ═══════════════════════════════════════════════════════════════════════════════

def run_m3(train_df: pd.DataFrame,
           test_df:  pd.DataFrame,
           lm3d_aligned: dict = None,
           epochs: int   = C.M3_EPOCHS,
           lr:     float = C.M3_LR,
           wd:     float = C.M3_WEIGHT_DECAY):
    """
    Train M3: 3D Landmarks + Procrustes + MLP.
    This is OPTIONAL – only run if 3D landmarks were extracted in Phase 1.

    Returns:
        preds, targets, metrics
    """
    print("\n" + "═"*55)
    print("  PHASE 4 – M3: 3D Landmarks + Procrustes + MLP  [optional]")
    print("═"*55)

    if lm3d_aligned is None:
        lm3d_aligned = _load_lm3d_aligned()

    train_loader, test_loader = get_landmark_loaders(train_df, test_df, lm3d_aligned)
    print(f"  Batches – train: {len(train_loader)}  test: {len(test_loader)}")

    model     = LandmarkMLP(input_dim=C.LANDMARK_DIM_3D,   # 1404 (3×468)
                             hidden_dims=C.M3_HIDDEN)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=C.VERBOSE
    )

    preds, targets, history = fit(
        model           = model,
        train_loader    = train_loader,
        val_loader      = test_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        epochs          = epochs,
        checkpoint_path = C.M3_CHECKPOINT,
        model_name      = "M3-MLP-3D",
        is_image_model  = False,
    )

    np.save(C.M3_PREDS_PATH, preds)
    print(f"  Predictions saved → {C.M3_PREDS_PATH}")

    metrics = compute_metrics(preds, targets)
    print(f"\n  M3 Final Metrics (test set):")
    print(f"    Pearson ρ : {metrics['pearson_r']:.4f}")
    print(f"    MAE       : {metrics['mae']:.4f}  (1-5 scale)")
    print(f"    RMSE      : {metrics['rmse']:.4f}")

    return preds, targets, metrics


def compare_m2_m3(metrics_m2: dict, metrics_m3: dict):
    """Print improvement Δρ from M2 → M3."""
    delta = metrics_m3["pearson_r"] - metrics_m2["pearson_r"]
    print(f"\n  Δρ (M3 vs M2) = {delta:+.4f}  "
          f"({'3D Procrustes helps' if delta > 0 else '3D Procrustes does NOT help'})")


# ─── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M2 (and optionally M3)")
    parser.add_argument("--m3",       action="store_true", help="Also train M3 (3D)")
    parser.add_argument("--only-m3",  action="store_true", help="Only train M3")
    parser.add_argument("--epochs2",  type=int,   default=C.M2_EPOCHS)
    parser.add_argument("--epochs3",  type=int,   default=C.M3_EPOCHS)
    parser.add_argument("--lr",       type=float, default=C.M2_LR)
    args = parser.parse_args()

    train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
    test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
    if not os.path.exists(train_csv):
        raise RuntimeError("Run phase1_data_prep.py first.")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    m2_metrics = None
    m3_metrics = None

    if not args.only_m3:
        _, _, m2_metrics = run_m2(train_df, test_df, epochs=args.epochs2, lr=args.lr)

    if args.m3 or args.only_m3:
        _, _, m3_metrics = run_m3(train_df, test_df, epochs=args.epochs3, lr=args.lr)

    if m2_metrics and m3_metrics:
        compare_m2_m3(m2_metrics, m3_metrics)
