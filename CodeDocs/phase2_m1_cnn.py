"""
phase2_m1_cnn.py
────────────────
Phase 2: Training the Appearance-based Texture Model (M1).

Model: 
    M1ImageCNN (ResNet-18)
    
Focus: 
    Analyzing the skin quality, symmetry, and color features of the face 
    to predict aesthetic beauty scores.
    
Usage:
    Import `run_m1` and provide training/testing dataframes.

  • Input : RGB image 224×224
  • Output: scalar [0,1]  (de-normalise: score = pred * 4 + 1)
  • Loss  : MSE
  • Optim : Adam lr=1e-3  weight_decay=1e-4
  • Sched : ReduceLROnPlateau (patience=3, factor=0.5)
  • Early stop : patience=5

Usage:
    python phase2_m1_cnn.py
    python phase2_m1_cnn.py --epochs 50 --lr 5e-4
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import config as C
from datasets import get_image_loaders
from models   import M1ImageCNN
from trainer  import fit, compute_metrics


def run_m1(train_df: pd.DataFrame,
           test_df:  pd.DataFrame,
           epochs:   int   = C.M1_EPOCHS,
           lr:       float = C.M1_LR,
           wd:       float = C.M1_WEIGHT_DECAY):
    """
    Train M1 and save predictions + checkpoint.

    Returns:
        preds   : np.ndarray [N_test]  normalised predictions [0,1]
        targets : np.ndarray [N_test]  normalised labels
        metrics : dict  (pearson_r, mae, rmse)
    """
    print("\n" + "═"*55)
    print("  PHASE 2 – M1: Full-Image CNN (ResNet-18)")
    print("═"*55)

    # ── Data loaders ─────────────────────────────────────────
    train_loader, test_loader = get_image_loaders(train_df, test_df)
    print(f"  Batches – train: {len(train_loader)}  test: {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────
    model = M1ImageCNN(pretrained=True)

    # ── Optimizer & scheduler ────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=C.VERBOSE
    )

    # ── Train ────────────────────────────────────────────────
    preds, targets, history = fit(
        model        = model,
        train_loader = train_loader,
        val_loader   = test_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        epochs       = epochs,
        checkpoint_path = C.M1_CHECKPOINT,
        model_name   = "M1-CNN",
        is_image_model = True,
    )

    # ── Save predictions ─────────────────────────────────────
    np.save(C.M1_PREDS_PATH, preds)
    print(f"  Predictions saved → {C.M1_PREDS_PATH}")

    metrics = compute_metrics(preds, targets)
    print(f"\n  M1 Final Metrics (test set):")
    print(f"    Pearson ρ : {metrics['pearson_r']:.4f}")
    print(f"    MAE       : {metrics['mae']:.4f}  (1-5 scale)")
    print(f"    RMSE      : {metrics['rmse']:.4f}")

    return preds, targets, metrics


# ─── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M1 – Full-Image CNN")
    parser.add_argument("--epochs", type=int,   default=C.M1_EPOCHS)
    parser.add_argument("--lr",     type=float, default=C.M1_LR)
    parser.add_argument("--wd",     type=float, default=C.M1_WEIGHT_DECAY)
    args = parser.parse_args()

    # Load cached splits (created by phase1)
    import os
    train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
    test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
    if not os.path.exists(train_csv):
        raise RuntimeError("Run phase1_data_prep.py first to create data splits.")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    run_m1(train_df, test_df, epochs=args.epochs, lr=args.lr, wd=args.wd)
