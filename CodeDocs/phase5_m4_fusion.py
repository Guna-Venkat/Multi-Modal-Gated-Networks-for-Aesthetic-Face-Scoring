"""
phase5_m4_fusion.py
───────────────────
Phase 5: Training the Adaptive Gated Fusion Model (M4).

Model:
    M4AdaptiveFusion (Multimodal)

Methodology:
    This phase implements the gating mechanism that calculates content-dependent
    trust weights (alpha and beta). It combines the texture features from M1
    with the geometric features from M2/M3 using a softmax-based gating network.
    
    Entropy regularization is applied to prevent the gating network from 
    degenerating into a simple average.

Architecture:
  ŷ = α(I)·y_img(I)  +  β(I)·y_land(L)
  α + β = 1  (softmax output of gating network)

Loss:
  L = MSE(ŷ, y)  +  λ · (−α·logα − β·logβ)

Key novelty: per-face weights → interpretable per-sample geometry vs. texture
importance.

Usage:
    python phase5_m4_fusion.py
    python phase5_m4_fusion.py --lm 2d          # use 2D landmarks (M2 style)
    python phase5_m4_fusion.py --lm 3d          # use 3D landmarks (M3 style)  [default]
    python phase5_m4_fusion.py --epochs 60 --lr 5e-5
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import config as C
from datasets import get_fusion_loaders
from models   import M4AdaptiveFusion
from trainer  import fit_fusion, compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper – load landmark dict based on user choice
# ═══════════════════════════════════════════════════════════════════════════════

def _load_landmarks(lm_type: str) -> dict:
    """
    lm_type: '2d'  → anchor-normalised 2D  [1404]
             '3d'  → Procrustes-aligned 3D [1404]
    """
    if lm_type == "2d":
        path = os.path.join(C.CACHE_DIR, "norm_2d.npy")
        label = "2D (anchor-normalised)"
    else:
        path = os.path.join(C.CACHE_DIR, "aligned_3d.npy")
        label = "3D (Procrustes-aligned)"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Landmark file not found: {path}\n"
            f"Run phase1_data_prep.py first "
            f"({'--no3d' if lm_type == '3d' else ''})."
        )
    data = np.load(path, allow_pickle=True).item()
    print(f"[M4] Using {label} landmarks – {len(data)} samples")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  run_m4()
# ═══════════════════════════════════════════════════════════════════════════════

def run_m4(train_df: pd.DataFrame,
           test_df:  pd.DataFrame,
           landmarks_dict: dict = None,
           lm_type: str   = "3d",
           epochs:  int   = C.M4_EPOCHS,
           lr:      float = C.M4_LR,
           wd:      float = C.M4_WEIGHT_DECAY,
           lam:     float = C.M4_ENTROPY_LAMBDA):
    """
    Train M4 Adaptive Fusion model.

    Args:
        train_df, test_df  : DataFrames with [filename, filepath, score]
        landmarks_dict     : pre-loaded dict, or None (will be loaded from cache)
        lm_type            : '2d' or '3d'  (used only when landmarks_dict is None)
        epochs, lr, wd     : training hyper-parameters
        lam                : entropy regularisation weight

    Returns:
        preds, targets, alphas, betas, metrics
    """
    print("\n" + "═"*55)
    print("  PHASE 5 – M4: Adaptive Fusion (Gating Network)")
    print("═"*55)

    if landmarks_dict is None:
        landmarks_dict = _load_landmarks(lm_type)

    # ── Data loaders ─────────────────────────────────────────
    train_loader, test_loader = get_fusion_loaders(train_df, test_df, landmarks_dict)
    print(f"  Batches – train: {len(train_loader)}  test: {len(test_loader)}")

    # ── Model ────────────────────────────────────────────────
    lm_dim = next(iter(landmarks_dict.values())).shape[0]   # 1404
    model  = M4AdaptiveFusion(
        pretrained       = True,
        landmark_dim     = lm_dim,
        landmark_hidden  = [256, 128],
        gate_hidden      = 64,
        dropout          = 0.3,
    )

    # ── Optimizer & scheduler ────────────────────────────────
    # Use a lower lr than M1 since we're fine-tuning jointly
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=C.VERBOSE
    )

    # ── Train ────────────────────────────────────────────────
    preds, targets, alphas, betas, history = fit_fusion(
        model           = model,
        train_loader    = train_loader,
        val_loader      = test_loader,
        optimizer       = optimizer,
        scheduler       = scheduler,
        epochs          = epochs,
        lam             = lam,
        checkpoint_path = C.M4_CHECKPOINT,
    )

    # ── Save predictions and gates ────────────────────────────
    np.save(C.M4_PREDS_PATH, preds)
    np.save(C.M4_GATES_PATH, np.stack([alphas, betas], axis=1))  # [N, 2]
    print(f"  Predictions saved → {C.M4_PREDS_PATH}")
    print(f"  Gate weights saved → {C.M4_GATES_PATH}  (α, β)")

    # ── Final metrics ─────────────────────────────────────────
    metrics = compute_metrics(preds, targets)
    print(f"\n  M4 Final Metrics (test set):")
    print(f"    Pearson ρ : {metrics['pearson_r']:.4f}")
    print(f"    MAE       : {metrics['mae']:.4f}  (1-5 scale)")
    print(f"    RMSE      : {metrics['rmse']:.4f}")
    print(f"    Gate mean : α={alphas.mean():.3f}  β={betas.mean():.3f}")
    print(f"    Gate std  : α={alphas.std():.3f}  β={betas.std():.3f}")

    return preds, targets, alphas, betas, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Gating Analysis helpers  (used in Phase 6 and notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

def gating_analysis(test_df: pd.DataFrame, alphas: np.ndarray, betas: np.ndarray):
    """
    Per-face gating analysis.
    Returns a DataFrame with [filename, score, alpha, beta, geometry_dominant].
    """
    df = test_df.copy()
    df = df.iloc[:len(betas)].copy()   # align length
    df["alpha"] = alphas
    df["beta"]  = betas
    df["geometry_dominant"] = betas > 0.5
    df = df.sort_values("beta", ascending=False).reset_index(drop=True)
    return df


def print_gating_stats(gate_df: pd.DataFrame):
    geom_pct = gate_df["geometry_dominant"].mean() * 100
    print(f"\n  Gating Stats:")
    print(f"    % faces geometry-dominant (β > 0.5)  : {geom_pct:.1f}%")
    print(f"    % faces texture-dominant  (β < 0.5)  : {100 - geom_pct:.1f}%")
    print(f"\n  Top-5 geometry-driven faces (highest β):")
    print(gate_df[["filename", "score", "alpha", "beta"]].head(5).to_string(index=False))
    print(f"\n  Top-5 texture-driven faces  (lowest β):")
    print(gate_df[["filename", "score", "alpha", "beta"]].tail(5).to_string(index=False))


# ─── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M4 – Adaptive Fusion")
    parser.add_argument("--lm",      choices=["2d", "3d"], default="3d",
                        help="Which landmark type to use (default: 3d)")
    parser.add_argument("--epochs",  type=int,   default=C.M4_EPOCHS)
    parser.add_argument("--lr",      type=float, default=C.M4_LR)
    parser.add_argument("--lam",     type=float, default=C.M4_ENTROPY_LAMBDA)
    args = parser.parse_args()

    train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
    test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
    if not os.path.exists(train_csv):
        raise RuntimeError("Run phase1_data_prep.py first.")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    preds, targets, alphas, betas, metrics = run_m4(
        train_df, test_df,
        lm_type = args.lm,
        epochs  = args.epochs,
        lr      = args.lr,
        lam     = args.lam,
    )

    gate_df = gating_analysis(test_df, alphas, betas)
    print_gating_stats(gate_df)
    gate_df.to_csv(os.path.join(C.RESULTS_DIR, "m4_gate_analysis.csv"), index=False)
