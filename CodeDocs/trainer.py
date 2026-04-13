"""
trainer.py
──────────
Shared training utilities used by all phases:

  EarlyStopping       – stops training when val loss plateaus
  train_epoch()       – one forward+backward pass over a dataloader
  eval_epoch()        – evaluation pass, returns loss + predictions
  compute_metrics()   – Pearson ρ, MAE, RMSE (on de-normalised scores)
  save_checkpoint()   – save model state dict
  load_checkpoint()   – load model state dict
  fit()               – complete training loop (M1 / M2 / M3)
  fit_fusion()        – training loop for M4 with entropy regularisation
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

import config as C


# ═══════════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation loss does not improve for `patience` epochs.
    Saves the best model weights to a temporary buffer.
    """
    def __init__(self, patience: int = C.PATIENCE, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.best_state = None
        self.stop       = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module):
        """Load the best weights back into the model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds: np.ndarray, targets: np.ndarray, denorm: bool = True):
    """
    Args:
        preds, targets : 1-D arrays in [0, 1] (normalised) or [1, 5] raw
        denorm         : if True, scale back to 1-5 before computing MAE/RMSE

    Returns dict: {pearson_r, pearson_p, mae, rmse}
    """
    if denorm:
        preds_d   = preds   * 4.0 + 1.0
        targets_d = targets * 4.0 + 1.0
    else:
        preds_d   = preds
        targets_d = targets

    r, p      = pearsonr(preds_d, targets_d)
    mae       = np.mean(np.abs(preds_d - targets_d))
    rmse      = np.sqrt(np.mean((preds_d - targets_d) ** 2))

    return {"pearson_r": r, "pearson_p": p, "mae": mae, "rmse": rmse}


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-branch training (M1 / M2 / M3)
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, is_image_model: bool):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if is_image_model:
            x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, is_image_model: bool):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_tgts   = []

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * y.size(0)
        all_preds.append(pred.cpu().numpy())
        all_tgts.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, preds, targets


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, path: str, extra: dict = None):
    state = {"model": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(model, path: str, device=None):
    if device is None:
        device = C.DEVICE
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  [Checkpoint] Loaded ← {path}")
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  fit() – generic loop for M1, M2, M3
# ═══════════════════════════════════════════════════════════════════════════════

def fit(model, train_loader, val_loader,
        optimizer, scheduler=None,
        epochs: int = 30,
        checkpoint_path: str = None,
        model_name: str = "model",
        is_image_model: bool = True):
    """
    Standard training loop for a single-input model.

    Returns:
        preds_val   : np.ndarray  [N]  on validation set (normalised [0,1])
        targets_val : np.ndarray  [N]
        history     : dict with train_loss, val_loss per epoch
    """
    device    = torch.device(C.DEVICE)
    model     = model.to(device)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=C.PATIENCE)
    history   = {"train_loss": [], "val_loss": []}

    print(f"\n{'═'*55}")
    print(f"  Training {model_name}  |  {epochs} epochs max  |  {C.DEVICE}")
    print(f"{'═'*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer,
                              criterion, device, is_image_model)
        vl_loss, preds, targets = eval_epoch(model, val_loader,
                                             criterion, device, is_image_model)
        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        metrics = compute_metrics(preds, targets)
        elapsed = time.time() - t0
        if C.VERBOSE or epoch % 5 == 0:
            print(f"  Ep {epoch:03d}/{epochs}  "
                  f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                  f"ρ={metrics['pearson_r']:.4f}  MAE={metrics['mae']:.4f}  "
                  f"({elapsed:.1f}s)")

        es(vl_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    es.restore(model)

    # Final evaluation with best weights
    _, preds_val, targets_val = eval_epoch(model, val_loader,
                                           criterion, device, is_image_model)
    final_metrics = compute_metrics(preds_val, targets_val)
    print(f"\n  ✓ Best val  ρ={final_metrics['pearson_r']:.4f}  "
          f"MAE={final_metrics['mae']:.4f}  RMSE={final_metrics['rmse']:.4f}")

    if checkpoint_path:
        save_checkpoint(model, checkpoint_path,
                        extra={"metrics": final_metrics, "history": history})

    return preds_val, targets_val, history


# ═══════════════════════════════════════════════════════════════════════════════
#  fit_fusion() – M4 with entropy regularisation
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy_loss(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    -E[α·log(α) + β·log(β)]
    Encourages both branches to contribute (avoids collapse to one branch).
    Clamp to avoid log(0).
    """
    eps = 1e-8
    ent = -(alpha * torch.log(alpha + eps) + beta * torch.log(beta + eps))
    return ent.mean()


def train_epoch_fusion(model, loader, optimizer, criterion, device, lam: float):
    model.train()
    total_loss = 0.0
    for img, lm, y in loader:
        img, lm, y = img.to(device), lm.to(device), y.to(device)

        optimizer.zero_grad()
        y_fused, alpha, beta, _, _ = model(img, lm)

        mse  = criterion(y_fused, y)
        ent  = _entropy_loss(alpha, beta)
        loss = mse + lam * ent

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += mse.item() * y.size(0)   # track MSE only for comparability

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch_fusion(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts, all_alpha, all_beta = [], [], [], []

    for img, lm, y in loader:
        img, lm, y = img.to(device), lm.to(device), y.to(device)
        y_fused, alpha, beta, _, _ = model(img, lm)
        loss = criterion(y_fused, y)
        total_loss += loss.item() * y.size(0)
        all_preds.append(y_fused.cpu().numpy())
        all_tgts.append(y.cpu().numpy())
        all_alpha.append(alpha.cpu().numpy())
        all_beta.append(beta.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_tgts)
    alphas  = np.concatenate(all_alpha)
    betas   = np.concatenate(all_beta)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, preds, targets, alphas, betas


def fit_fusion(model, train_loader, val_loader,
               optimizer, scheduler=None,
               epochs: int = 40,
               lam: float = C.M4_ENTROPY_LAMBDA,
               checkpoint_path: str = None):
    """
    Training loop for M4 Adaptive Fusion.

    Returns:
        preds_val, targets_val, alphas_val, betas_val, history
    """
    device    = torch.device(C.DEVICE)
    model     = model.to(device)
    criterion = nn.MSELoss()
    es        = EarlyStopping(patience=C.PATIENCE)
    history   = {"train_loss": [], "val_loss": []}

    print(f"\n{'═'*55}")
    print(f"  Training M4 (Adaptive Fusion)  |  λ={lam}  |  {C.DEVICE}")
    print(f"{'═'*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch_fusion(model, train_loader, optimizer,
                                     criterion, device, lam)
        vl_loss, preds, targets, alphas, betas = eval_epoch_fusion(
            model, val_loader, criterion, device
        )

        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        metrics = compute_metrics(preds, targets)
        elapsed = time.time() - t0

        if C.VERBOSE or epoch % 5 == 0:
            print(f"  Ep {epoch:03d}/{epochs}  "
                  f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                  f"ρ={metrics['pearson_r']:.4f}  MAE={metrics['mae']:.4f}  "
                  f"ᾱ={alphas.mean():.3f}  β̄={betas.mean():.3f}  "
                  f"({elapsed:.1f}s)")

        es(vl_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch}.")
            break

    es.restore(model)

    # Final pass
    _, preds_val, targets_val, alphas_val, betas_val = eval_epoch_fusion(
        model, val_loader, criterion, device
    )
    final_metrics = compute_metrics(preds_val, targets_val)
    print(f"\n  ✓ Best val  ρ={final_metrics['pearson_r']:.4f}  "
          f"MAE={final_metrics['mae']:.4f}  RMSE={final_metrics['rmse']:.4f}")
    print(f"     Gate mean: α={alphas_val.mean():.3f}  β={betas_val.mean():.3f}")

    if checkpoint_path:
        save_checkpoint(model, checkpoint_path,
                        extra={"metrics": final_metrics, "history": history})

    return preds_val, targets_val, alphas_val, betas_val, history
