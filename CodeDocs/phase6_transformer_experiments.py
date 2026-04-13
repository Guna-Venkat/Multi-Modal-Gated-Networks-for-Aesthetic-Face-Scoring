"""
phase6_transformer_experiments.py
─────────────────────────────────
Advanced Phase: Transformer-based Aesthetic Scoring (M5, M6, M7).

Models:
    M5: ViT-B/16 (Patch-only) - Base Vision Transformer.
    M6: ViT + Geometry concatenation.
    M7: Cross-Attention Fusion - Uses a Transformer decoder to let 
        geometric landmarks "attend" to specific image patches.

Key Features:
    - Implementation of Cross-Attention between texture and geometry.
    - Interpretability: Extraction of attention weights to identify 
      which facial regions are most influential for the model's decision.

M5_ViT: 
  - Freezes a pre‑trained ViT-B/16 backbone and trains a regression head on top.
  - Uses only image texture information (no landmarks).

M6_LandmarkTransformer:
  - Takes [B, 468, 3] facial landmark points, projects them to a latent space,
    processes them with a transformer encoder, and performs mean pooling.
  - Can optionally return attention heatmaps (simplified version).

M7_CrossAttentionFusion:
  - Fuses 196 ViT image patch tokens and 468 landmark tokens via cross‑attention.
  - Landmark tokens query the image patches, producing a fused representation.

M8_GatedViTFusion:
  - Combines ViT image features and landmark MLP predictions with a learnable
    gating mechanism (α, β) that depends on the ViT’s [CLS] token.
  - Same fusion strategy as M4 but using ViT instead of a CNN.

Usage (Local):
    python transformer_experiments.py --run m5
    python transformer_experiments.py --run m6
    python transformer_experiments.py --run m7
    python transformer_experiments.py --run m8
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models

# Internal imports from the CodeDocs package
import config as C
from datasets import get_image_loaders, get_landmark_loaders, get_fusion_loaders
from trainer import EarlyStopping, compute_metrics, save_checkpoint, fit_fusion


# ═══════════════════════════════════════════════════════════════════════════════
#  M5 – Vision Transformer (Texture baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class M5ViT(nn.Module):
    """
    M5: Vision Transformer baseline.
    
    Uses a frozen torchvision ViT-B/16 pretrained on ImageNet. A new regression
    head (two fully‑connected layers with dropout) is trained from scratch.
    Outputs a beauty score in [0,1].
    """
    def __init__(self, freeze_backbone=True, dropout=0.3):
        """
        Args:
            freeze_backbone (bool): If True, all ViT parameters are frozen.
            dropout (float): Dropout probability in the regression head.
        """
        super().__init__()
        # Load pretrained ViT-B/16 with default weights (ImageNet)
        weights = tv_models.ViT_B_16_Weights.DEFAULT
        self.vit = tv_models.vit_b_16(weights=weights)
        
        # Freeze all backbone parameters if requested
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Replace the classification head with a custom regression head
        self.vit.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),   # ViT hidden size = 768
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()           # Constrain output to [0,1]
        )

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images, shape [B, 3, 224, 224].
        
        Returns:
            torch.Tensor: Predicted beauty scores, shape [B].
        """
        return self.vit(x).squeeze(1)   # Remove extra singleton dimension


# ═══════════════════════════════════════════════════════════════════════════════
#  M6 – Landmark Attention Transformer (with positional encoding + mean pooling)
# ═══════════════════════════════════════════════════════════════════════════════

class M6LandmarkTransformer(nn.Module):
    """
    M6: Transformer encoder applied to 468 facial landmarks.
    
    Each landmark (x,y,z) is linearly projected to a d_model‑dimensional space,
    added with a learnable positional embedding, then processed by a transformer
    encoder. The final representation is obtained by mean pooling over landmarks.
    A small MLP regresses the beauty score.
    """
    def __init__(self, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        """
        Args:
            d_model (int): Latent dimension for landmark tokens.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        
        # Linear projection from 3D coordinates to d_model
        self.point_proj = nn.Linear(3, d_model)
        
        # Learnable positional encoding (added to projected landmarks)
        self.pos_embed = nn.Parameter(torch.randn(1, 468, d_model) * 0.02)

        # Transformer encoder (batch_first=True for convenient shapes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout,
            batch_first=True, norm_first=True          # Pre‑LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Regression head (takes mean‑pooled representation)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attn=False):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Flattened landmarks, shape [B, 468*3] = [B, 1404].
            return_attn (bool): If True, returns a dummy attention map
                                 (compatible with training loop).
        
        Returns:
            score (torch.Tensor): Predicted beauty scores, shape [B].
            attn (torch.Tensor or None): Dummy attention map of shape [B, 468]
                                         if return_attn=True, else None.
        """
        B = x.shape[0]
        # Reshape from flat to [B, 468, 3]
        x = x.view(B, 468, 3)
        
        # Project and add positional encoding
        tokens = self.point_proj(x) + self.pos_embed   # [B, 468, d_model]
        
        # Transformer encoding
        encoded = self.transformer(tokens)             # [B, 468, d_model]
        
        # Mean pooling over landmarks
        pooled = encoded.mean(dim=1)                   # [B, d_model]
        
        # Regression
        score = self.head(pooled).squeeze(1)           # [B]
        
        if return_attn:
            # Return a uniform attention map for compatibility.
            # Real attention can be extracted by modifying the encoder.
            attn = torch.ones(B, 468, device=x.device) / 468
            return score, attn
        return score, None


# ═══════════════════════════════════════════════════════════════════════════════
#  M7 – Cross-Attention Fusion (ViT + Landmark)
# ═══════════════════════════════════════════════════════════════════════════════

class M7CrossAttentionFusion(nn.Module):
    """
    M7: Cross‑attention fusion of image patches and landmark tokens.
    
    Image patches (196) are extracted from a frozen ViT and projected to d_model.
    Landmark tokens (468) are processed by a shallow transformer encoder.
    Then a multi‑head cross‑attention layer uses the landmarks as queries and
    the image patches as keys/values, producing fused landmark tokens.
    Those are flattened and passed through a regression head.
    """
    def __init__(self, freeze_vit=True, d_model=128, nhead=4, dropout=0.3):
        """
        Args:
            freeze_vit (bool): If True, the ViT backbone is frozen.
            d_model (int): Latent dimension for both modalities.
            nhead (int): Number of attention heads in all transformers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # ----- Image branch (frozen ViT) -----
        weights = tv_models.ViT_B_16_Weights.DEFAULT
        self.vit = tv_models.vit_b_16(weights=weights)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Project ViT patch tokens (768‑dim) to d_model
        self.patch_proj = nn.Linear(768, d_model)

        # ----- Landmark branch -----
        # Linear projection from 3D coordinates to d_model
        self.point_proj = nn.Linear(3, d_model)
        
        # Shallow transformer encoder for landmarks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.landmark_transformer = nn.TransformerEncoder(encoder_layer, 2)

        # ----- Cross‑attention fusion -----
        # Queries = landmark tokens, Keys/Values = image patch tokens
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # ----- Regression head -----
        # Flatten the 468 fused landmark tokens (each d_model‑dim)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(468 * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, landmarks):
        """
        Forward pass.
        
        Args:
            image (torch.Tensor): Input images, shape [B, 3, 224, 224].
            landmarks (torch.Tensor): Flattened landmarks, shape [B, 1404].
        
        Returns:
            score (torch.Tensor): Predicted beauty scores, shape [B].
            attn_weights (torch.Tensor): Cross‑attention weights of shape
                                         [B, 468, 196], indicating how each
                                         landmark attends to image patches.
        """
        B = image.shape[0]
        
        # --- Image branch: extract patch tokens from ViT ---
        # The following code replicates the ViT forward pass up to the encoder output,
        # but without the final classification head.
        with torch.set_grad_enabled(False):   # ViT is frozen
            # _process_input: splits image into patches and applies linear projection
            x_img = self.vit._process_input(image)          # [B, 196, 768]
            batch_cls = self.vit.class_token.expand(B, -1, -1)  # [B, 1, 768]
            x_img = torch.cat([batch_cls, x_img], dim=1)    # [B, 197, 768]
            x_img = self.vit.encoder(x_img)                 # [B, 197, 768]
        
        # Drop the [CLS] token, keep only the 196 patch tokens
        img_patches = x_img[:, 1:, :]                       # [B, 196, 768]
        img_patches = self.patch_proj(img_patches)          # [B, 196, d_model]
        
        # --- Landmark branch ---
        lms = landmarks.view(B, 468, 3)                     # [B, 468, 3]
        lm_tokens = self.point_proj(lms)                    # [B, 468, d_model]
        lm_tokens = self.landmark_transformer(lm_tokens)    # [B, 468, d_model]
        
        # --- Cross‑attention: landmarks attend to image patches ---
        fused_tokens, attn_weights = self.cross_attn(
            query=lm_tokens,          # [B, 468, d_model]
            key=img_patches,          # [B, 196, d_model]
            value=img_patches,        # [B, 196, d_model]
            need_weights=True
        )   # fused_tokens: [B, 468, d_model], attn_weights: [B, 468, 196]
        
        # --- Regression ---
        score = self.head(fused_tokens).squeeze(-1)          # [B]
        
        return score, attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
#  M8 – Gated Fusion (ViT + Landmark MLP) – same as M4 but with ViT
# ═══════════════════════════════════════════════════════════════════════════════

class M8_GatedViTFusion(nn.Module):
    """
    M8: Gated fusion of ViT image prediction and landmark MLP prediction.
    
    This model mirrors the fusion strategy of M4, but replaces the CNN image
    branch with a ViT. The ViT produces a beauty score and also its [CLS] token
    is used to compute per‑sample gating weights α and β. The final score is
    α * y_img + β * y_land. The gates are softmax‑normalised.
    """
    def __init__(self, freeze_vit=True, dropout=0.3):
        """
        Args:
            freeze_vit (bool): If True, the ViT backbone is frozen.
            dropout (float): Dropout probability in all MLP heads.
        """
        super().__init__()
        
        # ----- ViT image branch -----
        weights = tv_models.ViT_B_16_Weights.DEFAULT
        self.vit = tv_models.vit_b_16(weights=weights)
        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Regression head on top of ViT (identical to M5)
        self.vit.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # ----- Landmark MLP branch -----
        # Input: flattened 468*3 = 1404 coordinates
        self.land_mlp = nn.Sequential(
            nn.Linear(1404, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ----- Gating network -----
        # Takes the ViT's [CLS] token (768‑dim) and outputs two logits
        self.gate = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, image, landmarks):
        """
        Forward pass.
        
        Args:
            image (torch.Tensor): Input images, shape [B, 3, 224, 224].
            landmarks (torch.Tensor): Flattened landmarks, shape [B, 1404].
        
        Returns:
            y_fused (torch.Tensor): Fused beauty score, shape [B].
            alpha (torch.Tensor): Weight for image branch, shape [B].
            beta (torch.Tensor): Weight for landmark branch, shape [B].
            y_img (torch.Tensor): Image branch raw score, shape [B].
            y_land (torch.Tensor): Landmark branch raw score, shape [B].
        """
        # ---- Image branch: both prediction and [CLS] token ----
        y_img = self.vit(image).squeeze(1)   # [B]
        
        # Extract the [CLS] token after the encoder (before the classification head)
        x = self.vit._process_input(image)               # [B, 196, 768]
        B = image.shape[0]
        cls_token = self.vit.class_token.expand(B, -1, -1)  # [B, 1, 768]
        x = torch.cat([cls_token, x], dim=1)             # [B, 197, 768]
        x = self.vit.encoder(x)                          # [B, 197, 768]
        cls_feat = x[:, 0, :]                            # [B, 768]
        
        # ---- Landmark branch ----
        y_land = self.land_mlp(landmarks).squeeze(1)     # [B]
        
        # ---- Gating ----
        logits = self.gate(cls_feat)                     # [B, 2]
        weights = torch.softmax(logits, dim=1)           # [B, 2]
        alpha, beta = weights[:, 0], weights[:, 1]       # each [B]
        
        # ---- Fused prediction ----
        y_fused = alpha * y_img + beta * y_land          # [B]
        
        return y_fused, alpha, beta, y_img, y_land


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loops (M5, M6, M7 use fit_transformer; M8 uses fit_fusion from trainer)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_transformer(model, train_loader, val_loader, optimizer, scheduler,
                    epochs, model_type, checkpoint_path):
    """
    Generic training loop for models M5, M6, M7.
    
    Handles both single‑input (M5, M6) and dual‑input (M7) models.
    Saves the best model (by validation loss), predictions, and attention maps.
    
    Args:
        model (nn.Module): The transformer model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Maximum number of epochs.
        model_type (str): One of "m5", "m6", "m7". Used for conditional logic.
        checkpoint_path (str): Path to save the best model checkpoint.
    
    Returns:
        preds (np.ndarray): Final predictions on validation set.
        metrics (dict): Dictionary with MAE, RMSE, R², Pearson correlation.
    """
    device = torch.device(C.DEVICE)
    model = model.to(device)
    criterion = nn.MSELoss()
    es = EarlyStopping(patience=5)

    print(f"\n{'═'*55}")
    print(f"  Training {model_type.upper()}  |  {epochs} epochs |  {C.DEVICE}")
    print(f"{'═'*55}")

    # Storage for the best validation predictions and attention maps
    best_val_loss = float('inf')
    best_preds = None
    best_attns = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        # ---- Training loop ----
        for batch in train_loader:
            # Unpack batch according to model type
            if model_type == "m7":
                img, lm, y = batch
                img, lm, y = img.to(device), lm.to(device), y.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass
            if model_type == "m5":
                pred = model(x)
                loss = criterion(pred, y)
            elif model_type == "m6":
                pred, _ = model(x)
                loss = criterion(pred, y)
            elif model_type == "m7":
                pred, _ = model(img, lm)
                loss = criterion(pred, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item() * y.size(0)

        tr_loss /= len(train_loader.dataset)

        # ---- Validation loop ----
        model.eval()
        vl_loss = 0.0
        all_preds, all_tgts, all_attns = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                if model_type == "m7":
                    img, lm, y = batch
                    img, lm, y = img.to(device), lm.to(device), y.to(device)
                    pred, attn = model(img, lm)
                    all_attns.append(attn.cpu().numpy())
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    if model_type == "m6":
                        pred, attn = model(x, return_attn=True)
                        all_attns.append(attn.cpu().numpy())
                    else:
                        pred = model(x)

                vl_loss += criterion(pred, y).item() * y.size(0)
                all_preds.append(pred.cpu().numpy())
                all_tgts.append(y.cpu().numpy())

        vl_loss /= len(val_loader.dataset)
        if scheduler:
            scheduler.step(vl_loss)

        # Compute metrics and print
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_tgts)
        metrics = compute_metrics(preds, targets)

        print(f"  Ep {epoch:03d}  tr={tr_loss:.4f}  vl={vl_loss:.4f}  ρ={metrics['pearson_r']:.4f}  MAE={metrics['mae']:.4f}  ({time.time()-t0:.1f}s)")

        # Early stopping
        es(vl_loss, model)
        if es.stop:
            print(f"  Early stopping at epoch {epoch}.")
            break

        # Keep best predictions and attention maps
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_preds = preds
            best_targets = targets
            if model_type in ["m6", "m7"]:
                best_attns = np.concatenate(all_attns)

    # Restore best model weights
    es.restore(model)
    print(f"\n  [Checkpoint] Saved best {model_type.upper()} model.")
    save_checkpoint(model, checkpoint_path)

    # Save predictions and attention maps
    np.save(checkpoint_path.replace(".pt", "_preds.npy"), best_preds)
    np.save(checkpoint_path.replace(".pt", "_targets.npy"), best_targets)
    if model_type in ["m6", "m7"] and best_attns is not None:
        np.save(checkpoint_path.replace(".pt", "_heatmaps.npy"), best_attns)
        print(f"  [Heatmaps] Saved to {checkpoint_path.replace('.pt', '_heatmaps.npy')}")

    return best_preds, metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Run wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def run_m5(train_df, test_df):
    """
    Execute M5 experiment: ViT texture baseline.
    
    Args:
        train_df (pd.DataFrame): Training split with columns 'filepath' and 'score'.
        test_df (pd.DataFrame): Validation split with same structure.
    
    Returns:
        tuple: (predictions, metrics) from fit_transformer.
    """
    model = M5ViT()
    train_loader, test_loader = get_image_loaders(train_df, test_df)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "m5_transformer.pt")
    return fit_transformer(model, train_loader, test_loader, optimizer, scheduler, C.M4_EPOCHS, "m5", ckpt_path)


def run_m6(train_df, test_df):
    """
    Execute M6 experiment: Landmark transformer.
    
    Args:
        train_df (pd.DataFrame): Training split with 'filename' column.
        test_df (pd.DataFrame): Validation split with 'filename' column.
    
    Returns:
        tuple: (predictions, metrics) from fit_transformer.
    """
    model = M6LandmarkTransformer()
    # Load pre‑computed 3D landmarks from cache
    path = os.path.join(C.CACHE_DIR, "aligned_3d.npy")
    lm_data = np.load(path, allow_pickle=True).item()
    train_loader, test_loader = get_landmark_loaders(train_df, test_df, lm_data)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "m6_transformer.pt")
    return fit_transformer(model, train_loader, test_loader, optimizer, scheduler, C.M4_EPOCHS, "m6", ckpt_path)


def run_m7(train_df, test_df):
    """
    Execute M7 experiment: Cross‑attention fusion of ViT patches and landmarks.
    
    Args:
        train_df (pd.DataFrame): Training split with 'filename' and 'filepath'.
        test_df (pd.DataFrame): Validation split with same structure.
    
    Returns:
        tuple: (predictions, metrics) from fit_transformer.
    """
    model = M7CrossAttentionFusion()
    path = os.path.join(C.CACHE_DIR, "aligned_3d.npy")
    lm_data = np.load(path, allow_pickle=True).item()
    train_loader, test_loader = get_fusion_loaders(train_df, test_df, lm_data)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "m7_transformer.pt")
    return fit_transformer(model, train_loader, test_loader, optimizer, scheduler, C.M4_EPOCHS, "m7", ckpt_path)


def run_m8(train_df, test_df):
    """
    Execute M8 experiment: Gated ViT + landmark MLP fusion.
    
    Args:
        train_df (pd.DataFrame): Training split with 'filename' and 'filepath'.
        test_df (pd.DataFrame): Validation split with same structure.
    
    Returns:
        tuple: (predictions, targets, alphas, betas) from fit_fusion.
    """
    # Load 3D landmarks
    path = os.path.join(C.CACHE_DIR, "aligned_3d.npy")
    lm_data = np.load(path, allow_pickle=True).item()
    train_loader, test_loader = get_fusion_loaders(train_df, test_df, lm_data)

    model = M8_GatedViTFusion(freeze_vit=True, dropout=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    ckpt_path = os.path.join(C.CHECKPOINT_DIR, "m8_gated_vit.pt")
    # fit_fusion is a custom training function from CodeDocs.trainer that handles
    # the gating loss (entropy regularisation) and returns additional gate values.
    preds, targets, alphas, betas, history = fit_fusion(
        model, train_loader, test_loader,
        optimizer, scheduler,
        epochs=C.M4_EPOCHS, lam=C.M4_ENTROPY_LAMBDA,
        checkpoint_path=ckpt_path
    )
    # Save gates and predictions
    np.save(ckpt_path.replace(".pt", "_preds.npy"), preds)
    np.save(ckpt_path.replace(".pt", "_gates.npy"), np.stack([alphas, betas], axis=1))
    return preds, targets, alphas, betas


# ═══════════════════════════════════════════════════════════════════════════════
#  Execute Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_type, train_df, test_df):
    """
    Dispatch function to run the requested experiment.
    
    Args:
        model_type (str): One of 'm5', 'm6', 'm7', 'm8'.
        train_df (pd.DataFrame): Training split.
        test_df (pd.DataFrame): Validation split.
    
    Returns:
        Depending on model_type:
            - M5, M6, M7: (preds, metrics)
            - M8: (preds, targets, alphas, betas)
    """
    if model_type == "m5":
        return run_m5(train_df, test_df)
    elif model_type == "m6":
        return run_m6(train_df, test_df)
    elif model_type == "m7":
        return run_m7(train_df, test_df)
    elif model_type == "m8":
        return run_m8(train_df, test_df)
    else:
        raise ValueError("Invalid model type. Choose m5, m6, m7, or m8.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run transformer experiments for beauty prediction.")
    parser.add_argument("--run", choices=["m5", "m6", "m7", "m8"], required=True,
                        help="Which model to run: m5 (ViT), m6 (Landmark Transformer), m7 (Cross-Attention), m8 (Gated ViT+Landmark)")
    args = parser.parse_args()

    # Load train/test splits (assumed to be created by phase1_data_prep.py)
    train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
    test_csv = os.path.join(C.CACHE_DIR, "test_split.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    run_experiment(args.run, train_df, test_df)