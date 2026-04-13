"""
models.py
─────────
All four model architectures.

  M1ImageCNN          – ResNet-18 fine-tuned for beauty regression
  M2LandmarkMLP       – MLP on normalised 2D landmarks
  M3LandmarkMLP       – MLP on Procrustes-aligned 3D landmarks (same arch as M2)
  M4AdaptiveFusion    – Gating network (α·y_img + β·y_land)

All models output a scalar in [0, 1]  (normalised from 1-5 scale in datasets.py).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config as C


# ═══════════════════════════════════════════════════════════════════════════════
#  M1 – Full-image CNN (ResNet-18)
# ═══════════════════════════════════════════════════════════════════════════════

class M1ImageCNN(nn.Module):
    """
    ResNet-18 pretrained on ImageNet.
    Final FC replaced with: Linear(512, 1) + Sigmoid  → scalar [0,1].

    Architecture choice: Sigmoid output keeps predictions in [0,1],
    consistent with the normalised label range.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Feature extractor: everything before the FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # [B, 512, 1, 1] after adaptive avg pool

        self.head = nn.Sequential(
            nn.Flatten(),          # [B, 512]
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),          # [B, 1]
        )

    def forward(self, x):
        feat = self.features(x)    # [B, 512, 1, 1]
        return self.head(feat).squeeze(1)   # [B]


# ═══════════════════════════════════════════════════════════════════════════════
#  M2 / M3 – Landmark MLP
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkMLP(nn.Module):
    """
    Three-layer MLP for landmark-based regression.
    Input:   [B, input_dim]  (1404 for both 2D-anchor and 3D-Procrustes)
    Output:  scalar [B]

    hidden_dims: list of hidden layer sizes (default [256, 128])
    """
    def __init__(self,
                 input_dim: int = C.LANDMARK_DIM_3D,
                 hidden_dims: list = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)   # [B]

    def forward_feat(self, x):
        """Return the last hidden representation (for analysis)."""
        for layer in self.net[:-2]:     # everything before final Linear+Sigmoid
            x = layer(x)
        return x


# Aliases matching model names used in config
M2LandmarkMLP = LandmarkMLP
M3LandmarkMLP = LandmarkMLP


# ═══════════════════════════════════════════════════════════════════════════════
#  M4 – Adaptive Fusion (Gating Network)
# ═══════════════════════════════════════════════════════════════════════════════

class M4AdaptiveFusion(nn.Module):
    """
    Adaptive fusion model from Section 2.5 of the project plan.

    Forward pass:
        y_img,  z_img = CNN(image)          # z_img: [B, 512]
        y_land        = MLP(landmarks)
        [α, β]        = softmax(gate(z_img))
        ŷ             = α·y_img + β·y_land

    Loss (computed externally in training loop):
        L = MSE(ŷ, y)  +  λ * (-α·log α - β·log β)   [entropy regularisation]

    The gating network takes the CNN feature vector so that α/β depend
    on the image content (per-face weights).
    """

    def __init__(self,
                 pretrained: bool = True,
                 landmark_dim: int = C.LANDMARK_DIM_3D,
                 landmark_hidden: list = None,
                 gate_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()

        if landmark_hidden is None:
            landmark_hidden = [256, 128]

        # ── Image branch ──────────────────────────────────────────────────────
        weights   = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone  = tv_models.resnet18(weights=weights)
        # Remove last FC; keep up to AdaptiveAvgPool → [B, 512, 1, 1]
        self.cnn  = nn.Sequential(*list(backbone.children())[:-1])
        self.img_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        # ── Landmark branch ──────────────────────────────────────────────────
        lm_layers = []
        prev = landmark_dim
        for h in landmark_hidden:
            lm_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        lm_layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.land_mlp = nn.Sequential(*lm_layers)

        # ── Gating network ────────────────────────────────────────────────────
        # Input: z_img [B, 512]  →  logits [B, 2]  →  softmax → (α, β)
        self.gate = nn.Sequential(
            nn.Linear(512, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 2),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, image, landmarks):
        """
        Args:
            image     : [B, 3, H, W]
            landmarks : [B, landmark_dim]
        Returns:
            y_fused : [B]        final prediction
            alpha   : [B]        image branch weight
            beta    : [B]        landmark branch weight
            y_img   : [B]        raw image branch prediction
            y_land  : [B]        raw landmark branch prediction
        """
        # Image branch
        feat   = self.cnn(image)               # [B, 512, 1, 1]
        feat   = feat.view(feat.size(0), -1)   # [B, 512]
        y_img  = self.img_head(
            torch.cat([feat], dim=1) if False else feat
        )  # kept clean
        # recompute through head properly
        y_img  = torch.sigmoid(
            self._img_linear(feat)
        )

        # Landmark branch
        y_land = self.land_mlp(landmarks)      # [B]

        # Gate
        logits      = self.gate(feat)              # [B, 2]
        weights     = torch.softmax(logits, dim=1) # [B, 2]
        alpha       = weights[:, 0]                # [B]
        beta        = weights[:, 1]                # [B]

        y_fused = alpha * y_img + beta * y_land    # [B]

        return y_fused, alpha, beta, y_img, y_land

    def _img_linear(self, feat):
        """Pass feat through the img_head linear+dropout layers."""
        x = feat
        for layer in self.img_head:
            if isinstance(layer, nn.Sigmoid):
                break
            x = layer(x)
        return x.squeeze(1)

    def get_feature(self, image):
        """Expose the CNN feature vector (used in analysis)."""
        with torch.no_grad():
            feat = self.cnn(image)
            return feat.view(feat.size(0), -1)


def _build_m4_clean(pretrained=True,
                    landmark_dim=C.LANDMARK_DIM_3D,
                    landmark_hidden=None,
                    gate_hidden=64,
                    dropout=0.3):
    """
    Cleaner re-implementation of M4 that avoids the forward() workaround.
    This version is what actually gets used.
    """
    if landmark_hidden is None:
        landmark_hidden = [256, 128]

    class _M4(nn.Module):
        def __init__(self):
            super().__init__()
            weights  = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # → [B,512,1,1]

            self.img_dropout = nn.Dropout(dropout)
            self.img_linear  = nn.Linear(512, 1)

            lm_layers = []
            prev = landmark_dim
            for h in landmark_hidden:
                lm_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                               nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            lm_layers += [nn.Linear(prev, 1)]
            self.land_mlp = nn.Sequential(*lm_layers)

            self.gate = nn.Sequential(
                nn.Linear(512, gate_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 2),
            )

        def forward(self, image, landmarks):
            # ── image branch ──
            feat   = self.cnn(image).view(image.size(0), -1)   # [B, 512]
            y_img  = torch.sigmoid(self.img_linear(
                self.img_dropout(feat)
            )).squeeze(1)                                        # [B]

            # ── landmark branch ──
            y_land = torch.sigmoid(
                self.land_mlp(landmarks)
            ).squeeze(1)                                         # [B]

            # ── gate ──
            weights = torch.softmax(self.gate(feat), dim=1)     # [B, 2]
            alpha   = weights[:, 0]
            beta    = weights[:, 1]

            y_fused = alpha * y_img + beta * y_land              # [B]
            return y_fused, alpha, beta, y_img, y_land

        def get_feature(self, image):
            with torch.no_grad():
                return self.cnn(image).view(image.size(0), -1)

    return _M4()


# Expose the clean version as M4AdaptiveFusion
M4AdaptiveFusion = _build_m4_clean


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility – print model parameter counts
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary():
    m1 = M1ImageCNN()
    m2 = LandmarkMLP(C.LANDMARK_DIM_3D)
    m4 = M4AdaptiveFusion()
    print(f"M1 (CNN)         params: {count_params(m1):>10,}")
    print(f"M2/M3 (MLP)      params: {count_params(m2):>10,}")
    print(f"M4 (Fusion)      params: {count_params(m4):>10,}")


if __name__ == "__main__":
    print_model_summary()
    # Quick forward pass test
    B = 4
    img  = torch.randn(B, 3, 224, 224)
    lm   = torch.randn(B, C.LANDMARK_DIM_3D)

    m1 = M1ImageCNN()
    out = m1(img)
    print(f"M1 output: {out.shape}  range [{out.min():.3f}, {out.max():.3f}]")

    m2 = LandmarkMLP(C.LANDMARK_DIM_3D)
    out = m2(lm)
    print(f"M2 output: {out.shape}  range [{out.min():.3f}, {out.max():.3f}]")

    m4 = M4AdaptiveFusion()
    y_fused, alpha, beta, y_img, y_land = m4(img, lm)
    print(f"M4 fused:  {y_fused.shape}")
    print(f"   alpha mean={alpha.mean():.3f}  beta mean={beta.mean():.3f}")
    assert torch.allclose(alpha + beta, torch.ones(B), atol=1e-5), "α+β ≠ 1 !"
    print("✓ α + β = 1 holds.")
