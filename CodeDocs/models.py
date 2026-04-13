"""
models.py
─────────
Core neural architectures for Facial Beauty Prediction.

This module implements the four primary models described in our methodology:
1. M1: Appearance-based model (ResNet-18) focusing on texture and skin quality.
2. M2: Geometric model using 2D landmarks with anchor-based normalization.
3. M3: Geometric model using 3D landmarks with Procrustes alignment.
4. M4: Adaptive Gated Fusion combining M1 and M2/M3 branches with a content-dependent gating network.

All models perform regression to predict a beauty score normalized in [0, 1].
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config as C


# ═══════════════════════════════════════════════════════════════════════════════
#  M1 – Full-image CNN (Texture-dominant Branch)
# ═══════════════════════════════════════════════════════════════════════════════

class M1ImageCNN(nn.Module):
    """
    Fine-tuned ResNet-18 for beauty regression.
    
    Architecture:
    - Pretrained ResNet-18 backbone (ImageNet-1K weights).
    - Global Average Pooling (GAP) reduces spatial dimensions to [B, 512, 1, 1].
    - Regression head (Flatten -> Dropout -> Linear -> Sigmoid).
    
    Rationale:
    ResNet-18 is used as a robust feature extractor for 'appearance' features such as 
    skin texture, color symmetry, and global facial lighting.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Use V1 weights for the standard pretrained ResNet
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        # Feature extractor handles feature extraction but discards the original 1000-class FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),  # Squash output to [0, 1] range
        )

    def forward(self, x):
        """
        Input: Tensor of shape [B, 3, 224, 224]
        Output: Tensor of shape [B]
        """
        feat = self.features(x)
        return self.head(feat).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  M2 / M3 – Landmark MLP (Geometric-dominant Branch)
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkMLP(nn.Module):
    """
    Deep MLP architecture for analyzing facial proportions and landmark geometry.
    
    Used for both M2 (2D landmarks) and M3 (3D procrustes-aligned landmarks).
    Inputs are flattened vectors of shape [B, 1404].
    """
    def __init__(self,
                 input_dim: int = C.LANDMARK_DIM_3D,
                 hidden_dims: list = None,
                 dropout: float = 0.3):
        """
        Args:
            input_dim (int): Typically 1404 (468 landmarks * 3 coords).
            hidden_dims (list): Neurons per layer.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev = input_dim
        # Construct layers dynamically based on hidden_dims
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h), 
                nn.BatchNorm1d(h), 
                nn.ReLU(), 
                nn.Dropout(dropout)
            ]
            prev = h
            
        # Final regression layer
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: [B, input_dim]
        Output: [B]
        """
        return self.net(x).squeeze(1)

    def forward_feat(self, x):
        """
        Return the penultimate hidden layer representation.
        Useful for visualization or secondary analysis.
        """
        for layer in self.net[:-2]:
            x = layer(x)
        return x


# Model aliases for clarity in config and trainer logic
M2LandmarkMLP = LandmarkMLP
M3LandmarkMLP = LandmarkMLP


# ═══════════════════════════════════════════════════════════════════════════════
#  M4 – Adaptive Gated Fusion (Multimodal Branch)
# ═══════════════════════════════════════════════════════════════════════════════

def _M4Factory(pretrained=True,
               landmark_dim=C.LANDMARK_DIM_3D,
               landmark_hidden=None,
               gate_hidden=64,
               dropout=0.3):
    """
    Factory function for constructing the M4 Adaptive Fusion architecture.
    
    Mathematical Formulation:
        z_img = CNN(image)
        y_img = ImageBranch(z_img)
        y_geo = LandmarkBranch(landmarks)
        [alpha, beta] = Softmax(Gate(z_img)) 
        y_fused = (alpha * y_img) + (beta * y_geo)
        
    where alpha + beta = 1.
    """
    if landmark_hidden is None:
        landmark_hidden = [256, 128]

    class AdaptiveFusionModel(nn.Module):
        """
        Implementation of the Gated Multimodal fusion.
        Distinctively allows the model to shift trust between texture (CNN) 
        and geometry (MLP) on a per-image basis.
        """
        def __init__(self):
            super().__init__()
            # 1. Texture Branch (CNN)
            weights  = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.resnet18(weights=weights)
            self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # Through GAP
            
            self.img_linear = nn.Linear(512, 1)
            self.img_dropout = nn.Dropout(dropout)

            # 2. Geometry Branch (MLP)
            lm_layers = []
            prev = landmark_dim
            for h in landmark_hidden:
                lm_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), 
                               nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            lm_layers += [nn.Linear(prev, 1)]
            self.land_mlp = nn.Sequential(*lm_layers)

            # 3. Gating Head
            # Takes visual features to decide relative weights (alpha, beta)
            self.gate = nn.Sequential(
                nn.Linear(512, gate_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, 2), # 2 logits for alpha & beta
            )

        def forward(self, image, landmarks):
            """
            Args:
                image (batch): [B, 3, 224, 224]
                landmarks (batch): [B, 1404]
                
            Returns:
                y_fused, alpha, beta, y_img, y_land
            """
            # Extract image features
            feat = self.cnn(image).view(image.size(0), -1)  # [B, 512]
            
            # Prediction from individual branches
            # Note: We apply sigmoid individually inside forward to keep branches interpretable
            y_img = torch.sigmoid(self.img_linear(self.img_dropout(feat))).squeeze(1)
            y_land = torch.sigmoid(self.land_mlp(landmarks)).squeeze(1)

            # Compute gating weights
            gate_logits = self.gate(feat)
            gate_weights = torch.softmax(gate_logits, dim=1)
            alpha = gate_weights[:, 0]  # Bias towards texture
            beta = gate_weights[:, 1]   # Bias towards geometry

            # Perform the Adaptive Fusion
            y_fused = (alpha * y_img) + (beta * y_land)
            return y_fused, alpha, beta, y_img, y_land

        def get_feature(self, image):
            """Returns the CNN feature representation for visualization tasks."""
            with torch.no_grad():
                return self.cnn(image).view(image.size(0), -1)

    return AdaptiveFusionModel()


# Expose M4 under a descriptive name
M4AdaptiveFusion = _M4Factory


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility & Model Summaries
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary():
    """Prints a comparison of model sizes (M1, M2, M4)."""
    m1 = M1ImageCNN()
    m2 = LandmarkMLP(C.LANDMARK_DIM_3D)
    m4 = M4AdaptiveFusion()
    
    print("-" * 40)
    print(f"{'Model Architecture':<20} | {'Parameters':<15}")
    print("-" * 40)
    print(f"{'M1 (Texture CNN)':<20} | {count_params(m1):>15,}")
    print(f"{'M2/M3 (Geo MLP)':<20} | {count_params(m2):>15,}")
    print(f"{'M4 (Gated Fusion)':<20} | {count_params(m4):>15,}")
    print("-" * 40)


if __name__ == "__main__":
    # Test block to verify architecture construction
    print_model_summary()
    
    # Verify forward passes with dummy tensors
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    dummy_land = torch.randn(batch_size, C.LANDMARK_DIM_3D)

    print("\n[M1] testing forward...")
    m1 = M1ImageCNN()
    out1 = m1(dummy_img)
    assert out1.shape == (batch_size,)

    print("[M4] testing fusion logic...")
    m4 = M4AdaptiveFusion()
    y_fused, alpha, beta, y_img, y_land = m4(dummy_img, dummy_land)
    
    # Verification of gating constraint: alpha + beta must always equal 1.0
    gate_sum = (alpha + beta)
    assert torch.allclose(gate_sum, torch.ones(batch_size), atol=1e-5)
    print("✓ Architecture verification successful: alpha + beta = 1.0")
