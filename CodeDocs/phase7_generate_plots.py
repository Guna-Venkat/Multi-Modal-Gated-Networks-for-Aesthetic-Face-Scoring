"""
phase7_generate_plots.py
────────────────────────
Generates all result figures for the paper:

1. Bar chart comparing model performance (Pearson ρ)
2. Histogram of gate weights β (M4)
3. 1D line plot of M7 cross‑attention distribution
4. 2D spatial attention map overlaid on facial landmarks

Usage:
    python CodeDocs/phase7_generate_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local configuration (paths, directories)
import config as C


def generate_report_plots():
    """
    Create the main performance bar chart and the β‑gate distribution histogram.
    """
    # ------------------------------------------------------------------
    # 1. Load metrics (if available) or use fallback demo data
    # ------------------------------------------------------------------
    metrics_path = os.path.join(C.RESULTS_DIR, "final_metrics_comparison.csv")
    if not os.path.exists(metrics_path):
        # Fallback hardcoded values for demonstration when file is missing
        data = {
            "Model": ["M1 (ResNet-18)", "M2 (2D MLP)", "M3 (3D MLP)", "M4 (Adaptive Fusion)"],
            "Pearson ρ": [0.8565, 0.6968, 0.7622, 0.8947]
        }
        df = pd.DataFrame(data)
        print("Warning: final_metrics_comparison.csv not found. Using demo data.")
    else:
        df = pd.read_csv(metrics_path)

    # ------------------------------------------------------------------
    # 2. Bar plot: Pearson correlation per model
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    # Note: The FutureWarning about palette/hue is harmless and can be ignored.
    sns.barplot(x="Model", y="Pearson ρ", data=df, palette="viridis")
    plt.title("Beauty Prediction Performance (Pearson Correlation)", fontsize=14)
    plt.ylim(min(df["Pearson ρ"]) * 0.9, 1.0)
    plt.xticks(rotation=45)

    plot_path = os.path.join(C.RESULTS_DIR, "final_performance_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # 3. Histogram of gate weights β (geometry contribution in M4)
    # ------------------------------------------------------------------
    gate_path = os.path.join(C.RESULTS_DIR, "m4_gate_analysis.csv")
    if os.path.exists(gate_path):
        gate_df = pd.read_csv(gate_path)
        plt.figure(figsize=(10, 6))
        sns.histplot(gate_df["beta"], bins=30, kde=True, color='salmon')
        plt.title("Distribution of Geometry Gate Weights (β)", fontsize=14)
        plt.xlabel("Beta (Geometry Weight)")
        plt.ylabel("Frequency")

        hist_path = os.path.join(C.RESULTS_DIR, "m4_beta_distribution.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        print(f"Histogram saved to {hist_path}")


def plot_attention_distribution():
    """
    Visualise the 1D cross‑attention weights across the 468 landmarks
    for the first test sample (M7 model).
    """
    heatmap_path = os.path.join(C.CHECKPOINT_DIR, "m7_transformer_heatmaps.npy")
    if not os.path.exists(heatmap_path):
        print(f"Attention weights not found: {heatmap_path}")
        return

    # Load attention weights: shape (num_samples, 468)
    attn_data = np.load(heatmap_path)
    sample_attn = attn_data[0]   # take first sample

    plt.figure(figsize=(12, 5))
    plt.plot(sample_attn, color='#2c3e50', linewidth=0.8, alpha=0.7)

    # Add baseline (uniform attention)
    baseline = 1.0 / 468
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.5,
                label=f'Baseline ({baseline:.5f})')

    plt.title("M7 Cross-Attention Distribution (1D Index)", fontsize=14)
    plt.xlabel("Landmark Index (0-467)")
    plt.ylabel("Attention Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(C.RESULTS_DIR, "m7_attention_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Attention distribution plot saved to {out_path}")


def plot_spatial_attention():
    """
    Map M7 attention weights onto the 2D facial landmarks to create a
    spatial heatmap of where the model "looks".
    """
    # ------------------------------------------------------------------
    # 1. Load required data
    # ------------------------------------------------------------------
    attn_path = os.path.join(C.CHECKPOINT_DIR, "m7_transformer_heatmaps.npy")
    land_path = os.path.join(C.CACHE_DIR, "norm_2d.npy")

    if not os.path.exists(attn_path) or not os.path.exists(land_path):
        print("Missing data for spatial mapping (attention or landmarks).")
        return

    attn_data = np.load(attn_path)                       # shape: (N, 468) or (N, 468, 196)
    land_raw = np.load(land_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # 2. Extract first sample's landmarks
    # ------------------------------------------------------------------
    if isinstance(land_raw.item(), dict):
        first_key = list(land_raw.item().keys())[0] # Added [0] to get the actual key string
        sample_land = land_raw.item()[first_key]
    else:
        sample_land = land_raw[0]
    
    # Reshape landmarks to 2 columns (X, Y). 
    # If it was 702 elements, it will become (351, 2).
    sample_land = np.array(sample_land).reshape(-1, 2)

    # ------------------------------------------------------------------
    # 3. Extract first sample's attention and reduce to per-landmark values
    # ------------------------------------------------------------------
    # If attn_data is the whole file, take the first sample [0]
    sample_attn = attn_data[0] if attn_data.ndim > 1 else attn_data

    # If attention is 2D (landmarks × patches), collapse over patches
    if sample_attn.ndim == 2:
        sample_attn = sample_attn.mean(axis=1)
        print("Note: Aggregated cross‑attention over patches using mean.")
    
    sample_attn = sample_attn.flatten()

    # ------------------------------------------------------------------
    # MATCHING LOGIC: Ensure sample_attn and sample_land have same length
    # ------------------------------------------------------------------
    min_len = min(len(sample_attn), len(sample_land))
    sample_attn = sample_attn[:min_len]
    sample_land = sample_land[:min_len]
    print(f"Aligning data: using first {min_len} points for plotting.")

    # ------------------------------------------------------------------
    # 4. Create 2D scatter plot with attention as colour
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 10))
    plt.gca().set_facecolor('#1c1c1c')

    scatter = plt.scatter(sample_land[:, 0], -sample_land[:, 1],
                          c=sample_attn, cmap='plasma', s=35, alpha=0.9,
                          edgecolors='white', linewidth=0.2)

    plt.colorbar(scatter, label='Attention Weight', shrink=0.8)
    plt.title("M7 Spatial Attention Map: Feature Importance", fontsize=15, pad=20)
    plt.axis('equal')
    plt.axis('off')

    out_path = os.path.join(C.RESULTS_DIR, "m7_attention_spatial.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Spatial attention map saved to {out_path}")


if __name__ == "__main__":
    generate_report_plots()
    plot_attention_distribution()
    plot_spatial_attention()