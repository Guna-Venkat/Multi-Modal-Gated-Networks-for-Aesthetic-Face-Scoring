import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import config as C

def compute_metrics(preds, targets):
    # Scale from 0-1 back to 1-5 for MAE/RMSE calculation
    preds_d   = preds   * 4.0 + 1.0
    targets_d = targets * 4.0 + 1.0
    
    r, _ = pearsonr(preds_d, targets_d)
    mae  = np.mean(np.abs(preds_d - targets_d))
    rmse = np.sqrt(np.mean((preds_d - targets_d) ** 2))
    return r, mae, rmse

"""
phase7_final_results_collector.py
─────────────────────────────────
Consolidates training logs and evaluation CSVs into a single master summary.
Used for generating the final comprehensive performance table in the report.
"""
def collect_all_metrics():
    # 1. Load Ground Truth
    test_csv = os.path.join(C.CACHE_DIR, "test_split.csv")
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return
    
    test_df = pd.read_csv(test_csv)
    targets = test_df["score"].values # This should be normalised 0-1 if phase1 did that, 
    # but phase1_data_prep actually saves the RAW score in the CSV and the DataLoader normalises it.
    # Let's check datasets.py to confirm.
    
    # Actually, phase1_data_prep.py:
    # row["score"] is raw 1-5.
    # In datasets.py: target = torch.tensor((row["score"] - 1.0) / 4.0, dtype=torch.float32)
    # So targets should be (targets - 1.0) / 4.0 for metric calculation if preds are 0-1.
    targets_norm = (targets - 1.0) / 4.0

    model_registry = {
        "M1 (ResNet-18)":        os.path.join(C.RESULTS_DIR, "m1_preds.npy"),
        "M2 (2D MLP)":           os.path.join(C.RESULTS_DIR, "m2_preds.npy"),
        "M3 (3D MLP)":           os.path.join(C.RESULTS_DIR, "m3_preds.npy"),
        "M4 (Adaptive Fusion)":  os.path.join(C.RESULTS_DIR, "m4_preds.npy"),
        "M5 (ViT Texture)":      os.path.join(C.CHECKPOINT_DIR, "m5_transformer_preds.npy"),
        "M6 (Landmark Trans)":   os.path.join(C.CHECKPOINT_DIR, "m6_transformer_preds.npy"),
        "M7 (Cross-Attention)":  os.path.join(C.CHECKPOINT_DIR, "m7_transformer_preds.npy"),
        "M8 (Gated ViT)":       os.path.join(C.CHECKPOINT_DIR, "m8_gated_vit_preds.npy"),
    }

    results = []
    for name, path in model_registry.items():
        if os.path.exists(path):
            preds = np.load(path)
            # Ensure shape matches
            if len(preds) != len(targets_norm):
                print(f"Warning: {name} prediction length ({len(preds)}) mismatch with test set ({len(targets_norm)}). Truncating.")
                preds = preds[:len(targets_norm)]
            
            r, mae, rmse = compute_metrics(preds, targets_norm)
            results.append({
                "Model": name,
                "Pearson ρ": round(r, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4)
            })
        else:
            print(f"Skipping {name}: {path} not found.")

    res_df = pd.DataFrame(results)
    output_path = os.path.join(C.RESULTS_DIR, "final_metrics_comparison.csv")
    res_df.to_csv(output_path, index=False)
    
    print("\n" + "═"*40)
    print("      FINAL METRICS COMPARISON")
    print("═"*40)
    print(res_df.to_markdown(index=False))
    print("═"*40)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    collect_all_metrics()
