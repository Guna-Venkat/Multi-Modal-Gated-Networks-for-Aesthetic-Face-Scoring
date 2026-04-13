"""
kaggle_notebook.py
──────────────────
Self-contained Kaggle notebook (run as a Python script or copy cells into
a Kaggle Notebook).  Each ### CELL block maps to one notebook cell.

Before running:
  1. Add the SCUT-FBP5500 dataset to your Kaggle notebook.
  2. Set the DATASET_DIR path in Cell 0 (config) if needed.
  3. Run cells top-to-bottom.

Toggle USE_3D to include / exclude M3 (Procrustes 3D).
Toggle TRAIN_M3 to include / exclude M3 training.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 0 – Install dependencies                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip_install("mediapipe")
pip_install("openpyxl")   # for reading .xlsx

print("✓ Dependencies installed")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 1 – Configuration                                         ║
# ╚══════════════════════════════════════════════════════════════════╝
# ── User toggles ─────────────────────────────────────────────────────
ENV        = "kaggle"   # "kaggle" | "local"
USE_3D     = True       # extract 3D landmarks (needed for M4 with 3d mode)
TRAIN_M3   = False      # train M3 model (optional)
M4_LM_TYPE = "3d"      # "2d" or "3d"  — which landmarks M4 uses

# Override config.py's ENV at import time
import os
os.environ["BEAUTY_ENV"] = ENV   # read by config.py if you wire it up

# ── Now import config (edit config.py ENV line to match above) ────
import sys
sys.path.insert(0, "/kaggle/working")   # ensure our modules are found

import config as C
C.print_config()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 2 – Phase 1: Data preparation + landmark extraction       ║
# ╚══════════════════════════════════════════════════════════════════╝
from phase1_data_prep import run_phase1

phase1_result = run_phase1(use_3d=USE_3D, force_landmarks=False)

train_df     = phase1_result["train_df"]
test_df      = phase1_result["test_df"]
lm2d_norm    = phase1_result["lm2d_norm"]
lm3d_aligned = phase1_result["lm3d_aligned"]

print(f"\nTrain: {len(train_df)}  |  Test: {len(test_df)}")
print(f"Sample score range: {train_df['score'].min():.2f} – {train_df['score'].max():.2f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 3 – Phase 2: Train M1 (Full-image CNN)                   ║
# ╚══════════════════════════════════════════════════════════════════╝
from phase2_m1_cnn import run_m1

preds_m1, targets, metrics_m1 = run_m1(train_df, test_df)
print(f"\nM1  ρ={metrics_m1['pearson_r']:.4f}  MAE={metrics_m1['mae']:.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 4 – Phase 3: Train M2 (2D landmarks + MLP)               ║
# ╚══════════════════════════════════════════════════════════════════╝
from phase3_m2_landmarks import run_m2

preds_m2, _, metrics_m2 = run_m2(train_df, test_df, lm2d_norm=lm2d_norm)
print(f"\nM2  ρ={metrics_m2['pearson_r']:.4f}  MAE={metrics_m2['mae']:.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 5 – Phase 4: Train M3 (3D + Procrustes)  [OPTIONAL]      ║
# ╚══════════════════════════════════════════════════════════════════╝
metrics_m3 = None
preds_m3   = None

if TRAIN_M3 and lm3d_aligned is not None:
    from phase3_m2_landmarks import run_m3, compare_m2_m3
    preds_m3, _, metrics_m3 = run_m3(train_df, test_df, lm3d_aligned=lm3d_aligned)
    print(f"\nM3  ρ={metrics_m3['pearson_r']:.4f}  MAE={metrics_m3['mae']:.4f}")
    compare_m2_m3(metrics_m2, metrics_m3)
else:
    print("[Cell 5] Skipping M3 (TRAIN_M3=False or no 3D data).")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 6 – Phase 5: Train M4 (Adaptive Fusion)                  ║
# ╚══════════════════════════════════════════════════════════════════╝
from phase5_m4_fusion import run_m4, gating_analysis, print_gating_stats

# Pick landmark dict for M4
lm_for_m4 = lm3d_aligned if (M4_LM_TYPE == "3d" and lm3d_aligned) else lm2d_norm

preds_m4, _, alphas, betas, metrics_m4 = run_m4(
    train_df, test_df,
    landmarks_dict = lm_for_m4,
    lm_type        = M4_LM_TYPE,
)
print(f"\nM4  ρ={metrics_m4['pearson_r']:.4f}  MAE={metrics_m4['mae']:.4f}")

gate_df = gating_analysis(test_df, alphas, betas)
print_gating_stats(gate_df)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 7 – Phase 6: Evaluation & plots                          ║
# ╚══════════════════════════════════════════════════════════════════╝
from phase6_evaluation import run_evaluation

run_evaluation(use_m3=TRAIN_M3)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 8 – Display plots inline (Kaggle)                        ║
# ╚══════════════════════════════════════════════════════════════════╝
from IPython.display import Image as IPImage, display

plots = [
    ("Scatter Plots",      "scatter_plots.png"),
    ("Correlation Matrix", "corr_matrix.png"),
    ("Gate Histogram",     "gate_histogram.png"),
    ("Face Examples",      "face_examples.png"),
]

for title, fname in plots:
    fpath = os.path.join(C.RESULTS_DIR, fname)
    if os.path.exists(fpath):
        print(f"\n── {title} ──")
        display(IPImage(fpath))
    else:
        print(f"[skip] {fname} not found")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CELL 9 – Print final summary table                            ║
# ╚══════════════════════════════════════════════════════════════════╝
import pandas as pd
table_path = os.path.join(C.RESULTS_DIR, "comparison_table.csv")
if os.path.exists(table_path):
    print(pd.read_csv(table_path).to_string(index=False))
