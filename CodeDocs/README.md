# Facial Beauty Prediction — CV Project
### Geometry vs. Texture Disentanglement

---

## Project Structure

```
facial_beauty/
├── config.py                 ← ★ SET ENV HERE (kaggle / local)
├── requirements.txt
│
├── phase1_data_prep.py       ← Data loading, landmark extraction, normalisation
├── phase2_m1_cnn.py          ← M1: Full-image ResNet-18 CNN
├── phase3_m2_landmarks.py    ← M2: 2D anchor-norm MLP  |  M3: 3D Procrustes MLP
├── phase5_m4_fusion.py       ← M4: Adaptive Fusion (gating network)  ← KEY NOVELTY
├── phase6_evaluation.py      ← Comparison table, plots, gating analysis
├── run_all.py                ← Master script (runs everything)
│
├── datasets.py               ← PyTorch Dataset / DataLoader classes
├── models.py                 ← All model architectures
├── trainer.py                ← Training loop, early stopping, metrics
│
└── kaggle_notebook.py        ← Cell-by-cell Kaggle notebook version
```

---

## Quick Start

### 1. Set your environment

Open `config.py` and set:
```python
ENV = "kaggle"   # or "local"
```

### 2. Install dependencies (local)
```bash
pip install -r requirements.txt
```
For Kaggle: dependencies are installed in Cell 0 of `kaggle_notebook.py`.

### 3a. Run everything (recommended)
```bash
# M1 + M2 + M4  (M3 optional, add --include-m3)
python run_all.py

# Include M3 (3D Procrustes)
python run_all.py --include-m3

# Use 2D landmarks for M4 instead of 3D
python run_all.py --lm 2d

# Skip landmark re-extraction (use cache)
python run_all.py --skip-phase1

# Only evaluate (use saved predictions)
python run_all.py --skip-training

# Custom epochs
python run_all.py --epochs-m1 50 --epochs-m4 60
```

### 3b. Run phases individually
```bash
python phase1_data_prep.py                  # extract + cache landmarks
python phase2_m1_cnn.py --epochs 40        # train M1
python phase3_m2_landmarks.py              # train M2
python phase3_m2_landmarks.py --m3         # train M2 + M3
python phase5_m4_fusion.py --lm 3d         # train M4
python phase6_evaluation.py                # generate all plots + table
```

### 3c. Kaggle
Copy `kaggle_notebook.py` cells into a Kaggle Notebook, or upload all
`.py` files as a dataset and run `kaggle_notebook.py` as a script.

---

## Models

| ID | Name                    | Input               | Invariance                     |
|----|-------------------------|---------------------|--------------------------------|
| M1 | Full-image CNN          | RGB 224×224         | —                              |
| M2 | 2D landmarks + anchor   | 468 pts, [1404]     | scale, translation             |
| M3 | 3D landmarks + Procrustes | 468 pts, [1404]   | scale, translation, rotation   |
| M4 | Adaptive Fusion (gating)| image + landmarks   | same as M3 + per-face weights  |

---

## M4 Gating (key novelty)

```
ŷ = α(I) · y_img  +  β(I) · y_land        α + β = 1
```
- α = image (texture) weight  
- β = geometry weight  
- The gating network takes the CNN feature vector → both weights depend on image content  
- Entropy regularisation prevents collapse to one branch:  
  `L = MSE(ŷ, y)  +  λ · (−α·logα − β·logβ)`

---

## Dataset – SCUT-FBP5500

Download from: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release  
Place at `DATASET_DIR` as set in `config.py`.

Expected layout:
```
SCUT-FBP5500/
  Images/           ← JPEG face images
  All_Ratings.xlsx  ← per-rater scores (1-5)
  train.txt         ← official split (optional)
  test.txt
```

---

## Outputs

All saved to `results/`:
- `comparison_table.csv`   — ρ, MAE, RMSE, contribution score per model
- `scatter_plots.png`      — predicted vs actual for each model
- `corr_matrix.png`        — cross-model Pearson correlation
- `gate_histogram.png`     — β distribution across test faces
- `face_examples.png`      — geometry-driven vs texture-driven faces
- `gate_analysis.csv`      — per-face α, β, geometry_dominant flag
- `error_analysis.csv`     — where M4 outperforms M1 / M2

Checkpoints saved to `checkpoints/`.

---

## Notes & Assumptions

- Images are resized to 224×224 for both CNN and landmark extraction
- Scores normalised to [0,1] for training; de-normalised to [1,5] for metrics
- MediaPipe Face Mesh gives 468 landmarks in normalised image coordinates
- M2 and M3 both produce [1404]-dim vectors (different normalisation strategies)
- M4 can use either 2D or 3D landmarks for the landmark branch (set `--lm`)
- Early stopping patience = 5 epochs (configurable in `config.py`)
- All randomness seeded via `RANDOM_SEED = 42` in `config.py`
