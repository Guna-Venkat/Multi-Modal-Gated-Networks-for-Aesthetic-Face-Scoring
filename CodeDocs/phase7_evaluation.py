"""
phase6_evaluation.py
────────────────────
Phase 6 – Evaluation & Analysis  (Days 15–16)

Loads saved predictions for M1, M2, (M3), M4 and produces:
  1. Comparison table  (ρ, MAE, RMSE, Contribution score)
  2. Scatter plots     predicted vs. actual
  3. Gating histogram  (β distribution across test faces)
  4. Cross-model correlation matrix
  5. Error analysis    (where M4 wins / loses vs M1 and M2)
  6. Example face grid (high-β geometry-driven vs low-β texture-driven)

Usage:
    python phase6_evaluation.py
    python phase6_evaluation.py --no-m3   # skip M3 column
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Kaggle)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from PIL import Image

import config as C
from trainer import compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Load predictions from disk
# ═══════════════════════════════════════════════════════════════════════════════

def load_preds(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return None
    return np.load(path)


def load_all_preds(use_m3: bool = True):
    return {
        "M1": load_preds(C.M1_PREDS_PATH),
        "M2": load_preds(C.M2_PREDS_PATH),
        "M3": load_preds(C.M3_PREDS_PATH) if use_m3 else None,
        "M4": load_preds(C.M4_PREDS_PATH),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(preds_dict: dict, targets: np.ndarray) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per model:
        Model | Pearson ρ | MAE | RMSE | Contribution (ρ_m / ρ_M1)
    """
    rows = []
    rho_m1 = None

    for name, preds in preds_dict.items():
        if preds is None:
            continue
        n = min(len(preds), len(targets))
        m = compute_metrics(preds[:n], targets[:n])
        row = {
            "Model":     name,
            "Pearson ρ": round(m["pearson_r"], 4),
            "MAE":       round(m["mae"],       4),
            "RMSE":      round(m["rmse"],      4),
        }
        if name == "M1":
            rho_m1 = m["pearson_r"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if rho_m1:
        df["Contribution"] = (df["Pearson ρ"] / rho_m1).round(4)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Scatter plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scatter(preds_dict: dict, targets: np.ndarray, save_path: str):
    active = {k: v for k, v in preds_dict.items() if v is not None}
    n_cols = len(active)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    targets_raw = targets * 4.0 + 1.0
    for ax, (name, preds) in zip(axes, active.items()):
        n = min(len(preds), len(targets))
        p_raw = preds[:n] * 4.0 + 1.0
        t_raw = targets_raw[:n]
        rho   = pearsonr(p_raw, t_raw)[0]

        ax.scatter(t_raw, p_raw, alpha=0.4, s=8, color="#2563EB")
        ax.plot([1, 5], [1, 5], "r--", lw=1)
        ax.set_title(f"{name}  (ρ={rho:.3f})", fontsize=13)
        ax.set_xlabel("Ground Truth (1–5)")
        ax.set_ylabel("Predicted (1–5)")
        ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Scatter → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Gating histogram (β distribution)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gate_histogram(betas: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(betas, bins=40, color="#16A34A", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", lw=1.5, label="β=0.5 threshold")
    ax.set_xlabel("β  (geometry branch weight)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Per-Face Geometry Importance (β)", fontsize=13)
    ax.legend()

    geom_pct = (betas > 0.5).mean() * 100
    ax.text(0.97, 0.95, f"Geom-dominant: {geom_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Gate histogram → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Cross-model correlation matrix
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(preds_dict: dict, save_path: str):
    active = {k: v for k, v in preds_dict.items() if v is not None}
    names  = list(active.keys())
    preds  = [active[n] for n in names]

    # Align lengths
    min_len = min(len(p) for p in preds)
    preds   = [p[:min_len] for p in preds]

    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = pearsonr(preds[i], preds[j])[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)
    ax.set_title("Cross-Model Pearson Correlation", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot] Correlation matrix → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Error analysis  (where M4 outperforms M1 and M2)
# ═══════════════════════════════════════════════════════════════════════════════

def error_analysis(preds_dict: dict, targets: np.ndarray, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each test sample, compute |error| for each model.
    Return top-20 rows where M4 error is smallest relative to M1 and M2.
    """
    dfs = {}
    for name, preds in preds_dict.items():
        if preds is None:
            continue
        n = min(len(preds), len(targets))
        dfs[name] = np.abs(preds[:n] * 4.0 + 1.0 - (targets[:n] * 4.0 + 1.0))

    n = min(len(v) for v in dfs.values())
    df = test_df.iloc[:n].copy()
    for name, err in dfs.items():
        df[f"err_{name}"] = err[:n]

    if "err_M4" in df.columns and "err_M1" in df.columns:
        df["m4_gain_vs_m1"] = df["err_M1"] - df["err_M4"]
        df = df.sort_values("m4_gain_vs_m1", ascending=False)

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Example face grid  (geometry-driven vs texture-driven)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_example_faces(gate_df: pd.DataFrame, n: int = 6, save_path: str = None):
    """
    Show the `n` most geometry-driven and `n` most texture-driven faces.
    gate_df must have [filepath, beta, score].
    """
    # Sort by beta
    sorted_df = gate_df.sort_values("beta", ascending=False)
    geom_rows = sorted_df.head(n)
    text_rows = sorted_df.tail(n)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    fig.suptitle("Geometry-driven (top) vs Texture-driven (bottom)", fontsize=14)

    for col, (_, row) in enumerate(geom_rows.iterrows()):
        ax = axes[0][col]
        _show_face(ax, row, title=f"β={row['beta']:.2f}")

    for col, (_, row) in enumerate(text_rows.iterrows()):
        ax = axes[1][col]
        _show_face(ax, row, title=f"β={row['beta']:.2f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
        plt.close()
        print(f"  [Plot] Face examples → {save_path}")
    else:
        plt.show()


def _show_face(ax, row, title=""):
    fp = row.get("filepath", None)
    if fp and os.path.exists(str(fp)):
        img = Image.open(fp).resize((112, 112))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    score = row.get("score", "?")
    ax.set_title(f"{title}\nscore={score:.2f}", fontsize=9)
    ax.axis("off")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(use_m3: bool = True):
    print("\n" + "═"*55)
    print("  PHASE 6 – Evaluation & Analysis")
    print("═"*55)

    # Load predictions
    preds_dict = load_all_preds(use_m3)
    missing = [k for k, v in preds_dict.items() if v is None]
    if missing:
        print(f"  [Warn] Missing predictions for: {missing}")
    preds_dict = {k: v for k, v in preds_dict.items() if v is not None}

    if not preds_dict:
        print("  No predictions found. Run training phases first.")
        return

    # Load test targets
    test_csv = os.path.join(C.CACHE_DIR, "test_split.csv")
    test_df  = pd.read_csv(test_csv)
    targets  = ((test_df["score"].values - 1.0) / 4.0).astype(np.float32)

    # 1. Comparison table
    table = build_comparison_table(preds_dict, targets)
    print(f"\n  {'─'*55}")
    print(f"  Model Comparison Table")
    print(f"  {'─'*55}")
    print(table.to_string(index=False))
    table.to_csv(os.path.join(C.RESULTS_DIR, "comparison_table.csv"), index=False)

    # 2. Scatter plots
    plot_scatter(preds_dict, targets,
                 os.path.join(C.RESULTS_DIR, "scatter_plots.png"))

    # 3. Gating histogram (M4 only)
    gates_path = C.M4_GATES_PATH
    if os.path.exists(gates_path):
        gates  = np.load(gates_path)   # [N, 2]
        alphas = gates[:, 0]
        betas  = gates[:, 1]
        plot_gate_histogram(betas,
                            os.path.join(C.RESULTS_DIR, "gate_histogram.png"))

        # Save gate analysis CSV
        gate_df = test_df.iloc[:len(betas)].copy()
        gate_df["alpha"] = alphas
        gate_df["beta"]  = betas
        gate_df["geometry_dominant"] = betas > 0.5
        gate_df.to_csv(os.path.join(C.RESULTS_DIR, "gate_analysis.csv"), index=False)

        # 6. Example faces
        try:
            plot_example_faces(
                gate_df,
                n = 5,
                save_path = os.path.join(C.RESULTS_DIR, "face_examples.png")
            )
        except Exception as e:
            print(f"  [Warn] Face example plot failed: {e}")

    # 4. Cross-model correlation
    plot_correlation_matrix(preds_dict,
                            os.path.join(C.RESULTS_DIR, "corr_matrix.png"))

    # 5. Error analysis
    err_df = error_analysis(preds_dict, targets, test_df)
    err_path = os.path.join(C.RESULTS_DIR, "error_analysis.csv")
    err_df.to_csv(err_path, index=False)
    print(f"\n  Error analysis saved → {err_path}")
    print(f"  Top-10 samples where M4 beats M1:")
    cols = ["filename", "score"] + [f"err_{k}" for k in preds_dict]
    print(err_df[cols].head(10).to_string(index=False))

    print(f"\n  [Phase 6] ✓ All results saved to {C.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-m3", action="store_true")
    args = parser.parse_args()
    run_evaluation(use_m3=not args.no_m3)
