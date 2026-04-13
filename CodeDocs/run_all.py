"""
run_all.py
──────────
Master orchestration script.  Runs all phases in order.

Usage:
    python run_all.py                   # full pipeline (M1 + M2 + M4)
    python run_all.py --include-m3      # also train M3
    python run_all.py --skip-phase1     # skip landmark extraction (use cache)
    python run_all.py --skip-training   # only run evaluation (use saved preds)
    python run_all.py --lm 2d           # M4 uses 2D landmarks instead of 3D

Environment:
    Set ENV = "kaggle" or "local" in config.py before running.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

import config as C


def main():
    parser = argparse.ArgumentParser(description="Facial Beauty Prediction – Full Pipeline")
    parser.add_argument("--include-m3",    action="store_true",
                        help="Also train M3 (3D Procrustes)")
    parser.add_argument("--skip-phase1",   action="store_true",
                        help="Skip Phase 1 (use cached landmarks)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip all training, go straight to evaluation")
    parser.add_argument("--lm",            choices=["2d", "3d"], default="3d",
                        help="Landmark type for M4 (default: 3d)")
    parser.add_argument("--force-lm",      action="store_true",
                        help="Force re-extraction of landmarks")
    # Per-model epoch overrides
    parser.add_argument("--epochs-m1",  type=int, default=C.M1_EPOCHS)
    parser.add_argument("--epochs-m2",  type=int, default=C.M2_EPOCHS)
    parser.add_argument("--epochs-m3",  type=int, default=C.M3_EPOCHS)
    parser.add_argument("--epochs-m4",  type=int, default=C.M4_EPOCHS)
    args = parser.parse_args()

    t_start = time.time()
    C.print_config()

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 – Data Preparation & Landmark Extraction
    # ══════════════════════════════════════════════════════════════════════════
    train_df = test_df = None
    lm2d_norm = lm3d_aligned = None

    if args.skip_phase1 or args.skip_training:
        print("\n[run_all] Loading cached splits …")
        train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
        test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
        if not os.path.exists(train_csv):
            sys.exit("ERROR: No cached splits found. Run without --skip-phase1 first.")
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
    else:
        from phase1_data_prep import run_phase1
        result = run_phase1(
            use_3d          = args.include_m3 or (args.lm == "3d"),
            force_landmarks = args.force_lm,
        )
        train_df    = result["train_df"]
        test_df     = result["test_df"]
        lm2d_norm   = result["lm2d_norm"]
        lm3d_aligned = result["lm3d_aligned"]

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASES 2-5 – Training
    # ══════════════════════════════════════════════════════════════════════════
    all_metrics = {}

    if not args.skip_training:

        # ── Phase 2: M1 ──────────────────────────────────────────────────────
        print("\n" + "─"*55)
        from phase2_m1_cnn import run_m1
        _, targets, metrics_m1 = run_m1(
            train_df, test_df, epochs=args.epochs_m1
        )
        all_metrics["M1"] = metrics_m1

        # ── Phase 3: M2 ──────────────────────────────────────────────────────
        print("\n" + "─"*55)
        from phase3_m2_landmarks import run_m2
        _, _, metrics_m2 = run_m2(
            train_df, test_df,
            lm2d_norm = lm2d_norm,
            epochs    = args.epochs_m2,
        )
        all_metrics["M2"] = metrics_m2

        # ── Phase 4: M3 (optional) ───────────────────────────────────────────
        if args.include_m3:
            print("\n" + "─"*55)
            from phase3_m2_landmarks import run_m3
            _, _, metrics_m3 = run_m3(
                train_df, test_df,
                lm3d_aligned = lm3d_aligned,
                epochs       = args.epochs_m3,
            )
            all_metrics["M3"] = metrics_m3

        # ── Phase 5: M4 ──────────────────────────────────────────────────────
        print("\n" + "─"*55)
        from phase5_m4_fusion import run_m4
        _, _, alphas, betas, metrics_m4 = run_m4(
            train_df, test_df,
            lm_type = args.lm,
            epochs  = args.epochs_m4,
        )
        all_metrics["M4"] = metrics_m4

        # ── Quick summary ─────────────────────────────────────────────────────
        print("\n" + "═"*55)
        print("  QUICK SUMMARY")
        print("═"*55)
        rho_m1 = all_metrics.get("M1", {}).get("pearson_r", None)
        for name, m in all_metrics.items():
            contrib = f"  (contrib={m['pearson_r']/rho_m1:.3f})" if rho_m1 else ""
            print(f"  {name:4s}  ρ={m['pearson_r']:.4f}  "
                  f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}{contrib}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 6 – Evaluation & Visualisation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*55)
    from phase6_evaluation import run_evaluation
    run_evaluation(use_m3=args.include_m3)

    elapsed = time.time() - t_start
    print(f"\n[run_all] ✓ Pipeline complete in {elapsed/60:.1f} min")
    print(f"  Results directory: {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
