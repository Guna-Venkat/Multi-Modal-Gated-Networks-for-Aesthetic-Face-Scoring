"""
run_all.py
──────────
Master Orchestration Script for the Facial Beauty Prediction Pipeline.

This script coordinates the end-to-end execution of the project, including:
- Phase 1: Data cleaning and MediaPipe landmark extraction.
- Phase 2-5: Training of M1 (CNN), M2/M3 (MLP), and M4 (Fusion).
- Phase 6: Running transformer-based experiments.
- Phase 7: Comprehensive evaluation and result visualization.

Execution Modes:
1. Standard Run: Trains M1, M2, and M4.
2. Full Run: Trains all models including M3 (Procrustes).
3. Eval-Only: Skips training and runs evaluation using existing checkpoints.

Usage examples:
    python CodeDocs/run_all.py                          # Basic run
    python CodeDocs/run_all.py --include-m3             # Full geometric analysis
    python CodeDocs/run_all.py --skip-training          # Quick evaluation
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

import config as C


def main():
    """
    Main entry point for orchestrating the multi-phase experiment.
    Parses command-line arguments to toggle specific phases and models.
    """
    parser = argparse.ArgumentParser(description="End-to-End Facial Beauty Pipeline")
    
    # Execution Toggles
    parser.add_argument("--include-m3",    action="store_true",
                        help="Train M3 (3D Procrustes-aligned MLP)")
    parser.add_argument("--skip-phase1",   action="store_true",
                        help="Skip landmark extraction (uses cached .npy files)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training phase; immediately run evaluation")
    parser.add_argument("--lm",            choices=["2d", "3d"], default="3d",
                        help="Landmark type for M4 Adaptive Fusion (default: 3d)")
    parser.add_argument("--force-lm",      action="store_true",
                        help="Force re-extraction of landmarks even if cache exists")
    
    # Custom Hyperparameter Overrides
    parser.add_argument("--epochs-m1",  type=int, default=C.M1_EPOCHS)
    parser.add_argument("--epochs-m2",  type=int, default=C.M2_EPOCHS)
    parser.add_argument("--epochs-m3",  type=int, default=C.M3_EPOCHS)
    parser.add_argument("--epochs-m4",  type=int, default=C.M4_EPOCHS)
    
    args = parser.parse_args()

    # Track overall execution time
    t_start = time.time()
    C.print_config()

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1 – DATA PREPARATION (Landmark Extraction)
    # ══════════════════════════════════════════════════════════════════════════
    # We maintain references to landmarks and dataframe splits across phases
    train_df = test_df = None
    lm2d_norm = lm3d_aligned = None

    if args.skip_phase1 or args.skip_training:
        print("\n[run_all] Skipping Phase 1: Loading cached splits...")
        train_csv = os.path.join(C.CACHE_DIR, "train_split.csv")
        test_csv  = os.path.join(C.CACHE_DIR, "test_split.csv")
        
        if not os.path.exists(train_csv):
            sys.exit("ERROR: No cached splits found. Run without --skip-phase1 first.")
            
        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
    else:
        # Import run_phase1 here to avoid overhead if skipped
        from phase1_data_prep import run_phase1
        result = run_phase1(
            use_3d          = args.include_m3 or (args.lm == "3d"),
            force_landmarks = args.force_lm,
        )
        train_df      = result["train_df"]
        test_df       = result["test_df"]
        lm2d_norm     = result["lm2d_norm"]
        lm3d_aligned  = result["lm3d_aligned"]

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASES 2-5 – TRAINING (ML Models)
    # ══════════════════════════════════════════════════════════════════════════
    all_metrics = {}

    if not args.skip_training:

        # ── Phase 2: M1 - Image Branch ──
        print("\n" + "="*60)
        print(" PHASE 2: TRAINING M1 (TEXTURE CNN)")
        print("="*60)
        from phase2_m1_cnn import run_m1
        _, targets, metrics_m1 = run_m1(
            train_df, test_df, epochs=args.epochs_m1
        )
        all_metrics["M1"] = metrics_m1

        # ── Phase 3: M2 - Geometry Branch (2D) ──
        print("\n" + "="*60)
        print(" PHASE 3: TRAINING M2 (2D GEOMETRY MLP)")
        print("="*60)
        from phase3_m2_landmarks import run_m2
        _, _, metrics_m2 = run_m2(
            train_df, test_df,
            lm2d_norm = lm2d_norm,
            epochs    = args.epochs_m2,
        )
        all_metrics["M2"] = metrics_m2

        # ── Phase 4: M3 - Geometry Branch (3D Aligned) ──
        if args.include_m3:
            print("\n" + "="*60)
            print(" PHASE 4: TRAINING M3 (3D PROCRUSTES MLP)")
            print("="*60)
            from phase3_m2_landmarks import run_m3
            _, _, metrics_m3 = run_m3(
                train_df, test_df,
                lm3d_aligned = lm3d_aligned,
                epochs       = args.epochs_m3,
            )
            all_metrics["M3"] = metrics_m3

        # ── Phase 5: M4 - Gated Fusion (Image + Landmarks) ──
        print("\n" + "="*60)
        print(" PHASE 5: TRAINING M4 (ADAPTIVE GATED FUSION)")
        print("="*60)
        from phase5_m4_fusion import run_m4
        _, _, alphas, betas, metrics_m4 = run_m4(
            train_df, test_df,
            lm_type = args.lm,
            epochs  = args.epochs_m4,
        )
        all_metrics["M4"] = metrics_m4

        # ── Print Intermediary Summary ──
        print("\n" + "═"*60)
        print(f"{'Experiment Name':<20} | {'Pearson ρ':<10} | {'MAE':<8}")
        print("─"*60)
        for name, m in all_metrics.items():
            print(f"{name:<20} | {m['pearson_r']:>10.4f} | {m['mae']:>8.4f}")
        print("═"*60)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 6 & 7 – EVALUATION & VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print(" FINAL EVALUATION & PLOT GENERATION")
    print("="*60)
    from phase7_evaluation import run_evaluation
    run_evaluation(use_m3=args.include_m3)

    elapsed_min = (time.time() - t_start) / 60
    print(f"\n[run_all] \u2713 Full pipeline completed in {elapsed_min:.1f} minutes.")
    print(f"       Results saved to: {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
"] = metrics_m4

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
