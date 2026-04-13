"""
phase1_data_prep.py
───────────────────
Phase 1 – Setup & Data Preparation  (Days 1–3)

Steps:
  1. Load image paths + mean beauty scores from SCUT-FBP5500
  2. Create train/test split (80/20, no identity overlap)
  3. Extract 2D & 3D landmarks via MediaPipe Face Mesh  (cached as .npy)
  4. Anchor-based 2D normalisation  (M2)
  5. Procrustes alignment of 3D landmarks  (M3)  [optional]

Run:
    python phase1_data_prep.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

import config as C

# ─── MediaPipe ───────────────────────────────────────────────────────────────
try:
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
    
    _TASK_PATH = os.path.join(C.CACHE_DIR, "face_landmarker.task")
    if not os.path.exists(_TASK_PATH):
        print(f"[MediaPipe] Downloading face_landmarker.task to {_TASK_PATH} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            _TASK_PATH
        )
except ImportError:
    _MP_AVAILABLE = False
    print("[WARN] mediapipe not installed – landmark extraction will be skipped.")
    print("       Install: pip install mediapipe")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. LOAD DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def load_scut_fbp5500():
    """
    Returns a DataFrame with columns: [filename, score]
    score  = mean beauty rating (1-5 scale).

    SCUT-FBP5500 provides an Excel file with per-rater ratings.
    We average across raters per image.
    Falls back to a simple CSV if the xlsx is missing (useful for testing).
    """
    label_path = Path(C.LABEL_CSV)
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}\n"
            f"Download SCUT-FBP5500 and set DATASET_DIR in config.py."
        )

    if label_path.suffix == ".txt":
        df = pd.read_csv(label_path, sep=r'\s+', header=None, names=["filename", "score"])
    elif label_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(label_path)
    else:
        df = pd.read_csv(label_path)

    # Expected columns vary by version; try common layouts
    # Layout A:  [Filename, R1, R2, ..., R60]
    # Layout B:  [filename, score]
    if "Filename" in df.columns:
        df = df.rename(columns={"Filename": "filename"})

    score_cols = [c for c in df.columns if c not in ["filename", "Filename"]]
    if len(score_cols) > 1:
        df["score"] = df[score_cols].mean(axis=1)
    elif "score" not in df.columns:
        df["score"] = df[score_cols[0]]

    df = df[["filename", "score"]].copy()
    df["filepath"] = df["filename"].apply(
        lambda fn: os.path.join(C.IMAGE_DIR, fn)
    )

    # Keep only images that actually exist
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[Data] Loaded {len(df)} images with scores.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  2. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def make_split(df):
    """
    Use official split files if available, otherwise random 80/20.
    Returns (train_df, test_df).
    """
    train_path = Path(C.TRAIN_SPLIT_CSV)
    test_path  = Path(C.TEST_SPLIT_CSV)

    if train_path.exists() and test_path.exists():
        print("[Split] Using official split files.")
        train_names = set(train_path.read_text().strip().splitlines())
        test_names  = set(test_path.read_text().strip().splitlines())
        train_df = df[df["filename"].isin(train_names)].reset_index(drop=True)
        test_df  = df[df["filename"].isin(test_names)].reset_index(drop=True)
    else:
        print("[Split] Official split not found – random 80/20 split.")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=1 - C.TRAIN_RATIO,
            random_state=C.RANDOM_SEED
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

    print(f"[Split] Train: {len(train_df)}  |  Test: {len(test_df)}")
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
#  3. LANDMARK EXTRACTION  (MediaPipe Face Mesh)
# ═══════════════════════════════════════════════════════════════════════════════

_LANDMARKER = None

def get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        base_options = mp_python.BaseOptions(model_asset_path=_TASK_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER

def close_landmarker():
    global _LANDMARKER
    if _LANDMARKER is not None:
        _LANDMARKER.close()
        _LANDMARKER = None


def extract_landmarks(filepath: str):
    """
    Runs MediaPipe Face Mesh on one image.
    Returns:
        lm2d : np.ndarray [468, 2]  (x, y)  normalised to [0,1] by image dims
        lm3d : np.ndarray [468, 3]  (x, y, z)
        None, None if no face detected.
    """
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detector = get_landmarker()
    results = detector.detect(mp_image)

    if not results.face_landmarks:
        return None, None

    lms = results.face_landmarks[0][:468]  # MediaPipe returns 478, we need 468
    lm2d = np.array([[lm.x, lm.y]         for lm in lms], dtype=np.float32)
    lm3d = np.array([[lm.x, lm.y, lm.z]   for lm in lms], dtype=np.float32)
    return lm2d, lm3d


def extract_all_landmarks(df: pd.DataFrame, force: bool = False):
    """
    Extract and cache 2D & 3D landmarks for every image in df.
    Returns two dicts: {filename: array}
    Skips images where no face is detected (logged in failed_list).
    """
    cache_2d = {}
    cache_3d = {}
    failed   = []

    cache_file = os.path.join(C.CACHE_DIR, "landmarks_raw.pkl")
    if os.path.exists(cache_file) and not force:
        print(f"[Landmarks] Loading cached landmarks from {cache_file}")
        with open(cache_file, "rb") as f:
            saved = pickle.load(f)
        return saved["lm2d"], saved["lm3d"], saved["failed"]

    if not _MP_AVAILABLE:
        raise RuntimeError("mediapipe is required for landmark extraction.")

    print(f"[Landmarks] Extracting landmarks for {len(df)} images …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fn  = row["filename"]
        fp  = row["filepath"]
        lm2, lm3 = extract_landmarks(fp)
        if lm2 is None:
            failed.append(fn)
        else:
            cache_2d[fn] = lm2   # [468, 2]
            cache_3d[fn] = lm3   # [468, 3]

    print(f"[Landmarks] Success: {len(cache_2d)}  |  Failed: {len(failed)}")

    with open(cache_file, "wb") as f:
        pickle.dump({"lm2d": cache_2d, "lm3d": cache_3d, "failed": failed}, f)
    print(f"[Landmarks] Cached to {cache_file}")
    
    close_landmarker()
    return cache_2d, cache_3d, failed


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ANCHOR-BASED 2D NORMALISATION  (M2)
# ═══════════════════════════════════════════════════════════════════════════════

# MediaPipe Face Mesh landmark indices
# Left eye centre  ≈ mean of left eye contour
# Right eye centre ≈ mean of right eye contour
# Nose tip         = 4
_LEFT_EYE_IDX  = [33, 133, 160, 159, 158, 144, 145, 153]
_RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 373, 374, 380]
_NOSE_TIP_IDX  = 4


def anchor_normalise_2d(lm2d: np.ndarray) -> np.ndarray:
    """
    Anchor-based normalisation from the project plan (Section 2.1, step 7).
    
    p1 = left eye centre,  p2 = right eye centre,  p3 = nose tip
    d  = ||p1 - p2||_2  (inter-ocular distance)
    
    For each landmark pk:
        p̃k = ( ||pk-p1||/d,  ||pk-p2||/d,  ||pk-p3||/d )

    Returns flattened vector  X_2d  of shape [468 * 3] = [1404].
    (Note: the plan says 3×468 = 1404 for the MLP input)
    """
    p1 = lm2d[_LEFT_EYE_IDX].mean(axis=0)    # [2]
    p2 = lm2d[_RIGHT_EYE_IDX].mean(axis=0)   # [2]
    p3 = lm2d[_NOSE_TIP_IDX]                  # [2]
    d  = np.linalg.norm(p1 - p2) + 1e-8

    dists = np.stack([
        np.linalg.norm(lm2d - p1, axis=1) / d,   # [468]
        np.linalg.norm(lm2d - p2, axis=1) / d,
        np.linalg.norm(lm2d - p3, axis=1) / d,
    ], axis=1)   # [468, 3]

    return dists.flatten().astype(np.float32)    # [1404]


def normalise_all_2d(lm2d_dict: dict) -> dict:
    """Apply anchor_normalise_2d to every entry in the dict."""
    out = {}
    for fn, lm in lm2d_dict.items():
        out[fn] = anchor_normalise_2d(lm)
    np.save(os.path.join(C.CACHE_DIR, "norm_2d.npy"),
            {fn: v for fn, v in out.items()}, allow_pickle=True)
    print(f"[Norm-2D] Done. Shape per sample: {next(iter(out.values())).shape}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  5. PROCRUSTES 3D ALIGNMENT  (M3 – optional)
# ═══════════════════════════════════════════════════════════════════════════════

def procrustes_align(X: np.ndarray, Y: np.ndarray):
    """
    Align source X [N,3] to target Y [N,3] via Procrustes (scale+rotation+translation).
    Solves:  min_{s,R,t}  ||s*R*X + t - Y||^2_F

    Returns:
        X_aligned : [N,3]  aligned source
        (s, R, t) : transformation parameters
    """
    # Centre
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    X0  = X - muX
    Y0  = Y - muY

    # Scale
    ssX = (X0 ** 2).sum()
    ssY = (Y0 ** 2).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY

    # Rotation via SVD
    A   = Y0.T @ X0
    U, s_svd, Vt = np.linalg.svd(A)
    # Ensure proper rotation (det=+1)
    det_sign = np.linalg.det(U @ Vt)
    D = np.eye(3)
    D[2, 2] = np.sign(det_sign)
    R = U @ D @ Vt

    # Optimal scale
    traceTA = s_svd @ D.diagonal()
    s       = traceTA * normY / normX

    # Translation
    t = muY - s * R @ muX

    X_aligned = (s * (X @ R.T) + t)
    return X_aligned.astype(np.float32), (s, R, t)


def build_reference_shape(lm3d_dict: dict) -> np.ndarray:
    """
    Compute Procrustes mean shape from training set landmarks.
    Simple version: average over all shapes after initial centering + unit-scaling.
    """
    shapes = []
    for lm in lm3d_dict.values():
        c = lm - lm.mean(axis=0)
        c /= (np.sqrt((c**2).sum()) + 1e-8)
        shapes.append(c)
    ref = np.mean(shapes, axis=0)
    ref /= (np.sqrt((ref**2).sum()) + 1e-8)
    np.save(C.PROCRUSTES_REF, ref)
    print(f"[Procrustes] Reference shape saved → {C.PROCRUSTES_REF}")
    return ref.astype(np.float32)


def align_all_3d(lm3d_dict: dict, ref: np.ndarray) -> dict:
    """Align every 3D shape to the reference and flatten to [1404]."""
    out = {}
    for fn, lm in lm3d_dict.items():
        aligned, _ = procrustes_align(lm, ref)
        out[fn] = aligned.flatten().astype(np.float32)
    np.save(os.path.join(C.CACHE_DIR, "aligned_3d.npy"),
            {fn: v for fn, v in out.items()}, allow_pickle=True)
    print(f"[Procrustes] Alignment done. Shape per sample: {next(iter(out.values())).shape}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(use_3d: bool = True, force_landmarks: bool = False):
    """
    Run the full Phase 1 pipeline.

    Args:
        use_3d          : If False, skip Procrustes / 3D processing (M3 optional).
        force_landmarks : Re-extract even if cache exists.

    Returns dict with keys:
        train_df, test_df,
        lm2d_norm   (normalised 2D, dict filename→[1404]),
        lm3d_aligned (Procrustes 3D, dict filename→[1404])  or None,
        failed_list
    """
    C.print_config()

    # 1. Load dataset
    df = load_scut_fbp5500()

    # 2. Split
    train_df, test_df = make_split(df)

    # Save splits
    train_df.to_csv(os.path.join(C.CACHE_DIR, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(C.CACHE_DIR,  "test_split.csv"),  index=False)

    # 3. Extract landmarks
    all_df = pd.concat([train_df, test_df])
    lm2d_raw, lm3d_raw, failed = extract_all_landmarks(all_df, force=force_landmarks)

    # Remove failed images from splits
    if failed:
        train_df = train_df[~train_df["filename"].isin(failed)].reset_index(drop=True)
        test_df  = test_df[ ~test_df["filename"].isin(failed)].reset_index(drop=True)
        print(f"[Phase1] After removing failed: Train={len(train_df)}, Test={len(test_df)}")

    # 4. 2D normalisation (M2)
    lm2d_norm = normalise_all_2d(lm2d_raw)

    # 5. 3D Procrustes (M3 – optional)
    lm3d_aligned = None
    if use_3d:
        train_lm3d = {fn: lm3d_raw[fn] for fn in train_df["filename"] if fn in lm3d_raw}
        ref = build_reference_shape(train_lm3d)
        lm3d_aligned = align_all_3d(lm3d_raw, ref)
    else:
        print("[Phase1] Skipping 3D Procrustes (use_3d=False).")

    print("\n[Phase1] ✓ Complete.")
    return {
        "train_df":     train_df,
        "test_df":      test_df,
        "lm2d_norm":    lm2d_norm,
        "lm3d_aligned": lm3d_aligned,
        "failed":       failed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no3d",   action="store_true", help="Skip 3D Procrustes")
    parser.add_argument("--force",  action="store_true", help="Re-extract landmarks")
    args = parser.parse_args()

    results = run_phase1(use_3d=not args.no3d, force_landmarks=args.force)
    print(f"Train: {len(results['train_df'])}  Test: {len(results['test_df'])}")
