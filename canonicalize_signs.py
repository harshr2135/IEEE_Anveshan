"""
Canonical sign generator from multiple keypoint sequences.

Reads:
  data_np/<word>/*.npy   -> (T_i, K=75, 3)

Writes (per word) to canonical/<word>/ :
  canonical_median.npy/.json   : per-frame robust median after alignment
  canonical_medoid.npy/.json   : most representative sequence (closest to median)

Pipeline (FAST MODE, no DTW):
  1) Load all sequences for a word
  2) Normalize (center by hips; scale by shoulder width)
  3) Time-align via linear interpolation to TARGET_FRAMES
  4) Canonicalize (median + Euclidean medoid)
  5) Temporal smoothing
  6) Save .npy and .json
"""

import os
import glob
import json
import numpy as np
from typing import List, Tuple

# Paths relative to your project root:
# C:\Users\Jyothi\IEEE_Anveshan\data_np
# C:\Users\Jyothi\IEEE_Anveshan\canonical
DATA_DIR = "data_np"
OUTPUT_DIR = "canonical"

TARGET_FRAMES = 60           # uniform timeline length
POSE_K, HAND_K = 33, 21
TOTAL_K = POSE_K + 2 * HAND_K  # 75

# MediaPipe Pose indices we’ll use for centering and scale
LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12

SMOOTH_WINDOW = 5


def moving_average(arr, win=5):
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = np.nanmean(padded[t:t + win], axis=0)
    return out


def interp_nan_1d(y):
    """Fill NaNs by linear interpolation in 1D; if all NaN, return zeros."""
    x = np.arange(len(y))
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.zeros_like(y, dtype=np.float32)
    y0 = np.interp(x, x[mask], y[mask])
    return y0.astype(np.float32)


def fill_nans(seq):
    """Fill NaNs per (k,c) over time; seq: (T,K,3)."""
    T, K, C = seq.shape
    out = seq.copy()
    for k in range(K):
        for c in range(C):
            out[:, k, c] = interp_nan_1d(out[:, k, c])
    return out


def interpolate_sequence(seq: np.ndarray, L: int) -> np.ndarray:
    """Linear interpolate a (T,K,3) sequence (now assumed NaNs were filled)."""
    T, K, C = seq.shape
    if T == L:
        return seq
    x_src = np.linspace(0, 1, T)
    x_tgt = np.linspace(0, 1, L)
    out = np.empty((L, K, C), dtype=np.float32)
    for k in range(K):
        for c in range(C):
            out[:, k, c] = np.interp(x_tgt, x_src, seq[:, k, c])
    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Center by mid-hips; scale by median shoulder width."""
    seq = seq.copy()

    # Center
    mid_hips = 0.5 * (seq[:, LEFT_HIP_IDX, :] + seq[:, RIGHT_HIP_IDX, :])  # (T,3)
    seq -= mid_hips[:, None, :]

    # Scale (constant per sequence)
    shoulder_dist = np.linalg.norm(
        seq[:, LEFT_SHOULDER_IDX, :] - seq[:, RIGHT_SHOULDER_IDX, :],
        axis=1
    )
    scale = np.nanmedian(shoulder_dist)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = np.nanstd(seq)
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0
    seq /= scale
    return seq


def safe_median(a, axis=0):
    return np.nanmedian(a, axis=axis)


def save_npy_json(path_prefix: str, seq: np.ndarray):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(path_prefix + ".npy", seq)
    frames = []
    for t in range(seq.shape[0]):
        pts = [{"x": float(x), "y": float(y), "z": float(z)} for (x, y, z) in seq[t]]
        frames.append({"frame": int(t), "points": pts})
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(frames, f)


def process_word(word_dir: str, out_dir: str):
    word = os.path.basename(word_dir.rstrip("/"))
    print(f"\n[{word}] Starting...")

    files = sorted(glob.glob(os.path.join(word_dir, "*.npy")))
    print(f"[{word}] Found {len(files)} sequences")

    if not files:
        print(f"[{word}] No .npy files found, skipping.")
        return

    # 1) Load + fill NaNs over time
    print(f"[{word}] Loading & filling NaNs...")
    seqs = [fill_nans(np.load(f)) for f in files]

    # 2) Normalize
    print(f"[{word}] Normalizing...")
    seqs = [normalize_sequence(s) for s in seqs]

    # 3) Resample to common length
    print(f"[{word}] Interpolating to {TARGET_FRAMES} frames...")
    seqs = [interpolate_sequence(s, TARGET_FRAMES) for s in seqs]

    # 4 & 5) Canonicals (FAST: no DTW)
    print(f"[{word}] Computing canonical median...")
    stack = np.stack(seqs, axis=0)  # (N,L,K,3)
    canonical_median = safe_median(stack, axis=0)

    print(f"[{word}] Selecting Euclidean medoid (closest to median)...")
    # Compute simple Euclidean distance of each sequence to the median
    diff = stack - canonical_median[None, :, :, :]
    # Sum of squared differences over (T,K,C)
    dists = np.sqrt(np.sum(diff ** 2, axis=(1, 2, 3)))  # (N,)
    medoid_idx = int(np.argmin(dists))
    canonical_medoid = seqs[medoid_idx]

    # 6) Smooth
    print(f"[{word}] Smoothing...")
    canonical_median = moving_average(canonical_median, SMOOTH_WINDOW)
    canonical_medoid = moving_average(canonical_medoid, SMOOTH_WINDOW)

    # 7) Save
    base = os.path.join(out_dir, f"{word}_canonical_median")
    save_npy_json(base, canonical_median)

    base = os.path.join(out_dir, f"{word}_canonical_medoid")
    save_npy_json(base, canonical_medoid)



def main():
    print("Running canonicalization (FAST, no DTW)...")
    print("DATA_DIR =", os.path.abspath(DATA_DIR))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    words = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(d)]
    print("Found word dirs:", [os.path.basename(w) for w in words])

    if not words:
        print("No word folders found in data_np/")
        return

    for wd in words:
        process_word(wd, OUTPUT_DIR)


if __name__ == "__main__":
    main()
