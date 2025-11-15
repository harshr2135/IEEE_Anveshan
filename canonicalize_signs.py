"""
Canonical sign generator from multiple keypoint sequences.

Reads:
  data_np/<word>/*.npy   -> (T_i, K=75, 3)

Writes (per word) to canonical/<word>/ :
  canonical_median.npy/.json   : per-frame robust median after alignment
  canonical_medoid.npy/.json   : most representative sequence by DTW medoid

Pipeline:
  1) Load all sequences for a word
  2) Normalize (center by hips; scale by shoulder width)
  3) Time-align via linear interpolation to TARGET_FRAMES
  4) Drop outliers using DTW distance z-score
  5) Canonicalize (median + medoid)
  6) Temporal smoothing
  7) Save .npy and .json
"""

import os
import glob
import json
import math
import numpy as np
from typing import List, Tuple

DATA_DIR = "data_np"
OUTPUT_DIR = "canonical"

TARGET_FRAMES = 60           # uniform timeline length
POSE_K, HAND_K = 33, 21
TOTAL_K = POSE_K + 2*HAND_K  # 75

# MediaPipe Pose indices we’ll use for centering and scale
LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12

MAX_ZSCORE = 2.5
SMOOTH_WINDOW = 5

def moving_average(arr, win=5):
    if win <= 1:
        return arr
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (0,0), (0,0)), mode='edge')
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = np.nanmean(padded[t:t+win], axis=0)
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
            out[:,k,c] = interp_nan_1d(out[:,k,c])
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

    # center
    mid_hips = 0.5 * (seq[:, LEFT_HIP_IDX, :] + seq[:, RIGHT_HIP_IDX, :])  # (T,3)
    seq -= mid_hips[:, None, :]

    # scale (constant per sequence)
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

def frame_cost(f1: np.ndarray, f2: np.ndarray) -> float:
    diff = f1 - f2
    return float(np.sqrt(np.nansum(diff * diff)))

def dtw_distance(A: np.ndarray, B: np.ndarray) -> float:
    TA, TB = A.shape[0], B.shape[0]
    dp = np.full((TA+1, TB+1), np.inf, dtype=np.float32)
    dp[0,0] = 0.0
    for i in range(1, TA+1):
        fi = A[i-1]
        for j in range(1, TB+1):
            fj = B[j-1]
            c = frame_cost(fi, fj)
            dp[i,j] = c + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[TA, TB])

def pairwise_dtw(seqs: List[np.ndarray]) -> np.ndarray:
    n = len(seqs)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = dtw_distance(seqs[i], seqs[j])
            D[i,j] = D[j,i] = d
    return D

def medoid_index(D: np.ndarray) -> int:
    return int(np.argmin(D.sum(axis=1)))

def zscore(a: np.ndarray) -> np.ndarray:
    mu, sd = np.mean(a), np.std(a) + 1e-9
    return (a - mu) / sd

def safe_median(a, axis=0):
    return np.nanmedian(a, axis=axis)

def save_npy_json(path_prefix: str, seq: np.ndarray):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(path_prefix + ".npy", seq)
    frames = []
    for t in range(seq.shape[0]):
        pts = [{"x": float(x), "y": float(y), "z": float(z)} for (x,y,z) in seq[t]]
        frames.append({"frame": int(t), "points": pts})
    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(frames, f)

def process_word(word_dir: str, out_dir: str):
    word = os.path.basename(word_dir.rstrip("/"))
    files = sorted(glob.glob(os.path.join(word_dir, "*.npy")))
    if not files:
        print(f"[{word}] No .npy files found, skipping.")
        return

    # 1) Load + fill NaNs over time
    seqs = [fill_nans(np.load(f)) for f in files]

    # 2) Normalize
    seqs = [normalize_sequence(s) for s in seqs]

    # 3) Resample to common length
    seqs = [interpolate_sequence(s, TARGET_FRAMES) for s in seqs]

    # 4) Outlier filtering with DTW
    if len(seqs) > 2:
        D = pairwise_dtw(seqs)
        totals = D.sum(axis=1)
        z = zscore(totals)
        keep = [i for i, zi in enumerate(z) if zi <= MAX_ZSCORE]
        if len(keep) < max(3, len(seqs)//3):
            keep = list(range(len(seqs)))
        seqs = [seqs[i] for i in keep]
        print(f"[{word}] kept {len(seqs)} sequences after outlier filter.")
    else:
        print(f"[{word}] fewer than 3 sequences; skipping outlier filter.")

    # 5) Canonicals
    stack = np.stack(seqs, axis=0)  # (N,L,K,3)
    canonical_median = safe_median(stack, axis=0)

    if len(seqs) == 1:
        canonical_medoid = seqs[0]
    else:
        Dk = pairwise_dtw(seqs)
        m_idx = medoid_index(Dk)
        canonical_medoid = seqs[m_idx]

    # 6) Smooth
    canonical_median = moving_average(canonical_median, SMOOTH_WINDOW)
    canonical_medoid = moving_average(canonical_medoid, SMOOTH_WINDOW)

    # 7) Save
    base = os.path.join(out_dir, word)
    save_npy_json(os.path.join(base, "canonical_median"), canonical_median)
    save_npy_json(os.path.join(base, "canonical_medoid"), canonical_medoid)
    print(f"[{word}] saved canonical sequences -> {base}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    words = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(d)]
    if not words:
        print("No word folders found in data_np/")
        return
    for wd in words:
        process_word(wd, OUTPUT_DIR)

if __name__ == "__main__":
    main()