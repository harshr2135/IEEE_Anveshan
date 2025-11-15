"""
FAST canonical sign generator (no DTW).
Saves output in FLAT format:

canonical/
    Word_canonical_median.npy
    Word_canonical_median.json
    Word_canonical_medoid.npy
    Word_canonical_medoid.json
"""

import os
import glob
import json
import numpy as np

DATA_DIR = "data_np"
OUTPUT_DIR = "canonical"

TARGET_FRAMES = 60
POSE_K, HAND_K = 33, 21
TOTAL_K = POSE_K + 2*HAND_K

LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12

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
    x = np.arange(len(y))
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.zeros_like(y, dtype=np.float32)
    y0 = np.interp(x, x[mask], y[mask])
    return y0.astype(np.float32)

def fill_nans(seq):
    T, K, C = seq.shape
    out = seq.copy()
    for k in range(K):
        for c in range(C):
            out[:,k,c] = interp_nan_1d(out[:,k,c])
    return out

def interpolate_sequence(seq, L):
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

def normalize_sequence(seq):
    seq = seq.copy()
    mid_hips = 0.5 * (seq[:, LEFT_HIP_IDX, :] + seq[:, RIGHT_HIP_IDX, :])
    seq -= mid_hips[:, None, :]

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

def save_npy_json(path_prefix, seq):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    np.save(path_prefix + ".npy", seq)

    frames = []
    for t in range(seq.shape[0]):
        pts = [{"x": float(x), "y": float(y), "z": float(z)} for (x,y,z) in seq[t]]
        frames.append({"frame": int(t), "points": pts})

    with open(path_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(frames, f)

def process_word(word_dir, out_dir):
    word = os.path.basename(word_dir)
    print(f"\n[{word}] Starting...")

    files = sorted(glob.glob(os.path.join(word_dir, "*.npy")))
    print(f"[{word}] Found {len(files)} samples")

    if not files:
        print(f"[{word}] No .npy files found, skipping.")
        return

    seqs = [fill_nans(np.load(f)) for f in files]
    seqs = [normalize_sequence(s) for s in seqs]
    seqs = [interpolate_sequence(s, TARGET_FRAMES) for s in seqs]

    print(f"[{word}] Computing canonical median...")
    stack = np.stack(seqs, axis=0)
    canonical_median = safe_median(stack, axis=0)

    print(f"[{word}] Selecting Euclidean medoid...")
    diff = stack - canonical_median[None]
    dists = np.sqrt(np.sum(diff**2, axis=(1,2,3)))
    idx = int(np.argmin(dists))
    canonical_medoid = seqs[idx]

    print(f"[{word}] Smoothing...")
    canonical_median = moving_average(canonical_median, SMOOTH_WINDOW)
    canonical_medoid = moving_average(canonical_medoid, SMOOTH_WINDOW)

    print(f"[{word}] Saving flat outputs...")

    save_npy_json(
        os.path.join(out_dir, f"{word}_canonical_median"),
        canonical_median
    )

    save_npy_json(
        os.path.join(out_dir, f"{word}_canonical_medoid"),
        canonical_medoid
    )

    print(f"[{word}] Done.")

def main():
    print("Running canonicalization (FAST, FLAT)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    words = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) if os.path.isdir(d)]
    print("Words found:", [os.path.basename(w) for w in words])

    if not words:
        print("No folders in data_np/")
        return

    for wd in words:
        process_word(wd, OUTPUT_DIR)

if __name__ == "__main__":
    main()
