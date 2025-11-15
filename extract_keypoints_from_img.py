"""
Extract MediaPipe keypoints (Pose + two Hands) from images.

Input tree:
  DATA_IN/<word>/*.(jpg|jpeg|png|bmp|webp)

Output:
  data_np/<word>/<same_stem>.npy        -> (T, K, 3) float32, T=1, K = 33 (pose) + 21 (L-hand) + 21 (R-hand) = 75
  data_np/<word>/<same_stem>.meta.json  -> fps, frame_count, landmark_order

Notes:
- Coordinates: x,y in [0,1] image-normalized; z is MediaPipe's relative depth (negative is into screen).
- We keep missing landmarks as NaN (e.g., hand/pose not detected).
- If a hand is not detected in a given image, those 21 rows are NaN.
"""

import os
import glob
import json
import numpy as np
import cv2
import mediapipe as mp

# Change this to your image root
DATA_IN = "C:/Users/Jyothi/ISL_Anveshan/data"
DATA_OUT = "data_np"

# Landmark counts (fixed by MediaPipe)
POSE_K = 33
HAND_K = 21
TOTAL_K = POSE_K + HAND_K + HAND_K  # 75

# Order map for reference in metadata
LANDMARK_ORDER = {
    "pose": list(range(POSE_K)),
    "left_hand": list(range(POSE_K, POSE_K + HAND_K)),
    "right_hand": list(range(POSE_K + HAND_K, TOTAL_K)),
}

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def extract_from_image(path: str) -> tuple[np.ndarray, dict]:
    """Extract (T=1, K, 3) keypoints from a single image."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare (K,3) array filled with NaN
    kpts = np.full((TOTAL_K, 3), np.nan, dtype=np.float32)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=True) as pose, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1) as hands:

        # Pose (whole body)
        pose_res = pose.process(rgb)

        # Hands
        hands_res = hands.process(rgb)

        # Pose (33)
        if pose_res.pose_landmarks:
            for i, lm in enumerate(pose_res.pose_landmarks.landmark[:POSE_K]):
                kpts[i, 0] = lm.x
                kpts[i, 1] = lm.y
                kpts[i, 2] = lm.z

        # Hands (up to 2) — assign to left/right consistently
        left_hand_idx = list(range(POSE_K, POSE_K + HAND_K))
        right_hand_idx = list(range(POSE_K + HAND_K, TOTAL_K))

        hand_keypoints = []  # list of (21,3) arrays
        if hands_res.multi_hand_landmarks:
            for hlm in hands_res.multi_hand_landmarks:
                arr = np.full((HAND_K, 3), np.nan, dtype=np.float32)
                for i, lm in enumerate(hlm.landmark[:HAND_K]):
                    arr[i, 0] = lm.x
                    arr[i, 1] = lm.y
                    arr[i, 2] = lm.z
                hand_keypoints.append(arr)

            # decide left vs right by x barycenter
            if len(hand_keypoints) == 2:
                xs = [np.nanmean(a[:, 0]) for a in hand_keypoints]
                left_idx = int(np.argmin(xs))
                right_idx = 1 - left_idx
                kpts[left_hand_idx, :] = hand_keypoints[left_idx]
                kpts[right_hand_idx, :] = hand_keypoints[right_idx]
            elif len(hand_keypoints) == 1:
                one = hand_keypoints[0]
                # Try to classify using shoulders if available
                # pose indices: 11 (L shoulder), 12 (R shoulder)
                l_sh, r_sh = 11, 12
                if not np.isnan(kpts[l_sh, 0]) and not np.isnan(kpts[r_sh, 0]):
                    mid_x = 0.5 * (kpts[l_sh, 0] + kpts[r_sh, 0])
                    is_left = np.nanmean(one[:, 0]) < mid_x
                else:
                    # fallback: assume right
                    is_left = False
                if is_left:
                    kpts[left_hand_idx, :] = one
                else:
                    kpts[right_hand_idx, :] = one

    # Wrap into (T, K, 3) with T=1 so it matches video-based pipeline
    arr = kpts[None, :, :]  # shape (1, TOTAL_K, 3)

    meta = {
        "fps": 1.0,                         # dummy (single frame)
        "frame_count": int(arr.shape[0]),   # 1
        "landmark_order": LANDMARK_ORDER,
        "source": os.path.basename(path),
    }
    return arr, meta


def save_arrays(out_stem: str, arr: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    np.save(out_stem + ".npy", arr)
    with open(out_stem + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    os.makedirs(DATA_OUT, exist_ok=True)
    word_dirs = [d for d in glob.glob(os.path.join(DATA_IN, "*")) if os.path.isdir(d)]
    if not word_dirs:
        print(f"No word folders found under {DATA_IN}/")
        return

    # Supported image extensions
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]

    for wdir in word_dirs:
        word = os.path.basename(wdir)

        # Collect all images for this word
        images = []
        for ext in exts:
            images.extend(sorted(glob.glob(os.path.join(wdir, ext))))

        if not images:
            print(f"[{word}] No image files (jpg/jpeg/png/bmp/webp), skipping.")
            continue

        out_dir = os.path.join(DATA_OUT, word)
        os.makedirs(out_dir, exist_ok=True)

        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_stem = os.path.join(out_dir, stem)
            print(f"[{word}] Processing image {os.path.basename(img_path)} ...")
            arr, meta = extract_from_image(img_path)
            save_arrays(out_stem, arr, meta)
        print(f"[{word}] Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()
