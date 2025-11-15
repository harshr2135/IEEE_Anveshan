# ISL Animation

This project is designed to create animations of Indian Sign Language (ISL) signs from video recordings. It processes video files, extracts keypoint data, canonicalizes the signs, and provides tools for visualizing the animations.

## Folder Structure

```
ISL_animation/
├── .gitignore
├── README.md
├── avatar_viewer.html
├── canonical/
├── canonicalize_signs.py
├── check3D.py
├── data_np/
│   ├── Alright/
│   ├── Beautiful/
│   ├── Bed/
│   ├── ... (and so on for each word)
├── player.html
└── word_mapping.html
```

## Project Structure

- **data_np/**: This directory contains the raw keypoint data extracted from videos. Each subdirectory is named after a word, and inside are `.npy` files containing the keypoint sequences for different recordings of that word.
- **canonical/**: This directory stores the canonical forms of the signs. For each word, a `canonical_median.npy` and `canonical_median.json` file are generated, representing the robust median of the sign's keypoint sequences.
- **player.html**: An HTML file that provides a stick-figure animation of the sign. It can load and play the `.json` files from the `canonical/` directory.
- **avatar_viewer.html**: An HTML file that provides a more advanced avatar-based animation of the sign. It can load a custom avatar (in `.glb` or `.gltf` format) and a motion file (in `.json` format) to create a realistic animation.
- **canonicalize_signs.py**: A Python script that processes the raw keypoint data in `data_np/` to create canonical forms of the signs. It performs normalization, time-alignment, outlier filtering, and temporal smoothing to generate the canonical sign data.
- **word_mapping.html**: This file is used to map words to their corresponding sign language representations. It helps in organizing and accessing the sign data for different words.
- **check3D.py**: This script is a utility for verifying and analyzing the 3D aspects of the sign language data. It can be used to check for inconsistencies or errors in the 3D keypoint data.

## How to Use

1. **Data Extraction**: The first step is to extract keypoint data from video recordings of ISL signs. While the script for this (`extract_mediapipe_from_mp4.py`) is not present in the repository, the output of this process should be stored in the `data_np/` directory. Each word should have its own subdirectory containing `.npy` files of the keypoint data.

2. **Canonicalization**: Run the `canonicalize_signs.py` script to process the raw data and generate canonical forms of the signs.

   ```bash
   python canonicalize_signs.py
   ```

   This will create the `canonical/` directory and populate it with the canonical sign data.

3. **Visualization**:
   - **Stick Figure Animation**: Open the `player.html` file in a web browser. You can then load a `.json` file from the `canonical/` directory to view a stick-figure animation of the sign.
   - **Avatar Animation**: Open the `avatar_viewer.html` file in a web browser. You can load a custom avatar and a motion file (`.json` from the `canonical/` directory) to view a more realistic animation of the sign.

## File Descriptions

- **`canonicalize_signs.py`**: This script reads the `.npy` files from the `data_np/` directory, processes them, and saves the canonical forms in the `canonical/` directory. The main steps in the pipeline are:
  1. Load all sequences for a word.
  2. Normalize the sequences by centering them by the hips and scaling by the shoulder width.
  3. Time-align the sequences to a target number of frames.
  4. Drop outlier sequences using DTW distance z-score.
  5. Canonicalize the sequences by taking the robust median.
  6. Apply temporal smoothing.
  7. Save the canonical sequences as `.npy` and `.json` files.

- **`player.html`**: This file uses [Three.js](https://threejs.org/) to render a stick-figure animation of a sign. It can load `.json` files containing the keypoint data and provides controls for playing, pausing, and scrubbing through the animation.

- **`avatar_viewer.html`**: This file also uses [Three.js](https://threejs.org/) to render an avatar-based animation. It can load a 3D model of an avatar (in `.glb` or `.gltf` format) and a motion file (`.json`) to animate the avatar. This provides a more realistic and detailed visualization of the sign.

- **`word_mapping.html`**: This file is used to map words to their corresponding sign language representations. It helps in organizing and accessing the sign data for different words.

- **`check3D.py`**: This script is a utility for verifying and analyzing the 3D aspects of the sign language data. It can be used to check for inconsistencies or errors in the 3D keypoint data.
