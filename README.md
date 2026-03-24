# HTH_LineExtract

Image analysis tool for extracting and analyzing cell contours in microscopy images using segmentation and line extraction techniques.

## Overview

HTH_LineExtract is a Python-based image processing suite designed to analyze microscopy images (ND2 format) for cell morphology and migration patterns. The project combines multiple thresholding methods and machine learning segmentation to extract cell contours and compute morphological metrics.

## Features

- **Multi-channel image processing**: Supports ND2 microscopy image files with automatic maximum intensity projection (MIP)
- **Adaptive thresholding**: Uses three-tier fallback approach:
  - Li thresholding (primary method)
  - Otsu thresholding (secondary method)
  - Machine learning segmentation (tertiary method)
- **Contour extraction**: Automated cell boundary detection and analysis
- **Morphological metrics**: Computes contour length, shortest line distance, and migration indices
- **Batch processing**: Analyze entire folders of microscopy images at once
- **Image rotation**: Optional 90-degree rotation support for corrected image orientation
- **Visualization**: Generates annotated output images with contour overlays

## Project Structure

```
HTH_LineExtract/
├── analyse_folder.py       # Main batch analysis script
├── Untitled.ipynb          # Jupyter notebook with experimental code
├── src/
│   ├── ContourUtils.py     # Contour manipulation and analysis utilities
│   └── Segmenter.py        # Machine learning segmentation module
├── models/                 # Pre-trained ML models
│   └── 202511_cB.pic       # Pickled segmentation model
└── results/                # Output directory (generated)
    ├── log.txt             # Processing log
    ├── res.txt             # Results summary
    └── *.png               # Annotated output images
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-image
- nd2 (ND2 image reader)
- PIL/Pillow (image handling)

## Usage

### Batch Process a Folder

```bash
python analyse_folder.py /path/to/image/folder [options]
```

#### Options

- `--rotate`: Apply 90-degree rotation to all images before analysis

### Example

```bash
python analyse_folder.py ./images/ --rotate
```

This will:
1. Scan the folder for `.nd2` files (excluding brightfield and TIFF images)
2. Process each image through the segmentation pipeline
3. Generate a `results/` subdirectory containing:
   - `log.txt`: Processing log with skipped/failed files
   - `res.txt`: Detailed analysis results (contour metrics, cell position)
   - `.png` files: Annotated images with overlaid contours

## Output

### Results File Format

For each successfully processed image, the `res.txt` contains:

```
#### File: image.nd2
## Channel: 0
## Dimension: (height, width)
## Method: li
## Cells on the left
## Contour_length Shortest_line_length Ratio
[contour_length] [base_distance] [ratio]
```

### Visualization

Output PNG files show:
- Original microscopy image (grayscale)
- White contour overlay marking cell boundaries

Failed analyses are saved with `_FAILED.png` suffix.

## How It Works

1. **Image Loading**: Reads ND2 files and computes maximum intensity projections
2. **Thresholding**: Attempts Li thresholding first; falls back to Otsu then ML if needed
3. **Contour Detection**: Extracts binary masks and derives contour boundaries
4. **Metrics**: Computes migration index and geometric measurements
5. **Localization**: Determines whether cells are positioned on left or right side of image
6. **Output**: Saves results and visualization images

## Copyright

Copyright © Combriat, 2025

## License

[Specify your license here]

## Author

tomcombriat-pro

---

**Note**: This project includes a pre-trained machine learning model for challenging segmentation cases. Ensure the `models/202511_cB.pic` file is present in the models directory before running the analysis.
