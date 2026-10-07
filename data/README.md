# Dataset Structure

AnomalyVFM+ uses an image-based anomaly detection dataset.

The first benchmark dataset is MVTec AD.

Expected structure:

data/
└── raw/
    └── mvtec/
        ├── bottle/
        │   ├── train/
        │   │   └── good/
        │   ├── test/
        │   │   ├── good/
        │   │   ├── broken_large/
        │   │   └── ...
        │   └── ground_truth/
        │       ├── broken_large/
        │       └── ...
        │
        ├── cable/
        ├── capsule/
        └── ...

Training:
- Only normal images are used for the normal training set.
- Synthetic anomalies will be generated from normal images.

Testing:
- Normal and anomalous images are evaluated.
- Ground-truth masks are used for pixel-level evaluation.

Important:
The dataset itself is not committed to GitHub.
Only dataset-related code, configuration, and documentation are committed.