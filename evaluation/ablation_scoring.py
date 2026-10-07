from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.decoder.calibration import compute_adaptive_score
from evaluation.metrics import image_level_auroc


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

DATASET_ROOT = Path(
    "data/raw/mvtec/bottle/test"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_anomaly_vfm_plus():

    print()
    print("=" * 70)
    print("LOADING AnomalyVFM+ CHECKPOINT")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = AnomalyVFMPlus()

    model = model.to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    print(
        "Checkpoint keys:",
        list(checkpoint.keys())
    )

    # --------------------------------------------------------
    # Current checkpoint format
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "Using: model_state_dict"
        )

    # --------------------------------------------------------
    # Alternative checkpoint format
    # --------------------------------------------------------

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print(
            "Using: state_dict"
        )

    else:

        raise KeyError(
            "No compatible model state found.\n"
            f"Available keys: {list(checkpoint.keys())}"
        )

    # --------------------------------------------------------
    # Restore model
    # --------------------------------------------------------

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print()
    print(
        "Missing keys:",
        len(result.missing_keys)
    )

    print(
        "Unexpected keys:",
        len(result.unexpected_keys)
    )

    if result.missing_keys:

        print(
            "Missing keys:"
        )

        for key in result.missing_keys[:10]:

            print(
                "  ",
                key
            )

    if result.unexpected_keys:

        print(
            "Unexpected keys:"
        )

        for key in result.unexpected_keys[:10]:

            print(
                "  ",
                key
            )

    model.eval()

    print()
    print(
        "AnomalyVFM+ model loaded successfully."
    )

    return model


# ============================================================
# TEST DATA
# ============================================================

def collect_test_images():

    samples = []

    # --------------------------------------------------------
    # Normal images
    # --------------------------------------------------------

    good_dir = (
        DATASET_ROOT
        / "good"
    )

    for image_path in sorted(
        good_dir.glob("*.png")
    ):

        samples.append(
            (
                "good",
                image_path
            )
        )

    # --------------------------------------------------------
    # Anomalous images
    # --------------------------------------------------------

    for category_dir in sorted(
        DATASET_ROOT.iterdir()
    ):

        if not category_dir.is_dir():
            continue

        if category_dir.name == "good":
            continue

        for image_path in sorted(
            category_dir.glob("*.png")
        ):

            samples.append(
                (
                    category_dir.name,
                    image_path
                )
            )

    return samples


# ============================================================
# SCORING METHODS
# ============================================================

def max_score(anomaly_map):

    probabilities = torch.sigmoid(
        anomaly_map
    )

    flattened = probabilities.flatten(
        start_dim=1
    )

    return flattened.max(
        dim=1
    ).values


def adaptive_score(anomaly_map):

    return compute_adaptive_score(
        anomaly_map
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANOMALYVFM+ SCORING ABLATION")
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    samples = collect_test_images()

    if not samples:

        raise RuntimeError(
            "No MVTec test images found."
        )

    print()
    print(
        "Total test images:",
        len(samples)
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = []

    normal_count = 0
    anomaly_count = 0

    for category, _ in samples:

        if category == "good":

            labels.append(0)
            normal_count += 1

        else:

            labels.append(1)
            anomaly_count += 1

    labels = np.array(
        labels
    )

    print(
        "Normal images:",
        normal_count
    )

    print(
        "Anomalous images:",
        anomaly_count
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_anomaly_vfm_plus()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    max_scores = []

    adaptive_scores = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Running inference..."
    )

    for index, (
        category,
        image_path
    ) in enumerate(samples):

        image = Image.open(
            image_path
        ).convert("RGB")

        with torch.no_grad():

            anomaly_map, image_logit = (
                model.forward_with_score(
                    [image]
                )
            )

            # ------------------------------------------------
            # MAX PIXEL SCORE
            # ------------------------------------------------

            max_value = max_score(
                anomaly_map
            )[0].item()

            # ------------------------------------------------
            # ADAPTIVE SCORE
            # ------------------------------------------------

            adaptive_value = adaptive_score(
                anomaly_map
            )[0].item()

        max_scores.append(
            max_value
        )

        adaptive_scores.append(
            adaptive_value
        )

        if (
            (index + 1) % 10 == 0
            or
            index == len(samples) - 1
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(samples)}"
            )

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    max_scores = np.array(
        max_scores
    )

    adaptive_scores = np.array(
        adaptive_scores
    )

    # --------------------------------------------------------
    # AUROC
    # --------------------------------------------------------

    max_auroc = image_level_auroc(
        max_scores,
        labels
    )

    adaptive_auroc = image_level_auroc(
        adaptive_scores,
        labels
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    difference = (
        adaptive_auroc
        -
        max_auroc
    )

    if max_auroc != 0:

        relative_change = (
            difference
            /
            max_auroc
            *
            100
        )

    else:

        relative_change = 0.0

    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    normal_max = max_scores[
        labels == 0
    ]

    anomaly_max = max_scores[
        labels == 1
    ]

    normal_adaptive = adaptive_scores[
        labels == 0
    ]

    anomaly_adaptive = adaptive_scores[
        labels == 1
    ]

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("SCORING ABLATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Scoring Method':<30}"
        f"{'Image AUROC':>15}"
    )

    print("-" * 50)

    print(
        f"{'Maximum Pixel Score':<30}"
        f"{max_auroc:>15.4f}"
    )

    print(
        f"{'Adaptive Score':<30}"
        f"{adaptive_auroc:>15.4f}"
    )

    print("-" * 50)

    print(
        f"{'Improvement':<30}"
        f"{difference:>15.4f}"
    )

    print()

    print(
        "Relative change:",
        f"{relative_change:+.2f}%"
    )

    # --------------------------------------------------------
    # Score distribution statistics
    # --------------------------------------------------------

    print()
    print(
        "MAXIMUM SCORE STATISTICS"
    )

    print(
        f"Normal mean   : "
        f"{normal_max.mean():.6f}"
    )

    print(
        f"Anomaly mean  : "
        f"{anomaly_max.mean():.6f}"
    )

    print(
        f"Normal min    : "
        f"{normal_max.min():.6f}"
    )

    print(
        f"Normal max    : "
        f"{normal_max.max():.6f}"
    )

    print(
        f"Anomaly min   : "
        f"{anomaly_max.min():.6f}"
    )

    print(
        f"Anomaly max   : "
        f"{anomaly_max.max():.6f}"
    )

    print()
    print(
        "ADAPTIVE SCORE STATISTICS"
    )

    print(
        f"Normal mean   : "
        f"{normal_adaptive.mean():.6f}"
    )

    print(
        f"Anomaly mean  : "
        f"{anomaly_adaptive.mean():.6f}"
    )

    print(
        f"Normal min    : "
        f"{normal_adaptive.min():.6f}"
    )

    print(
        f"Normal max    : "
        f"{normal_adaptive.max():.6f}"
    )

    print(
        f"Anomaly min   : "
        f"{anomaly_adaptive.min():.6f}"
    )

    print(
        f"Anomaly max   : "
        f"{anomaly_adaptive.max():.6f}"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print(
        "INTERPRETATION"
    )

    if adaptive_auroc > max_auroc:

        print(
            "Adaptive scoring improves "
            "image-level anomaly discrimination."
        )

    elif adaptive_auroc < max_auroc:

        print(
            "Maximum scoring performs better "
            "on this test set."
        )

    else:

        print(
            "Both scoring methods produce "
            "the same AUROC."
        )

    print()

    print(
        "Test images:",
        len(samples)
    )

    print(
        "Normal:",
        normal_count
    )

    print(
        "Anomalous:",
        anomaly_count
    )

    print()
    print("=" * 70)
    print("SCORING ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()