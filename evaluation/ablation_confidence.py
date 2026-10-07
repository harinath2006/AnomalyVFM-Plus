from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.decoder.confidence_refinement import refine_anomaly_map
from evaluation.metrics import pixel_level_auroc


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

DATASET_ROOT = Path(
    "data/raw/mvtec/bottle/test"
)

GROUND_TRUTH_ROOT = Path(
    "data/raw/mvtec/bottle/ground_truth"
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
            "Using checkpoint key: model_state_dict"
        )

    # --------------------------------------------------------
    # Alternative format
    # --------------------------------------------------------

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

        print(
            "Using checkpoint key: state_dict"
        )

    else:

        raise KeyError(
            "Could not find a compatible model state.\n"
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
# DATASET
# ============================================================

def collect_test_images():

    samples = []

    # --------------------------------------------------------
    # Normal
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
    # Anomalous
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
# GROUND-TRUTH MASK
# ============================================================

def load_ground_truth(
    category,
    image_name
):

    # --------------------------------------------------------
    # Normal images have no anomaly mask
    # --------------------------------------------------------

    if category == "good":

        return np.zeros(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Defect mask
    # --------------------------------------------------------

    stem = Path(
        image_name
    ).stem

    mask_path = (
        GROUND_TRUTH_ROOT
        / category
        / f"{stem}_mask.png"
    )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Ground-truth mask not found:\n"
            f"{mask_path}"
        )

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.NEAREST
    )

    mask = np.array(
        mask
    )

    return (
        mask > 0
    ).astype(np.uint8)


# ============================================================
# CREATE CONFIDENCE MAP
# ============================================================

def create_confidence_map(
    anomaly_map
):
    """
    Creates a confidence map from the anomaly
    prediction itself.

    High-confidence anomaly regions receive
    stronger weights.

    The map is normalized independently for
    each image.
    """

    probabilities = torch.sigmoid(
        anomaly_map
    )

    minimum = probabilities.amin(
        dim=(2, 3),
        keepdim=True
    )

    maximum = probabilities.amax(
        dim=(2, 3),
        keepdim=True
    )

    confidence = (
        probabilities - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    # Keep a small minimum confidence so that
    # the complete anomaly map is not eliminated.
    confidence = (
        0.25
        +
        0.75 * confidence
    )

    return confidence.clamp(
        0.0,
        1.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANOMALYVFM+ CONFIDENCE REFINEMENT ABLATION")
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
    # Dataset
    # --------------------------------------------------------

    samples = collect_test_images()

    if not samples:

        raise RuntimeError(
            "No MVTec Bottle test images found."
        )

    print()
    print(
        "Total test images:",
        len(samples)
    )

    normal_count = sum(
        1
        for category, _
        in samples
        if category == "good"
    )

    anomaly_count = (
        len(samples)
        -
        normal_count
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

    original_predictions = []
    refined_predictions = []

    ground_truth_masks = []

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Running confidence refinement evaluation..."
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
            # Original anomaly probability map
            # ------------------------------------------------

            original_map = torch.sigmoid(
                anomaly_map
            )

            # ------------------------------------------------
            # Generate confidence map
            # ------------------------------------------------

            confidence_map = (
                create_confidence_map(
                    anomaly_map
                )
            )

            # ------------------------------------------------
            # Confidence refinement
            # ------------------------------------------------

            refined_map = (
                refine_anomaly_map(
                    anomaly_map,
                    confidence_map
                )
            )

            refined_probability = torch.sigmoid(
                refined_map
            )

            # ------------------------------------------------
            # Resize both maps
            # ------------------------------------------------

            original_map = F.interpolate(
                original_map,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                mode="bilinear",
                align_corners=False
            )

            refined_probability = F.interpolate(
                refined_probability,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                mode="bilinear",
                align_corners=False
            )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth = load_ground_truth(
            category,
            image_path.name
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        original_predictions.append(
            original_map[
                0,
                0
            ].cpu().numpy().flatten()
        )

        refined_predictions.append(
            refined_probability[
                0,
                0
            ].cpu().numpy().flatten()
        )

        ground_truth_masks.append(
            ground_truth.flatten()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

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
    # Combine arrays
    # --------------------------------------------------------

    original_predictions = np.concatenate(
        original_predictions
    )

    refined_predictions = np.concatenate(
        refined_predictions
    )

    ground_truth_masks = np.concatenate(
        ground_truth_masks
    )

    # --------------------------------------------------------
    # Pixel AUROC
    # --------------------------------------------------------

    original_auroc = pixel_level_auroc(
        original_predictions,
        ground_truth_masks
    )

    refined_auroc = pixel_level_auroc(
        refined_predictions,
        ground_truth_masks
    )

    improvement = (
        refined_auroc
        -
        original_auroc
    )

    if original_auroc != 0:

        relative_change = (
            improvement
            /
            original_auroc
            *
            100
        )

    else:

        relative_change = 0.0

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print("CONFIDENCE REFINEMENT ABLATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Configuration':<35}"
        f"{'Pixel AUROC':>15}"
    )

    print("-" * 55)

    print(
        f"{'Without confidence refinement':<35}"
        f"{original_auroc:>15.4f}"
    )

    print(
        f"{'With confidence refinement':<35}"
        f"{refined_auroc:>15.4f}"
    )

    print("-" * 55)

    print(
        f"{'Improvement':<35}"
        f"{improvement:>15.4f}"
    )

    print()

    print(
        "Relative change:",
        f"{relative_change:+.2f}%"
    )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()

    print(
        "INTERPRETATION"
    )

    if refined_auroc > original_auroc:

        print(
            "Confidence refinement improves "
            "pixel-level anomaly localization."
        )

    elif refined_auroc < original_auroc:

        print(
            "The unrefined anomaly map performs "
            "better on this test set."
        )

    else:

        print(
            "Both configurations produce "
            "the same pixel AUROC."
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
    print("CONFIDENCE REFINEMENT ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()