from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from inference.threshold import load_threshold


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

OUTPUT_DIR = Path(
    "results/visualizations"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("Loading AnomalyVFM+")
    print("=" * 60)

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # Current checkpoint format
    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        raise KeyError(
            "No compatible model state found in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    print(
        "Missing keys:",
        len(result.missing_keys)
    )

    print(
        "Unexpected keys:",
        len(result.unexpected_keys)
    )

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(
    model,
    image,
    threshold
):

    with torch.no_grad():

        anomaly_map, image_logit = (
            model.forward_with_score(
                [image]
            )
        )

        # ----------------------------------------------------
        # Image-level anomaly score
        # ----------------------------------------------------

        image_score = torch.sigmoid(
            image_logit
        ).item()

        # ----------------------------------------------------
        # Pixel-level anomaly map
        # ----------------------------------------------------

        anomaly_probability = torch.sigmoid(
            anomaly_map
        )

        anomaly_probability = F.interpolate(
            anomaly_probability,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        anomaly_map = (
            anomaly_probability[
                0,
                0
            ]
            .cpu()
            .numpy()
        )

    prediction = (
        "ANOMALOUS"
        if image_score >= threshold
        else "NORMAL"
    )

    return (
        image_score,
        anomaly_map,
        prediction
    )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    anomaly_map,
    prediction,
    score,
    threshold,
    image_path
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_array = np.asarray(
        image
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(15, 5)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    ax1 = figure.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        image_array
    )

    ax1.set_title(
        "Original Image"
    )

    ax1.axis("off")

    # --------------------------------------------------------
    # Anomaly heatmap
    # --------------------------------------------------------

    ax2 = figure.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        anomaly_map,
        cmap="jet",
        vmin=0.0,
        vmax=1.0
    )

    ax2.set_title(
        "Anomaly Heatmap"
    )

    ax2.axis("off")

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    ax3 = figure.add_subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        image_array
    )

    ax3.imshow(
        anomaly_map,
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0
    )

    ax3.set_title(
        "Anomaly Overlay"
    )

    ax3.axis("off")

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    figure.suptitle(
        (
            f"AnomalyVFM+ | "
            f"Prediction: {prediction} | "
            f"Score: {score:.4f} | "
            f"Threshold: {threshold:.4f}"
        ),
        fontsize=14
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # IMPORTANT:
    # Use prediction in filename so normal and anomalous
    # images never overwrite each other.
    # --------------------------------------------------------

    image_stem = Path(
        image_path
    ).stem

    prediction_name = prediction.lower()

    output_path = (
        OUTPUT_DIR
        /
        f"{image_stem}_{prediction_name}_result.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AnomalyVFM+ VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    print()
    print(
        f"Calibrated threshold: {threshold:.6f}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Ask for image
    # --------------------------------------------------------

    print()

    image_path = input(
        "Enter image path: "
    ).strip()

    if not image_path:

        raise ValueError(
            "No image path provided."
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    print()
    print(
        "Image loaded:",
        image_path
    )

    print(
        "Image size:",
        image.size
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Running AnomalyVFM+ inference..."
    )

    (
        score,
        anomaly_map,
        prediction
    ) = run_inference(
        model,
        image,
        threshold
    )

    # --------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------

    output_path = save_visualization(
        image=image,
        anomaly_map=anomaly_map,
        prediction=prediction,
        score=score,
        threshold=threshold,
        image_path=image_path
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VISUALIZATION RESULT")
    print("=" * 60)

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Score      : {score:.6f}"
    )

    print(
        f"Threshold  : {threshold:.6f}"
    )

    print(
        f"Output     : {output_path}"
    )

    print("=" * 60)
    print(
        "VISUALIZATION COMPLETE"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()