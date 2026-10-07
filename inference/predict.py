from pathlib import Path

import torch
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from inference.threshold import load_threshold
from inference.decision import make_decision, print_decision


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

IMAGE_SIZE = 224

# ------------------------------------------------------------
# Calibrated threshold
# ------------------------------------------------------------

ANOMALY_THRESHOLD = load_threshold()

# ------------------------------------------------------------
# Pixel-level evidence settings
# ------------------------------------------------------------

MAP_THRESHOLD = 0.60
SPATIAL_RATIO_THRESHOLD = 0.01


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Create AnomalyVFM+ and restore the trained checkpoint.
    """

    print()
    print("Creating AnomalyVFM+ model...")

    model = AnomalyVFMPlus()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Support multiple checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Restore weights
    # --------------------------------------------------------

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing_keys:

        print()
        print("WARNING: Missing checkpoint keys:")

        for key in missing_keys[:20]:
            print("   ", key)

    if unexpected_keys:

        print()
        print("WARNING: Unexpected checkpoint keys:")

        for key in unexpected_keys[:20]:
            print("   ", key)

    model.eval()

    print("Checkpoint restored successfully.")

    # --------------------------------------------------------
    # Checkpoint metadata
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "step" in checkpoint:

            print(
                "Checkpoint training step:",
                checkpoint["step"]
            )

        elif "training_step" in checkpoint:

            print(
                "Checkpoint training step:",
                checkpoint["training_step"]
            )

    return model


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an input image as RGB PIL image.
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    return image


# ============================================================
# ANOMALY MAP EXTRACTION
# ============================================================

def prepare_anomaly_map(anomaly_map):
    """
    Convert model anomaly map into a 2D tensor.

    Expected:

        [B, 1, H, W]

    Single-image result:

        [H, W]
    """

    if isinstance(anomaly_map, tuple):

        anomaly_map = anomaly_map[0]

    anomaly_map = torch.as_tensor(
        anomaly_map
    ).detach().float()

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    anomaly_map = anomaly_map.squeeze()

    if anomaly_map.ndim != 2:

        raise ValueError(
            "Expected a single 2D anomaly map, "
            f"got shape {tuple(anomaly_map.shape)}"
        )

    return anomaly_map


# ============================================================
# ANOMALY SCORE
# ============================================================

def compute_inference_score(anomaly_map):
    """
    Convert the raw decoder anomaly map into an image-level
    anomaly score.

    The decoder output in the current project is not a
    calibrated probability. Therefore we DO NOT apply sigmoid.

    Instead, the score is based directly on anomaly magnitude.

    Components:

        1. Mean anomaly magnitude
        2. 95th percentile anomaly magnitude
        3. Maximum anomaly magnitude

    The combination is useful because:

        mean       -> detects widespread anomalies
        percentile -> detects strong regional anomalies
        maximum    -> preserves strong localized anomalies
    """

    anomaly_map = torch.as_tensor(
        anomaly_map
    ).detach().float()

    # --------------------------------------------------------
    # Ensure valid values
    # --------------------------------------------------------

    anomaly_map = torch.nan_to_num(
        anomaly_map,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Decoder output may contain positive and negative values.
    #
    # Anomaly magnitude should not depend on sign.
    # --------------------------------------------------------

    magnitude = torch.abs(
        anomaly_map
    )

    flattened = magnitude.flatten()

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    mean_value = flattened.mean()

    percentile_value = torch.quantile(
        flattened,
        0.95
    )

    max_value = flattened.max()

    # --------------------------------------------------------
    # Combined anomaly score
    #
    # Weights:
    #
    # 30% mean
    # 50% upper percentile
    # 20% maximum
    # --------------------------------------------------------

    raw_score = (
        0.30 * mean_value
        + 0.50 * percentile_value
        + 0.20 * max_value
    )

    return float(
        raw_score.item()
    )


# ============================================================
# MAP NORMALIZATION
# ============================================================

def normalize_anomaly_map(anomaly_map):
    """
    Normalize anomaly map to [0, 1] for pixel-level
    visualization and spatial evidence.

    This normalization is performed per image.
    """

    anomaly_map = torch.abs(
        anomaly_map
    )

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    difference = maximum - minimum

    if difference <= 1e-8:

        return torch.zeros_like(
            anomaly_map
        )

    normalized = (
        anomaly_map - minimum
    ) / difference

    return normalized.clamp(
        0.0,
        1.0
    )


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_image(model, image):

    with torch.no_grad():

        anomaly_map, image_logit = (
            model.forward_with_score(
                [image]
            )
        )

        processed_map = (
            prepare_anomaly_map(
                anomaly_map
            )
        )

        image_score = torch.sigmoid(
            image_logit
        )

        score = float(
            image_score
            .detach()
            .cpu()
            .flatten()[0]
            .item()
        )

    decision = make_decision(
        anomaly_score=score,
        anomaly_map=processed_map,
        score_threshold=ANOMALY_THRESHOLD,
        map_threshold=MAP_THRESHOLD,
        spatial_ratio_threshold=SPATIAL_RATIO_THRESHOLD,
    )

    decision["anomaly_map"] = processed_map

    return decision

# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AnomalyVFM+ FINAL INFERENCE")
    print("=" * 60)

    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    image_path = input(
        "Enter image path: "
    ).strip()

    if not image_path:

        raise ValueError(
            "Image path cannot be empty."
        )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    print()
    print("Image loaded:")
    print(Path(image_path))

    print(
        "Original image size:",
        image.size
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    print()
    print(
        "Running AnomalyVFM+ inference..."
    )

    result = predict_image(
        model,
        image,
    )

    # --------------------------------------------------------
    # DISPLAY DECISION
    # --------------------------------------------------------

    print_decision(
        result
    )

    # --------------------------------------------------------
    # RAW ANOMALY MAP
    # --------------------------------------------------------

    print()
    print("ANOMALY MAP")
    print("-" * 60)

    anomaly_map = result[
        "anomaly_map"
    ]

    print(
        "Shape:",
        tuple(anomaly_map.shape)
    )

    print(
        "Minimum:",
        f"{anomaly_map.min().item():.6f}"
    )

    print(
        "Maximum:",
        f"{anomaly_map.max().item():.6f}"
    )

    print(
        "Mean:",
        f"{anomaly_map.mean().item():.6f}"
    )

    # --------------------------------------------------------
    # NORMALIZED MAP
    # --------------------------------------------------------

    anomaly_map = result[
        "anomaly_map"
    ]

    print()
    print("ANOMALY MAP")
    print("-" * 60)

    print(
        "Minimum:",
        f"{anomaly_map.min().item():.6f}"
    )

    print(
        "Maximum:",
        f"{anomaly_map.max().item():.6f}"
    )

    print(
        "Mean:",
        f"{anomaly_map.mean().item():.6f}"
    )

    # --------------------------------------------------------
    # FINAL USER-FACING RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Prediction   : "
        f"{result['prediction']}"
    )

    print(
        f"Anomaly score: "
        f"{result['anomaly_score']:.6f}"
    )

    print(
        f"Threshold    : "
        f"{result['score_threshold']:.6f}"
    )

    print(
        f"Pixel ratio  : "
        f"{result['anomalous_pixel_ratio']:.6f}"
    )

    print(
        f"Reason       : "
        f"{result['reason']}"
    )

    print("=" * 60)

    print()
    print("=== INFERENCE: SUCCESS ===")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()