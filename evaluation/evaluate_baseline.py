import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.decoder.image_predictor import ImagePredictor
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.foundation.dinov2 import DINOv2
from evaluation.metrics import image_level_auroc, pixel_level_auroc


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/baseline/checkpoint.pt"
)

DATASET_ROOT = Path(
    "data/raw/mvtec/bottle/test"
)

GROUND_TRUTH_ROOT = Path(
    "data/raw/mvtec/bottle/ground_truth"
)

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# BASELINE MODEL
# ============================================================

class BaselineModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        foundation = DINOv2()
        foundation.load()

        self.foundation = foundation
        self.foundation_model = foundation.model

        for parameter in self.foundation_model.parameters():
            parameter.requires_grad = False

        self.foundation_model.eval()

        self.decoder = AnomalyDecoder(
            in_channels=768
        )

        self.predictor = ImagePredictor(
            input_dim=768
        )

    def forward(self, image):

        inputs = self.foundation.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.foundation_model(
                **inputs,
                output_hidden_states=True
            )

        # CLS token
        summary = outputs.last_hidden_state[:, 0, :]

        # Final patch features
        patch_tokens = (
            outputs.last_hidden_state[:, 1:, :]
        )

        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        channels = patch_tokens.shape[2]

        grid_size = int(
            num_patches ** 0.5
        )

        feature_map = patch_tokens.reshape(
            batch_size,
            grid_size,
            grid_size,
            channels
        )

        feature_map = feature_map.permute(
            0,
            3,
            1,
            2
        )

        anomaly_map = self.decoder(
            feature_map
        )

        image_logit = self.predictor(
            summary
        )

        return anomaly_map, image_logit


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = Image.open(path).convert("RGB")

    return image


# ============================================================
# GROUND-TRUTH MASK
# ============================================================

def load_ground_truth(
    category,
    image_name
):

    if category == "good":
        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8
        )

    stem = Path(image_name).stem

    mask_path = (
        GROUND_TRUTH_ROOT
        / category
        / f"{stem}_mask.png"
    )

    if not mask_path.exists():

        raise FileNotFoundError(
            f"Ground-truth mask not found: {mask_path}"
        )

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.array(mask)

    mask = (
        mask > 0
    ).astype(np.uint8)

    return mask


# ============================================================
# COLLECT TEST IMAGES
# ============================================================

def collect_test_images():

    samples = []

    # Normal images
    good_dir = DATASET_ROOT / "good"

    for image_path in sorted(
        good_dir.glob("*.png")
    ):

        samples.append(
            (
                "good",
                image_path
            )
        )

    # Anomalous categories
    for category in sorted(
        DATASET_ROOT.iterdir()
    ):

        if not category.is_dir():
            continue

        if category.name == "good":
            continue

        for image_path in sorted(
            category.glob("*.png")
        ):

            samples.append(
                (
                    category.name,
                    image_path
                )
            )

    return samples


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BASELINE EVALUATION")
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
    # Load checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BaselineModel()

    model = model.to(DEVICE)

    # Only decoder + predictor need checkpoint weights.
    model.decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    model.predictor.load_state_dict(
        checkpoint["predictor_state_dict"]
    )

    model.eval()

    print(
        "Baseline checkpoint loaded."
    )

    # --------------------------------------------------------
    # Test samples
    # --------------------------------------------------------

    samples = collect_test_images()

    print(
        "Total test images:",
        len(samples)
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    image_scores = []
    image_labels = []

    pixel_predictions = []
    pixel_ground_truth = []

    normal_count = 0
    anomaly_count = 0

    for index, (
        category,
        image_path
    ) in enumerate(samples):

        image = load_image(
            image_path
        )

        with torch.no_grad():

            anomaly_map, image_logit = model(
                [image]
            )

            image_score = torch.sigmoid(
                image_logit
            ).item()

            pixel_map = torch.sigmoid(
                anomaly_map
            )[0, 0]

            pixel_map = torch.nn.functional.interpolate(
                pixel_map.unsqueeze(0).unsqueeze(0),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )[0, 0]

        ground_truth = load_ground_truth(
            category,
            image_path.name
        )

        # ----------------------------------------------------
        # Image-level labels
        # ----------------------------------------------------

        if category == "good":

            image_label = 0
            normal_count += 1

        else:

            image_label = 1
            anomaly_count += 1

        image_scores.append(
            image_score
        )

        image_labels.append(
            image_label
        )

        # ----------------------------------------------------
        # Pixel-level predictions
        # ----------------------------------------------------

        pixel_predictions.append(
            pixel_map.cpu().numpy().flatten()
        )

        pixel_ground_truth.append(
            ground_truth.flatten()
        )

        if (
            (index + 1) % 10 == 0
            or index == len(samples) - 1
        ):

            print(
                f"Processed "
                f"{index + 1}/{len(samples)}"
            )

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    image_scores = np.array(
        image_scores
    )

    image_labels = np.array(
        image_labels
    )

    pixel_predictions = np.concatenate(
        pixel_predictions
    )

    pixel_ground_truth = np.concatenate(
        pixel_ground_truth
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    image_auroc = image_level_auroc(
        image_scores,
        image_labels
    )

    pixel_auroc = pixel_level_auroc(
        pixel_predictions,
        pixel_ground_truth
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(
        f"Test images      : {len(samples)}"
    )

    print(
        f"Normal images    : {normal_count}"
    )

    print(
        f"Anomalous images : {anomaly_count}"
    )

    print()

    print(
        f"Image AUROC      : {image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC      : {pixel_auroc:.4f}"
    )

    print("=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()