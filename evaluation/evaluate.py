from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from evaluation.metrics import (
    image_level_auroc,
    pixel_level_auroc,
)

from data.processor_alignment import align_mask_to_processor


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

TEST_DIR = Path(
    "data/raw/mvtec/bottle/test"
)

MASK_DIR = Path(
    "data/raw/mvtec/bottle/ground_truth"
)

DEVICE = torch.device("cpu")

EVALUATION_SIZE = 224


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading AnomalyVFM+...")

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# COLLECT TEST SAMPLES
# ============================================================

def collect_test_samples():

    samples = []

    good_dir = TEST_DIR / "good"

    if not good_dir.exists():
        raise FileNotFoundError(good_dir)

    # Normal images

    for image_path in sorted(good_dir.glob("*.png")):

        samples.append({
            "image_path": image_path,
            "label": 0,
            "mask_path": None,
            "category": "good",
        })

    # Anomalous images

    anomaly_categories = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_dir() and path.name != "good"
    )

    for category_dir in anomaly_categories:

        category = category_dir.name

        for image_path in sorted(
            category_dir.glob("*.png")
        ):

            mask_path = (
                MASK_DIR
                / category
                / f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"Missing ground-truth mask: {mask_path}"
                )

            samples.append({
                "image_path": image_path,
                "label": 1,
                "mask_path": mask_path,
                "category": category,
            })

    if not samples:
        raise RuntimeError("No test images found.")

    return samples


# ============================================================
# PREPARE ANOMALY MAP
# ============================================================

def prepare_anomaly_map(
    anomaly_map,
    target_size=(224, 224),
):

    anomaly_map = torch.as_tensor(
        anomaly_map
    ).float()

    if anomaly_map.ndim == 2:

        anomaly_map = anomaly_map[
            None, None
        ]

    elif anomaly_map.ndim == 3:

        anomaly_map = anomaly_map.unsqueeze(0)

    if anomaly_map.ndim != 4:

        raise ValueError(
            f"Unexpected anomaly map shape: {anomaly_map.shape}"
        )

    anomaly_map = torch.sigmoid(
        anomaly_map
    )

    anomaly_map = F.interpolate(
        anomaly_map,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return anomaly_map.squeeze(0).squeeze(0)


# ============================================================
# LOAD AND ALIGN GROUND-TRUTH MASK
# ============================================================

def load_aligned_mask(
    mask_path,
    processor,
):

    mask_image = Image.open(
        mask_path
    ).convert("L")

    mask = align_mask_to_processor(
        mask_image,
        processor,
    )

    mask = (
        mask > 0
    ).float()

    return mask.squeeze(0)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("ANOMALYVFM+ CORRECTED MVTec BOTTLE EVALUATION")
    print("=" * 70)

    model = load_model()

    processor = model.foundation.processor

    samples = collect_test_samples()

    normal_count = sum(
        sample["label"] == 0
        for sample in samples
    )

    anomaly_count = sum(
        sample["label"] == 1
        for sample in samples
    )

    print()
    print("Total images:", len(samples))
    print("Normal images:", normal_count)
    print("Anomalous images:", anomaly_count)

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    image_scores = []
    image_labels = []

    anomaly_maps = []
    ground_truth_masks = []

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        image_path = sample["image_path"]

        label = sample["label"]

        image = Image.open(
            image_path
        ).convert("RGB")

        with torch.inference_mode():

            anomaly_map, image_logit = (
                model.forward_with_score(
                    [image]
                )
            )

            image_score = torch.sigmoid(
                image_logit
            ).reshape(-1)[0].item()

            prediction_map = prepare_anomaly_map(
                anomaly_map,
                target_size=(
                    EVALUATION_SIZE,
                    EVALUATION_SIZE,
                ),
            )

        # Image-level predictions

        image_scores.append(image_score)
        image_labels.append(label)

        # Pixel-level ground truth

        if sample["mask_path"] is not None:

            gt_mask = load_aligned_mask(
                sample["mask_path"],
                processor,
            )

        else:

            gt_mask = torch.zeros(
                EVALUATION_SIZE,
                EVALUATION_SIZE,
            )

        if gt_mask.shape != prediction_map.shape:

            raise ValueError(
                f"Mask/prediction mismatch: "
                f"{gt_mask.shape} vs {prediction_map.shape}"
            )

        anomaly_maps.append(
            prediction_map.cpu()
        )

        ground_truth_masks.append(
            gt_mask.cpu()
        )

        if index % 25 == 0:

            print(
                f"Processed {index}/{len(samples)} images"
            )

    # ========================================================
    # IMAGE AUROC
    # ========================================================

    image_auroc = image_level_auroc(
        image_scores,
        image_labels,
    )

    # ========================================================
    # PIXEL AUROC
    # ========================================================

    pixel_auroc = pixel_level_auroc(
        torch.stack(anomaly_maps),
        torch.stack(ground_truth_masks),
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("CORRECTED MVTec BOTTLE RESULTS")
    print("=" * 70)

    print(
        f"Image AUROC: {image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC: {pixel_auroc:.4f}"
    )

    print("=" * 70)

    print()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()