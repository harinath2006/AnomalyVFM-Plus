from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from evaluation.metrics import image_level_auroc, pixel_level_auroc


# ============================================================
# PATHS
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


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("Loading AnomalyVFM+...")

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("Checkpoint loaded.")

    return model


# ============================================================
# DATASET COLLECTION
# ============================================================

def collect_test_samples():

    samples = []

    # --------------------------------------------------------
    # Normal images
    # --------------------------------------------------------

    good_dir = TEST_DIR / "good"

    good_images = sorted(
        good_dir.glob("*.png")
    )

    for image_path in good_images:

        samples.append(
            {
                "image_path": image_path,
                "label": 0,
                "mask_path": None,
                "category": "good",
            }
        )

    # --------------------------------------------------------
    # Anomaly images
    # --------------------------------------------------------

    anomaly_categories = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_dir()
        and path.name != "good"
    )

    for category_dir in anomaly_categories:

        category = category_dir.name

        image_paths = sorted(
            category_dir.glob("*.png")
        )

        for image_path in image_paths:

            mask_path = (
                MASK_DIR
                / category
                / f"{image_path.stem}_mask.png"
            )

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"Missing mask: {mask_path}"
                )

            samples.append(
                {
                    "image_path": image_path,
                    "label": 1,
                    "mask_path": mask_path,
                    "category": category,
                }
            )

    return samples


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = torch.tensor(
        list(mask.getdata()),
        dtype=torch.float32,
    )

    mask = mask.reshape(
        1,
        900,
        900,
    )

    mask = (
        mask > 0
    ).float()

    return mask


# ============================================================
# RESIZE PREDICTION
# ============================================================

def prepare_anomaly_map(
    anomaly_map,
    target_size=(224, 224),
):

    anomaly_map = torch.as_tensor(
        anomaly_map
    ).float()

    if anomaly_map.ndim == 2:

        anomaly_map = (
            anomaly_map
            .unsqueeze(0)
            .unsqueeze(0)
        )

    elif anomaly_map.ndim == 3:

        anomaly_map = (
            anomaly_map
            .unsqueeze(0)
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

    return anomaly_map.squeeze()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("ANOMALYVFM+ FULL MVTec BOTTLE EVALUATION")
    print("=" * 70)

    model = load_model()

    samples = collect_test_samples()

    print()
    print(
        "Total test images:",
        len(samples),
    )

    normal_count = sum(
        sample["label"] == 0
        for sample in samples
    )

    anomaly_count = sum(
        sample["label"] == 1
        for sample in samples
    )

    print(
        "Normal images:",
        normal_count,
    )

    print(
        "Anomaly images:",
        anomaly_count,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    image_scores = []

    image_labels = []

    anomaly_maps = []

    ground_truth_masks = []

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for index, sample in enumerate(
        samples,
        start=1,
    ):

        image_path = sample[
            "image_path"
        ]

        label = sample[
            "label"
        ]

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
            # IMAGE SCORE
            # ------------------------------------------------

            image_score = torch.sigmoid(
                image_logit
            )[0].item()

            # ------------------------------------------------
            # PIXEL MAP
            # ------------------------------------------------

            prediction_map = (
                prepare_anomaly_map(
                    anomaly_map
                )
            )

        image_scores.append(
            image_score
        )

        image_labels.append(
            label
        )

        # ----------------------------------------------------
        # Only anomalous samples have GT masks
        # ----------------------------------------------------

        if sample["mask_path"] is not None:

            # Anomalous image: load real ground-truth mask
            gt_mask = load_mask(
                sample["mask_path"]
            )

            gt_mask = F.interpolate(
                gt_mask.unsqueeze(0),
                size=prediction_map.shape,
                mode="nearest",
            ).squeeze()

        else:

            # Normal image: ground-truth mask is completely zero
            gt_mask = torch.zeros_like(
                prediction_map
            )


        # Include BOTH normal and anomalous images
        anomaly_maps.append(
            prediction_map
        )

        ground_truth_masks.append(
            gt_mask
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

    if anomaly_maps:

        pixel_auroc = pixel_level_auroc(
            torch.stack(
                anomaly_maps
            ),
            torch.stack(
                ground_truth_masks
            ),
        )

    else:

        pixel_auroc = float("nan")

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL MVTec BOTTLE RESULTS")
    print("=" * 70)

    print(
        f"Image AUROC : {image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC : {pixel_auroc:.4f}"
    )

    print("=" * 70)

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()