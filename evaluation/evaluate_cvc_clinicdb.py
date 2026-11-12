from pathlib import Path
import json
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

# Ensure imports work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.cvc_clinicdb import CVCClinicDBTestDataset
from models.anomaly_vfm_plus import AnomalyVFMPlus


CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "plus" / "checkpoint.pt"
)

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "medical" / "cvc_clinicdb"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_PATH = RESULTS_DIR / "cvc_clinicdb_results.json"

DEVICE = torch.device("cpu")

# Start with 10 images to verify inference and evaluation.
# Change to None after the initial test to evaluate all images.
MAX_IMAGES = None


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading AnomalyVFM+ model...")

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


def prepare_anomaly_map(anomaly_map, target_size):
    """
    Convert the model output into a 2D anomaly-score map.

    The sigmoid is appropriate if the decoder returns logits.
    """
    anomaly_map = anomaly_map.detach().float().cpu()

    if anomaly_map.ndim == 2:
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    elif anomaly_map.ndim == 3:
        anomaly_map = anomaly_map.unsqueeze(1)
    elif anomaly_map.ndim != 4:
        raise ValueError(
            f"Unexpected anomaly-map shape: {tuple(anomaly_map.shape)}"
        )

    anomaly_map = torch.sigmoid(anomaly_map)

    anomaly_map = F.interpolate(
        anomaly_map,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return anomaly_map[0, 0].numpy()


def prepare_mask(mask, target_size):
    """
    Resize the binary ground-truth mask using nearest-neighbor
    interpolation to preserve its binary labels.
    """
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
    elif isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D ground-truth mask, got {mask.shape}"
        )

    binary_mask = (mask > 0.5).astype(np.uint8)

    mask_image = Image.fromarray(binary_mask * 255)

    mask_image = mask_image.resize(
        (target_size[1], target_size[0]),
        resample=Image.Resampling.NEAREST,
    )

    return (np.array(mask_image) > 127).astype(np.uint8)


def evaluate():
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"CVC-ClinicDB dataset not found: {DATASET_PATH}"
        )

    print("Loading CVC-ClinicDB dataset...")

    dataset = CVCClinicDBTestDataset(
        path=DATASET_PATH,
        max_samples=MAX_IMAGES,
    )

    if len(dataset) == 0:
        raise ValueError("The CVC-ClinicDB dataset is empty.")

    model = load_model()

    image_labels = []
    image_scores = []

    all_pixel_scores = []
    all_pixel_masks = []

    samples = []

    print("\nRunning CVC-ClinicDB evaluation...")
    print(f"Device: {DEVICE}")
    print(f"Images: {len(dataset)}")

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]

            image_path = DATASET_PATH / sample["path"]

            # The model expects PIL images. Its own processor
            # handles the model-specific preprocessing.
            with Image.open(image_path) as source:
                pil_image = source.convert("RGB")

            anomaly_map, image_logit = model.forward_with_score(
                [pil_image]
            )

            image_score = torch.sigmoid(
                image_logit
            ).reshape(-1)[0].item()

            target_size = tuple(sample["mask"].shape[-2:])

            prediction_map = prepare_anomaly_map(
                anomaly_map,
                target_size=target_size,
            )

            ground_truth = prepare_mask(
                sample["mask"],
                target_size=target_size,
            )

            label = int(ground_truth.any())

            image_labels.append(label)
            image_scores.append(float(image_score))

            all_pixel_scores.append(prediction_map.reshape(-1))
            all_pixel_masks.append(ground_truth.reshape(-1))

            samples.append({
                "path": str(sample["path"]),
                "label": label,
                "image_score": float(image_score),
                "image_height": int(target_size[0]),
                "image_width": int(target_size[1]),
                "annotated_pixels": int(ground_truth.sum()),
            })

            if (
                (index + 1) % 2 == 0
                or index + 1 == len(dataset)
            ):
                print(
                    f"Processed {index + 1}/{len(dataset)} images"
                )

    # Image-level AUROC requires both classes.
    image_auroc = None

    if len(set(image_labels)) >= 2:
        image_auroc = float(
            roc_auc_score(image_labels, image_scores)
        )
    else:
        print(
            "\nImage-level AUROC unavailable: "
            "the evaluated subset contains only one image class."
        )

    # Pixel-level AUROC compares predicted anomaly scores with
    # the ground-truth masks.
    pixel_scores = np.concatenate(all_pixel_scores)
    pixel_masks = np.concatenate(all_pixel_masks)

    pixel_auroc = None

    if len(np.unique(pixel_masks)) >= 2:
        pixel_auroc = float(
            roc_auc_score(pixel_masks, pixel_scores)
        )
    else:
        print(
            "\nPixel-level AUROC unavailable: "
            "ground-truth pixels contain only one class."
        )

    results = {
        "dataset": "CVC-ClinicDB",
        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),
        "evaluation_type": (
            "cross-domain medical image anomaly evaluation"
        ),
        "device": str(DEVICE),
        "num_images": len(dataset),
        "image_level_auroc": image_auroc,
        "pixel_level_auroc": pixel_auroc,
        "total_pixels_evaluated": int(pixel_masks.size),
        "ground_truth_positive_pixels": int(pixel_masks.sum()),
        "ground_truth_negative_pixels": int(
            pixel_masks.size - pixel_masks.sum()
        ),
        "warning": (
            "The checkpoint was trained on MVTec Bottle. "
            "This experiment evaluates cross-domain generalization "
            "on CVC-ClinicDB and does not establish medical "
            "diagnostic performance. "
            "The anomaly-map sigmoid assumes the decoder outputs logits; "
            "verify this against the decoder implementation."
        ),
        "samples": samples,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n========== CVC-CLINICDB RESULTS ==========")
    print(f"Images evaluated : {len(dataset)}")
    print(f"Image AUROC      : {image_auroc}")
    print(f"Pixel AUROC      : {pixel_auroc}")
    print(f"Pixels evaluated : {pixel_masks.size}")
    print(f"Results saved to : {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()