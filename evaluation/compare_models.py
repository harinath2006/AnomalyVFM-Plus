from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.decoder.calibration import compute_adaptive_score


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cpu")

DATASET_ROOT = Path("data/raw/mvtec/bottle")
TEST_DIR = DATASET_ROOT / "test"
MASK_DIR = DATASET_ROOT / "ground_truth"

BASELINE_CHECKPOINT = Path(
    "experiments/baseline/checkpoint.pt"
)

LORA_CHECKPOINT = Path(
    "experiments/lora/checkpoint.pt"
)

PLUS_CHECKPOINT = Path(
    "experiments/plus/checkpoint.pt"
)

RESULT_DIR = Path("results/evaluation")

IMAGE_SIZE = 224


# ============================================================
# Dataset
# ============================================================

def collect_test_samples():

    samples = []

    categories = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    for category_dir in categories:

        category = category_dir.name

        image_paths = sorted(
            category_dir.glob("*.png")
        )

        for image_path in image_paths:

            if category == "good":

                label = 0
                mask_path = None

            else:

                label = 1

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
                    "label": label,
                    "mask_path": mask_path,
                    "category": category
                }
            )

    return samples


# ============================================================
# Image loading
# ============================================================

def load_image(path):

    return Image.open(path).convert("RGB")


def load_mask(path):

    if path is None:

        return torch.zeros(
            1,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    mask = Image.open(path).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.NEAREST
    )

    mask = transforms.ToTensor()(mask)

    mask = (mask > 0).float()

    return mask


# ============================================================
# Baseline model
# ============================================================

def create_baseline_model():

    print("\nCreating BASELINE model...")

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    foundation = DINOv2()

    foundation.load()

    foundation.model.to(DEVICE)

    foundation.model.eval()

    decoder = AnomalyDecoder(
        in_channels=768
    ).to(DEVICE)

    decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    decoder.eval()

    print(
        "Baseline checkpoint step:",
        checkpoint["step"]
    )

    print("Baseline model restored.")

    return foundation, decoder


# ============================================================
# LoRA model
# ============================================================

def create_lora_model():

    print("\nCreating LoRA model...")

    checkpoint = torch.load(
        LORA_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    foundation = DINOv2()

    foundation.load()

    injected = inject_lora(
        foundation.model,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ),
        rank=4,
        alpha=8
    )

    print(
        "LoRA layers:",
        injected
    )

    foundation.model.load_state_dict(
        checkpoint["lora_state"],
        strict=False
    )

    foundation.model.to(DEVICE)

    foundation.model.eval()

    decoder = AnomalyDecoder(
        in_channels=768
    ).to(DEVICE)

    decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    decoder.eval()

    print(
        "LoRA checkpoint step:",
        checkpoint["step"]
    )

    print("LoRA model restored.")

    return foundation, decoder


# ============================================================
# AnomalyVFM+ model
# ============================================================

def create_plus_model():

    print("\nCreating AnomalyVFM+ model...")

    checkpoint = torch.load(
        PLUS_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model = AnomalyVFMPlus()

    model.foundation_model.load_state_dict(
        checkpoint["lora_state"],
        strict=False
    )

    model.fusion.load_state_dict(
        checkpoint["fusion_state"]
    )

    model.decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    model.to(DEVICE)

    model.eval()

    print(
        "AnomalyVFM+ checkpoint step:",
        checkpoint["step"]
    )

    print("AnomalyVFM+ model restored.")

    return model


# ============================================================
# Baseline prediction
# ============================================================

def predict_baseline(
    foundation,
    decoder,
    images
):

    with torch.no_grad():

        features = foundation.extract_features(
            images,
            trainable=False
        )

        logits = decoder(features)

        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        scores = compute_adaptive_score(
            logits
        )

    return logits, scores


# ============================================================
# LoRA prediction
# ============================================================

def predict_lora(
    foundation,
    decoder,
    images
):

    with torch.no_grad():

        features = foundation.extract_features(
            images,
            trainable=False
        )

        logits = decoder(features)

        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        scores = compute_adaptive_score(
            logits
        )

    return logits, scores


# ============================================================
# AnomalyVFM+ prediction
# ============================================================

def predict_plus(
    model,
    images
):

    with torch.no_grad():

        logits = model(
            images
        )

        logits = F.interpolate(
            logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        scores = model.compute_anomaly_score(
            logits
        )

    return logits, scores


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    scores,
    labels,
    anomaly_maps,
    masks
):

    scores = (
        scores
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        labels
        .detach()
        .cpu()
        .numpy()
    )

    anomaly_maps = (
        anomaly_maps
        .detach()
        .cpu()
        .numpy()
    )

    masks = (
        masks
        .detach()
        .cpu()
        .numpy()
    )

    image_auroc = float(
        roc_auc_score(
            labels,
            scores
        )
    )

    pixel_scores = anomaly_maps.reshape(-1)

    pixel_labels = masks.reshape(-1)

    pixel_auroc = float(
        roc_auc_score(
            pixel_labels,
            pixel_scores
        )
    )

    return image_auroc, pixel_auroc


# ============================================================
# Save result
# ============================================================

def save_result(
    model_name,
    image_auroc,
    pixel_auroc,
    num_images
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {
        "model": model_name,
        "dataset": "MVTec Bottle",
        "num_images": num_images,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc
    }

    import json

    output_path = (
        RESULT_DIR
        / f"{model_name.lower().replace(' ', '_')}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        "Result saved:",
        output_path
    )


# ============================================================
# Main evaluation
# ============================================================

def main():

    print(
        "=================================================="
    )

    print(
        " AnomalyVFM+ EXPERIMENTAL EVALUATION"
    )

    print(
        "=================================================="
    )

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    samples = collect_test_samples()

    print(
        "\nTotal test images:",
        len(samples)
    )

    labels = torch.tensor(
        [
            sample["label"]
            for sample in samples
        ],
        dtype=torch.long
    )

    masks = torch.stack(
        [
            load_mask(
                sample["mask_path"]
            )
            for sample in samples
        ]
    )

    images = [
        load_image(
            sample["image_path"]
        )
        for sample in samples
    ]

    print(
        "Labels:",
        labels.shape
    )

    print(
        "Masks:",
        masks.shape
    )

    # ========================================================
    # BASELINE
    # ========================================================

    foundation, decoder = (
        create_baseline_model()
    )

    baseline_maps, baseline_scores = (
        predict_baseline(
            foundation,
            decoder,
            images
        )
    )

    baseline_image_auc, baseline_pixel_auc = (
        calculate_metrics(
            baseline_scores,
            labels,
            baseline_maps,
            masks
        )
    )

    print(
        "\nBASELINE RESULTS"
    )

    print(
        "Image AUROC:",
        baseline_image_auc
    )

    print(
        "Pixel AUROC:",
        baseline_pixel_auc
    )

    save_result(
        "baseline",
        baseline_image_auc,
        baseline_pixel_auc,
        len(samples)
    )

    # ========================================================
    # LoRA
    # ========================================================

    foundation, decoder = (
        create_lora_model()
    )

    lora_maps, lora_scores = (
        predict_lora(
            foundation,
            decoder,
            images
        )
    )

    lora_image_auc, lora_pixel_auc = (
        calculate_metrics(
            lora_scores,
            labels,
            lora_maps,
            masks
        )
    )

    print(
        "\nLORA RESULTS"
    )

    print(
        "Image AUROC:",
        lora_image_auc
    )

    print(
        "Pixel AUROC:",
        lora_pixel_auc
    )

    save_result(
        "lora",
        lora_image_auc,
        lora_pixel_auc,
        len(samples)
    )

    # ========================================================
    # AnomalyVFM+
    # ========================================================

    plus_model = (
        create_plus_model()
    )

    plus_maps, plus_scores = (
        predict_plus(
            plus_model,
            images
        )
    )

    plus_image_auc, plus_pixel_auc = (
        calculate_metrics(
            plus_scores,
            labels,
            plus_maps,
            masks
        )
    )

    print(
        "\nANOMALYVFM+ RESULTS"
    )

    print(
        "Image AUROC:",
        plus_image_auc
    )

    print(
        "Pixel AUROC:",
        plus_pixel_auc
    )

    save_result(
        "anomalyvfm_plus",
        plus_image_auc,
        plus_pixel_auc,
        len(samples)
    )

    # ========================================================
    # Final comparison
    # ========================================================

    print(
        "\n=================================================="
    )

    print(
        " FINAL EXPERIMENT COMPARISON"
    )

    print(
        "=================================================="
    )

    print(
        f"{'Model':<20}"
        f"{'Image AUROC':>15}"
        f"{'Pixel AUROC':>15}"
    )

    print(
        "-" * 50
    )

    print(
        f"{'Baseline':<20}"
        f"{baseline_image_auc:>15.4f}"
        f"{baseline_pixel_auc:>15.4f}"
    )

    print(
        f"{'LoRA':<20}"
        f"{lora_image_auc:>15.4f}"
        f"{lora_pixel_auc:>15.4f}"
    )

    print(
        f"{'AnomalyVFM+':<20}"
        f"{plus_image_auc:>15.4f}"
        f"{plus_pixel_auc:>15.4f}"
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    image_improvement = (
        plus_image_auc
        - baseline_image_auc
    )

    pixel_improvement = (
        plus_pixel_auc
        - baseline_pixel_auc
    )

    print(
        "\nAnomalyVFM+ improvement over baseline:"
    )

    print(
        "Image AUROC:",
        f"{image_improvement:+.4f}"
    )

    print(
        "Pixel AUROC:",
        f"{pixel_improvement:+.4f}"
    )

    print(
        "\n=== EXPERIMENTAL EVALUATION: SUCCESS ==="
    )


if __name__ == "__main__":
    main()