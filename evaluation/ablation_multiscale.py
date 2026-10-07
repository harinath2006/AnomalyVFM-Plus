from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.decoder.image_predictor import ImagePredictor
from models.anomaly_vfm_plus import AnomalyVFMPlus
from evaluation.metrics import (
    image_level_auroc,
    pixel_level_auroc,
)


# ============================================================
# CONFIGURATION
# ============================================================

PLUS_CHECKPOINT = Path(
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


# Possible locations for the single-scale LoRA checkpoint.
# The script will automatically use the first one that exists.
SINGLE_SCALE_CHECKPOINT_CANDIDATES = [
    Path("experiments/ablation/single_scale/checkpoint.pt"),
    Path("experiments/ablation/single_scale_checkpoint.pt"),
    Path("experiments/single_scale/checkpoint.pt"),
    Path("experiments/lora/checkpoint.pt"),
    Path("experiments/plus/single_scale_checkpoint.pt"),
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_single_scale_checkpoint():
    """
    Find the existing single-scale LoRA checkpoint.
    """

    for path in SINGLE_SCALE_CHECKPOINT_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "\nCould not find the single-scale LoRA checkpoint.\n\n"
        "Checked:\n"
        + "\n".join(
            f"  - {path}"
            for path in SINGLE_SCALE_CHECKPOINT_CANDIDATES
        )
        + "\n\n"
        "If your checkpoint is stored somewhere else, "
        "change SINGLE_SCALE_CHECKPOINT_CANDIDATES "
        "at the top of this file."
    )


def load_checkpoint(path):
    """
    Load checkpoint safely.
    """

    print()
    print("Loading checkpoint:")
    print(path)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    print(
        "Checkpoint keys:",
        list(checkpoint.keys())
    )

    return checkpoint


def get_state_dict(checkpoint):
    """
    Extract a model state dictionary from several
    checkpoint formats used during development.
    """

    possible_keys = [
        "model_state_dict",
        "state_dict",
        "lora_state",
    ]

    for key in possible_keys:

        if key in checkpoint:

            print(
                f"Using checkpoint key: {key}"
            )

            return checkpoint[key]

    # Some checkpoints may themselves be state_dicts.
    if all(
        isinstance(key, str)
        for key in checkpoint.keys()
    ):

        tensor_values = [
            value
            for value in checkpoint.values()
            if torch.is_tensor(value)
        ]

        if tensor_values:

            print(
                "Using checkpoint directly as state_dict."
            )

            return checkpoint

    raise KeyError(
        "Could not find model state in checkpoint.\n"
        f"Available keys: {list(checkpoint.keys())}"
    )


def strip_prefix(
    state_dict,
    prefix
):
    """
    Remove a prefix from state-dict keys.
    """

    result = {}

    for key, value in state_dict.items():

        if key.startswith(prefix):

            new_key = key[len(prefix):]

            result[new_key] = value

    return result


def print_load_result(
    name,
    result
):

    print()
    print(
        f"{name} checkpoint loading:"
    )

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
            "First missing keys:"
        )

        for key in result.missing_keys[:10]:
            print(
                "  ",
                key
            )

    if result.unexpected_keys:

        print(
            "First unexpected keys:"
        )

        for key in result.unexpected_keys[:10]:
            print(
                "  ",
                key
            )


# ============================================================
# SINGLE-SCALE MODEL
# ============================================================

class SingleScaleLoRAModel(nn.Module):
    """
    Baseline for the multi-scale ablation.

    Architecture:

        DINOv2
           |
          LoRA
           |
      final-layer
       features
           |
        decoder
           |
      anomaly map

        CLS token
           |
      image predictor
           |
      image score
    """

    def __init__(
        self,
        lora_rank=4,
        lora_alpha=8
    ):

        super().__init__()

        # ----------------------------------------------------
        # DINOv2
        # ----------------------------------------------------

        foundation = DINOv2()

        foundation.load()

        self.foundation = foundation
        self.foundation_model = foundation.model

        # ----------------------------------------------------
        # LoRA
        # ----------------------------------------------------

        self.lora_layers = inject_lora(
            self.foundation_model,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ),
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = AnomalyDecoder(
            in_channels=768
        )

        # ----------------------------------------------------
        # Image predictor
        # ----------------------------------------------------

        self.predictor = ImagePredictor(
            input_dim=768
        )

    def forward(
        self,
        image
    ):

        inputs = self.foundation.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        outputs = self.foundation_model(
            **inputs,
            output_hidden_states=True
        )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        summary = (
            outputs.last_hidden_state[:, 0, :]
        )

        # ----------------------------------------------------
        # FINAL PATCH FEATURES
        # ----------------------------------------------------

        patch_tokens = (
            outputs.last_hidden_state[:, 1:, :]
        )

        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        channels = patch_tokens.shape[2]

        grid_size = int(
            num_patches ** 0.5
        )

        if (
            grid_size * grid_size
            != num_patches
        ):

            raise ValueError(
                f"Invalid patch count: "
                f"{num_patches}"
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

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        anomaly_map = self.decoder(
            feature_map
        )

        # ----------------------------------------------------
        # Image score
        # ----------------------------------------------------

        image_logit = self.predictor(
            summary
        )

        return (
            anomaly_map,
            image_logit
        )


# ============================================================
# LOAD SINGLE-SCALE MODEL
# ============================================================

def load_single_scale_model(
    checkpoint_path
):

    print()
    print("=" * 70)
    print("LOADING SINGLE-SCALE LoRA MODEL")
    print("=" * 70)

    model = SingleScaleLoRAModel()

    model = model.to(DEVICE)

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    # --------------------------------------------------------
    # New complete model checkpoint
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        result = model.load_state_dict(
            state_dict,
            strict=False
        )

        print_load_result(
            "Single-scale",
            result
        )

    else:

        # ----------------------------------------------------
        # Older checkpoint format
        # ----------------------------------------------------

        # LoRA
        if "lora_state" in checkpoint:

            lora_state = checkpoint[
                "lora_state"
            ]

            result = model.foundation_model.load_state_dict(
                lora_state,
                strict=False
            )

            print_load_result(
                "LoRA",
                result
            )

        else:

            state_dict = get_state_dict(
                checkpoint
            )

            # Try full model state first.
            result = model.load_state_dict(
                state_dict,
                strict=False
            )

            print_load_result(
                "Single-scale",
                result
            )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        if (
            "decoder_state_dict"
            in checkpoint
        ):

            result = model.decoder.load_state_dict(
                checkpoint[
                    "decoder_state_dict"
                ],
                strict=False
            )

            print_load_result(
                "Decoder",
                result
            )

        # ----------------------------------------------------
        # Predictor
        # ----------------------------------------------------

        if (
            "predictor_state_dict"
            in checkpoint
        ):

            result = model.predictor.load_state_dict(
                checkpoint[
                    "predictor_state_dict"
                ],
                strict=False
            )

            print_load_result(
                "Predictor",
                result
            )

    model.eval()

    print()
    print(
        "Single-scale LoRA layers:",
        model.lora_layers
    )

    print(
        "Single-scale model ready."
    )

    return model


# ============================================================
# LOAD ANOMALYVFM+ MODEL
# ============================================================

def load_plus_model():

    print()
    print("=" * 70)
    print("LOADING AnomalyVFM+ MODEL")
    print("=" * 70)

    model = AnomalyVFMPlus()

    model = model.to(DEVICE)

    checkpoint = load_checkpoint(
        PLUS_CHECKPOINT
    )

    # --------------------------------------------------------
    # Current checkpoint format
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        result = model.load_state_dict(
            state_dict,
            strict=False
        )

        print_load_result(
            "AnomalyVFM+",
            result
        )

    else:

        # ----------------------------------------------------
        # Older checkpoint format
        # ----------------------------------------------------

        state_dict = get_state_dict(
            checkpoint
        )

        result = model.load_state_dict(
            state_dict,
            strict=False
        )

        print_load_result(
            "AnomalyVFM+",
            result
        )

    model.eval()

    print()
    print(
        "AnomalyVFM+ model ready."
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
    # Defect categories
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


def load_ground_truth(
    category,
    image_name
):

    # --------------------------------------------------------
    # Normal image
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
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    samples,
    is_plus=False
):

    print()
    print("=" * 70)
    print(
        f"EVALUATING: {model_name}"
    )
    print("=" * 70)

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

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.no_grad():

            if is_plus:

                anomaly_map, image_logit = (
                    model.forward_with_score(
                        [image]
                    )
                )

            else:

                anomaly_map, image_logit = (
                    model(
                        [image]
                    )
                )

            # ------------------------------------------------
            # Image-level score
            # ------------------------------------------------

            image_score = torch.sigmoid(
                image_logit
            ).flatten()[0].item()

            # ------------------------------------------------
            # Pixel-level map
            # ------------------------------------------------

            pixel_map = torch.sigmoid(
                anomaly_map
            )

            pixel_map = F.interpolate(
                pixel_map,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                mode="bilinear",
                align_corners=False
            )

            pixel_map = pixel_map[
                0,
                0
            ]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth = load_ground_truth(
            category,
            image_path.name
        )

        # ----------------------------------------------------
        # Image label
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
        # Pixel data
        # ----------------------------------------------------

        pixel_predictions.append(
            pixel_map.cpu().numpy().flatten()
        )

        pixel_ground_truth.append(
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
    # Arrays
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

    print()
    print(
        f"{model_name} RESULTS"
    )

    print(
        f"Image AUROC : "
        f"{image_auroc:.4f}"
    )

    print(
        f"Pixel AUROC : "
        f"{pixel_auroc:.4f}"
    )

    return {
        "image_auroc": float(
            image_auroc
        ),
        "pixel_auroc": float(
            pixel_auroc
        ),
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ABLATION A: SINGLE-SCALE vs MULTI-SCALE")
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    print(
        "Dataset:",
        DATASET_ROOT
    )

    # --------------------------------------------------------
    # Find checkpoint
    # --------------------------------------------------------

    single_scale_checkpoint = (
        find_single_scale_checkpoint()
    )

    print()
    print(
        "Single-scale checkpoint:"
    )

    print(
        single_scale_checkpoint
    )

    print()
    print(
        "AnomalyVFM+ checkpoint:"
    )

    print(
        PLUS_CHECKPOINT
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not PLUS_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"AnomalyVFM+ checkpoint not found:\n"
            f"{PLUS_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    samples = collect_test_images()

    print()
    print(
        "Total test images:",
        len(samples)
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    single_scale_model = (
        load_single_scale_model(
            single_scale_checkpoint
        )
    )

    plus_model = load_plus_model()

    # --------------------------------------------------------
    # Evaluate single-scale
    # --------------------------------------------------------

    single_scale_results = evaluate_model(
        single_scale_model,
        "SINGLE-SCALE LoRA",
        samples,
        is_plus=False
    )

    # --------------------------------------------------------
    # Evaluate multi-scale
    # --------------------------------------------------------

    plus_results = evaluate_model(
        plus_model,
        "AnomalyVFM+ MULTI-SCALE",
        samples,
        is_plus=True
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    image_difference = (
        plus_results["image_auroc"]
        -
        single_scale_results["image_auroc"]
    )

    pixel_difference = (
        plus_results["pixel_auroc"]
        -
        single_scale_results["pixel_auroc"]
    )

    print()
    print()
    print("=" * 70)
    print("MULTI-SCALE ABLATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Model':<30}"
        f"{'Image AUROC':>15}"
        f"{'Pixel AUROC':>15}"
    )

    print("-" * 70)

    print(
        f"{'Single-scale LoRA':<30}"
        f"{single_scale_results['image_auroc']:>15.4f}"
        f"{single_scale_results['pixel_auroc']:>15.4f}"
    )

    print(
        f"{'AnomalyVFM+ Multi-scale':<30}"
        f"{plus_results['image_auroc']:>15.4f}"
        f"{plus_results['pixel_auroc']:>15.4f}"
    )

    print("-" * 70)

    print(
        f"{'Improvement':<30}"
        f"{image_difference:>15.4f}"
        f"{pixel_difference:>15.4f}"
    )

    print()

    if (
        single_scale_results["image_auroc"]
        != 0
    ):

        image_relative = (
            image_difference
            /
            single_scale_results["image_auroc"]
            *
            100
        )

        print(
            "Image AUROC relative change:"
            f" {image_relative:+.2f}%"
        )

    if (
        single_scale_results["pixel_auroc"]
        != 0
    ):

        pixel_relative = (
            pixel_difference
            /
            single_scale_results["pixel_auroc"]
            *
            100
        )

        print(
            "Pixel AUROC relative change:"
            f" {pixel_relative:+.2f}%"
        )

    print()

    print(
        "Test images:",
        len(samples)
    )

    print(
        "Normal:",
        single_scale_results[
            "normal_count"
        ]
    )

    print(
        "Anomalous:",
        single_scale_results[
            "anomaly_count"
        ]
    )

    print()
    print("=" * 70)
    print("MULTI-SCALE ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()