import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from models.foundation.dinov2 import DINOv2
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.decoder.image_predictor import ImagePredictor


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw/mvtec/bottle/train/good")
CHECKPOINT_PATH = Path("experiments/baseline/checkpoint.pt")

IMAGE_SIZE = 224
BATCH_SIZE = 2
TRAIN_STEPS = 100

LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 12

ANOMALY_PROBABILITY = 0.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATASET
# ============================================================

class BottleDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(Path(root).glob("*.png"))

        if not self.paths:
            raise RuntimeError(
                f"No PNG images found in: {root}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        tensor = (
            torch.from_numpy(
                np.array(image)
            )
            .float()
            .permute(2, 0, 1)
            / 255.0
        )

        return tensor


# ============================================================
# SYNTHETIC ANOMALY GENERATION
# ============================================================

def create_irregular_mask(batch_size, height, width, device):
    """
    Creates irregular ellipse/blob-like anomaly masks.
    """

    masks = torch.zeros(
        batch_size,
        1,
        height,
        width,
        device=device
    )

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij"
    )

    for b in range(batch_size):

        cx = random.randint(
            width // 5,
            4 * width // 5
        )

        cy = random.randint(
            height // 5,
            4 * height // 5
        )

        rx = random.randint(
            max(10, width // 12),
            max(11, width // 4)
        )

        ry = random.randint(
            max(10, height // 12),
            max(11, height // 4)
        )

        ellipse = (
            ((xx - cx).float() / rx) ** 2
            +
            ((yy - cy).float() / ry) ** 2
            <= 1.0
        )

        masks[b, 0] = ellipse.float()

    return masks


def create_synthetic_anomaly(images):
    """
    Creates synthetic anomalous images and corresponding masks.

    Returns:
        anomalous_images
        masks
    """

    batch_size, channels, height, width = images.shape

    masks = create_irregular_mask(
        batch_size,
        height,
        width,
        images.device
    )

    # Random anomaly appearance
    anomaly_type = random.choice(
        ["bright", "dark", "noise", "color"]
    )

    if anomaly_type == "bright":

        anomaly = torch.ones_like(images)

    elif anomaly_type == "dark":

        anomaly = torch.zeros_like(images)

    elif anomaly_type == "noise":

        anomaly = torch.rand_like(images)

    else:

        anomaly = torch.zeros_like(images)

        anomaly[:, 0] = 1.0
        anomaly[:, 1] = 0.15
        anomaly[:, 2] = 0.15

    # Small blur to avoid completely artificial sharp boundaries
    if random.random() < 0.5:

        anomaly_cpu = anomaly.detach().cpu()

        blurred = []

        for image in anomaly_cpu:

            pil = Image.fromarray(
                (
                    image.permute(1, 2, 0).numpy() * 255
                ).clip(0, 255).astype(np.uint8)
            )

            pil = pil.filter(
                ImageFilter.GaussianBlur(radius=1.5)
            )

            tensor = (
                torch.from_numpy(
                    np.array(pil)
                )
                .float()
                .permute(2, 0, 1)
                / 255.0
            )

            blurred.append(tensor)

        anomaly = torch.stack(
            blurred
        ).to(images.device)

    anomalous_images = (
        images * (1.0 - masks)
        +
        anomaly * masks
    )

    return anomalous_images.clamp(0.0, 1.0), masks


# ============================================================
# LOSSES
# ============================================================

def binary_focal_loss(
    logits,
    targets,
    alpha=0.75,
    gamma=2.0
):
    """
    Binary focal loss.
    """

    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    probabilities = torch.sigmoid(logits)

    pt = (
        probabilities * targets
        +
        (1.0 - probabilities) * (1.0 - targets)
    )

    alpha_factor = (
        alpha * targets
        +
        (1.0 - alpha) * (1.0 - targets)
    )

    focal_weight = (
        alpha_factor
        * (1.0 - pt).pow(gamma)
    )

    return (
        focal_weight * bce
    ).mean()


def segmentation_loss(predictions, masks):
    """
    Segmentation loss = focal loss + L1 loss.
    """

    if predictions.shape[-2:] != masks.shape[-2:]:

        masks = F.interpolate(
            masks,
            size=predictions.shape[-2:],
            mode="nearest"
        )

    focal = binary_focal_loss(
        predictions,
        masks
    )

    predicted_probability = torch.sigmoid(
        predictions
    )

    l1 = F.l1_loss(
        predicted_probability,
        masks
    )

    return focal + l1


# ============================================================
# DINOv2 BASELINE
# ============================================================

class BaselineModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Frozen DINOv2
        # ----------------------------------------------------

        foundation = DINOv2()
        foundation.load()

        self.foundation = foundation
        self.foundation_model = foundation.model

        # Freeze EVERYTHING in DINOv2.
        for parameter in self.foundation_model.parameters():
            parameter.requires_grad = False

        self.foundation_model.eval()

        # ----------------------------------------------------
        # Simple decoder
        # ----------------------------------------------------

        self.decoder = AnomalyDecoder(
            in_channels=768
        )

        # ----------------------------------------------------
        # Image-level predictor
        # ----------------------------------------------------

        self.predictor = ImagePredictor(
            input_dim=768
        )

    def extract_features(self, images):

        inputs = self.foundation.processor(
            images=images,
            return_tensors="pt"
        )

        # Move processor output to same device
        device = next(
            self.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.foundation_model(
                **inputs,
                output_hidden_states=True
            )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        summary = outputs.last_hidden_state[:, 0, :]

        # ----------------------------------------------------
        # FINAL DINOv2 PATCH FEATURES
        # ----------------------------------------------------

        tokens = outputs.last_hidden_state

        patch_tokens = tokens[:, 1:, :]

        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        channels = patch_tokens.shape[2]

        grid_size = int(
            num_patches ** 0.5
        )

        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"Invalid patch count: {num_patches}"
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

        return feature_map, summary

    def forward(self, images):

        feature_map, summary = self.extract_features(
            images
        )

        anomaly_map = self.decoder(
            feature_map
        )

        image_logit = self.predictor(
            summary
        )

        return anomaly_map, image_logit


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AnomalyVFM+ BASELINE TRAINING")
    print("=" * 70)
    print()

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Dataset:", DATA_DIR)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Training steps:", TRAIN_STEPS)
    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = BottleDataset(
        DATA_DIR
    )

    print(
        "Training images:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BaselineModel()

    model = model.to(device)

    # --------------------------------------------------------
    # Count trainable parameters
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "Total parameters:",
        total_count
    )

    print(
        "Trainable parameters:",
        trainable_count
    )

    print(
        "Frozen parameters:",
        total_count - trainable_count
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    step = 0

    while step < TRAIN_STEPS:

        for images in loader:

            if step >= TRAIN_STEPS:
                break

            # ------------------------------------------------
            # Convert PIL images to tensors
            # ------------------------------------------------

            normal_images = images.to(device)

            batch_size = normal_images.shape[0]

            # ------------------------------------------------
            # Create synthetic anomalies
            # ------------------------------------------------

            labels = torch.zeros(
                batch_size,
                device=device
            )

            masks = torch.zeros(
                batch_size,
                1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                device=device
            )

            input_images = normal_images.clone()

            for i in range(batch_size):

                if random.random() < ANOMALY_PROBABILITY:

                    anomalous_image, anomaly_mask = (
                        create_synthetic_anomaly(
                            normal_images[i:i + 1]
                        )
                    )

                    input_images[i] = (
                        anomalous_image[0]
                    )

                    masks[i] = (
                        anomaly_mask[0]
                    )

                    labels[i] = 1.0

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            anomaly_map, image_logit = model(
                input_images
            )

            # ------------------------------------------------
            # Image-level loss
            # ------------------------------------------------

            image_loss = binary_focal_loss(
                image_logit,
                labels
            )

            # ------------------------------------------------
            # Pixel-level loss
            # ------------------------------------------------

            mask_loss = segmentation_loss(
                anomaly_map,
                masks
            )

            # ------------------------------------------------
            # Combined loss
            # ------------------------------------------------

            loss = (
                image_loss
                +
                mask_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            step += 1

            # ------------------------------------------------
            # Monitoring
            # ------------------------------------------------

            with torch.no_grad():

                scores = torch.sigmoid(
                    image_logit
                )

                mean_score = scores.mean().item()

                anomaly_count = int(
                    labels.sum().item()
                )

            if (
                step == 1
                or step % 10 == 0
                or step == TRAIN_STEPS
            ):

                print(
                    f"Step {step:04d}/{TRAIN_STEPS} "
                    f"| Loss {loss.item():.4f} "
                    f"| ImageLoss {image_loss.item():.4f} "
                    f"| MaskLoss {mask_loss.item():.4f} "
                    f"| MeanScore {mean_score:.4f} "
                    f"| Anomalies "
                    f"{anomaly_count}/{batch_size}"
                )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "decoder_state_dict":
            model.decoder.state_dict(),

        "predictor_state_dict":
            model.predictor.state_dict(),

        "config": {

            "model": "DINOv2",

            "type": "baseline",

            "image_size": IMAGE_SIZE,

            "train_steps": TRAIN_STEPS,

            "batch_size": BATCH_SIZE,

            "learning_rate": LR,

            "weight_decay": WEIGHT_DECAY,

            "seed": SEED
        }
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )

    print()
    print(
        "Checkpoint saved:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()