import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

from models.anomaly_vfm_plus import AnomalyVFMPlus


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(
    "data/raw/mvtec/bottle/train/good"
)

CHECKPOINT_DIR = Path(
    "experiments/plus"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "checkpoint.pt"
)

IMAGE_SIZE = 224

BATCH_SIZE = 2

TRAIN_STEPS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

SEED = 12


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class SyntheticBottleDataset(Dataset):

    def __init__(
        self,
        image_dir,
        image_size=224,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def create_mask(self):

        mask = np.zeros(
            (
                self.image_size,
                self.image_size,
            ),
            dtype=np.float32,
        )

        cx = random.randint(
            40,
            self.image_size - 40,
        )

        cy = random.randint(
            40,
            self.image_size - 40,
        )

        rx = random.randint(15, 45)
        ry = random.randint(15, 45)

        y, x = np.ogrid[
            :self.image_size,
            :self.image_size,
        ]

        ellipse = (
            ((x - cx) / rx) ** 2
            +
            ((y - cy) / ry) ** 2
            <= 1
        )

        mask[ellipse] = 1.0

        mask_image = Image.fromarray(
            (mask * 255).astype(np.uint8)
        )

        mask_image = mask_image.filter(
            ImageFilter.GaussianBlur(
                radius=1
            )
        )

        mask = (
            np.asarray(mask_image)
            .astype(np.float32)
            / 255.0
        )

        return mask

    def create_anomaly(
        self,
        image,
        mask,
    ):

        image_array = np.asarray(
            image
        ).astype(np.float32)

        noise = np.random.normal(
            loc=0.0,
            scale=45.0,
            size=image_array.shape,
        )

        anomaly = image_array + noise

        # Strong color alteration
        anomaly[:, :, 0] *= 1.35
        anomaly[:, :, 1] *= 0.65
        anomaly[:, :, 2] *= 0.65

        anomaly = np.clip(
            anomaly,
            0,
            255,
        )

        mask_3 = mask[:, :, None]

        result = (
            image_array * (1.0 - mask_3)
            +
            anomaly * mask_3
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(result)

    def __getitem__(self, index):

        image_path = self.images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            (
                self.image_size,
                self.image_size,
            )
        )

        # 50% normal / 50% synthetic anomaly
        is_anomaly = (
            random.random() < 0.5
        )

        if is_anomaly:

            mask = self.create_mask()

            image = self.create_anomaly(
                image,
                mask,
            )

        else:

            mask = np.zeros(
                (
                    self.image_size,
                    self.image_size,
                ),
                dtype=np.float32,
            )

        mask = torch.tensor(
            mask,
            dtype=torch.float32,
        ).unsqueeze(0)

        label = torch.tensor(
            float(is_anomaly),
            dtype=torch.float32,
        )

        return (
            image,
            mask,
            label,
        )


# ============================================================
# FOCAL LOSS
# ============================================================

def binary_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    probabilities = torch.sigmoid(
        logits
    )

    p_t = (
        probabilities * targets
        +
        (1.0 - probabilities)
        * (1.0 - targets)
    )

    alpha_t = (
        alpha * targets
        +
        (1.0 - alpha)
        * (1.0 - targets)
    )

    loss = (
        alpha_t
        * (1.0 - p_t) ** gamma
        * bce
    )

    return loss.mean()


# ============================================================
# MASK LOSS
# ============================================================

def mask_loss(
    mask_logits,
    mask_target,
):

    # Focal loss
    focal = binary_focal_loss(
        mask_logits,
        mask_target,
    )

    # L1 loss
    probabilities = torch.sigmoid(
        mask_logits
    )

    l1 = F.l1_loss(
        probabilities.clamp(
            0.1,
            0.9,
        ),
        mask_target,
    )

    return (
        5.0 * focal
        +
        l1
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "step":
                step,

            "image_size":
                IMAGE_SIZE,

            "lora_rank":
                4,

            "lora_alpha":
                8,
        },
        CHECKPOINT_PATH,
    )

    print()
    print(
        "Checkpoint saved:",
        CHECKPOINT_PATH,
    )


# ============================================================
# TRAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANOMALYVFM+ TRAINING")
    print("=" * 70)

    print(
        "Dataset:",
        DATA_DIR,
    )

    print(
        "Image size:",
        IMAGE_SIZE,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Training steps:",
        TRAIN_STEPS,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SyntheticBottleDataset(
        DATA_DIR,
        IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            [item[0] for item in batch],

            torch.stack(
                [item[1] for item in batch]
            ),

            torch.stack(
                [item[2] for item in batch]
            ),
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating AnomalyVFM+ model...")

    model = AnomalyVFMPlus(
        lora_rank=4,
        lora_alpha=8,
    )

    model.train()

    # --------------------------------------------------------
    # Trainable parameters
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

    print(
        "Trainable parameters:",
        trainable_count,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    data_iterator = iter(loader)

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        try:

            images, masks, labels = next(
                data_iterator
            )

        except StopIteration:

            data_iterator = iter(loader)

            images, masks, labels = next(
                data_iterator
            )

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        anomaly_map, image_logits = (
            model.forward_with_score(
                images
            )
        )

        # ----------------------------------------------------
        # Match mask resolution
        # ----------------------------------------------------

        if (
            anomaly_map.shape[-2:]
            !=
            masks.shape[-2:]
        ):

            masks = F.interpolate(
                masks,
                size=anomaly_map.shape[-2:],
                mode="nearest",
            )

        # ----------------------------------------------------
        # Image-level loss
        # ----------------------------------------------------

        image_loss = binary_focal_loss(
            image_logits,
            labels,
        )

        # ----------------------------------------------------
        # Pixel-level loss
        # ----------------------------------------------------

        segmentation_loss = mask_loss(
            anomaly_map,
            masks,
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            image_loss
            +
            segmentation_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step == 1
            or step % 10 == 0
        ):

            with torch.no_grad():

                scores = torch.sigmoid(
                    image_logits
                )

                mean_score = (
                    scores.mean().item()
                )

                anomaly_count = int(
                    labels.sum().item()
                )

            print(
                f"Step {step:04d}/{TRAIN_STEPS} | "
                f"Loss {loss.item():.4f} | "
                f"ImageLoss {image_loss.item():.4f} | "
                f"MaskLoss {segmentation_loss.item():.4f} | "
                f"MeanScore {mean_score:.4f} | "
                f"Anomalies {anomaly_count}/{BATCH_SIZE}"
            )

        # ----------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------

        if step == TRAIN_STEPS:

            save_checkpoint(
                model,
                optimizer,
                step,
            )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()