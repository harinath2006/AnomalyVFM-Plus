import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

from models.anomaly_vfm_plus import AnomalyVFMPlus
from data.processor_alignment import align_mask_to_processor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(
    "data/raw/mvtec/bottle/train/good"
)

CHECKPOINT_DIR = Path(
    "experiments/plus_v3a"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "checkpoint.pt"
)

IMAGE_SIZE = 224

BATCH_SIZE = 2

TRAIN_STEPS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LORA_RANK = 4

LORA_ALPHA = 8

SEED = 12

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# SYNTHETIC BOTTLE DATASET
# ============================================================

class SyntheticBottleDataset(Dataset):

    def __init__(
        self,
        image_dir,
        processor,
        image_size=224,
    ):

        self.image_dir = Path(image_dir)

        self.processor = processor

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

    # ========================================================
    # CREATE ELLIPTICAL MASK
    # ========================================================

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

    # ========================================================
    # CREATE SYNTHETIC ANOMALY
    # ========================================================

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

        # Strong colour alteration.
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

    # ========================================================
    # LOAD SAMPLE
    # ========================================================

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

        # Same 50/50 sampling strategy as V1.
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

        # ----------------------------------------------------
        # CORRECTED PROCESSOR ALIGNMENT
        # ----------------------------------------------------

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).float().unsqueeze(0)

        mask_tensor = align_mask_to_processor(
            mask_tensor,
            self.processor,
        )

        label = torch.tensor(
            float(is_anomaly),
            dtype=torch.float32,
        )

        return (
            image,
            mask_tensor,
            label,
        )


# ============================================================
# BATCH COLLATION
# ============================================================

def collate_batch(batch):

    images = [
        item[0]
        for item in batch
    ]

    masks = torch.stack(
        [
            item[1]
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            item[2]
            for item in batch
        ]
    )

    return images, masks, labels


# ============================================================
# ORIGINAL V1 FOCAL LOSS
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
        (1.0 - alpha) * (1.0 - targets)
    )

    loss = (
        alpha_t
        * (1.0 - p_t) ** gamma
        * bce
    )

    return loss.mean()


# ============================================================
# ORIGINAL V1 MASK LOSS
# ============================================================

def mask_loss(
    mask_logits,
    mask_target,
):

    focal = binary_focal_loss(
        mask_logits,
        mask_target,
    )

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
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
):

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "step":
            step,

        "loss":
            float(loss),

        "image_size":
            IMAGE_SIZE,

        "lora_rank":
            LORA_RANK,

        "lora_alpha":
            LORA_ALPHA,

        "seed":
            SEED,

        "architecture":
            "AnomalyVFMPlus",

        "training_version":
            "v3a",

        "mask_alignment":
            "dinov2_processor_resize_center_crop",

    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    print(
        f"Checkpoint saved at step {step}: "
        f"{CHECKPOINT_PATH}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("ANOMALYVFM+ V3-A TRAINING")
    print("CONTROLLED MASK ALIGNMENT EXPERIMENT")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Training steps:", TRAIN_STEPS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading AnomalyVFM+...")

    model = AnomalyVFMPlus(
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    )

    model.to(DEVICE)

    # Keep the same training-mode behaviour as V1.
    model.train()

    # --------------------------------------------------------
    # TRAINABLE PARAMETERS
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

    if trainable_count == 0:

        raise RuntimeError(
            "No trainable parameters found."
        )

    print(
        "Trainable parameters:",
        trainable_count,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Preparing training dataset...")

    dataset = SyntheticBottleDataset(
        image_dir=DATA_DIR,
        processor=model.foundation.processor,
        image_size=IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(
        "Training images:",
        len(dataset),
    )

    data_iterator = iter(loader)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    print()
    print("Starting training...")
    print("-" * 70)

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

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        anomaly_map, image_logits = (
            model.forward_with_score(
                images
            )
        )

        # ----------------------------------------------------
        # MASK RESOLUTION
        # ----------------------------------------------------

        masks = F.interpolate(
            masks,
            size=anomaly_map.shape[-2:],
            mode="nearest",
        )

        # ----------------------------------------------------
        # IMAGE LOSS
        # ----------------------------------------------------

        image_loss = binary_focal_loss(
            image_logits,
            labels,
        )

        # ----------------------------------------------------
        # PIXEL LOSS
        # ----------------------------------------------------

        segmentation_loss = mask_loss(
            anomaly_map,
            masks,
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            image_loss
            +
            segmentation_loss
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # LOGGING
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
                f"Step {step:03d}/{TRAIN_STEPS} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Image Loss: {image_loss.item():.4f} | "
                f"Pixel Loss: {segmentation_loss.item():.4f} | "
                f"Mean Score: {mean_score:.4f} | "
                f"Anomalies: {anomaly_count}/{BATCH_SIZE}"
            )

        # ----------------------------------------------------
        # PERIODIC CHECKPOINTS
        # ----------------------------------------------------

        if (
            step % 25 == 0
            or step == TRAIN_STEPS
        ):

            save_checkpoint(
                model,
                optimizer,
                step,
                loss.item(),
            )

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANOMALYVFM+ V3-A TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":

    main()