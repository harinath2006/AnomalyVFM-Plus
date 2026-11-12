import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw
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
    "experiments/plus_v2"
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

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ============================================================
# SYNTHETIC ANOMALY DATASET
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
            [
                path
                for path in self.image_dir.glob("*")
                if path.suffix.lower()
                in [".png", ".jpg", ".jpeg"]
            ]
        )

        if not self.images:

            raise RuntimeError(
                f"No training images found: {self.image_dir}"
            )

    def __len__(self):

        return len(self.images)

    # ========================================================
    # CREATE RANDOM DEFECT MASK
    # ========================================================

    def create_mask(self):

        size = self.image_size

        mask_image = Image.new(
            "L",
            (size, size),
            0,
        )

        draw = ImageDraw.Draw(mask_image)

        shape_type = random.choice(
            [
                "ellipse",
                "rectangle",
                "polygon",
            ]
        )

        x1 = random.randint(20, 110)

        y1 = random.randint(20, 110)

        x2 = min(
            size - 10,
            x1 + random.randint(25, 80),
        )

        y2 = min(
            size - 10,
            y1 + random.randint(25, 80),
        )

        if shape_type == "ellipse":

            draw.ellipse(
                (x1, y1, x2, y2),
                fill=255,
            )

        elif shape_type == "rectangle":

            draw.rectangle(
                (x1, y1, x2, y2),
                fill=255,
            )

        else:

            points = [
                (
                    random.randint(x1, x2),
                    random.randint(y1, y2),
                )
                for _ in range(7)
            ]

            draw.polygon(
                points,
                fill=255,
            )

        mask = np.asarray(
            mask_image,
            dtype=np.float32,
        ) / 255.0

        return mask

    # ========================================================
    # GENERATE SYNTHETIC ANOMALY
    # ========================================================

    def create_anomaly(
        self,
        image,
        mask,
    ):

        original = np.asarray(
            image,
            dtype=np.float32,
        ).copy()

        result = original.copy()

        height, width, channels = original.shape

        mode = random.choice(
            [
                "noise",
                "color",
                "texture",
                "contrast",
            ]
        )

        # ----------------------------------------------------
        # TYPE 1: RANDOM NOISE
        # ----------------------------------------------------

        if mode == "noise":

            noise = np.random.normal(
                loc=0,
                scale=random.uniform(25, 65),
                size=original.shape,
            )

            anomaly = original + noise

        # ----------------------------------------------------
        # TYPE 2: COLOR DISTORTION
        # ----------------------------------------------------

        elif mode == "color":

            factors = np.random.uniform(
                0.45,
                1.7,
                size=(1, 1, channels),
            )

            offsets = np.random.uniform(
                -35,
                35,
                size=(1, 1, channels),
            )

            anomaly = (
                original * factors
                + offsets
            )

        # ----------------------------------------------------
        # TYPE 3: TEXTURE / PATCH REPLACEMENT
        # ----------------------------------------------------

        elif mode == "texture":

            shift_x = random.randint(
                15,
                width - 15,
            )

            shift_y = random.randint(
                15,
                height - 15,
            )

            anomaly = np.roll(
                original,
                shift=(
                    shift_y,
                    shift_x,
                ),
                axis=(0, 1),
            )

            noise = np.random.normal(
                0,
                15,
                size=original.shape,
            )

            anomaly = anomaly + noise

        # ----------------------------------------------------
        # TYPE 4: CONTRAST DISTORTION
        # ----------------------------------------------------

        else:

            factor = random.uniform(
                0.3,
                2.0,
            )

            anomaly = (
                original * factor
                + random.uniform(-30, 30)
            )

        anomaly = np.clip(
            anomaly,
            0,
            255,
        )

        # ----------------------------------------------------
        # APPLY ANOMALY ONLY INSIDE MASK
        # ----------------------------------------------------

        mask_3d = mask[:, :, None]

        result = (
            original * (1.0 - mask_3d)
            +
            anomaly * mask_3d
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

        return Image.fromarray(result)

    # ========================================================
    # DATASET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path = self.images[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        # Resize original image.
        image = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BICUBIC,
        )

        # 50% normal, 50% synthetic anomaly.
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
        # ALIGN MASK WITH DINOv2 PROCESSOR
        # ----------------------------------------------------

        mask_tensor = torch.from_numpy(
            mask.copy()
        ).float().unsqueeze(0)

        mask_tensor = align_mask_to_processor(
            mask_tensor,
            self.processor,
        )

        mask_tensor = (
            mask_tensor > 0
        ).float()

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
# COLLATE FUNCTION
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
# FOCAL LOSS
# ============================================================

def focal_loss(
    logits,
    targets,
    alpha=0.75,
    gamma=2.0,
):

    logits = logits.reshape(-1)

    targets = targets.reshape(-1)

    if logits.shape != targets.shape:

        raise ValueError(
            f"Logit/target mismatch: "
            f"{logits.shape} vs {targets.shape}"
        )

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
        (1 - probabilities) * (1 - targets)
    )

    alpha_t = (
        alpha * targets
        +
        (1 - alpha) * (1 - targets)
    )

    loss = (
        alpha_t
        * (1 - p_t).pow(gamma)
        * bce
    )

    return loss.mean()


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(
        start_dim=1
    )

    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        +
        targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return (
        1.0 - dice
    ).mean()


# ============================================================
# SEGMENTATION LOSS
# ============================================================

def segmentation_loss(
    anomaly_logits,
    masks,
):

    focal = focal_loss(
        anomaly_logits,
        masks,
        alpha=0.75,
    )

    dice = dice_loss(
        anomaly_logits,
        masks,
    )

    return (
        focal + dice
    )


# ============================================================
# SAVE CHECKPOINT
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
            "v2",

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
    print("ANOMALYVFM+ V2 TRAINING")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Training steps:", TRAIN_STEPS)
    print("Learning rate:", LEARNING_RATE)

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print()
    print("Loading AnomalyVFM+...")

    model = AnomalyVFMPlus(
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # TRAINING MODE
    # --------------------------------------------------------

    model.train()

    # Keep pretrained DINOv2 in evaluation mode.
    # LoRA parameters still receive gradients.
    model.foundation_model.eval()

    # --------------------------------------------------------
    # VERIFY TRAINABLE PARAMETERS
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

    frozen_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    if trainable_count == 0:

        raise RuntimeError(
            "No trainable parameters found."
        )

    # Verify that pretrained non-LoRA weights remain frozen.
    for name, parameter in model.foundation_model.named_parameters():

        if (
            "lora_A" not in name
            and "lora_B" not in name
            and parameter.requires_grad
        ):

            raise RuntimeError(
                f"Unexpected trainable backbone parameter: {name}"
            )

    print()
    print("Trainable parameters:", trainable_count)
    print("Frozen parameters:", frozen_count)

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

    print("Training images:", len(dataset))

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

        masks = masks.to(DEVICE)

        labels = labels.to(DEVICE)

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
        # IMAGE LOSS
        # ----------------------------------------------------

        image_logits = image_logits.reshape(-1)

        image_loss = focal_loss(
            image_logits,
            labels,
            alpha=0.5,
        )

        # ----------------------------------------------------
        # PIXEL LOSS
        # ----------------------------------------------------

        masks_small = F.interpolate(
            masks,
            size=anomaly_map.shape[-2:],
            mode="nearest",
        )

        pixel_loss = segmentation_loss(
            anomaly_map,
            masks_small,
        )

        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            image_loss + pixel_loss
        )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
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

                image_scores = torch.sigmoid(
                    image_logits
                )

                mean_score = (
                    image_scores.mean().item()
                )

                anomaly_count = int(
                    labels.sum().item()
                )

            print(
                f"Step {step:03d}/{TRAIN_STEPS} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Image Loss: {image_loss.item():.4f} | "
                f"Pixel Loss: {pixel_loss.item():.4f} | "
                f"Mean Score: {mean_score:.4f} | "
                f"Anomalies: {anomaly_count}/{BATCH_SIZE}"
            )

        # ----------------------------------------------------
        # PERIODIC CHECKPOINT
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
    print("ANOMALYVFM+ V2 TRAINING COMPLETE")
    print("=" * 70)

    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":

    main()