import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

from models.anomaly_vfm_plus import AnomalyVFMPlus
from training.losses import anomaly_loss, focal_loss


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw/mvtec/bottle/train/good")
CHECKPOINT_DIR = Path("experiments/plus")
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 4
TRAIN_STEPS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 12


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DOMAIN-GENERALIZED SYNTHETIC ANOMALY DATASET
# ============================================================

class DomainGeneralizedAnomalyDataset(Dataset):
    """
    Generates realistic foreground-constrained structural & intensity anomalies:
      1. Hyper/hypo-intense focal blobs (mimics MRI lesions / calcifications)
      2. Smooth texture-warped / blurred / sharpened patches (mimics mucosal polyps & surface defects)
      3. Self-blended Poisson-like patch transplants (mimics structural tissue displacement)
    """

    def __init__(self, image_dir, image_size=224):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")
        self.images = []
        for ext in exts:
            self.images.extend(sorted(self.image_dir.rglob(ext)))

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def _foreground_mask(self, img_arr):
        gray = img_arr.mean(axis=-1)
        thresh = gray.min() + 0.08 * (gray.max() - gray.min() + 1e-6)
        fg = (gray > thresh).astype(np.float32)
        if fg.mean() < 0.05:
            fg = np.ones_like(fg)
        return fg

    def create_mask(self, fg_mask):
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        ys, xs = np.where(fg_mask > 0.5)

        if len(xs) > 0:
            idx = random.randint(0, len(xs) - 1)
            cx, cy = int(xs[idx]), int(ys[idx])
        else:
            cx = random.randint(40, self.image_size - 40)
            cy = random.randint(40, self.image_size - 40)

        # Multi-scale irregular ellipse / organic blob
        rx = random.randint(8, 36)
        ry = random.randint(8, 36)
        angle = random.uniform(0, np.pi)

        y, x = np.ogrid[: self.image_size, : self.image_size]
        dx = x - cx
        dy = y - cy
        rot_x = dx * np.cos(angle) + dy * np.sin(angle)
        rot_y = -dx * np.sin(angle) + dy * np.cos(angle)

        ellipse = ((rot_x / rx) ** 2 + (rot_y / ry) ** 2) <= 1.0
        mask[ellipse] = 1.0

        # Constrain anomaly strictly inside foreground tissue
        mask = mask * fg_mask

        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.5)
        )
        mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        return mask

    def create_anomaly(self, image, mask):
        arr = np.asarray(image, dtype=np.float32)
        mode = random.choice(["intensity", "texture_blur", "self_blend"])

        if mode == "intensity":
            # Hyperintense or hypointense local shift + subtle textural noise
            factor = random.choice([random.uniform(1.25, 1.65), random.uniform(0.45, 0.75)])
            noise = np.random.normal(0.0, 18.0, size=arr.shape)
            altered = arr * factor + noise

        elif mode == "texture_blur":
            # Smooth local tissue homogenization / edge disruption
            blurred = np.asarray(
                image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 5.0))),
                dtype=np.float32,
            )
            contrast = random.uniform(1.15, 1.45)
            mean_val = blurred.mean(axis=(0, 1), keepdims=True)
            altered = (blurred - mean_val) * contrast + mean_val

        else:
            # Self-blended shifted patch (CutPaste-style structural anomaly)
            shift_y = random.randint(-35, 35)
            shift_x = random.randint(-35, 35)
            shifted = np.roll(arr, shift=(shift_y, shift_x), axis=(0, 1))
            altered = 0.75 * shifted + 0.25 * arr

        altered = np.clip(altered, 0, 255)
        mask_3 = mask[:, :, None]
        result = arr * (1.0 - mask_3) + altered * mask_3
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        arr = np.asarray(image, dtype=np.float32)
        fg_mask = self._foreground_mask(arr)

        is_anomaly = random.random() < 0.5
        if is_anomaly:
            mask = self.create_mask(fg_mask)
            if mask.max() < 0.1:
                is_anomaly = False
                mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                image = self.create_anomaly(image, mask)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(float(is_anomaly), dtype=torch.float32)
        return image, mask_tensor, label_tensor


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, step):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "image_size": IMAGE_SIZE,
            "lora_rank": 8,
            "lora_alpha": 16,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nCheckpoint saved: {CHECKPOINT_PATH}")


# ============================================================
# TRAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"ANOMALYVFM+ UPGRADED TRAINING | Device: {device}")
    print("=" * 70)

    dataset = DomainGeneralizedAnomalyDataset(DATA_DIR, IMAGE_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.stack([item[1] for item in batch]),
            torch.stack([item[2] for item in batch]),
        ),
    )

    model = AnomalyVFMPlus(
        lora_rank=8,
        lora_alpha=16,
        output_size=IMAGE_SIZE,
        apply_roi_mask=True,
    ).to(device)
    model.train()

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameters:", sum(p.numel() for p in trainable_parameters))

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_STEPS, eta_min=1e-6
    )

    data_iterator = iter(loader)

    for step in range(1, TRAIN_STEPS + 1):
        try:
            images, masks, labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            images, masks, labels = next(data_iterator)

        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        anomaly_map, image_logits = model.forward_with_score(images)

        if image_logits.ndim > labels.ndim:
            image_logits = image_logits.squeeze(-1)

        image_loss = focal_loss(image_logits, labels, alpha=0.5, gamma=2.0)
        segmentation_loss = anomaly_loss(anomaly_map, masks)
        loss = 0.5 * image_loss + 1.5 * segmentation_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            with torch.no_grad():
                mean_score = torch.sigmoid(image_logits).mean().item()
                anomaly_count = int(labels.sum().item())
            print(
                f"Step {step:04d}/{TRAIN_STEPS} | "
                f"Loss {loss.item():.4f} | "
                f"ImgLoss {image_loss.item():.4f} | "
                f"SegLoss {segmentation_loss.item():.4f} | "
                f"MeanScore {mean_score:.4f} | "
                f"Anomalies {anomaly_count}/{len(images)}"
            )

        if step == TRAIN_STEPS:
            save_checkpoint(model, optimizer, step)

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()