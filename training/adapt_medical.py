import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader

from models.anomaly_vfm_plus import AnomalyVFMPlus
from training.losses import anomaly_loss, focal_loss


BASE_CHECKPOINT = Path("experiments/plus/checkpoint.pt")
MEDICAL_CHECKPOINT = Path("experiments/plus/checkpoint_medical.pt")
BRAIN_MRI_ROOT = Path("data/raw/medical/brain_mri")

IMAGE_SIZE = 224
BATCH_SIZE = 2
ADAPT_STEPS = 100
LEARNING_RATE = 6e-5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(8)


class FewShotNormalBrainMRIDataset(Dataset):
    """
    Uses ONLY healthy 'notumor' MRI slices (zero tumor labels) to calibrate
    normal brain anatomy and synthesize multi-scale focal & infiltrative lesions
    across the entire intracranial cavity (including central/pituitary midline).
    """

    def __init__(self, root_dir: Path, image_size: int = 224, max_normal: int = 48):
        self.image_size = image_size
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

        train_notumor = root_dir / "Training" / "notumor"
        test_notumor = root_dir / "Testing" / "notumor"

        if train_notumor.is_dir():
            all_imgs = sorted(
                p for p in train_notumor.iterdir() if p.is_file() and p.suffix.lower() in exts
            )
            self.images = all_imgs[:max_normal]
            print(f"Using {len(self.images)} healthy slices from Training/notumor (zero test overlap).")
        elif test_notumor.is_dir():
            all_imgs = sorted(
                p for p in test_notumor.iterdir() if p.is_file() and p.suffix.lower() in exts
            )
            self.images = all_imgs[50 : 50 + max_normal] if len(all_imgs) > 55 else all_imgs[:16]
            print(f"Using {len(self.images)} healthy slices from Testing/notumor.")
        else:
            raise FileNotFoundError(f"Could not find notumor directory in {root_dir}")

    def __len__(self):
        return len(self.images)

    def _intracranial_mask(self, gray: np.ndarray) -> np.ndarray:
        """Keep full intracranial region including skull-base/pituitary midline."""
        thresh = gray.min() + 0.08 * (gray.max() - gray.min() + 1e-6)
        brain = (gray > thresh).astype(np.float32)
        pil_m = Image.fromarray((brain * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=2)
        )
        mask = (np.asarray(pil_m, dtype=np.float32) / 255.0 > 0.45).astype(np.float32)
        return mask if mask.sum() > 200 else brain

    def _synthesize_mri_lesion(self, image: Image.Image, brain_mask: np.ndarray):
        arr = np.asarray(image, dtype=np.float32)
        ys, xs = np.where(brain_mask > 0.5)

        # 30% chance to place a small focal nodule near the central/pituitary midline
        if random.random() < 0.30:
            cx = random.randint(int(self.image_size * 0.40), int(self.image_size * 0.60))
            cy = random.randint(int(self.image_size * 0.42), int(self.image_size * 0.68))
            rx = random.randint(6, 14)
            ry = random.randint(6, 14)
        elif len(xs) > 0:
            idx = random.randint(0, len(xs) - 1)
            cx, cy = int(xs[idx]), int(ys[idx])
            rx = random.randint(9, 28)
            ry = random.randint(9, 28)
        else:
            cx, cy = self.image_size // 2, self.image_size // 2
            rx, ry = 14, 14

        angle = random.uniform(0, np.pi)
        y, x = np.ogrid[: self.image_size, : self.image_size]
        dx, dy = x - cx, y - cy
        rot_x = dx * np.cos(angle) + dy * np.sin(angle)
        rot_y = -dx * np.sin(angle) + dy * np.cos(angle)
        dist_sq = (rot_x / rx) ** 2 + (rot_y / ry) ** 2

        core = (dist_sq <= 1.0).astype(np.float32) * brain_mask
        core_img = Image.fromarray((core * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=1.8)
        )
        soft_mask = np.asarray(core_img, dtype=np.float32) / 255.0

        lesion_type = random.choice(["hyperintense", "subtle_infiltrative", "ring_enhancing", "hypointense"])
        if lesion_type == "hyperintense":
            boost = random.uniform(55.0, 120.0)
            texture = np.random.normal(0.0, 10.0, size=(self.image_size, self.image_size, 1))
            altered = arr + boost + texture
        elif lesion_type == "subtle_infiltrative":
            # Mimics diffuse glioma with mild intensity shift + local texture smoothing
            blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=3.0)), dtype=np.float32)
            altered = blurred + random.uniform(25.0, 55.0)
        elif lesion_type == "ring_enhancing":
            ring = np.exp(-((dist_sq - 0.72) ** 2) / 0.16) * brain_mask
            altered = arr * 0.70 + (ring[:, :, None] * 125.0)
        else:
            altered = arr * random.uniform(0.35, 0.65)

        altered = np.clip(altered, 0, 255)
        m3 = soft_mask[:, :, None]
        blended = np.clip(arr * (1.0 - m3) + altered * m3, 0, 255).astype(np.uint8)
        binary_target = (soft_mask > 0.20).astype(np.float32)
        return Image.fromarray(blended), binary_target

    def __getitem__(self, index):
        path = self.images[index % len(self.images)]
        image = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        gray = np.asarray(image, dtype=np.float32).mean(axis=-1)
        brain_mask = self._intracranial_mask(gray)

        is_anomaly = random.random() < 0.55
        if is_anomaly:
            image, mask = self._synthesize_mri_lesion(image, brain_mask)
            if mask.sum() < 10:
                is_anomaly = False
                mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        return (
            image,
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            torch.tensor(float(is_anomaly), dtype=torch.float32),
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"FEW-SHOT FULL-BRAIN MRI CALIBRATION | Device: {device}")
    print("=" * 70)

    dataset = FewShotNormalBrainMRIDataset(BRAIN_MRI_ROOT, IMAGE_SIZE)
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
    )
    ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPT_STEPS, eta_min=1e-6)

    data_iter = iter(loader)
    for step in range(1, ADAPT_STEPS + 1):
        try:
            images, masks, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, masks, labels = next(data_iter)

        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        anomaly_map, image_logits = model.forward_with_score(images)
        if image_logits.ndim > labels.ndim:
            image_logits = image_logits.squeeze(-1)

        img_loss = focal_loss(image_logits, labels, alpha=0.55, gamma=2.0)
        seg_loss = anomaly_loss(anomaly_map, masks)
        loss = 1.2 * img_loss + 1.0 * seg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            with torch.no_grad():
                probs = torch.sigmoid(image_logits)
                norm_score = probs[labels == 0].mean().item() if (labels == 0).any() else -1.0
                anom_score = probs[labels == 1].mean().item() if (labels == 1).any() else -1.0
            print(
                f"Step {step:03d}/{ADAPT_STEPS} | Loss {loss.item():.4f} | "
                f"NormalProb {norm_score:.4f} | LesionProb {anom_score:.4f}"
            )

    MEDICAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": ADAPT_STEPS,
            "image_size": IMAGE_SIZE,
            "lora_rank": 8,
            "lora_alpha": 16,
            "adaptation": "few-shot normal-only Brain MRI calibration (full intracranial)",
        },
        MEDICAL_CHECKPOINT,
    )
    print(f"\nSaved medical-adapted checkpoint: {MEDICAL_CHECKPOINT}")


if __name__ == "__main__":
    main()