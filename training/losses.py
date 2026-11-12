import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return (1.0 - dice).mean()


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return loss.mean()


def anomaly_loss(predictions, targets):
    """
    Combined Focal + Soft Dice loss for medical & industrial anomaly segmentation.
    """
    if predictions.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.float(),
            size=predictions.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    targets = targets.float()
    return 2.0 * focal_loss(predictions, targets) + soft_dice_loss(predictions, targets)