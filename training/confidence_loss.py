import torch
import torch.nn.functional as F


def confidence_weighted_loss(
    predictions,
    targets,
    confidence_map
):
    """
    Confidence-weighted binary cross entropy.

    predictions:
        [B, 1, H, W] raw logits

    targets:
        [B, 1, H, W] ground-truth anomaly mask

    confidence_map:
        [B, 1, H, W] values in [0, 1]
    """

    # Make sure target and confidence map match prediction size.
    if targets.shape[-2:] != predictions.shape[-2:]:
        targets = F.interpolate(
            targets.float(),
            size=predictions.shape[-2:],
            mode="nearest"
        )

    if confidence_map.shape[-2:] != predictions.shape[-2:]:
        confidence_map = F.interpolate(
            confidence_map.float(),
            size=predictions.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    targets = targets.float()
    confidence_map = confidence_map.float()

    # Keep weights away from zero.
    # This prevents uncertain regions from being completely ignored.
    weights = 0.25 + 0.75 * confidence_map

    # Pixel-wise BCE.
    pixel_loss = F.binary_cross_entropy_with_logits(
        predictions,
        targets,
        reduction="none"
    )

    # Apply confidence weighting.
    weighted_loss = pixel_loss * weights

    # Normalize by total weight.
    loss = weighted_loss.sum() / (
        weights.sum() + 1e-8
    )

    return loss


if __name__ == "__main__":
    print("=== Confidence Weighted Loss Test ===")

    predictions = torch.randn(
        2, 1, 224, 224,
        requires_grad=True
    )

    targets = torch.zeros(
        2, 1, 224, 224
    )

    targets[:, :, 80:140, 80:140] = 1.0

    confidence_map = torch.rand(
        2, 1, 224, 224
    )

    loss = confidence_weighted_loss(
        predictions,
        targets,
        confidence_map
    )

    print("Prediction shape:", predictions.shape)
    print("Target shape:", targets.shape)
    print("Confidence shape:", confidence_map.shape)
    print("Loss:", loss.item())

    loss.backward()

    print(
        "Prediction gradient received:",
        predictions.grad is not None
    )

    print("Confidence weighted loss test: SUCCESS")