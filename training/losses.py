import torch
import torch.nn.functional as F


def anomaly_loss(predictions, targets):
    """
    Calculate pixel-level anomaly detection loss.

    Args:
        predictions:
            Raw decoder output (logits), shape [B, 1, H, W]

        targets:
            Ground-truth anomaly mask, shape [B, 1, H, W]
            Values should be 0 for normal and 1 for anomaly.

    Returns:
        Binary cross-entropy loss.
    """

    if predictions.shape[-2:] != targets.shape[-2:]:
        targets = F.interpolate(
            targets.float(),
            size=predictions.shape[-2:],
            mode="nearest"
        )

    loss = F.binary_cross_entropy_with_logits(
        predictions,
        targets.float()
    )

    return loss


if __name__ == "__main__":
    print("=== Anomaly Loss Test ===")

    # Simulated decoder output
    predictions = torch.randn(2, 1, 16, 16)

    # Simulated ground-truth masks
    targets = torch.randint(
        0,
        2,
        (2, 1, 16, 16)
    ).float()

    loss = anomaly_loss(predictions, targets)

    print("Prediction shape:", predictions.shape)
    print("Target shape:", targets.shape)
    print("Loss:", loss.item())
    print("Loss test: SUCCESS")