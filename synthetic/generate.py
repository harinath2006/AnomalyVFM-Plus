import torch

from .masks import create_rectangle_mask


def apply_synthetic_anomaly(image, mask):
    """
    Apply a simple synthetic anomaly to an image.

    Args:
        image: [B, 3, H, W], values in [0, 1]
        mask:  [B, 1, H, W], 0 = normal, 1 = anomaly

    Returns:
        anomalous_image: [B, 3, H, W]
    """

    # Create a bright artificial defect
    anomaly_pattern = torch.ones_like(image)

    anomalous_image = (
        image * (1.0 - mask)
        + anomaly_pattern * mask
    )

    return anomalous_image


if __name__ == "__main__":
    print("=== Synthetic Anomaly Generation Test ===")

    image = torch.rand(1, 3, 224, 224)

    mask = create_rectangle_mask(
        batch_size=1,
        height=224,
        width=224,
        top=80,
        left=60,
        rectangle_height=40,
        rectangle_width=70
    )

    anomalous_image = apply_synthetic_anomaly(
        image,
        mask
    )

    print("Original image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Anomalous image shape:", anomalous_image.shape)

    print(
        "Original range:",
        image.min().item(),
        "to",
        image.max().item()
    )

    print(
        "Anomalous range:",
        anomalous_image.min().item(),
        "to",
        anomalous_image.max().item()
    )

    print("Synthetic anomaly test: SUCCESS")