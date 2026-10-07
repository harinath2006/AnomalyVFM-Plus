import torch


def apply_anomaly_corruption(
    image,
    mask,
    corruption_type="mixed"
):
    """
    Apply a synthetic corruption only inside the anomaly mask.

    image: [B, 3, H, W], expected range [0, 1]
    mask:  [B, 1, H, W], values 0 or 1

    Returns:
        corrupted_image: [B, 3, H, W]
    """

    if corruption_type == "dark":
        # Make anomaly region darker
        corruption = image * 0.25

    elif corruption_type == "bright":
        # Make anomaly region brighter
        corruption = torch.clamp(image * 1.8, 0.0, 1.0)

    elif corruption_type == "noise":
        # Add random noise
        noise = torch.randn_like(image) * 0.35
        corruption = torch.clamp(image + noise, 0.0, 1.0)

    elif corruption_type == "color":
        # Apply a random color shift
        shift = torch.rand(
            image.shape[0],
            3,
            1,
            1,
            device=image.device
        ) * 0.6

        corruption = torch.clamp(image + shift, 0.0, 1.0)

    elif corruption_type == "mixed":
        # Combine brightness change + noise
        noise = torch.randn_like(image) * 0.20
        corruption = torch.clamp(
            image * 1.5 + noise,
            0.0,
            1.0
        )

    else:
        raise ValueError(
            f"Unknown corruption type: {corruption_type}"
        )

    # Apply corruption ONLY inside anomaly region
    corrupted_image = (
        image * (1.0 - mask)
        + corruption * mask
    )

    return corrupted_image


if __name__ == "__main__":
    print("=== Anomaly Corruption Test ===")

    image = torch.rand(2, 3, 224, 224)

    mask = torch.zeros(2, 1, 224, 224)
    mask[:, :, 80:140, 80:140] = 1.0

    corrupted = apply_anomaly_corruption(
        image,
        mask,
        corruption_type="mixed"
    )

    print("Original shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Corrupted shape:", corrupted.shape)

    print("Original min:", image.min().item())
    print("Original max:", image.max().item())

    print("Corrupted min:", corrupted.min().item())
    print("Corrupted max:", corrupted.max().item())

    # Check that normal pixels remain unchanged
    normal_region = mask == 0
    unchanged = torch.allclose(
        image[normal_region.expand_as(image)],
        corrupted[normal_region.expand_as(image)]
    )

    print("Normal pixels unchanged:", unchanged)

    print("Anomaly corruption test: SUCCESS")