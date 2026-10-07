import torch


def create_irregular_mask(
    batch_size,
    height,
    width,
    min_radius=10,
    max_radius=35,
    num_blobs=3
):
    """
    Create irregular synthetic anomaly masks.

    Output:
        [B, 1, H, W]

    Values:
        0 = normal
        1 = anomaly
    """

    masks = torch.zeros(
        batch_size,
        1,
        height,
        width
    )

    for batch in range(batch_size):

        for _ in range(num_blobs):

            center_y = torch.randint(
                min_radius,
                height - min_radius,
                (1,)
            ).item()

            center_x = torch.randint(
                min_radius,
                width - min_radius,
                (1,)
            ).item()

            radius_y = torch.randint(
                min_radius,
                max_radius + 1,
                (1,)
            ).item()

            radius_x = torch.randint(
                min_radius,
                max_radius + 1,
                (1,)
            ).item()

            y_start = max(
                0,
                center_y - radius_y
            )

            y_end = min(
                height,
                center_y + radius_y
            )

            x_start = max(
                0,
                center_x - radius_x
            )

            x_end = min(
                width,
                center_x + radius_x
            )

            yy, xx = torch.meshgrid(
                torch.arange(
                    y_start,
                    y_end
                ),
                torch.arange(
                    x_start,
                    x_end
                ),
                indexing="ij"
            )

            ellipse = (
                ((yy - center_y) / radius_y) ** 2
                +
                ((xx - center_x) / radius_x) ** 2
                <= 1
            )

            masks[
                batch,
                0,
                y_start:y_end,
                x_start:x_end
            ][ellipse] = 1.0

    return masks


if __name__ == "__main__":

    print("=== Irregular Anomaly Mask Test ===")

    mask = create_irregular_mask(
        batch_size=2,
        height=224,
        width=224
    )

    print("Mask shape:", mask.shape)
    print("Minimum:", mask.min().item())
    print("Maximum:", mask.max().item())

    anomaly_pixels = mask.sum().item()

    print("Anomaly pixels:", anomaly_pixels)

    print("Irregular mask test: SUCCESS")