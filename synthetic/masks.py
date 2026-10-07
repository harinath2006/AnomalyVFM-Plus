import torch


def create_rectangle_mask(batch_size,height,width,top,left,rectangle_height,rectangle_width):
    """
    Create a rectangular synthetic anomaly mask.

    Output:
        [B, 1, H, W]

    0 = normal
    1 = anomaly
    """

    mask = torch.zeros(
        batch_size,
        1,
        height,
        width
    )

    bottom = min(top + rectangle_height, height)
    right = min(left + rectangle_width, width)

    mask[:, :, top:bottom, left:right] = 1.0

    return mask


if __name__ == "__main__":
    print("=== Synthetic Mask Test ===")

    mask = create_rectangle_mask(
        batch_size=1,
        height=224,
        width=224,
        top=80,
        left=60,
        rectangle_height=40,
        rectangle_width=70
    )

    print("Mask shape:", mask.shape)
    print("Minimum value:", mask.min().item())
    print("Maximum value:", mask.max().item())
    print("Anomaly pixels:", int(mask.sum().item()))
    print("Synthetic mask test: SUCCESS")