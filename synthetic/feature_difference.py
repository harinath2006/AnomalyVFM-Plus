import torch
import torch.nn.functional as F


def compute_feature_difference(
    normal_features,
    anomaly_features,
    output_size=(224, 224)
):
    """
    Compute feature-level difference between
    normal and anomalous images.

    Input:
        normal_features:  [B, C, Hf, Wf]
        anomaly_features: [B, C, Hf, Wf]

    Output:
        anomaly_map:  [B, 1, H, W]
        confidence_map: [B, 1, H, W]
    """

    if normal_features.shape != anomaly_features.shape:
        raise ValueError(
            "Normal and anomaly features must have the same shape."
        )

    # Normalize feature vectors across channels.
    normal_features = F.normalize(
        normal_features,
        p=2,
        dim=1
    )

    anomaly_features = F.normalize(
        anomaly_features,
        p=2,
        dim=1
    )

    # Cosine distance at every spatial location.
    similarity = (
        normal_features * anomaly_features
    ).sum(dim=1, keepdim=True)

    difference = 1.0 - similarity

    # Normalize difference to [0, 1].
    difference_min = difference.amin(
        dim=(2, 3),
        keepdim=True
    )

    difference_max = difference.amax(
        dim=(2, 3),
        keepdim=True
    )

    anomaly_map = (
        difference - difference_min
    ) / (
        difference_max - difference_min + 1e-8
    )

    # Resize from feature resolution to image resolution.
    anomaly_map = F.interpolate(
        anomaly_map,
        size=output_size,
        mode="bilinear",
        align_corners=False
    )

    # Confidence is high where feature difference is low.
    confidence_map = 1.0 - anomaly_map

    return anomaly_map, confidence_map


if __name__ == "__main__":
    print("=== Feature Difference Test ===")

    # Simulate DINOv2 feature maps.
    normal_features = torch.randn(
        2, 768, 16, 16
    )

    anomaly_features = normal_features.clone()

    # Introduce artificial feature differences.
    anomaly_features[:, :, 6:10, 6:10] += 1.0

    anomaly_map, confidence_map = compute_feature_difference(
        normal_features,
        anomaly_features
    )

    print(
        "Normal feature shape:",
        normal_features.shape
    )

    print(
        "Anomaly feature shape:",
        anomaly_features.shape
    )

    print(
        "Anomaly map shape:",
        anomaly_map.shape
    )

    print(
        "Confidence map shape:",
        confidence_map.shape
    )

    print(
        "Anomaly map min:",
        anomaly_map.min().item()
    )

    print(
        "Anomaly map max:",
        anomaly_map.max().item()
    )

    print(
        "Confidence min:",
        confidence_map.min().item()
    )

    print(
        "Confidence max:",
        confidence_map.max().item()
    )

    print("Feature difference test: SUCCESS")