import torch


def refine_anomaly_map(
    anomaly_map,
    confidence_map
):
    """
    Refine anomaly predictions using confidence.

    Inputs:
        anomaly_map:    [B, 1, H, W]
        confidence_map: [B, 1, H, W]

    Output:
        refined_map:    [B, 1, H, W]
    """

    if anomaly_map.shape != confidence_map.shape:
        raise ValueError(
            "Anomaly map and confidence map "
            "must have the same shape."
        )

    confidence_map = confidence_map.clamp(
        0.0,
        1.0
    )

    refined_map = (
        anomaly_map * confidence_map
    )

    return refined_map


if __name__ == "__main__":

    print(
        "=== Confidence Refinement Test ==="
    )

    anomaly_map = torch.rand(
        2,
        1,
        224,
        224
    )

    confidence_map = torch.rand(
        2,
        1,
        224,
        224
    )

    refined_map = refine_anomaly_map(
        anomaly_map,
        confidence_map
    )

    print(
        "Anomaly map:",
        anomaly_map.shape
    )

    print(
        "Confidence map:",
        confidence_map.shape
    )

    print(
        "Refined map:",
        refined_map.shape
    )

    assert refined_map.shape == (
        2,
        1,
        224,
        224
    )

    assert torch.all(
        refined_map <= anomaly_map
    )

    assert torch.all(
        refined_map >= 0
    )

    print(
        "Confidence refinement test: SUCCESS"
    )